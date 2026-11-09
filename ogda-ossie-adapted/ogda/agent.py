"""One smolagents CodeAgent visit: a task, tools, an isolated worker and a checked final answer."""

from __future__ import annotations

import importlib.resources
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from smolagents import CodeAgent
from smolagents.memory import ActionStep
from smolagents.models import ChatMessage, MessageRole, Model, get_clean_message_list
from smolagents.monitoring import TokenUsage
from smolagents.utils import AgentGenerationError
import yaml

from .llm import BudgetExceeded, Generation, LLMError
from .sandbox import IMPORTS, IsolatedExecutor
from .sandbox.executor import CellRecorder

TOOL_CHOICES = {"required", "auto", "none"}
CODE_TOOL = {"type": "function", "function": {
    "name": "code",
    "description": "Run Python code in the persistent Python session. Write your reasoning first, then the code.",
    "parameters": {
        "type": "object",
        "properties": {
            "thought": {"type": "string", "description": "Your reasoning for this step."},
            "code": {"type": "string", "description": "Python source to run. Call final_answer(...) here to finish."},
        },
        "required": ["thought", "code"],
        "additionalProperties": False,
    },
}}
TRUNCATED_NOTE = ("Your previous response was cut off at the output token limit, so none of it ran. "
                  "Send a shorter step: keep the thought brief and split long work over several steps, "
                  "keeping intermediate results in variables.")
LAST_STEPS = 3  # From this many remaining steps, each observation says how many remain.
STEPS_NOTE = ("{left} of {total} steps remain. Work that no accepted answer holds is lost when the steps run out: "
              "call final_answer with the best result you have before then.")
CODE_TOOL_NOTE = ("Run code by calling the `code` function: put your reasoning in `thought` and the Python source in "
                  "`code`. Call `code` in every step, also for final_answer(...).")
NO_CALL_NOTE = "Your response called no `code` function, so nothing ran. Call `code` with the Python for this step."


def load_model_profile(model_name: str) -> dict:
    """Read model-specific overrides, leaving other settings to CodeAgent's defaults."""
    directory = importlib.resources.files("ogda").joinpath("models")
    settings = {"tool_choice": "auto", "return_reasoning": False}
    name = model_name.rsplit("/", 1)[-1].casefold()
    for path in sorted(directory.iterdir(), key=lambda path: path.name):
        if not path.name.endswith(".yaml"):
            continue
        profile = yaml.safe_load(path.read_text(encoding="utf-8"))
        names = profile.get("model_names", [])
        if not isinstance(names, list) or any(not isinstance(alias, str) for alias in names):
            raise ValueError(f"{path.name}: model_names must be a list of strings")
        if name in {alias.rsplit("/", 1)[-1].casefold() for alias in names}:
            settings.update(profile["code_agent"])
            break
    if "system_prompt" in settings:
        prompt = settings["system_prompt"]
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Model profile needs a nonempty code_agent.system_prompt")
    if "code_block_tags" in settings:
        tags = settings["code_block_tags"]
        if not isinstance(tags, list) or len(tags) != 2 or any(not isinstance(tag, str) or not tag for tag in tags):
            raise ValueError("Model profile needs two nonempty code_agent.code_block_tags strings")
        settings["code_block_tags"] = tuple(tags)
    if not isinstance(settings["return_reasoning"], bool):
        raise ValueError("Model profile code_agent.return_reasoning must be true or false")
    if settings["tool_choice"] not in TOOL_CHOICES:
        raise ValueError(f"Model profile code_agent.tool_choice must be one of {sorted(TOOL_CHOICES)}")
    return settings


@dataclass(frozen=True)
class AgentConfig:
    max_steps: int = 20
    cell_timeout_seconds: int = 30
    memory_mb: int = 4096
    max_output_chars: int = 20000


@dataclass
class Visit:
    answer: Any  # None when no final_answer was accepted.
    steps: int
    artifact_root: Path

    @property
    def completed(self) -> bool:
        return self.answer is not None


@dataclass
class Budget:
    """A total token cap for all model calls of one construction or one Task, counted from `start` tokens."""

    max_tokens: int
    start: int = 0

    def check(self, llm) -> None:
        if llm.usage["total_tokens"] - self.start >= self.max_tokens:
            raise BudgetExceeded(f"Token budget of {self.max_tokens} is spent")


def code_text(generation: Generation, tags: tuple[str, str]) -> str:
    """Turn a `code` tool call into the text form CodeAgent parses. Text without a call passes through.

    A truncated response becomes a correction without code: CodeAgent would otherwise run its partial code,
    or an empty cell when nothing but hidden reasoning fit in the limit.
    """
    if generation.truncated:
        return TRUNCATED_NOTE
    call = next((c for c in generation.tool_calls if c["name"] == "code"), None)
    if call is None:
        return generation.text
    try:
        arguments = json.loads(call["arguments"])
        thought, source = arguments.get("thought", ""), arguments["code"]
    except (ValueError, TypeError, KeyError, AttributeError) as error:
        # No code block: CodeAgent reports a parsing error that includes this text, and the agent retries.
        return (f"The `code` call had invalid arguments ({type(error).__name__}: {error}). "
                "Call `code` again with a JSON object that has string fields `thought` and `code`.")
    text = "" if tags[0] in generation.text else generation.text.strip()  # Run only the call's code.
    lead = "\n".join(part for part in (text, f"Thought: {thought}".strip()) if part)
    return f"{lead}\n{tags[0]}\n{source}\n{tags[1]}"


def _message_text(message: ChatMessage) -> str:
    if isinstance(message.content, list):
        return "\n".join(part["text"] for part in message.content if part.get("type") == "text")
    return message.content or ""


def _key(text: str, tags: tuple[str, str]) -> str:
    """A step's text as CodeAgent keeps it: stripped, and ending with the closing tag, which it appends if missing."""
    text = text.strip()
    return text if text.endswith(tags[1]) else text + tags[1]


def _add(messages: list[dict], role: str, text: str) -> None:
    if messages and messages[-1]["role"] == role and "tool_calls" not in messages[-1]:
        messages[-1]["content"] += "\n" + text
    else:
        messages.append({"role": role, "content": text})


class LLMModel(Model):
    """Send smolagents messages through the application LLM client for one role.

    With tool_choice "required" or "auto", the request offers one native function, `code(thought, code)`. A call
    comes back to CodeAgent as a text code block, and earlier steps go to the model as native `code` calls with
    `tool` results, so it only ever sees the format it is asked for. "none" sends no tools and text history.
    """

    def __init__(self, llm, role: str, budget: Budget | None = None, *, tool_choice: str = "none",
                 tags: tuple[str, str] = ("<code>", "</code>"), return_reasoning: bool = False):
        super().__init__(model_id=llm.model_name(role))
        self.llm, self.role, self.budget = llm, role, budget
        self.tool_choice, self.tags, self.return_reasoning = tool_choice, tags, return_reasoning
        self.turns: dict[str, list[dict]] = {}  # Text given to CodeAgent -> the native turns behind it, in order.
        self.calls = 0

    def native_turn(self, generation: Generation, text: str) -> dict:
        """The native assistant turn behind `text`: a `code` call, and a note to send instead of a parse error."""
        if generation.truncated:
            return {"call": None, "content": "", "note": TRUNCATED_NOTE}
        call = next((c for c in generation.tool_calls if c["name"] == "code"), None)
        content = "" if self.tags[0] in generation.text else generation.text.strip()
        if call is not None:
            arguments = call["arguments"]
            try:
                json.loads(arguments)
            except ValueError:
                arguments = "{}"  # Only valid JSON can go back to the provider.
            note = None if self.tags[0] in text else text  # code_text gave a correction instead of code.
        else:
            # A text code block becomes the call it stands for, so the history holds only `code` calls.
            opening, closing = map(re.escape, self.tags)
            block = re.search(f"{opening}(.*?)(?:{closing}|$)", text, re.S)
            if block is None:
                return {"call": None, "content": text, "note": NO_CALL_NOTE}
            thought = text[:block.start()].strip().removeprefix("Thought:").strip()
            arguments = json.dumps({"thought": thought, "code": block.group(1).strip("\n")}, ensure_ascii=False)
            content, note = "", None
        self.calls += 1
        return {"call": {"id": f"call_{self.calls}", "type": "function",
                         "function": {"name": "code", "arguments": arguments}},
                "content": content, "note": note, "reasoning": generation.reasoning}

    def native_messages(self, messages) -> list[dict]:
        """CodeAgent's memory as native turns: each step is an assistant `code` call followed by its `tool` result."""
        native, turn, seen = [], None, Counter()
        for message in messages:
            message = message if isinstance(message, ChatMessage) else ChatMessage.from_dict(message)
            role, text = message.role, _message_text(message)
            if role == MessageRole.ASSISTANT:
                key = _key(text, self.tags)
                turns = self.turns.get(key, [])
                turn = turns[seen[key]] if seen[key] < len(turns) else None
                seen[key] += 1
                if turn is None:
                    _add(native, "assistant", text)
                elif turn["call"] is not None:
                    native.append({"role": "assistant", "content": turn["content"] or None, "tool_calls": [turn["call"]]})
                    if self.return_reasoning:  # A thinking model can require it, even when empty.
                        native[-1]["reasoning_content"] = turn["reasoning"]
                elif turn["content"]:
                    _add(native, "assistant", turn["content"])
            elif role == MessageRole.TOOL_CALL:
                if turn is None:  # CodeAgent's record of a call; the native call already stands for it.
                    _add(native, "assistant", text)
            elif role == MessageRole.TOOL_RESPONSE:
                # CodeAgent's error for a step without code tells the model to write a text code block; the note
                # says what went wrong with the call instead.
                error = text.startswith(("Error:", "Call id:"))
                result = turn["note"] if turn is not None and turn["note"] and error else text
                if turn is None or turn["call"] is None:
                    _add(native, "user", result)
                elif native[-1].get("tool_call_id") == turn["call"]["id"]:
                    native[-1]["content"] += "\n" + result
                else:
                    native.append({"role": "tool", "tool_call_id": turn["call"]["id"], "content": result})
            else:
                turn = None
                _add(native, "system" if role == MessageRole.SYSTEM else "user", text)
        return native

    def generate(self, messages, stop_sequences=None, **kwargs) -> ChatMessage:
        if self.budget is not None:
            self.budget.check(self.llm)
        before = dict(self.llm.usage)
        if self.tool_choice == "none":
            messages = get_clean_message_list(messages, role_conversions={
                MessageRole.TOOL_CALL: MessageRole.ASSISTANT, MessageRole.TOOL_RESPONSE: MessageRole.USER,
            }, flatten_messages_as_text=True)
            text = code_text(self.llm.generate(self.role, messages, stop_sequences), self.tags)
        else:
            # A required call ends the response, so stop sequences could only cut the code short.
            stop = None if self.tool_choice == "required" else stop_sequences
            generation = self.llm.generate(self.role, self.native_messages(messages), stop, tools=[CODE_TOOL],
                                           tool_choice=self.tool_choice)
            text = code_text(generation, self.tags)
            self.turns.setdefault(_key(text, self.tags), []).append(self.native_turn(generation, text))
        usage = {key: self.llm.usage[key] - before.get(key, 0) for key in ("input_tokens", "output_tokens")}
        return ChatMessage(role=MessageRole.ASSISTANT, content=text,
                           token_usage=TokenUsage(input_tokens=usage["input_tokens"], output_tokens=usage["output_tokens"]))


class _CodeAgent(CodeAgent):
    def _handle_max_steps_reached(self, task):
        return None  # No extra model call: a visit without an accepted final answer is incomplete.


def run_agent(llm, *, role: str, instructions: str, task: str, validate: Callable[[Any], Any],
              lake_root: Path, scratch_root: Path, artifact_root: Path, config: AgentConfig,
              tools: list = (), budget: Budget | None = None, lake: str = "write",
              workspace: Path | None = None, readable: Iterable[str | Path] = (),
              recorder: CellRecorder | None = None) -> Visit:
    """Run one visit. `validate` gets the final_answer value and raises ValueError to send a correction back;
    it may return a cleaned value, which the Visit keeps instead of the submitted one.

    `lake`, `workspace` and `readable` set the worker's access (see IsolatedExecutor). A `recorder` sees each cell
    and gets `current_step`, which returns the step the agent is running.
    """
    accepted: dict = {}

    def check_answer(answer, memory, agent=None):
        cleaned = validate(answer)
        accepted["answer"] = answer if cleaned is None else cleaned
        return True

    def steps_left(step, agent=None):
        left = config.max_steps - step.step_number
        if 0 < left <= LAST_STEPS:
            note = STEPS_NOTE.format(left=left, total=config.max_steps)
            step.observations = f"{step.observations}\n{note}" if step.observations else note

    profile = load_model_profile(llm.model_name(role))
    tags = profile.get("code_block_tags", ("<code>", "</code>"))
    model = LLMModel(llm, role, budget, tool_choice=profile["tool_choice"], tags=tags,
                     return_reasoning=profile["return_reasoning"])
    if profile["tool_choice"] != "none":
        instructions = f"{CODE_TOOL_NOTE}\n\n{instructions}"
    templates = None
    if "system_prompt" in profile:
        templates = yaml.safe_load(importlib.resources.files("smolagents.prompts").joinpath("code_agent.yaml").read_text())
        templates["system_prompt"] = profile["system_prompt"]
    executor = IsolatedExecutor(lake_root, scratch_root, artifact_root, timeout=config.cell_timeout_seconds,
                                memory_mb=config.memory_mb, max_output_chars=config.max_output_chars,
                                lake=lake, workspace=workspace, readable=readable, recorder=recorder)
    agent = _CodeAgent(
        tools=list(tools), model=model, executor=executor, instructions=instructions, prompt_templates=templates,
        code_block_tags=tags,
        additional_authorized_imports=IMPORTS, max_steps=config.max_steps,
        max_print_outputs_length=config.max_output_chars, final_answer_checks=[check_answer], verbosity_level=-1,
        step_callbacks={ActionStep: steps_left},
    )
    if recorder is not None:
        recorder.current_step = lambda: agent.step_number
    try:
        agent.run(task)
    except AgentGenerationError as error:
        # Model errors (budget, provider, transport) stop the construction; resume continues it.
        if isinstance(error.__cause__, LLMError):
            raise error.__cause__ from None
        raise
    finally:
        executor.cleanup()
        artifact_root.mkdir(parents=True, exist_ok=True)
        (artifact_root / "steps.json").write_text(json.dumps(agent.memory.get_succinct_steps(), indent=1, default=str))
    steps = sum(isinstance(step, ActionStep) for step in agent.memory.steps)
    return Visit(accepted.get("answer"), steps, artifact_root)
