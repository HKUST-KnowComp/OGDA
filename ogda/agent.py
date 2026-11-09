"""One smolagents CodeAgent visit: a task, tools, an isolated worker and a checked final answer."""

from __future__ import annotations

import importlib.resources
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from smolagents import CodeAgent
from smolagents.memory import ActionStep
from smolagents.models import ChatMessage, MessageRole, Model, get_clean_message_list
from smolagents.monitoring import TokenUsage
from smolagents.utils import AgentGenerationError
import yaml

from .llm import BudgetExceeded, Generation, LLMError
from .sandbox import IMPORTS, IsolatedExecutor

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
                  "Send a shorter `code` call: keep the thought brief and split long work over several steps, "
                  "keeping intermediate results in variables.")
LAST_STEPS = 3  # From this many remaining steps, each observation says how many remain.
STEPS_NOTE = ("{left} of {total} steps remain. A visit that ends without an accepted final_answer has no answer: "
              "call final_answer with the best result you have before the steps run out.")
CODE_TOOL_NOTE = ("Run code by calling the `code` function: put your reasoning in `thought` and the Python source in "
                  "`code`. Call `code` in every step, also for final_answer(...).")


def load_model_profile(model_name: str) -> dict:
    """Read model-specific overrides, leaving other settings to CodeAgent's defaults."""
    directory = importlib.resources.files("ogda").joinpath("models")
    settings = {"tool_choice": "auto"}
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
            data_agent = profile.get("data_agent", {})
            if not isinstance(data_agent, dict):
                raise ValueError("Model profile data_agent must be a mapping")
            if "instructions" in data_agent:
                settings["data_agent_instructions"] = data_agent["instructions"]
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
    if settings["tool_choice"] not in TOOL_CHOICES:
        raise ValueError(f"Model profile code_agent.tool_choice must be one of {sorted(TOOL_CHOICES)}")
    if "data_agent_instructions" in settings:
        instructions = settings["data_agent_instructions"]
        if not isinstance(instructions, str) or not instructions.strip():
            raise ValueError("Model profile needs nonempty data_agent.instructions")
    return settings


@dataclass(frozen=True)
class AgentConfig:
    max_steps: int = 20
    cell_timeout_seconds: int = 30
    memory_mb: int = 4096
    max_output_chars: int = 20000


@dataclass
class Visit:
    answer: dict | None
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


class LLMModel(Model):
    """Send smolagents messages through the application LLM client for one role.

    With tool_choice "required" or "auto", the request offers one native function, `code(thought, code)`,
    and a call comes back to CodeAgent as a text code block. "none" sends no tools.
    """

    def __init__(self, llm, role: str, budget: Budget | None = None, *, tool_choice: str = "none",
                 tags: tuple[str, str] = ("<code>", "</code>")):
        super().__init__(model_id=llm.model_name(role))
        self.llm, self.role, self.budget = llm, role, budget
        self.tool_choice, self.tags = tool_choice, tags

    def generate(self, messages, stop_sequences=None, **kwargs) -> ChatMessage:
        if self.budget is not None:
            self.budget.check(self.llm)
        messages = get_clean_message_list(messages, role_conversions={
            MessageRole.TOOL_CALL: MessageRole.ASSISTANT, MessageRole.TOOL_RESPONSE: MessageRole.USER,
        }, flatten_messages_as_text=True)
        before = dict(self.llm.usage)
        if self.tool_choice == "none":
            text = code_text(self.llm.generate(self.role, messages, stop_sequences), self.tags)
        else:
            # A required call ends the response, so stop sequences could only cut the code short.
            stop = None if self.tool_choice == "required" else stop_sequences
            text = code_text(self.llm.generate(self.role, messages, stop, tools=[CODE_TOOL],
                                               tool_choice=self.tool_choice), self.tags)
        usage = {key: self.llm.usage[key] - before.get(key, 0) for key in ("input_tokens", "output_tokens")}
        return ChatMessage(role=MessageRole.ASSISTANT, content=text,
                           token_usage=TokenUsage(input_tokens=usage["input_tokens"], output_tokens=usage["output_tokens"]))


class _CodeAgent(CodeAgent):
    def _handle_max_steps_reached(self, task):
        return None  # No extra model call: a visit without an accepted final answer is incomplete.


def run_agent(llm, *, role: str, instructions: str, task: str, validate: Callable[[dict], Any],
              lake_root: Path, scratch_root: Path, artifact_root: Path, config: AgentConfig,
              tools: list = (), budget: Budget | None = None, lake: str = "write", workspace: Path | None = None,
              long_tools: bool = False, gate: Callable[[], Any] | None = None,
              on_files: Callable[[list[dict]], Any] | None = None,
              on_step: Callable[[ActionStep], Any] | None = None) -> Visit:
    """Run one visit. `validate` raises ValueError to send a correction back to the agent.

    `lake`, `workspace`, `long_tools`, `gate` and `on_files` set the worker's access and file log (see IsolatedExecutor).
    `on_step` can add observations to each completed action, including failed actions, before it enters memory.
    """
    accepted: dict = {}

    def check_answer(answer, memory, agent=None):
        if not isinstance(answer, dict):
            raise ValueError("final_answer needs one JSON object")
        validate(answer)
        accepted["answer"] = answer
        return True

    def steps_left(step, agent=None):
        if on_step is not None:
            on_step(step)
        left = config.max_steps - step.step_number
        if 0 < left <= LAST_STEPS:
            note = STEPS_NOTE.format(left=left, total=config.max_steps)
            step.observations = f"{step.observations}\n{note}" if step.observations else note

    profile = load_model_profile(llm.model_name(role))
    tags = profile.get("code_block_tags", ("<code>", "</code>"))
    model = LLMModel(llm, role, budget, tool_choice=profile["tool_choice"], tags=tags)
    if role == "data_agent" and "data_agent_instructions" in profile:
        instructions = f"{instructions}\n\n{profile['data_agent_instructions']}"
    if profile["tool_choice"] != "none":
        instructions = f"{CODE_TOOL_NOTE}\n\n{instructions}"
    templates = None
    if "system_prompt" in profile:
        templates = yaml.safe_load(importlib.resources.files("smolagents.prompts").joinpath("code_agent.yaml").read_text())
        templates["system_prompt"] = profile["system_prompt"]
    executor = IsolatedExecutor(lake_root, scratch_root, artifact_root, timeout=config.cell_timeout_seconds,
                                memory_mb=config.memory_mb, max_output_chars=config.max_output_chars,
                                lake=lake, workspace=workspace, long_tools=long_tools, gate=gate, on_files=on_files)
    agent = _CodeAgent(
        tools=list(tools), model=model, executor=executor, instructions=instructions, prompt_templates=templates,
        code_block_tags=tags,
        additional_authorized_imports=IMPORTS, max_steps=config.max_steps,
        max_print_outputs_length=config.max_output_chars, final_answer_checks=[check_answer], verbosity_level=-1,
        step_callbacks={ActionStep: steps_left},
    )
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
