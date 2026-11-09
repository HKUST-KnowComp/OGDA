"""Answer a Task with a finished Harness: OGDA, one CodeAgent that grounds the data it needs in the Ontology.

The task message holds the Task and then the Lake map. The agent gets data only through `ground`, which prints what a
Dataset, Field or File means and returns its data: a Dataset's rows as a DataFrame, a Field's as a Series, a File's
path; `search` prints the Fields whose words match. Its worker defines no LAKE_ROOT: it reads the Lake copy, which
stays read-only so the Harness never changes, and the table cache, and writes only the Task's Workspace. Code records
the agent's steps as the Task graph and checks the Task and the shape of the answer, never its content.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Callable

from smolagents import tool

from .agent import AgentConfig, Budget, run_agent
from .grounding import Grounding, lake_map
from .lineage import TaskGraph
from .llm import BudgetExceeded, LLMError
from .ontology import OntologyStore
from .ontology_search import NO_MATCH, OntologySearch
from .prompts import DATA_AGENT
from .sandbox.executor import ToolCorrection

ROLE = "data_agent"
TABLES = ".tables"    # The grounded Datasets as Parquet, in the Task scratch directory; the worker reads them.
WORKERS = ".workers"  # Worker temporary directories, beside the workspace in the Task scratch directory.
WORKSPACE = "workspace"


@dataclass(frozen=True)
class DataAgentConfig:
    max_tokens: int | None = None  # Optional total token cap per Task.
    agent: AgentConfig = field(default_factory=AgentConfig)  # The agent answers within these steps.

    @classmethod
    def from_dict(cls, data: dict | None) -> "DataAgentConfig":
        data = dict(data or {})
        agent = AgentConfig(**(data.pop("agent", None) or {}))
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown data_agent settings: {sorted(unknown)}")
        return cls(agent=agent, **data)


def check_task(task) -> dict:
    """A Task is {"id", "question", optional "answer_format"}."""
    if not isinstance(task, dict) or set(task) - {"id", "question", "answer_format"}:
        raise ValueError("A Task is a JSON object with id, question and optional answer_format")
    if not isinstance(task.get("id"), str) or not task["id"].strip():
        raise ValueError("A Task needs a nonempty string id")
    if not isinstance(task.get("question"), str) or not task["question"].strip():
        raise ValueError("A Task needs a nonempty string question")
    if not isinstance(task.get("answer_format", ""), str):
        raise ValueError("answer_format must be a string")
    return dict(task)


def serialized_string(value) -> bool:
    """A string such as '"Singapore"' that json.loads turns into another string."""
    if not isinstance(value, str) or len(value) < 2 or value[0] != '"' or value[-1] != '"':
        return False
    try:
        return isinstance(json.loads(value), str)
    except ValueError:
        return False


def unquote(answer):
    """Unwrap an answer string sent as JSON text (json.dumps("Bank Account")), alone or as list items, once."""
    if isinstance(answer, list):
        return [json.loads(value) if serialized_string(value) else value for value in answer]
    return json.loads(answer) if serialized_string(answer) else answer


def check_answer(answer, check: Callable[[Any], None] | None = None):
    """Return the final_answer value to save, or reject one that is missing, wrapped in {"answer": ...}, still
    JSON text after one unwrap, or the wrong shape.

    A string scorer compares '"Bank Account"' with its quotation marks, so a JSON-quoted string is unwrapped.
    """
    if answer is None:
        raise ValueError("Call final_answer(answer) with the answer itself, in the Task's answer_format.")
    if isinstance(answer, dict) and "answer" in answer and set(answer) <= {"answer", "explanation"}:
        raise ValueError('Pass the answer itself: final_answer(value), not final_answer({"answer": value}).')
    answer = unquote(answer)
    values = answer if isinstance(answer, list) else [answer]
    if quoted := next((value for value in values if serialized_string(value)), None):
        raise ValueError(f"The answer string {quoted} is JSON text with quotation marks around it. "
                         "Give the Python string itself, not json.dumps of it.")
    if check is not None:
        try:
            check(answer)
        except ValueError as error:
            raise ValueError(f"The answer does not have the required shape: {error}") from None
    return answer


def grounded_names(graph: dict) -> list[str]:
    """The names a Task graph grounded, in the order they were first grounded."""
    return [node["id"] for node in graph["nodes"] if node["kind"] != "derived"]


class TaskRun:
    """The state of one Task: its Harness and Lake map, what it grounds, its Task graph and the saved result."""

    def __init__(self, store: OntologyStore, llm, task: dict, config: DataAgentConfig, scratch_root: Path,
                 output_root: Path, check: Callable[[Any], None] | None = None):
        self.store, self.llm, self.task, self.config, self.check = store, llm, task, config, check
        self.scratch_root, self.output_root = Path(scratch_root).resolve(), Path(output_root).resolve()
        self.workspace, self.tables = self.scratch_root / WORKSPACE, self.scratch_root / TABLES
        self.log_root = self.output_root / "logs"
        harness = store.load()
        self.index = OntologySearch(harness)
        self.lake_map = lake_map(harness, store.lake_root, self.index)
        self.grounding = Grounding(harness, store.lake_root, self.tables, self.index,
                                   max_output_chars=config.agent.max_output_chars)
        self.graph = TaskGraph()
        self.budget = Budget(config.max_tokens, start=llm.usage["total_tokens"]) if config.max_tokens is not None else None
        self.steps: int | None = None
        self.started, self.usage_before = time.monotonic(), dict(llm.usage)

    def tools(self) -> list:
        @tool
        def ground(names: list) -> dict:
            """Ground tables, columns and Files: print what each name means and return its data, one value for one
            name and a tuple for several. The names may also be given as separate arguments:
            a, b = ground("x", "y.z").

            Args:
                names: Each a table of the Lake map ("x"), one of its columns ("x.y"), a Lake path or a pattern
                    such as "dir/*.csv".
            """
            names = list(names) if isinstance(names, (list, tuple)) else [names]
            started = time.monotonic()
            try:
                found = self.grounding.ground(names)
            except ValueError as error:
                # No names, a name that is not a string, an unknown name, a pattern that matches no File, or views
                # too long for one call: the agent corrects the call. A table whose File fails to read is grounded
                # as its File, and its view says why.
                raise ToolCorrection(str(error)) from None
            for node in found:
                self.graph.grounded(node.name, node.kind, node.files, node.rows)
            self.log("ground", names=names, rows=[node.rows for node in found],
                     seconds=round(time.monotonic() - started, 3))
            return {"__print__": "\n\n".join(node.view for node in found),
                    "__values__": [node.value for node in found], "__single__": len(found) == 1}

        @tool
        def search(words: str, limit: int = 15) -> dict:
            """Print the columns of all tables whose names and meanings match the words, best first, one line each:
            table.column, header text, datatype and meaning.

            Args:
                words: Words for the data you need, such as "fraud losses by state".
                limit: How many columns to print.
            """
            try:
                text = self.index.search(words, limit)
            except ValueError as error:  # A limit that is not a whole number: the agent corrects it.
                raise ToolCorrection(str(error)) from None
            printed = [] if text == NO_MATCH else [line.split("  ", 1)[0] for line in text.splitlines()]
            self.log("search", words=words, limit=limit, fields=printed)
            return {"__print__": text, "__values__": [], "__single__": True}

        return [ground, search]

    def check_answer(self, answer):
        try:
            cleaned = check_answer(answer, self.check)
        except ValueError as error:
            self.log("answer_rejected", error=str(error), answer=answer)
            raise
        if cleaned != answer:
            self.log("answer_unwrapped", submitted=answer, answer=cleaned)
        return cleaned

    def result(self, status: str, answer=None, stop_reason: str = "") -> dict:
        return {
            "task": self.task, "status": status, "stop_reason": stop_reason, "answer": answer,
            "graph": self.graph.to_json(), "steps": self.steps,
            "usage": {key: self.llm.usage[key] - self.usage_before.get(key, 0) for key in self.llm.usage},
            "wall_seconds": round(time.monotonic() - self.started, 3),
        }

    def save(self, status: str, answer=None, stop_reason: str = "") -> dict:
        result = self.result(status, answer, stop_reason)
        self.output_root.mkdir(parents=True, exist_ok=True)
        temporary = self.output_root / "result.json.tmp"
        temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        temporary.replace(self.output_root / "result.json")
        return result

    def log(self, event: str, **data) -> None:
        self.log_root.mkdir(parents=True, exist_ok=True)
        with (self.log_root / "data_agent.jsonl").open("a") as stream:
            stream.write(json.dumps({"event": event, "time": time.time(), "task_id": self.task["id"], **data},
                                    ensure_ascii=False, default=str) + "\n")


def answer(store: OntologyStore, llm, task: dict, config: DataAgentConfig, scratch_root: Path,
           output_root: Path, check: Callable[[Any], None] | None = None) -> dict:
    """Answer one Task. Save `result.json` in `output_root`, with logs in `output_root/logs`, and return it.

    `check` receives the agent's answer value and raises ValueError when its shape is wrong; the agent sees the
    error and can repair the answer, instead of a benchmark adapter rejecting it after the Task ends.
    """
    task = check_task(task)
    scratch_root, output_root = Path(scratch_root).resolve(), Path(output_root).resolve()
    if (output_root / "result.json").exists() or (scratch_root.is_dir() and any(scratch_root.iterdir())):
        raise FileExistsError("Use a new output and scratch directory for each Task")
    run = TaskRun(store, llm, task, config, scratch_root, output_root, check)
    # Made once and never replaced during the visit: Landlock grants the worker this directory as it is at its start.
    run.tables.mkdir(parents=True)
    run.log("task_start", task=task, model=llm.model_name(ROLE))
    agent_task = {"task": {key: task[key] for key in ("question", "answer_format") if key in task},
                  "max_steps": config.agent.max_steps, "workspace": str(run.workspace)}
    final, stop_reason = None, "answered"
    try:
        visit = run_agent(
            llm, role=ROLE, instructions=DATA_AGENT,
            task=json.dumps(agent_task, ensure_ascii=False) + "\n\n" + run.lake_map,
            validate=run.check_answer, tools=run.tools(), lake_root=store.lake_root,
            scratch_root=scratch_root / WORKERS, artifact_root=run.log_root / ROLE, config=config.agent,
            budget=run.budget, lake="ground", workspace=run.workspace, readable=[run.tables], recorder=run.graph)
        run.steps, final = visit.steps, visit.answer
        if final is None:
            stop_reason = "incomplete"
            run.log("incomplete", steps=visit.steps)
    except LLMError as error:  # Save what the Task has so far.
        stop_reason = "budget" if isinstance(error, BudgetExceeded) else "llm_error"
        run.log(stop_reason, message=f"{type(error).__name__}: {error}")
    result = run.save("incomplete" if final is None else "answered", final, stop_reason)
    run.log("task_end", status=result["status"], stop_reason=stop_reason, answer=result["answer"],
            steps=result["steps"], grounded=grounded_names(result["graph"]), usage=result["usage"],
            wall_seconds=result["wall_seconds"])
    return result
