"""Answer a Task with a finished Harness: one CodeAgent grounds a task graph in the Ontology and computes over it.

The agent searches the Harness with `search_ontology` and `get_schema` and adds the objects it needs to the task
graph with `ground`. Its worker reads the Lake copy, which stays read-only so the Harness never changes, but can
open only the Lake Files that the task graph makes readable, and it writes the Task's workspace. The worker's file
log turns each workspace File it writes into a derived node of the graph, and the accepted answer becomes the
`answer` node. Code checks references and the shape of the answer, never its content.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Callable

from smolagents import tool
from smolagents.memory import ActionStep

from .agent import AgentConfig, Budget, run_agent
from .claims import lake_path
from .llm import BudgetExceeded, LLMError
from .ontology import OntologyStore
from .prompts import DATA_AGENT
from .retrieval import retrieve, schema_detail

ROLE = "data_agent"
WORKERS = ".workers"  # Worker temporary directories, beside the workspace in the Task scratch directory.
WORKSPACE = "workspace"
LISTED = 20  # Tables and readable Files listed in one ground result; the rest are counted.


@dataclass(frozen=True)
class DataAgentConfig:
    max_tokens: int | None = None  # Optional total token cap per Task.
    agent: AgentConfig = field(default_factory=lambda: AgentConfig(max_steps=40))  # The agent answers within these steps.

    @classmethod
    def from_dict(cls, data: dict | None) -> "DataAgentConfig":
        data = dict(data or {})
        agent = AgentConfig(**{"max_steps": 40, **(data.pop("agent", None) or {})})
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


class TaskGraph:
    """The Ontology objects that one Task grounded, the Files it derived and the edges between them.

    Nodes are keyed by (kind, id), since a Schema and a Table may share an id; a node is {"in_ontology"}. An edge is
    (source, kind, target). A Lake File is readable when it is the File of a Table in the graph, a Documentation File
    of a Schema in the graph, or a File node.
    """

    def __init__(self, ontology, lake_root: Path):
        self.ontology, self.lake_root = ontology, Path(lake_root).resolve()
        self.fields = ontology.fields()
        self.schema_of = {f.id: s.id for s in ontology.schemas.values() for f in s.fields}
        self.nodes: dict[tuple[str, str], dict] = {}
        self.edges: list[tuple[str, str, str]] = []
        self.reads: list[str] = []  # Files read so far, in order: Lake paths and workspace/<path>.
        self.shown: set[str] = set()  # Experience ids already shown.

    def add(self, node: str, kind: str, in_ontology: bool = True) -> None:
        self.nodes.setdefault((kind, node), {"in_ontology": in_ontology})

    def of(self, kind: str) -> list[str]:
        return [node for k, node in self.nodes if k == kind]

    def edge(self, source: str, kind: str, target: str) -> None:
        if (source, kind, target) not in self.edges:
            self.edges.append((source, kind, target))

    def readable(self) -> set[str]:
        paths = {self.ontology.tables[n].path for n in self.of("table")}
        paths |= {p for n in self.of("schema") for p in self.ontology.schemas[n].documentation}
        return paths | set(self.of("file"))

    def gate(self) -> list[str]:
        """The absolute paths that the worker may open in the Lake copy."""
        return [str((self.lake_root / path).resolve()) for path in self.readable()]

    def schema(self, schema_id: str, tables: bool = False) -> None:
        self.add(schema_id, "schema")
        if tables:
            for table in self.ontology.tables.values():
                if table.schema_id == schema_id:
                    self.table(table.id)

    def table(self, table_id: str) -> None:
        table = self.ontology.tables[table_id]
        self.add(table_id, "table")
        self.schema(table.schema_id)
        self.edge(table_id, "table_of", table.schema_id)
        self.edge(table_id, "file_of", table.path)

    def field(self, field_id: str) -> None:
        self.add(field_id, "field")
        self.schema(self.schema_of[field_id])
        self.edge(field_id, "field_of", self.schema_of[field_id])

    def concept(self, concept_id: str) -> None:
        self.add(concept_id, "concept")
        for link in self.ontology.links.values():
            if concept_id not in (link.source, link.target):
                continue
            self.edge(link.source, link.kind, link.target)
            other = link.target if link.source == concept_id else link.source
            if other in self.fields:
                self.field(other)
            elif other in self.ontology.tables:
                self.table(other)
            elif other in self.ontology.concepts:
                self.add(other, "concept")
            else:
                self.add(other, "file")  # A `mentions` Link starts at a File path.

    def ground(self, ids) -> dict:
        """Add Ontology objects or Lake Files to the graph and describe what was added."""
        ids = [ids] if isinstance(ids, str) else ids
        if not isinstance(ids, list) or any(not isinstance(i, str) for i in ids):
            return {"errors": ["ground takes one id or a list of ids: Schema, Table, Field or Concept ids or Lake paths"]}
        before, edges_before, readable_before = set(self.nodes), len(self.edges), self.readable()
        errors = []
        for item in ids:
            if item in self.ontology.schemas:
                self.schema(item, tables=True)
            elif item in self.ontology.tables:
                self.table(item)
            elif item in self.fields:
                self.field(item)
            elif item in self.ontology.concepts:
                self.concept(item)
            else:
                try:
                    path = lake_path(item, self.lake_root)
                except ValueError:
                    errors.append(f"{item!r} is no Schema, Table, Field or Concept id and no File in the Lake; "
                                  "search_ontology finds ids")
                    continue
                mapped = [t.id for t in self.ontology.tables.values() if t.path == path]
                for table_id in mapped:
                    self.table(table_id)
                if not mapped:
                    documented = any(path in s.documentation for s in self.ontology.schemas.values())
                    self.add(path, "file", in_ontology=documented)
        return self.describe([key for key in self.nodes if key not in before], self.edges[edges_before:],
                             sorted(self.readable() - readable_before), errors)

    def describe(self, added: list[tuple[str, str]], edges: list, readable: list[str], errors: list[str]) -> dict:
        ontology = self.ontology
        kinds = {kind: [n for k, n in added if k == kind] for kind in ("schema", "table", "field", "concept", "file")}
        anchors = {node for _, node in self.nodes} | self.readable()
        experience = [e for e in ontology.live_experience() if e.id not in self.shown and anchors & set(e.anchors)]
        self.shown |= {e.id for e in experience}
        result = {
            "added": list(dict.fromkeys(node for _, node in added)),
            "schemas": [schema_detail(ontology, s) for s in kinds["schema"]],
            "tables": [ontology.tables[t].model_dump(mode="json") for t in kinds["table"][:LISTED]],
            "table_count": len(kinds["table"]),
            "fields": [{**self.fields[f].model_dump(mode="json"), "schema_id": self.schema_of[f]} for f in kinds["field"]],
            "concepts": [ontology.concepts[c].model_dump(mode="json") for c in kinds["concept"]],
            "files": [{"path": f, "in_ontology": self.nodes["file", f]["in_ontology"]} for f in kinds["file"]],
            "edges": [{"source": s, "kind": k, "target": t} for s, k, t in edges],
            "readable": readable[:LISTED], "readable_count": len(readable),
            "experience": [e.labelled() for e in experience],
        }
        if errors:
            result["errors"] = errors
        return result

    def record(self, files: list[dict]) -> None:
        """Turn the worker's file log into reads and derived workspace nodes."""
        for entry in files:
            node = entry["path"] if entry["area"] == "lake" else f"{WORKSPACE}/{entry['path']}"
            if entry["mode"] == "read":
                if node not in self.reads:
                    self.reads.append(node)
                continue
            self.add(node, "derived")
            for source in self.reads:
                if source != node:
                    self.edge(node, "derived_from", source)

    def answer(self) -> None:
        self.add("answer", "answer")
        for source in self.reads:
            self.edge("answer", "derived_from", source)

    def to_json(self) -> dict:
        return {"nodes": [{"id": n, "kind": k, **node} for (k, n), node in self.nodes.items()],
                "edges": [{"source": s, "kind": k, "target": t} for s, k, t in self.edges],
                "reads": list(self.reads),
                "outside_ontology": sorted(n for n in self.of("file") if not self.nodes["file", n]["in_ontology"])}


def serialized_string(value) -> bool:
    """A string such as '"Singapore"' that json.loads turns into another string."""
    if not isinstance(value, str) or len(value) < 2 or value[0] != '"' or value[-1] != '"':
        return False
    try:
        return isinstance(json.loads(value), str)
    except ValueError:
        return False


class TaskRun:
    """The state of one Task: its task graph, the agent's tools and visit, and the saved result."""

    def __init__(self, store: OntologyStore, llm, task: dict, config: DataAgentConfig, scratch_root: Path,
                 output_root: Path, check: Callable[[Any], None] | None = None):
        self.store, self.llm, self.task, self.config, self.check = store, llm, task, config, check
        self.ontology = store.load()
        self.graph = TaskGraph(self.ontology, store.lake_root)
        self.scratch_root, self.output_root = Path(scratch_root).resolve(), Path(output_root).resolve()
        self.workspace = self.scratch_root / WORKSPACE
        self.log_root = self.output_root / "logs"
        self.budget = Budget(config.max_tokens, start=llm.usage["total_tokens"]) if config.max_tokens is not None else None
        self.pending_ground_context: list[dict] = []
        self.steps: int | None = None
        self.started, self.usage_before = time.monotonic(), dict(llm.usage)

    def tools(self) -> list:
        ontology = self.ontology

        @tool
        def search_ontology(query: str, hops: int = 2) -> dict:
            """Return the Concepts, Fields, Schemas, Tables and Files that match a query and their neighbours, the edges
            and paths walked, and the Experience anchored to them.

            Args:
                query: Words to match against Concept, Schema, Field and Table names and descriptions.
                hops: How many edges to walk from the matched objects.
            """
            found = retrieve(ontology, query, hops=hops)
            self.log("search_ontology", query=query, hops=hops, matched=found["matched"])
            return found

        @tool
        def get_schema(schema_id: str) -> dict:
            """Return one Schema with all its Fields, its first Tables and its table_count.

            Args:
                schema_id: The id of a Schema from search_ontology.
            """
            return schema_detail(ontology, schema_id)

        @tool
        def ground(ids: Any) -> dict:
            """Add Ontology objects or Lake Files to the task graph so that their Files become readable. Return what
            was added: Schemas with their Fields, Tables, Fields, Concepts, Files, edges, newly readable Files, new
            Experience and any errors.

            Args:
                ids: One id or a list of ids: a Schema id (the Schema and all its Tables), a Table id, a Field id, a
                    Concept id (with its Links and the objects they connect) or a Lake path (the Tables that map it,
                    or the File).
            """
            result = self.graph.ground(ids)
            if result.get("added") or result.get("experience") or result.get("errors"):
                self.pending_ground_context.append(result)
            self.log("ground", ids=ids, added=result.get("added", []), errors=result.get("errors", []))
            return result

        return [search_ontology, get_schema, ground]

    def observe_ground_context(self, step: ActionStep) -> None:
        """Expose authoritative ground results even when Python did not print them or its cell failed."""
        if not self.pending_ground_context:
            return
        context = ("Ground context (automatically provided; review before final_answer):\n"
                   + json.dumps(self.pending_ground_context, ensure_ascii=False))
        step.observations = f"{step.observations}\n\n{context}" if step.observations else context
        self.pending_ground_context.clear()

    def check_answer(self, answer: dict) -> None:
        if self.pending_ground_context:
            raise ValueError("ground() returned new context that you have not yet received. Your computation has "
                             "already run; review the ground context automatically provided in this step's "
                             "observation, then reuse the existing variables to submit or revise final_answer "
                             "in the next step.")
        if "answer" not in answer:
            raise ValueError('Call final_answer({"answer": ...})')
        if extra := sorted(set(answer) - {"answer", "explanation"}):
            # Keys beside "answer" would be dropped, leaving only what "answer" holds.
            raise ValueError(f'final_answer takes only "answer" and an optional "explanation", but it got {extra}. '
                             f'If they belong to the answer, put them inside its value: '
                             f'final_answer({{"answer": {{"{extra[0]}": ...}}}})')
        if "explanation" in answer and not isinstance(answer["explanation"], str):
            raise ValueError("explanation must be a string when provided")
        values = answer["answer"] if isinstance(answer["answer"], list) else [answer["answer"]]
        if quoted := next((value for value in values if serialized_string(value)), None):
            raise ValueError(f"The answer string {quoted} is JSON text with quotation marks around it. "
                             "Give the Python string itself, not json.dumps of it.")
        if self.check is not None:
            try:
                self.check(answer["answer"])
            except ValueError as error:
                self.log("answer_rejected", error=str(error), answer=answer["answer"])
                raise ValueError(f"The answer does not have the required shape: {error}") from None

    def result(self, status: str, final: dict | None = None, stop_reason: str = "") -> dict:
        return {
            "task": self.task, "status": status, "stop_reason": stop_reason,
            "answer": final.get("answer") if final else None, "explanation": final.get("explanation", "") if final else "",
            "graph": self.graph.to_json(), "steps": self.steps,
            "usage": {key: self.llm.usage[key] - self.usage_before.get(key, 0) for key in self.llm.usage},
            "wall_seconds": round(time.monotonic() - self.started, 3),
        }

    def save(self, status: str, final: dict | None = None, stop_reason: str = "") -> dict:
        result = self.result(status, final, stop_reason)
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
    run.log("task_start", task=task, model=llm.model_name(ROLE))
    agent_task = {"task": {key: task[key] for key in ("question", "answer_format") if key in task},
                  "max_steps": config.agent.max_steps, "workspace": str(run.workspace)}
    final, stop_reason = None, "answered"
    try:
        visit = run_agent(
            llm, role=ROLE, instructions=DATA_AGENT, task=json.dumps(agent_task, ensure_ascii=False),
            validate=run.check_answer, tools=run.tools(), lake_root=store.lake_root,
            scratch_root=scratch_root / WORKERS, artifact_root=run.log_root / ROLE, config=config.agent,
            budget=run.budget, lake="read", workspace=run.workspace, long_tools=True,
            gate=run.graph.gate, on_files=run.graph.record, on_step=run.observe_ground_context)
        run.steps, final = visit.steps, visit.answer
        if final is None:
            stop_reason = "incomplete"
            run.log("incomplete", steps=visit.steps)
        else:
            run.graph.answer()
    except LLMError as error:  # Save what the Task has so far.
        stop_reason = "budget" if isinstance(error, BudgetExceeded) else "llm_error"
        run.log(stop_reason, message=f"{type(error).__name__}: {error}")
    result = run.save("answered" if final else "incomplete", final, stop_reason)
    run.log("task_end", status=result["status"], stop_reason=stop_reason, answer=result["answer"],
            explanation=result["explanation"], steps=result["steps"], nodes=len(result["graph"]["nodes"]),
            outside_ontology=result["graph"]["outside_ontology"], usage=result["usage"],
            wall_seconds=result["wall_seconds"])
    return result
