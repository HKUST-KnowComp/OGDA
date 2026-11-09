"""Legacy, opt-in live-model scenarios for the former planner/analyst data agent.

This harness still expects the old schemas/tables, plans, and step-result interfaces.
It has not been migrated to the current Ossie Harness and single data-agent API.
It is skipped unless OGDA_SCENARIO_CONFIG and OGDA_SCENARIO_ONTOLOGY are set;
it should not be treated as validation of the current implementation.
"""

import hashlib
import json
import os
import time
from pathlib import Path

import pandas as pd
import pytest
import yaml

from ogda.data_agent import DataAgentConfig, answer
from ogda.llm import OpenAICompatibleClient
from ogda.ontology import OntologyStore

pytestmark = pytest.mark.skipif(not (os.environ.get("OGDA_SCENARIO_CONFIG") and os.environ.get("OGDA_SCENARIO_ONTOLOGY")),
                                reason="Scenario tests call a live model; set OGDA_SCENARIO_CONFIG and OGDA_SCENARIO_ONTOLOGY")
ONLY = {name.strip() for name in os.environ.get("OGDA_SCENARIO_ONLY", "").split(",") if name.strip()}
STRICT = os.environ.get("OGDA_SCENARIO_STRICT") == "1"


def harness_hashes(store: OntologyStore) -> dict:
    paths = [store.path] + sorted(p for p in store.lake_root.rglob("*") if p.is_file())
    return {str(p.relative_to(store.root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


class Session:
    def __init__(self):
        run_id = os.environ.get("OGDA_SCENARIO_RUN_ID") or time.strftime("scenarios-%Y%m%dT%H%M%SZ", time.gmtime())
        self.output = Path(os.environ.get("OGDA_SCENARIO_OUTPUT") or f"results/dis-meta/{run_id}").resolve()
        self.scratch = Path(os.environ.get("OGDA_SCENARIO_SCRATCH") or f"env/dis-meta/{run_id}").resolve()
        config_path = os.environ["OGDA_SCENARIO_CONFIG"]
        document = yaml.safe_load(Path(config_path).read_text()) or {}
        self.config = DataAgentConfig.from_dict(document.get("data_agent"))
        self.store = OntologyStore(os.environ["OGDA_SCENARIO_ONTOLOGY"])
        self.ontology = self.store.load()
        self.llm = OpenAICompatibleClient(config_path, record_path=self.output / "logs" / "llm.jsonl")
        self.hashes = harness_hashes(self.store)
        self.report = self.output / "scenario-report.md"
        self.output.mkdir(parents=True, exist_ok=True)
        self.say(f"# Data agent scenarios\n\nontology: {self.store.root}\noutput: {self.output}\nscratch: {self.scratch}\n"
                 f"planner model: {self.llm.model_name('planner')}, analyst model: {self.llm.model_name('analyst')}\n"
                 f"config: {self.config}\n")

    def say(self, text: str) -> None:
        print(text)
        with self.report.open("a") as stream:
            stream.write(text + "\n")

    def field(self, column: str) -> str:
        found = [f.id for f in self.ontology.fields().values() if f.column == column]
        if not found:
            pytest.skip(f"The Harness has no Field with column {column!r}")
        return found[0]

    def frame(self, column: str) -> pd.DataFrame | None:
        """The Table that holds a column, read for ground truth when its region is a plain header-first CSV."""
        field_id = self.field(column)
        schema_id = next(s.id for s in self.ontology.schemas.values() if any(f.id == field_id for f in s.fields))
        for table in self.ontology.tables.values():
            if (table.schema_id == schema_id and table.path.endswith(".csv") and table.header_row == 0
                    and table.start_row == 1 and table.end_row is None):
                return pd.read_csv(self.store.lake_root / table.path)
        return None

    def run(self, name: str, task: dict, expectations) -> dict:
        """Answer a Task, log what happened, check contracts and report expectations."""
        output, scratch = self.output / "tasks" / task["id"], self.scratch / "tasks" / task["id"]
        self.say(f"\n## {name}: {task['id']}\n\nquestion: {task['question']}\nanswer_format: {task.get('answer_format', '')}")
        result = answer(self.store, self.llm, task, self.config, scratch, output)
        events = [json.loads(line) for line in (output / "logs" / "data_agent.jsonl").read_text().splitlines()]
        self.transcript(events, result)

        checks = [(f"{label}: {detail}" if detail else label, bool(ok)) for label, ok, detail in expectations(result, events)]
        for label, ok in checks:
            self.say(f"- expectation {'MET' if ok else 'NOT MET'}: {label}")
        with (self.output / "scenario-report.jsonl").open("a") as stream:
            stream.write(json.dumps({"scenario": name, "task": task, "status": result["status"],
                                     "stop_reason": result["stop_reason"], "answer": result["answer"],
                                     "plan_versions": len(result["plans"]),
                                     "steps": [{"id": s["step"]["id"], "status": s["result"]["status"]}
                                               for s in result["steps"]],
                                     "expectations": dict(checks), "usage": result["usage"],
                                     "wall_seconds": result["wall_seconds"], "output": str(output)},
                                    ensure_ascii=False, default=str) + "\n")

        assert result["status"] == "answered", f"{name} stopped: {result['stop_reason']}"
        assert json.loads((output / "result.json").read_text())["status"] == "answered"
        for step in result["steps"]:
            assert step["result"]["status"] in {"done", "blocked"}
            assert all((scratch / "workspace" / path).is_file() for path in step["result"]["files"])
        assert harness_hashes(self.store) == self.hashes, "The Data agent changed the Harness"
        if STRICT:
            assert all(ok for _, ok in checks), [label for label, ok in checks if not ok]
        return result

    def transcript(self, events: list[dict], result: dict) -> None:
        for event in events:
            kind = event["event"]
            if kind == "task_start":
                self.say(f"task: {event['task']['question']}")
            elif kind in ("search_ontology", "read_harness"):
                self.say(f"planner {kind}: {event.get('query', '')} -> {event.get('matched_fields', '')}")
            elif kind == "plan_rejected":
                self.say(f"plan REJECTED: {event['errors']}")
            elif kind == "plan_accepted":
                self.say(f"plan v{event['version']} accepted (not run: {event['pending']}):")
                for step in event["steps"]:
                    self.say(f"  {step['id']} <- {step['inputs']}: {step['instruction']}\n"
                             f"     scope={step['scope']}\n     guidance={step['guidance']!r}")
            elif kind == "step_start":
                self.say(f"step {event['step']['id']} starts; experience shown: {event['experience']}; "
                         f"inputs: {event['inputs']}")
            elif kind == "step_result":
                r = event["result"]
                self.say(f"step {event['step_id']} -> {r['status']} ({event['analyst_steps']} agent steps, "
                         f"{event['tokens']} tokens, {event['wall_seconds']}s, completed={event['completed']})\n"
                         f"     result: {r['result'][:600]}\n     notes: {r['notes'][:600]}\n     files: {r['files']}\n"
                         f"     workspace changes: {event['workspace_changes']}")
            elif kind == "run_step_refused":
                self.say(f"run_step({event['step_id']}) refused: {event['error'][:300]}")
            elif kind in ("planner_incomplete", "budget", "llm_error"):
                self.say(f"STOP {kind}: {event}")
        self.say(f"status: {result['status']} ({result['stop_reason']})\nanswer: {json.dumps(result['answer'], ensure_ascii=False)}\n"
                 f"usage: {result['usage']}, wall: {result['wall_seconds']}s")

    def close(self) -> None:
        self.llm.close()
        unchanged = harness_hashes(self.store) == self.hashes
        self.say(f"\nHarness unchanged: {unchanged}. Total LLM usage: {self.llm.usage}")


@pytest.fixture(autouse=True)
def selected(request):
    if ONLY and request.node.name.removeprefix("test_") not in ONLY:
        pytest.skip("Not in OGDA_SCENARIO_ONLY")


@pytest.fixture(scope="module")
def session():
    session = Session()
    yield session
    session.close()


def scopes(result: dict, key: str) -> set[str]:
    return {entry for step in result["steps"] for entry in step["step"]["scope"][key]}


def text_of(result: dict) -> str:
    return " ".join(f"{s['result']['result']} {s['result']['notes']}" for s in result["steps"]).casefold()


def test_field_without_concept(session):
    seniority = session.field("seniority.o")
    linked = any(l.kind == "key_of" and l.source == seniority for l in session.ontology.links.values())
    session.say(f"precondition: {seniority} has a key_of Link: {linked}")
    task = {"id": "scenario-field-without-concept",
            "question": "Is replication success related to the seniority of the original authors?",
            "answer_format": "One hypothesis sentence with the supporting statistics."}
    session.run("field_without_concept", task, lambda result, events: [
        ("a Scope cites the seniority Field directly", seniority in scopes(result, "fields"), seniority),
        ("the answer is a nonempty sentence", isinstance(result["answer"], str) and result["answer"].strip(), ""),
    ])


def test_documentation_value_mismatch(session):
    project_y, replicated = session.field("project.y"), session.field("replicated")
    frame = session.frame("project.y")
    truth = None
    if frame is not None:
        values = frame["project.y"].value_counts().to_dict()
        session.say(f"precondition: project.y values {values}")
        psychology = frame[frame["project.y"] == "rpp"]
        truth = round(float(psychology["replicated"].mean()), 3) if len(psychology) else None
    task = {"id": "scenario-documentation-value-mismatch",
            "question": "Using the project.y column to select them, what fraction of the studies in the Psychology "
                        "replication project were successfully replicated?",
            "answer_format": "A decimal fraction between 0 and 1, rounded to 3 decimal places."}

    def expectations(result, events):
        blocked = [s["step"]["id"] for s in result["steps"] if s["result"]["status"] == "blocked"]
        try:
            close = truth is not None and abs(float(result["answer"]) - truth) <= 0.005
        except (TypeError, ValueError):
            close = False
        return [
            ("a Scope cites project.y", project_y in scopes(result, "fields"), project_y),
            ("a Scope cites replicated", replicated in scopes(result, "fields"), replicated),
            ("the Analyst reported the value mismatch (blocked Step, or 'rpp' in a result or notes)",
             blocked or "rpp" in text_of(result), f"blocked={blocked}"),
            ("a blocked Step led to a revised Plan", not blocked or len(result["plans"]) > 1,
             f"plan versions={len(result['plans'])}"),
            ("the answer matches the data", close, f"answer={result['answer']!r}, truth={truth}"),
        ]

    session.run("documentation_value_mismatch", task, expectations)


def test_file_without_table(session):
    files = session.ontology.metadata.get("progress", {}).get("files", {})
    with_table = {t.path for t in session.ontology.tables.values()}
    unresolved = sorted(p for p, item in files.items() if item["status"] == "unresolved")
    documentation = sorted(p for p, item in files.items() if item["status"] == "documentation" and p not in with_table)
    if unresolved:
        path = unresolved[0]
        question = (f"What does the Lake File {path} contain, and which records or Tables of the Lake does it "
                    "describe or add to?")
    elif documentation:
        path = documentation[0]
        question = ("What general guidance about preparing the data for analysis does the Lake's Documentation give, "
                    "beyond the descriptions of individual columns?")
    else:
        pytest.skip("The Harness has no File without a Table")
    session.say(f"precondition: File without a Table: {path} ({files[path]['status']}: {files[path].get('reason', '')})")
    task = {"id": "scenario-file-without-table", "question": question,
            "answer_format": "A short paragraph."}
    session.run("file_without_table", task, lambda result, events: [
        ("a Scope names the File in files", path in scopes(result, "files"), path),
    ])


def test_experience_contradicted(session):
    entries = [e for e in session.ontology.live_experience() if "prediction" in e.text.casefold()]
    if not entries:
        pytest.skip("The Harness has no Experience entry about the prediction market")
    entry = entries[0]
    session.field("endprice")
    frame = session.frame("endprice")
    truth = None
    if frame is not None and "project.y" in frame:
        truth = int(frame.loc[frame["project.y"] == "rpp", "endprice"].notna().sum())
    session.say(f"precondition: Experience {entry.id}: {entry.text}\nground truth: rpp rows with endprice = {truth}")
    task = {"id": "scenario-experience-contradicted",
            "question": "In the Psychology replication project, how many studies have a recorded final prediction-market "
                        "price, and is that price related to replication success?",
            "answer_format": "The count, then one sentence with the supporting statistic."}

    def expectations(result, events):
        shown = {e for event in events if event["event"] == "step_start" for e in event["experience"]}
        cited = entry.id in " ".join(s["step"]["guidance"] for s in result["steps"])
        text = text_of(result)
        return [
            ("the Analyst was shown the entry (Scope anchor or guidance)", entry.id in shown or cited, entry.id),
            ("a Step result or notes report the contradiction",
             entry.id.casefold() in text or any(word in text for word in ("contradict", "disagree", "not empty",
                                                                          "inconsistent", "despite")), ""),
            ("the answer has the true count", truth is not None and str(truth) in json.dumps(result["answer"]),
             f"truth={truth}"),
        ]

    session.run("experience_contradicted", task, expectations)


def test_derived_variable(session):
    session.field("relative_es")
    session.field("no")
    task = {"id": "scenario-derived-variable",
            "question": "Is the relative effect size larger for studies whose original sample size is above the median "
                        "original sample size than for the other studies?",
            "answer_format": "One sentence with both group means, group sizes and a test statistic."}

    def expectations(result, events):
        saved = {s["step"]["id"] for s in result["steps"] if s["result"]["files"]}
        handed = [s["step"]["id"] for s in result["steps"] if set(s["step"]["inputs"]) & saved]
        return [
            ("a Step built a workspace file", saved, f"steps with files={sorted(saved)}"),
            ("a later Step used that file through inputs", handed, f"steps={handed}"),
        ]

    session.run("derived_variable", task, expectations)


def test_kramabench_number(session):
    session.field("replicated")
    frame = session.frame("project.y")
    truth = int(frame.loc[frame["project.y"] == "ee", "replicated"].sum()) if frame is not None else None
    task = {"id": "scenario-kramabench-number",
            "question": "How many studies in the Experimental Economics replication project were successfully replicated?",
            "answer_format": "A single integer."}
    session.run("kramabench_number", task, lambda result, events: [
        ("the answer is an integer", isinstance(result["answer"], int) and not isinstance(result["answer"], bool),
         repr(result["answer"])),
        ("the answer matches the data", result["answer"] == truth, f"truth={truth}"),
    ])


def test_kramabench_list(session):
    session.field("discipline")
    frame = session.frame("discipline")
    truth = None
    if frame is not None:
        rates = frame.groupby("discipline")["replicated"].mean()
        truth = sorted(rates[rates < 0.4].index)
        session.say(f"ground truth replication rates: {rates.round(3).to_dict()}")
    task = {"id": "scenario-kramabench-list",
            "question": "Which disciplines of the original papers have a replication rate below 40%?",
            "answer_format": "A JSON list of discipline names, sorted alphabetically."}
    session.run("kramabench_list", task, lambda result, events: [
        ("the answer is a list", isinstance(result["answer"], list), repr(result["answer"])),
        ("the answer matches the data", result["answer"] == truth, f"truth={truth}"),
    ])
