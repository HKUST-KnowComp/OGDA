"""Shared fixtures for scripts/benchmark.py tests.

`scripts/` is not an installed package; insert it (and `tests/`) at the front of
`sys.path` so `from benchmarks import ...` and `from helpers import ...` resolve.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

from ogda.sandbox import IsolatedExecutor

REPO = Path(__file__).resolve().parent.parent.parent

for _path in (REPO / "scripts", REPO / "tests"):
    _str_path = str(_path)
    if _str_path not in sys.path:
        sys.path.insert(0, _str_path)

_sandbox_available: bool | None = None  # Session cache: the probe below runs at most once per run.


def _sandbox_probe() -> bool:
    global _sandbox_available
    if _sandbox_available is None:
        with tempfile.TemporaryDirectory() as lake, tempfile.TemporaryDirectory() as scratch, \
                tempfile.TemporaryDirectory() as artifacts:
            executor = IsolatedExecutor(lake, scratch, artifacts)
            try:
                executor("1")
                _sandbox_available = True
            except Exception:
                _sandbox_available = False
            finally:
                executor.cleanup()
    return _sandbox_available


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "requires_sandbox: needs the code sandbox (Landlock/seccomp, or srt on macOS)")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if not any(item.get_closest_marker("requires_sandbox") for item in items):
        return
    if _sandbox_probe():
        return
    skip = pytest.mark.skip(reason="code sandbox unavailable")
    for item in items:
        if item.get_closest_marker("requires_sandbox"):
            item.add_marker(skip)


@pytest.fixture
def fake_kramabench(tmp_path: Path) -> Path:
    """A minimal KramaBench root: one domain, two tasks, one of each answer_type."""
    input_dir = tmp_path / "data" / "environment" / "input"
    input_dir.mkdir(parents=True)
    (input_dir / "a.csv").write_text("col\n1\n2\n")

    workload_dir = tmp_path / "workload"
    workload_dir.mkdir(parents=True)
    tasks = [
        {"id": "t1", "query": "How many rows?", "answer": 2, "answer_type": "numeric_exact"},
        {
            "id": "t2",
            "query": "Which values appear in the column?",
            "answer": [1, 2],
            "answer_type": "list_exact",
        },
    ]
    (workload_dir / "environment.json").write_text(json.dumps(tasks))

    fixtures_dir = tmp_path / "benchmark" / "fixtures"
    fixtures_dir.mkdir(parents=True)
    fixtures = [
        {"name": "numeric_exact", "metrics": ["success"]},
        {"name": "list_exact", "metrics": ["f1"]},
    ]
    (fixtures_dir / "answer_type_fixtures.json").write_text(json.dumps(fixtures))

    return tmp_path


@pytest.fixture
def fake_discoverybench(tmp_path: Path) -> Path:
    """A minimal DiscoveryBench root: one dataset, one metadata file, one query."""
    dataset_dir = tmp_path / "discoverybench" / "real" / "test" / "toy"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "data.csv").write_text("x,y\n1,2\n")

    metadata = {
        "datasets": [
            {
                "name": "data.csv",
                "description": "Toy data.",
                "columns": {"raw": [{"name": "x"}, {"name": "y"}]},
            }
        ],
        "hypotheses": [],
        "intermediate": [],
        "workflow": [],
        "queries": [[{"qid": 0, "question": "Is y rising?"}]],
    }
    (dataset_dir / "metadata_0.json").write_text(json.dumps(metadata))

    answer_key_dir = tmp_path / "eval"
    answer_key_dir.mkdir(parents=True)
    (answer_key_dir / "answer_key_real.csv").write_text(
        "dataset,metadataid,query_id,gold_hypo\ntoy,0,0,\"y rises\"\n"
    )

    return tmp_path


# A one-Dataset construction of the fake KramaBench Lake (`a.csv`): the Proposer induces a.csv as Readings identified
# by their value, the Verifier verifies it, and one concept-layer visit reviews both Concepts.
A_CLAIMS = [
    {"kind": "concept", "concept": {"concept": "ReadingValue", "type": "ValueType", "extends": ["Integer"]}},
    {"kind": "concept", "concept": {"concept": "Reading", "type": "EntityType", "identify_by": ["value"]}},
    {"kind": "relationship", "concept": "Reading", "relationship": {
        "name": "value", "roles": [{"concept": "ReadingValue"}], "multiplicity": "OneToOne",
        "verbalizes": ["{Reading} has value {ReadingValue}"]}},
    {"kind": "dataset", "dataset": {
        "name": "a", "path": "a.csv", "header_row": 0, "start_row": 1, "end_row": 3, "origin": "inferred",
        "documentation": [], "description": "One count per row",
        "fields": [{"name": "col", "column": "col", "datatype": "Integer", "description": "A count, e.g. 1"}],
        "rows": "Reading", "binding": [{"concept": "Reading", "identify": {"value": "col"}}]}},
]


def one_dataset_script():
    from helpers import Script, answer, verdict

    induction = {"claims": A_CLAIMS, "summary": "One table", "unresolved": [],
                 "file_dispositions": [{"path": "a.csv", "status": "mapped", "complete": True, "datasets": ["a"]}]}
    review = {"claims": [], "reviewed": [{"concept": name, "reason": "The only Concept of its kind"}
                                         for name in ("Reading", "ReadingValue")]}
    return Script([answer(induction), answer(review)], verifier=[verdict("verified")])


class BenchmarkEnv:
    """Paths and seams for one scripts/benchmark.py run on the fake KramaBench root."""

    def __init__(self, root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        self.root = root
        self.lake_dir = tmp_path / "lake" / "Kramabench"
        self.results = tmp_path / "results" / "Kramabench"
        self.envdir = tmp_path / "env" / "Kramabench"
        self.config = tmp_path / "config.yaml"
        self.tmp_path = tmp_path
        self.monkeypatch = monkeypatch
        self.script = one_dataset_script()
        self.clients = []  # Every client make_client returned, in order.

    def answer_task(self, system: str, prompt: str) -> str:
        """A data_agent-role answer that passes kramabench.check: a number for t1, a list for t2.

        OGDA (its prompt names `ground(`) reads the Dataset `a`; the baseline reads a.csv under LAKE_ROOT.
        """
        from helpers import code

        value = "len(df)" if "How many rows?" in prompt else "sorted(df['col'].tolist())"
        if "ground(" in system:
            read = "df = ground('a')\n"
        else:
            read = "import pandas as pd\ndf = pd.read_csv(LAKE_ROOT / 'a.csv')\n"
        return code(f"{read}final_answer({value})")

    def respond(self, role: str, system: str, prompt: str):
        return self.answer_task(system, prompt) if role == "data_agent" else self.script(role, system, prompt)

    def build_ontology(self, run_id: str) -> None:
        """Run the construct stage once so `run_id` holds a finished one-Dataset Ontology of `environment`."""
        import benchmark

        assert benchmark.main(self.argv("construct", run_id=run_id)) == 0

    def infer_run(self, run_id: str, predictions: bool = True) -> Path:
        """An infer run folder with an answered prediction for t1 and t2, or without predictions.jsonl."""
        run = self.results / "inference" / run_id
        run.mkdir(parents=True)
        (run / "run.json").write_text(json.dumps({"stage": "infer", "run_id": run_id}))
        if predictions:
            rows = [{"task_id": "t1", "lake_id": "environment", "status": "answered", "answer": 2,
                     "pipeline_code": "len(df)"},
                    {"task_id": "t2", "lake_id": "environment", "status": "answered", "answer": [1, 2],
                     "pipeline_code": "sorted(df.col)"}]
            (run / "predictions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        return run

    def make_client(self, config: Path, record_path: Path):
        from ogda.llm import ScriptedClient

        class Client(ScriptedClient):
            def close(self) -> None:
                pass

        client = Client(self.respond, "openai/scripted-model")
        self.clients.append(client)
        return client

    def argv(self, stage_or_problem: str, *, method: str | None = None, run_id: str | None = None,
             extra: tuple[str, ...] = ()) -> list[str]:
        """CLI args for a stage, or for a construct run that fails the named preflight check."""
        import benchmark

        stage, config, lake_dir, args = stage_or_problem, self.config, self.lake_dir, list(extra)
        if stage_or_problem in {"no_lake_dir", "unknown_lake", "unknown_task", "existing_run", "no_sandbox",
                                "bad_config"}:
            stage = "construct"
        if stage_or_problem == "no_lake_dir":
            lake_dir = self.tmp_path / "missing"
        elif stage_or_problem == "unknown_lake":
            args += ["--lakes", "nowhere"]
        elif stage_or_problem == "unknown_task":
            args += ["--tasks", "no-such-task"]
        elif stage_or_problem == "existing_run":
            run_id = "taken"
            (self.results / "construction" / run_id).mkdir(parents=True)
        elif stage_or_problem == "no_sandbox":
            def unavailable(scratch: Path) -> None:
                raise benchmark.PreflightError("code sandbox unavailable: srt is not installed")
            self.monkeypatch.setattr(benchmark, "check_sandbox", unavailable)
        elif stage_or_problem == "bad_config":
            config = self.tmp_path / "bad-config.yaml"
            config.write_text("llm:\n  base_url: http://localhost:1/v1\n  api_key: test-key\n")
            self.monkeypatch.setattr(benchmark, "role_model", self.real_role_model)
        argv = ["--benchmark", "Kramabench", "--stage", stage, "--config", str(config),
                "--benchmark-root", str(self.root), "--lake-dir", str(lake_dir),
                "--results-dir", str(self.results), "--env-dir", str(self.envdir), "--workers", "1"]
        if method is not None:
            argv += ["--method", method]
        if run_id is not None:
            argv += ["--run-id", run_id]
        return argv + args


@pytest.fixture
def env(fake_kramabench: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> BenchmarkEnv:
    """A prepared fake KramaBench Lake, a small YAML config and scripted seams for scripts/benchmark.py."""
    import benchmark
    from prepare_lakes import prepare

    env = BenchmarkEnv(fake_kramabench, tmp_path, monkeypatch)
    prepare("Kramabench", env.root, env.lake_dir)
    env.config.write_text(
        "llm:\n  model: openai/scripted-model\n  base_url: http://localhost:1/v1\n  api_key: test-key\n"
        "  roles:\n    evaluator: {model: openai/gpt-5-mini}\n"
        "construction:\n  max_passes: 1\n  agent:\n    max_steps: 3\n")
    env.real_role_model = benchmark.role_model
    monkeypatch.setattr(benchmark, "make_client", env.make_client)
    monkeypatch.setattr(benchmark, "check_sandbox", lambda scratch: None)
    monkeypatch.setattr(benchmark, "check_judge", lambda judge: None)
    monkeypatch.setattr(benchmark, "role_model", lambda config, role: "openai/scripted-model")
    return env
