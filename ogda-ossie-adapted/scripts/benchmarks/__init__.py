"""Benchmark registry for scripts/benchmark.py.

Each benchmark module (``kramabench``, ``discoverybench``) implements the shared
contract: ``lakes``, ``tasks``, ``answer_format``,
``check``, ``prediction``, ``evaluate`` and ``summarize``. This package also holds
the judge model used by the scorers and the subprocess helper that runs them,
since scoring happens out-of-process to avoid importing KramaBench's own
``benchmark`` package (its name clashes with ``scripts/benchmark.py``).
"""

from __future__ import annotations

import importlib
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

from ogda.llm import OpenAICompatibleClient

REPO = Path(__file__).resolve().parent.parent.parent

MODULES = {"Kramabench": "kramabench", "discoverybench": "discoverybench"}

# External benchmark checkouts live in the enclosing project, one level above this folder; each one holds
# both its Tasks and its official scorer.
ROOTS = {
    "Kramabench": REPO.parent / "benchmark/Kramabench",
    "discoverybench": REPO.parent / "benchmark/discoverybench",
}

SCORERS = REPO / "scripts" / "scorers"

SCORER_ROOTS = ROOTS


def get(benchmark: str) -> ModuleType:
    if benchmark not in MODULES:
        raise ValueError(f"Unsupported benchmark: {benchmark}")
    return importlib.import_module(f".{MODULES[benchmark]}", __name__)


@dataclass(frozen=True)
class Judge:
    model: str
    api_key: str = field(repr=False)
    base_url: str


def judge_from_config(config: Path) -> Judge:
    """Build the Judge from the evaluator role; both scorers call the openai SDK."""
    client = OpenAICompatibleClient(config)
    try:
        settings = client.role_config("evaluator")
    finally:
        client.close()
    model = settings["model"]
    if not model.startswith("openai/"):
        raise ValueError(
            "The evaluator model must start with openai/ because both scorers "
            "use the OpenAI SDK against base_url"
        )
    return Judge(
        model=model[len("openai/") :],
        api_key=settings["credential"],
        base_url=settings["endpoint"],
    )


def run_scorer(script: Path, job: dict, judge: Judge, cwd: Path, log: Path | None = None) -> dict:
    """Run a KramaBench/DiscoveryBench scorer script in its own subprocess; `log` keeps its stderr, the judge log."""
    env = os.environ.copy()
    env.update(
        OPENAI_API_KEY=judge.api_key,
        OPENAI_BASE_URL=judge.base_url,
        JUDGE_MODEL=judge.model,
        PYTHONPATH=str(cwd),
    )
    result = subprocess.run(
        [sys.executable, str(script)],
        input=json.dumps(job),
        capture_output=True,
        text=True,
        cwd=cwd,
        env=env,
    )
    if log is not None:
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"{script} failed:\n{result.stderr[-2000:]}")
    last_line = next((line for line in reversed(result.stdout.splitlines()) if line.strip()), None)
    if last_line is None:
        raise RuntimeError(f"{script.name} printed no result")
    return json.loads(last_line)


def generation_failed(prediction: dict, field: str) -> bool:
    """True when infer produced no answer to score: the Task did not finish or its answer field is empty."""
    return prediction["status"] in {"incomplete", "error"} or prediction.get(field) in (None, "")


def valid_score(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= 1
