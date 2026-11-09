"""One module per benchmark: what the benchmark decides, and nothing that the method decides.

Every module defines:
- `inventory(root, lake_root)`: the clean source Lakes and the Tasks, with generation fields only.
- `check_reference(task, root)`: the saved question still matches the official one.
- `answer_format(task)`: the benchmark's own output rule. It is the only benchmark text an agent sees.
- `check(value, task, workspace)`: raise ValueError when the Data agent's answer has the wrong shape. The Data agent sees the
  error and can repair its answer.
- `prediction(value, task, workspace, artifact, explanation)`: the fields saved with a completed Task.
- `accepted(prediction, task)`: whether a saved completed prediction can be kept on resume.
- `evaluate(job, prediction)`: the score from the benchmark's own scorer.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import ModuleType

from ..artifacts import input_hashes

# Benchmark names are the directory names under the runtime, the datalake and the results.
MODULES = {"Kramabench": "kramabench", "discoverybench": "discoverybench"}
BENCHMARKS = tuple(MODULES)
# The Task fields that a prediction and a score repeat; other fields (such as metadata paths) stay in the inventory.
IDENTIFIERS = ("task_id", "lake_id", "question", "answer_type", "domain", "category", "split", "dataset", "metadata_id", "query_id", "case_id")
GENERATION_FAILURES = {"failed", "incomplete"}
RESOLVED_SCORES = {"scored", "generation_failed"}


def get(benchmark: str) -> ModuleType:
    if benchmark not in MODULES:
        raise ValueError(f"Unsupported benchmark: {benchmark}")
    return importlib.import_module(f"{__name__}.{MODULES[benchmark]}")


def identifiers(task: dict) -> dict:
    return {key: task[key] for key in IDENTIFIERS if key in task}


def load_metadata(path: Path) -> dict:
    """Read benchmark JSON metadata, some of which is cp1252 rather than UTF-8."""
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("cp1252")
    return json.loads(text)


def clean_lake(path: Path, expected: dict[str, str]) -> dict:
    if not path.is_dir():
        raise FileNotFoundError(f"Clean datalake is missing: {path}")
    if any(item.is_symlink() for item in path.rglob("*")):
        raise ValueError(f"Clean datalake contains symlinks: {path}")
    actual = input_hashes(path)
    if not expected or actual != expected:
        raise ValueError(f"Clean datalake inputs or documentation differ from their task-blind sources: {path}")
    return actual


def judge_config(path: Path):
    from ogda.adaptor.llm import load_config, select_role, validate_openai_compatible_judge

    config = select_role(load_config(path), "evaluator")
    validate_openai_compatible_judge(config)
    return config


def public_judge(config) -> dict:
    from ogda.adaptor.llm import model_request_kwargs

    judge = config.judge or config
    options = model_request_kwargs(judge)
    return {"provider": judge.provider, "model": judge.provider_model, "base_url": judge.base_url,
            "reasoning_effort": judge.reasoning_effort, "request_options": options,
            "thinking": options.get("thinking", options.get("extra_body", {}).get("thinking"))}
