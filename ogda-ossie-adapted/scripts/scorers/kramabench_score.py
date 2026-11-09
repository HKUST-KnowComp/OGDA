"""Score one KramaBench answer, or aggregate scores, with the upstream code and your judge model.

Run by `benchmarks.run_scorer` with cwd and PYTHONPATH set to the KramaBench root, so `benchmark` is KramaBench's own
package. Reads one JSON job from stdin and prints one JSON line; upstream progress output goes to stderr.

- `{"mode": "score", "root", "domain", "task_id", "answer", "code", "out"}` prints `{"native": [...]}`, the result of
  `Evaluator._evaluate_result_for_task` with pipeline judging.
- `{"mode": "aggregate", "root", "rows": [...]}` prints `{"aggregates": [...]}`, the records of `aggregate_results`.

Every upstream judge is a `GPTInterface` hardcoded to `gpt-5-mini`; `use_judge` points each one at `JUDGE_MODEL` on
`OPENAI_BASE_URL` with `OPENAI_API_KEY`.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import sys
from pathlib import Path

from openai import OpenAI


def use_judge(cls: type, model: str) -> None:
    """After `cls.__init__` runs, replace its model and its client (built from a module-level key) with your judge."""
    original = cls.__init__

    def __init__(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.model = model
        self.client = OpenAI()

    cls.__init__ = __init__


def score(job: dict) -> dict:
    import benchmark.llm_tools.gpt_interface
    from benchmark.evaluator import Evaluator

    use_judge(benchmark.llm_tools.gpt_interface.GPTInterface, os.environ["JUDGE_MODEL"])
    root = Path(job["root"])
    workload_path = root / "workload" / f"{job['domain']}.json"
    reference = next(task for task in json.loads(workload_path.read_text()) if task["id"] == job["task_id"])
    evaluator = Evaluator(workload_path=workload_path, task_fixture_directory=root / "benchmark/fixtures",
                          results_directory=job["out"], evaluate_pipeline=True)
    response = {"task_id": job["task_id"], "code": job["code"], "model_output": {"answer": job["answer"]}}
    return {"native": evaluator._evaluate_result_for_task(response, reference, evaluate_pipeline=True)}


def aggregate(job: dict) -> dict:
    import pandas as pd

    spec = importlib.util.spec_from_file_location("kramabench_evaluate", Path(job["root"]) / "evaluate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    records = module.aggregate_results("ogda", pd.DataFrame(job["rows"]))
    return {"aggregates": json.loads(records.to_json(orient="records"))}


def main() -> None:
    job = json.loads(sys.stdin.read())
    with contextlib.redirect_stdout(sys.stderr):
        result = {"score": score, "aggregate": aggregate}[job["mode"]](job)
    print(json.dumps(result, default=lambda value: value.item() if hasattr(value, "item") else str(value)))


if __name__ == "__main__":
    main()
