"""KramaBench adaptor for scripts/benchmark.py.

Reads the KramaBench checkout's own `workload/<domain>.json` and
`data/<domain>/input` layout directly; never imports KramaBench's `benchmark`
package in-process (its name clashes with scripts/benchmark.py), so scoring
runs in a subprocess: `scripts/scorers/kramabench_score.py`.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from . import SCORERS, Judge, generation_failed, run_scorer, valid_score

DOMAINS = ("archeology", "astronomy", "biomedical", "environment", "legal", "wildfire")

# The metrics upstream `aggregate_results` sums into its total score; each answer_type has exactly one.
PRIMARY_METRICS = {"success", "llm_paraphrase", "rae_score", "f1", "f1_approximate"}
SCORER = SCORERS / "kramabench_score.py"
SUT = "ogda"

_KIND_BY_PREFIX = {"numeric": "number", "list": "list", "string": "string"}


def _domains(root: Path):
    for domain in DOMAINS:
        workload = root / "workload" / f"{domain}.json"
        source = root / "data" / domain / "input"
        if workload.exists() and source.exists():
            yield domain, workload, source


def lakes(root: Path) -> list[dict]:
    return [
        {"lake_id": domain, "source": source, "exclude": [], "strip_keys": []}
        for domain, _workload, source in _domains(root)
    ]


def tasks(root: Path) -> list[dict]:
    result = []
    for domain, workload, _source in _domains(root):
        for item in json.loads(workload.read_text()):
            task = dict(item)
            task_id = task.pop("id")
            query = task.pop("query")
            task.update(task_id=task_id, lake_id=domain, question=query)
            result.append(task)
    return result


def _kind(answer_type: str) -> str:
    return _KIND_BY_PREFIX[answer_type.split("_", 1)[0]]


def answer_format(task: dict) -> str:
    kind = _kind(task["answer_type"])
    empty = {
        "number": " Zero is a valid answer.",
        "list": " An empty list is a valid answer.",
        "string": "",
    }[kind]
    return (
        "The answer is just the answer to the question, with no explanation: "
        f"a JSON {kind}. Keep the units and precision the question asks for.{empty}"
    )


def check(value, task: dict) -> None:
    """Reject an answer that is not the Task's JSON kind; messages name only the kind, never the answer_type."""
    kind = _kind(task["answer_type"])
    if isinstance(value, bool):
        raise ValueError(f"The answer must be a JSON {kind}, not a boolean")
    if not isinstance(value, {"number": (int, float), "list": list, "string": str}[kind]):
        raise ValueError(f"The answer must be a JSON {kind}")
    json.dumps(value, allow_nan=False)


def prediction(result: dict, task: dict, pipeline: str) -> dict:
    return {"answer": result["answer"], "pipeline_code": pipeline}


def _answer_metrics(answer_type: str, scorer_root: Path) -> list[str]:
    fixtures = json.loads((scorer_root / "benchmark/fixtures/answer_type_fixtures.json").read_text())
    return next(fixture["metrics"] for fixture in fixtures if fixture["name"] == answer_type)


def _primary_metric(answer_metrics: list[str]) -> str:
    [metric] = [name for name in answer_metrics if name in PRIMARY_METRICS]
    return metric


def evaluate(prediction: dict, task: dict, judge: Judge, root: Path, scorer_root: Path, out: Path) -> dict:
    """Score one prediction with upstream `Evaluator._evaluate_result_for_task`, pipeline judging included."""
    task_id = prediction["task_id"]
    answer_metrics = _answer_metrics(task["answer_type"], scorer_root)
    metric = _primary_metric(answer_metrics)
    # Every row names its domain and primary metric, so summarize can count a failed Task as 0.
    identity = {"task_id": task_id, "workload": task["lake_id"], "metric": metric}
    if generation_failed(prediction, "answer"):
        return {**identity, "status": "generation_failed", "score": 0.0}
    job = {"mode": "score", "root": str(scorer_root), "domain": task["lake_id"], "task_id": task_id,
           "answer": prediction["answer"], "code": prediction["pipeline_code"], "out": str(out)}
    evaluation = run_scorer(SCORER, job, judge, scorer_root, log=out / "logs" / f"{task_id}.log")["native"][0]
    score = evaluation.get(metric)
    return {**identity, "status": "scored" if valid_score(score) else "evaluator_failed",
            "score": float(score) if valid_score(score) else 0.0,
            "metrics": {name: evaluation.get(name) for name in [*answer_metrics, "llm_code_eval"]},
            "native_evaluation": evaluation}


def summarize(scores: list[dict], scorer_root: Path, judge: Judge) -> dict:
    """Upstream `aggregate_results` per domain and overall, over measures flattened as upstream evaluate.py does.

    A Task that was not scored counts as 0 on its primary metric with no pipeline judgments, so upstream counts it.
    """
    rows = []
    for score in scores:
        if score["status"] == "scored":
            measures = {metric: value for metric, value in (score.get("native_evaluation") or {}).items()
                        if metric not in {"task_id", "code", "model_output", "subresponses"}}
        else:
            workload, metric = score.get("workload"), score.get("metric")
            if workload is None or metric is None:  # An exception in evaluate wrote only the task_id.
                task = next(task for task in tasks(scorer_root) if task["task_id"] == score["task_id"])
                workload, metric = task["lake_id"], _primary_metric(_answer_metrics(task["answer_type"], scorer_root))
            score = {**score, "workload": workload}
            measures = {metric: 0, "llm_code_eval": []}
        for metric, value in measures.items():
            if isinstance(value, list):  # LLM code eval results, encoded as upstream does.
                value = f"\"{json.dumps(value)}\""
            for workload in (score["workload"], "overall"):
                rows.append({"sut": SUT, "workload": workload, "task_id": score["task_id"], "metric": metric,
                             "value": value})
    aggregates = []
    if rows:
        aggregates = run_scorer(SCORER, {"mode": "aggregate", "root": str(scorer_root), "rows": rows}, judge,
                                scorer_root)["aggregates"]
    pipeline_score = next((record["value_mean"] for record in aggregates
                           if record["workload"] == "overall.json" and record["metric"] == "llm_code_eval"), None)
    counts = Counter(score["status"] for score in scores)
    return {"aggregates": aggregates,
            "score": sum(score["score"] for score in scores) / len(scores) if scores else 0.0,
            "pipeline_score": pipeline_score, "scored": counts["scored"],
            "generation_failed": counts["generation_failed"], "evaluator_failed": counts["evaluator_failed"]}
