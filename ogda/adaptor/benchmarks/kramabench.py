"""KramaBench: six domain Lakes, typed answers and the benchmark's own metrics."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

from ..artifacts import file_digest, input_hashes, read_json, read_jsonl
from . import GENERATION_FAILURES, clean_lake, identifiers, judge_config

DOMAINS = ("archeology", "astronomy", "biomedical", "environment", "legal", "wildfire")
ANSWER_TYPES = {"numeric_exact", "numeric_approximate", "string_exact", "string_approximate", "list_exact", "list_approximate"}
PRIMARY_METRICS = {"success", "llm_paraphrase", "rae_score", "f1", "f1_approximate"}
EVALUATION_PROTOCOL = "kramabench-native-pipeline-v1"


def inventory(root: Path, lake_root: Path) -> tuple[list[dict], list[dict]]:
    lakes, tasks = [], []
    for domain in DOMAINS:
        source = root / "data" / domain / "input"
        hashes = clean_lake(lake_root / domain, input_hashes(source))
        lakes.append({"lake_id": domain, "source": str((lake_root / domain).resolve()), "source_files_sha256": hashes})
        for task in read_json(root / "workload" / (domain + ".json")):
            if task["answer_type"] not in ANSWER_TYPES:
                raise ValueError(f"Unsupported answer type: {task['answer_type']}")
            tasks.append({"task_id": task["id"], "lake_id": domain, "domain": domain,
                          "question": task["query"], "answer_type": task["answer_type"]})
    return lakes, tasks


def check_reference(task: dict, root: Path) -> None:
    primary_metric(task["answer_type"], root)


def answer_kind(answer_type: str) -> str:
    return "number" if answer_type.startswith("numeric_") else "list" if answer_type.startswith("list_") else "string"


def answer_format(task: dict) -> str:
    # KramaBench asks for "just the answer, no explanation needed" (systems/smolagents/smolagents_prompts.py) and never
    # names the answer type, whose "exact" or "approximate" is the scorer's tolerance. The JSON kind lets the answer parse.
    kind = answer_kind(task["answer_type"])
    empty = {"number": " Zero is a valid answer.", "list": " An empty list is a valid answer.", "string": ""}[kind]
    return f"The answer is just the answer to the question, with no explanation: a JSON {kind}. Keep the units and precision the question asks for.{empty}"


def validate_answer(value, answer_type: str) -> None:
    # The Planner reads this error, so it names the JSON kind and never the answer type.
    kind = answer_kind(answer_type)
    if answer_type not in ANSWER_TYPES or isinstance(value, bool) or not isinstance(value, {"number": (int, float), "list": list, "string": str}[kind]):
        raise ValueError(f"The answer must be a JSON {kind}")
    json.dumps(value, allow_nan=False)


def check(value, task: dict, workspace: Path) -> None:
    validate_answer(value, task["answer_type"])


def prediction(value, task: dict, workspace: Path, artifact: Path, explanation: str) -> dict:
    validate_answer(value, task["answer_type"])
    return {"answer": value, "workflow": explanation}


def accepted(prediction: dict, task: dict) -> bool:
    try:
        validate_answer(prediction["answer"], task["answer_type"])
    except (KeyError, TypeError, ValueError):
        return False
    return True


def primary_metric(answer_type: str, root: Path) -> str:
    fixtures = read_json(root / "benchmark/fixtures/answer_type_fixtures.json")
    entries = [entry for entry in fixtures if entry["name"] == answer_type]
    metrics = set(entries[0]["metrics"]) & PRIMARY_METRICS if len(entries) == 1 else set()
    if len(metrics) != 1:
        raise ValueError(f"Expected exactly one native primary metric for {answer_type}")
    return metrics.pop()


def valid_score(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= 1


def pipeline_evidence(prediction: dict, *, artifact: Path | None = None) -> dict:
    """Recover executed Analyst Python, including failed attempts, without using workflow prose."""
    if isinstance(prediction.get("pipeline_code"), str):
        return {"pipeline_code": prediction["pipeline_code"]}
    roots = [artifact] if artifact is not None else []
    for field in ("pipeline_result", "log_file", "artifact_root"):
        if prediction.get(field):
            path = Path(prediction[field])
            roots.append(path if field == "artifact_root" else path.parent)
    for root in dict.fromkeys(roots):
        # The Data agent, the former Analyst and legacy single-agent traces at the root.
        agent = next((root / "logs" / role for role in ("data_agent", "analyst") if (root / "logs" / role).exists()), None)
        paths = sorted(agent.rglob("execution.jsonl")) if agent else [root / "execution.jsonl"]
        events = [event for path in paths for event in read_jsonl(path)
                  if event.get("event") == "execution_start" and isinstance(event.get("code"), str)]
        code = [event["code"] for event in sorted(events, key=lambda event: event.get("time", 0))]
        if code or any(path.exists() for path in paths):
            return {"pipeline_code": "\n\n".join(code)}
    return {}


def evaluation_root(root: Path, domains) -> Path:
    """Use freshly assembled native code and confirm saved official references still match."""
    from ogda.adaptor.runtime import benchmark_path

    current = benchmark_path("Kramabench")
    references = [Path("workload") / (domain + ".json") for domain in domains]
    references.extend(path.relative_to(current) for path in sorted((current / "benchmark/fixtures").glob("*.json")))
    if any(not (root / path).is_file() or file_digest(root / path) != file_digest(current / path) for path in references):
        raise ValueError("Saved KramaBench workload or fixtures differ from the original benchmark")
    return current


def native_aggregates(rows: list[dict], root: Path | None = None) -> list[dict]:
    """Flatten native results as evaluate.py does, then use its unchanged aggregation."""
    import pandas as pd
    from ogda.adaptor.runtime import benchmark_path

    flat = []
    for row in rows:
        for result in row.get("native_evaluation", []):
            for metric, value in result.items():
                if metric in {"task_id", "code", "model_output", "subresponses"}:
                    continue
                value = f'"{json.dumps(value)}"' if isinstance(value, list) else value
                entry = {"sut": "ogda", "workload": row.get("domain", "environment"),
                         "task_id": result["task_id"], "metric": metric, "value": value}
                flat.extend([entry, {**entry, "workload": "overall"}])
    if not flat:
        return []
    root = root or benchmark_path("Kramabench")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    specification = importlib.util.spec_from_file_location("_ogda_kramabench_native_evaluate", root / "evaluate.py")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    aggregated = module.aggregate_results("ogda", pd.DataFrame(flat))
    return json.loads(aggregated.to_json(orient="records", double_precision=15))


def native_summary(rows: list[dict], root: Path | None = None) -> dict:
    aggregates = native_aggregates(rows, root)
    pipeline = next((entry for entry in aggregates if entry["workload"] == "overall.json" and entry["metric"] == "llm_code_eval"), {})
    return {"evaluation_protocol": EVALUATION_PROTOCOL, "native_aggregates": aggregates,
            "pipeline_score": pipeline.get("value_mean"), "pipeline_scored": pipeline.get("value_support", 0)}


def accepted_evaluation(result: dict, prediction: dict | None = None) -> bool:
    native = result.get("native_evaluation", [])
    if (result.get("evaluation_protocol") != EVALUATION_PROTOCOL or len(native) != 1
            or result.get("metrics", {}).get("llm_code_eval") is None):
        return False
    metrics = result["metrics"]
    if any(native[0].get(name) != value for name, value in metrics.items()):
        return False
    code = native[0].get("code")
    if not isinstance(code, str) or result.get("pipeline_code_sha256") != hashlib.sha256(code.encode()).hexdigest():
        return False
    if prediction is not None:
        evidence = pipeline_evidence(prediction)
        if "pipeline_code" not in evidence and prediction.get("status") not in GENERATION_FAILURES:
            return False
        if evidence.get("pipeline_code", "") != code:
            return False
    return result.get("score") == native[0].get(result.get("metric"))


def evaluate(job: dict, prediction: dict) -> dict:
    from ogda.adaptor.llm import apply_judge_environment
    from ogda.adaptor.kramabench_transport import configure_judge

    task = job["task"]
    root = evaluation_root(Path(job["benchmark_root"]), [task["domain"]])
    metric = primary_metric(task["answer_type"], root)
    config = judge_config(Path(job["evaluator_config"]))
    apply_judge_environment(config)
    result = dict(identifiers(task), generation_status=prediction["status"], metric=metric, evaluation_protocol=EVALUATION_PROTOCOL)
    if prediction["status"] not in GENERATION_FAILURES:
        validate_answer(prediction["answer"], task["answer_type"])
    references = [row for row in read_json(root / "workload" / (task["domain"] + ".json")) if row["id"] == task["task_id"]]
    if len(references) != 1 or references[0]["query"] != task["question"] or references[0]["answer_type"] != task["answer_type"]:
        raise ValueError("Prediction differs from the official main question")
    sys.path.insert(0, str(root))
    configure_judge(Path(job["journal"]) if job.get("journal") else Path(job["artifact"]) / "judge.jsonl")
    from benchmark.evaluator import Evaluator

    evidence = pipeline_evidence(prediction)
    if "pipeline_code" not in evidence and prediction["status"] not in GENERATION_FAILURES:
        raise ValueError("Saved prediction has no executed Analyst code for native pipeline evaluation")
    code = evidence.get("pipeline_code", "")
    response = {"task_id": task["task_id"], "code": code,
                "model_output": {"answer": prediction["answer"]} if "answer" in prediction else {}}
    evaluator = Evaluator(workload_path=root / "workload" / (task["domain"] + ".json"),
                          task_fixture_directory=root / "benchmark/fixtures", results_directory=Path(job["artifact"]),
                          evaluate_pipeline=True)
    native = evaluator._evaluate_result_for_task(response, references[0], evaluate_pipeline=True)
    names = evaluator.answer_to_metric_dict[task["answer_type"]] + ["llm_code_eval"]
    metrics = {name: native[0][name] for name in names}
    score = native[0][metric]
    resolved = valid_score(score) and metrics["llm_code_eval"] is not None
    result.update(status=("generation_failed" if prediction["status"] in GENERATION_FAILURES else "scored") if resolved else "evaluator_failed",
                  score=score, metrics=metrics, native_evaluation=native,
                  pipeline_code_sha256=hashlib.sha256(code.encode()).hexdigest(),
                  token_usage=sum(native[0].get("token_usage_" + name, 0) for name in ("metrics", "pipeline_eval")),
                  token_usage_input=sum(native[0].get("token_usage_" + name + "_input", 0) for name in ("metrics", "pipeline_eval")),
                  token_usage_output=sum(native[0].get("token_usage_" + name + "_output", 0) for name in ("metrics", "pipeline_eval")))
    if not resolved:
        result["error"] = "Native evaluator returned an unresolved answer or pipeline metric"
    result["pipeline_score"] = native_summary([result], root)["pipeline_score"]
    return result
