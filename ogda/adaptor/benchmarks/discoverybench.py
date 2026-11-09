"""DiscoveryBench: the real test datasets, a hypothesis with its workflow, and the benchmark's own judge."""

from __future__ import annotations

import csv
from pathlib import Path
import sys

from ..artifacts import file_digest
from . import RESOLVED_SCORES, clean_lake, identifiers, judge_config, load_metadata

# The hypothesis and workflow sentences are DiscoveryBench's own (utils/autonomous_single_agent.py), so the
# answer is asked for in the words its evaluator was built around.
ANSWER_FORMAT = """The answer is a JSON object with exactly two nonempty strings: {'hypothesis': '...', 'workflow': '...'}. In hypothesis, write down a scientific hypothesis in natural language, derived from the provided dataset, clearly stating the context of hypothesis (if any), variables chosen (if any) and relationship between those variables (if any) including any statistical significance. In workflow, write a summary of the full workflow starting from data loading that led to the final answer."""


def queries(metadata: dict) -> list[dict]:
    """The real release nests query groups; synthetic releases use flat lists."""
    result = []
    for group in metadata["queries"]:
        result.extend(group if isinstance(group, list) else [group])
    if any(not isinstance(query, dict) for query in result):
        raise ValueError("Metadata queries must be objects or lists of objects")
    return result


def inventory(root: Path, lake_root: Path) -> tuple[list[dict], list[dict]]:
    from ogda.adaptor.discovery_datalake import SCHEMA_PATH, discover_datalake

    lakes, tasks = [], []
    source = root / "discoverybench"
    raw_files, schemas = discover_datalake(source)
    for relative, schema in sorted(schemas.items()):
        category, split, dataset = relative.parts
        if category != "real" or split != "test":
            continue
        path = lake_root / relative
        expected = {p.relative_to(relative).as_posix(): file_digest(source / p)
                    for p in raw_files if p.is_relative_to(relative)}
        expected[SCHEMA_PATH.as_posix()] = file_digest(path / SCHEMA_PATH)
        if load_metadata(path / SCHEMA_PATH) != schema:
            raise ValueError(f"Sanitized DiscoveryBench documentation differs: {path}")
        hashes = clean_lake(path, expected)
        lake_id = "__".join(relative.parts)
        lakes.append({"lake_id": lake_id, "source": str(path.resolve()), "source_files_sha256": hashes,
                      "category": category, "split": split, "dataset": dataset})
        for metadata_path in sorted((source / relative).glob("metadata*.json")):
            metadata = load_metadata(metadata_path)
            for query in queries(metadata):
                task_id = f"{lake_id}__{metadata_path.stem}__q{query['qid']}"
                tasks.append({"task_id": task_id, "lake_id": lake_id, "category": category,
                              "split": split, "dataset": dataset,
                              "metadata_id": metadata_path.stem.removeprefix("metadata_"),
                              "query_id": query["qid"], "question": query["question"],
                              "metadata_path": str(metadata_path.resolve()), "metadata_sha256": file_digest(metadata_path)})
    return lakes, tasks


def reference(task: dict, root: Path, prediction: dict | None = None) -> dict:
    metadata_path = Path(task["metadata_path"])
    if file_digest(metadata_path) != task["metadata_sha256"]:
        raise ValueError(f"Official metadata changed: {task['task_id']}")
    metadata = load_metadata(metadata_path)
    matches = [query for query in queries(metadata) if str(query["qid"]) == str(task["query_id"])]
    if len(matches) != 1 or matches[0]["question"] != task["question"]:
        raise ValueError(f"Saved question differs from official metadata: {task['task_id']}")
    if task["split"] == "test":
        with (root / "eval" / f"answer_key_{task['category']}.csv").open(newline="") as stream:
            references = [row["gold_hypo"] for row in csv.DictReader(stream)
                          if (row["dataset"], row["metadataid"], row["query_id"]) ==
                          (task["dataset"], str(task["metadata_id"]), str(task["query_id"]))]
        if len(set(references)) != 1:
            raise ValueError(f"Missing or conflicting released test reference: {task['task_id']}")
        gold = references[0]
    else:
        gold = matches[0].get("true_hypothesis")
    if not isinstance(gold, str) or not gold.strip():
        raise ValueError(f"Missing gold hypothesis: {task['task_id']}")
    prediction = prediction or {}
    return {"query": task["question"], "gold_hypo": gold, "gold_workflow": metadata.get("workflow"),
            "gen_hypo": prediction.get("hypothesis", ""), "gen_workflow": prediction.get("workflow") or None,
            "dataset_meta": metadata, "dataset_type": task["category"], "use_column_metadata": True}


def check_reference(task: dict, root: Path) -> None:
    reference(task, root)


def answer_format(task: dict) -> str:
    return ANSWER_FORMAT


def validate_answer(answer: dict) -> None:
    if not isinstance(answer, dict):
        raise ValueError("final_answer needs a JSON object")
    for field in ("hypothesis", "workflow"):
        if not isinstance(answer.get(field), str) or not answer[field].strip():
            raise ValueError(f"final_answer needs a nonempty {field} string")


def check(value, task: dict, workspace: Path) -> None:
    validate_answer(value)


def prediction(value, task: dict, workspace: Path, artifact: Path, explanation: str) -> dict:
    validate_answer(value)
    return {"hypothesis": value["hypothesis"], "workflow": value["workflow"]}


def accepted(prediction: dict, task: dict) -> bool:
    try:
        validate_answer(prediction)
    except ValueError:
        return False
    return True


def evaluation_root(root: Path) -> Path:
    """Bind fresh native scoring to the saved benchmark's released references."""
    from ogda.adaptor.runtime import benchmark_path

    current = benchmark_path("discoverybench")
    for path in sorted((current / "eval").glob("answer_key_*.csv")):
        saved = root / "eval" / path.name
        if not saved.is_file() or file_digest(saved) != file_digest(path):
            raise ValueError("Saved DiscoveryBench answer keys differ from the original benchmark")
    return current


def native_evaluator(root: Path | None = None):
    from .. import discovery_evaluation

    if root is not None and str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return discovery_evaluation


def accepted_evaluation(result: dict, prediction: dict | None = None) -> bool:
    if native_evaluator().accepted_score(result) is None:
        return False
    if prediction is not None:
        native = result["native_evaluation"]
        return (native.get("HypoB") == prediction.get("hypothesis", "")
                and native.get("WorkflowB") == (prediction.get("workflow") or None))
    return True


def native_summary() -> dict:
    return {"evaluation_protocol": native_evaluator().EVALUATION_PROTOCOL,
            "generation_failure_policy": "nonempty hypotheses use native scoring; empty hypotheses remain unresolved"}


def evaluate(job: dict, prediction: dict) -> dict:
    from ogda.adaptor.llm import apply_judge_environment

    config = judge_config(Path(job["evaluator_config"]))
    apply_judge_environment(config)
    root, task = evaluation_root(Path(job["benchmark_root"])), job["task"]
    evaluator = native_evaluator(root)
    hypothesis = prediction.get("hypothesis")
    native_job = {**identifiers(task), "generation_status": prediction["status"],
                  "eligible": isinstance(hypothesis, str) and bool(hypothesis.strip()),
                  "judge_model": (config.judge or config).provider_model,
                  "arguments": reference(task, root, prediction)}
    result = evaluator.evaluate_job(native_job)
    if result["status"] == "generation_failure":
        result["status"] = "generation_failed"
    if result["status"] not in RESOLVED_SCORES:
        result["status"] = "evaluator_failed"
    return {**identifiers(task), **result}
