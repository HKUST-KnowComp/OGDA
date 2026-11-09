"""DiscoveryBench adaptor for scripts/benchmark.py.

Reads the DiscoveryBench checkout's own
`discoverybench/real/test/<dataset>/metadata_<n>.json` layout and the
`eval/answer_key_real.csv` answer key directly. Scoring runs the official
eval in a subprocess: `scripts/scorers/discoverybench_score.py`.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

from . import SCORERS, Judge, generation_failed, run_scorer, valid_score

SCORER = SCORERS / "discoverybench_score.py"

ANSWER_FORMAT = (
    "The answer is a JSON object with exactly two nonempty strings: "
    "{'hypothesis': '...', 'workflow': '...'}. In hypothesis, write down a "
    "scientific hypothesis in natural language, derived from the provided "
    "dataset, clearly stating the context of hypothesis (if any), variables "
    "chosen (if any) and relationship between those variables (if any) "
    "including any statistical significance. In workflow, write a summary of "
    "the full workflow starting from data loading that led to the final "
    "answer."
)

_METADATA_NUMBER = re.compile(r"metadata_(\d+)")


def _read_metadata(path: Path) -> dict:
    data = path.read_bytes()
    try:
        return json.loads(data.decode("utf-8"))
    except UnicodeDecodeError:
        return json.loads(data.decode("cp1252"))


def _metadata_files(dataset_dir: Path) -> list[Path]:
    return sorted(
        dataset_dir.glob("metadata_*.json"),
        key=lambda p: int(_METADATA_NUMBER.match(p.stem).group(1)),
    )


def _dataset_dirs(root: Path) -> list[Path]:
    test_root = root / "discoverybench" / "real" / "test"
    return sorted((d for d in test_root.iterdir() if d.is_dir()), key=lambda d: d.name)


def lakes(root: Path) -> list[dict]:
    return [
        {
            "lake_id": dataset_dir.name,
            "source": dataset_dir,
            "exclude": [],
            "strip_keys": ["queries", "hypotheses", "intermediate"],
        }
        for dataset_dir in _dataset_dirs(root)
    ]


def tasks(root: Path) -> list[dict]:
    result = []
    for dataset_dir in _dataset_dirs(root):
        dataset = dataset_dir.name
        for metadata_path in _metadata_files(dataset_dir):
            metadata_id = metadata_path.stem[len("metadata_") :]
            metadata = _read_metadata(metadata_path)
            for group in metadata.get("queries", []):
                items = group if isinstance(group, list) else [group]
                for item in items:
                    qid = str(item["qid"])
                    result.append(
                        {
                            "task_id": f"{dataset}__{metadata_path.stem}__q{qid}",
                            "lake_id": dataset,
                            "question": item["question"],
                            "dataset": dataset,
                            "metadata_id": metadata_id,
                            "query_id": qid,
                        }
                    )
    return result


def answer_format(task: dict) -> str:
    return ANSWER_FORMAT


def check(value, task: dict) -> None:
    if not isinstance(value, dict):
        raise ValueError("answer expects a JSON object with hypothesis and workflow")
    hypothesis = value.get("hypothesis")
    if not isinstance(hypothesis, str) or not hypothesis:
        raise ValueError("answer expects a nonempty hypothesis string")
    workflow = value.get("workflow")
    if not isinstance(workflow, str) or not workflow:
        raise ValueError("answer expects a nonempty workflow string")


def prediction(result: dict, task: dict, pipeline: str) -> dict:
    answer = result["answer"]
    if isinstance(answer, dict):
        hypothesis = answer.get("hypothesis", "")
        workflow = answer.get("workflow", "")
    else:
        hypothesis = ""
        workflow = ""
    return {"hypothesis": hypothesis, "workflow": workflow, "pipeline_code": pipeline}


def gold(task: dict, root: Path) -> dict:
    answer_key = root / "eval" / "answer_key_real.csv"
    matches = []
    with answer_key.open(newline="") as f:
        for row in csv.DictReader(f):
            if (
                row["dataset"] == task["dataset"]
                and row["metadataid"] == task["metadata_id"]
                and row["query_id"] == task["query_id"]
            ):
                matches.append(row)
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one answer key row for dataset={task['dataset']} "
            f"metadataid={task['metadata_id']} query_id={task['query_id']}, "
            f"found {len(matches)}"
        )
    dataset_dir = root / "discoverybench" / "real" / "test" / task["dataset"]
    metadata = _read_metadata(dataset_dir / f"metadata_{task['metadata_id']}.json")
    return {
        "query": task["question"],
        "gold_hypo": matches[0]["gold_hypo"],
        "gold_workflow": metadata.get("workflow"),
        "dataset_meta": metadata,
        "dataset_type": "real",
    }


def evaluate(prediction: dict, task: dict, judge: Judge, root: Path, scorer_root: Path, out: Path) -> dict:
    """Score one prediction with the official `run_eval_gold_vs_gen_NL_hypo_workflow`."""
    task_id = prediction["task_id"]
    if generation_failed(prediction, "hypothesis"):
        return {"task_id": task_id, "status": "generation_failed", "score": 0.0}
    job = {**gold(task, root), "gen_hypo": prediction["hypothesis"], "gen_workflow": prediction["workflow"]}
    record = run_scorer(SCORER, job, judge, scorer_root, log=out / "logs" / f"{task_id}.log")
    score = record.get("final_score")
    return {"task_id": task_id, "status": "scored" if valid_score(score) else "evaluator_failed",
            "score": float(score) if valid_score(score) else 0.0, "native_evaluation": record}


def summarize(scores: list[dict], scorer_root: Path, judge: Judge) -> dict:
    """The mean final_score over every scored prediction; a failed one counts 0."""
    counts = Counter(score["status"] for score in scores)
    return {"score": sum(score["score"] for score in scores) / len(scores) if scores else 0.0,
            "scored": counts["scored"], "generation_failed": counts["generation_failed"],
            "evaluator_failed": counts["evaluator_failed"]}
