"""Earlier KramaBench Environment runner: frozen predictions for the 20 main questions and native scoring.

The suite replaced it for benchmark runs. Its single-agent INSTRUCTIONS are a KramaBench-specific system prompt."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import re
import signal
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from . import data_evaluation as data
from ..artifacts import WORKSPACE, input_hashes, read_json
from ..benchmarks import judge_config, public_judge
from ..benchmarks import kramabench as native
from ..benchmarks.kramabench import primary_metric, valid_score

IDENTIFIERS = ("task_id", "question", "answer_type")
ANSWER_TYPES = {"numeric_exact", "string_exact", "list_exact", "string_approximate"}
CONFIG = "llm_config/Kramabench/config.rsi-environment-no-thinking.yaml"
EVALUATOR_CONFIG = "llm_config/Kramabench/config.rsi-environment-evaluator-high.yaml"
INSTRUCTIONS = """Answer the supplied question by executing Python against the actual Lake data. Use the frozen ontology and its search_ontology/get_schema tools to locate columns, understand their meaning, and inspect recorded data quirks. Compute evidence from the data; the ontology provides context. LAKE_ROOT contains your private copy of the whole Environment input Lake, including documentation; SCRATCH_ROOT is your working directory. Inspect missing values and units, preserve duplicate rows unless the analysis justifies another unit, and document filters, joins, deduplication, weighting, and aggregation choices. Finish by calling final_answer({'answer': your_answer, 'workflow': 'the concrete analysis steps, files, columns, filters, and calculations used'}). The answer must be a native JSON number for numeric_exact, a native JSON list for list_exact, and a string for string_exact or string_approximate. Return just the requested value in answer, with the requested precision or units. Zero and empty lists are valid answers. workflow must be a nonempty concrete analysis string. Do not invent measurements or return only a plan."""


def benchmark_root() -> Path:
    from method.adaptor.runtime import benchmark_path

    return benchmark_path("Kramabench")


def select_tasks(lake: Path, limit: int | None = None, *, root: Path | None = None) -> list[dict]:
    """Whitelist only main questions and give every task the complete input Lake."""
    root = root or benchmark_root()
    official_files = input_hashes(root / "data/environment/input")
    if not official_files or input_hashes(lake) != official_files:
        raise ValueError("Frozen Lake differs from the complete benchmark Environment input Lake")
    workload = read_json(root / "workload/environment.json")
    if not isinstance(workload, list) or len(workload) != 20:
        raise ValueError("Environment workload must contain exactly 20 main questions")
    ids = [task["id"] for task in workload]
    if len(set(ids)) != 20 or any(not re.fullmatch(r"environment-(easy|hard)-\d+", task_id) for task_id in ids):
        raise ValueError("Environment workload has duplicate or invalid main question identifiers")
    inputs = [{"name": name, "sha256": digest} for name, digest in official_files.items()]
    tasks = []
    for task in workload:
        if task["answer_type"] not in ANSWER_TYPES or not isinstance(task.get("query"), str) or not task["query"].strip():
            raise ValueError(f"Unsupported Environment question: {task['id']}")
        tasks.append({"task_id": task["id"], "question": task["query"], "answer_type": task["answer_type"], "inputs": inputs})
    return tasks[:limit] if limit is not None else tasks


def task_prompt(task: dict, ontology, lake: Path) -> str:
    from ogda.retrieval import retrieve

    return json.dumps({"stage": "data_evaluation", "question": task["question"], "answer_type": task["answer_type"],
                       "lake_files": sorted(input_hashes(lake)),
                       "ontology_context": retrieve(ontology, task["question"])}, ensure_ascii=False)


def validate_answer(answer: dict, answer_type: str | None = None) -> None:
    if not isinstance(answer, dict) or "answer" not in answer:
        raise ValueError("final_answer needs an answer value")
    value = answer["answer"]
    if isinstance(value, bool) or not isinstance(value, (int, float, str, list)):
        raise ValueError("answer must be a native JSON number, string, or list")
    if answer_type == "numeric_exact" and not isinstance(value, (int, float)):
        raise ValueError("numeric_exact answer must be a native JSON number")
    if answer_type == "list_exact" and not isinstance(value, list):
        raise ValueError("list_exact answer must be a native JSON list")
    if answer_type in {"string_exact", "string_approximate"} and not isinstance(value, str):
        raise ValueError("string answer must be a string")
    json.dumps(value, allow_nan=False)
    if not isinstance(answer.get("workflow"), str) or not answer["workflow"].strip():
        raise ValueError("final_answer needs a nonempty workflow string")


def positive_settings(parser, args) -> None:
    if (args.workers < 1 or getattr(args, "max_steps", 20) < 1 or
            (getattr(args, "limit", None) is not None and args.limit < 1) or
            not math.isfinite(args.task_timeout) or args.task_timeout <= 0):
        parser.error("Workers, steps, limit, and timeout must be positive")


def generation_options(parser) -> None:
    parser.add_argument("--config", type=data.workspace_path, default=data.workspace_path(CONFIG))
    parser.add_argument("--output", type=data.workspace_path)
    parser.add_argument("--scratch", type=data.workspace_path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--task-timeout", type=float, default=1200)
    parser.add_argument("--limit", type=int, help="Explicit smoke-test limit; otherwise all 20 main questions")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")


def generation_main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) == 2 and argv[0] == "--worker-job":
        return data.run_worker(Path(argv[1]))
    parser = argparse.ArgumentParser(description="Answer Environment main questions using a frozen OGDA ontology.")
    parser.add_argument("--ontology-run", required=True, type=data.workspace_path)
    generation_options(parser)
    args = parser.parse_args(argv)
    positive_settings(parser, args)
    try:
        args.benchmark, args.benchmark_root = "Kramabench", benchmark_root()
        return data.run(args)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


def evaluation_worker(job_path: Path) -> int:
    job = read_json(job_path)
    task, prediction = job["task"], read_json(Path(job["prediction"]))
    result = {key: task[key] for key in IDENTIFIERS}
    result.update(status="evaluator_failed", score=None, metric=job["metric"], generation_status=prediction.get("status"))
    started = time.monotonic()

    def terminate(signum, frame):
        raise TimeoutError("Native evaluator task deadline exceeded")

    signal.signal(signal.SIGTERM, terminate)
    try:
        config = judge_config(Path(job["config"]))
        result["judge"] = public_judge(config)
        if prediction.get("status") not in {"failed", "incomplete", "completed"}:
            raise ValueError("Prediction is missing a terminal generation status")
        if prediction["status"] == "completed":
            validate_answer(prediction, task["answer_type"])
        result.update(native.evaluate({"task": {**task, "domain": "environment"}, "benchmark_root": job["benchmark_root"],
                                       "evaluator_config": job["config"], "journal": job["journal"],
                                       "artifact": str(Path(job["journal"]).parent)}, prediction))
    except Exception as error:
        result.update(error=f"{type(error).__name__}: {error}")
    finally:
        result["runtime"] = time.monotonic() - started
        data.save_json(Path(job["result_path"]), result)
    return 0 if result["status"] in {"scored", "generation_failed"} else 1


def execute_evaluation(task: dict, args, root: Path, output: Path, scratch: Path) -> dict:
    attempt = "attempt-" + uuid.uuid4().hex[:12]
    work, artifact = scratch / task["task_id"] / attempt, output / "logs" / task["task_id"] / attempt
    work.mkdir(parents=True)
    artifact.mkdir(parents=True)
    job_path, result_path = work / "job.json", work / "result.json"
    metric = primary_metric(task["answer_type"], root)
    data.save_json(job_path, {"task": {key: task[key] for key in IDENTIFIERS}, "metric": metric,
                             "prediction": str(args.run / "tasks" / (task["task_id"] + ".json")),
                             "benchmark_root": str(root), "config": str(args.config), "journal": str(artifact / "judge.jsonl"),
                             "result_path": str(result_path)})
    exit_code, timed_out, runtime = data.execute_worker(WORKSPACE / "scripts/Kramabench/evaluate_rsi_ontology.py",
                                                       job_path, work, artifact / "console.log", args.task_timeout)
    result = read_json(result_path)
    if not result or timed_out or exit_code != 0:
        result.update({key: task[key] for key in IDENTIFIERS})
        result.update(status="evaluator_failed", metric=metric,
                      error=f"Task exceeded {args.task_timeout:g}s deadline" if timed_out else result.get("error", f"Evaluator exited {exit_code} without a score"))
        if not result.get("native_evaluation"):
            result["score"] = None
    result.update(wall_runtime=runtime, worker_exit_code=exit_code, artifact_root=str(artifact))
    data.save_json(output / "tasks" / (task["task_id"] + ".json"), result)
    return result


def scored(result: dict, task: dict, root: Path, prediction: dict | None = None) -> bool:
    metric = primary_metric(task["answer_type"], root)
    return (all(result.get(key) == task[key] for key in IDENTIFIERS) and result.get("metric") == metric
            and result.get("status") in {"scored", "generation_failed"} and valid_score(result.get("score"))
            and native.accepted_evaluation(result, prediction))


def evaluation_summary(tasks: list[dict], output: Path, root: Path, judge: dict) -> dict:
    rows = []
    generation_run = read_json(output / "run_config.json").get("generation_run")
    for task in tasks:
        result = read_json(output / "tasks" / (task["task_id"] + ".json"))
        prediction = read_json(Path(generation_run) / "tasks" / (task["task_id"] + ".json")) if generation_run else None
        accepted = scored(result, task, root, prediction)
        rows.append({"task_id": task["task_id"], "answer_type": task["answer_type"], "metric": primary_metric(task["answer_type"], root),
                     "score": result["score"] if accepted else None, "status": result.get("status", "pending") if not accepted else result["status"],
                     "pipeline_score": result.get("pipeline_score"),
                     "generation_status": result.get("generation_status"), "error": result.get("error", "")})
    complete = all(row["score"] is not None for row in rows)
    summary = {"benchmark": "Kramabench", "domain": "environment", "denominator": len(tasks), "official_main_questions": 20,
               "scored": sum(row["score"] is not None for row in rows), "unresolved": [row["task_id"] for row in rows if row["score"] is None],
               "generation_failures": sum(row["status"] == "generation_failed" for row in rows),
               "status": "completed" if complete else "incomplete", "judge_model": judge["model"], "judge": judge,
               "metric": "native answer metrics and pipeline judging", "overall_score": sum(row["score"] for row in rows) / len(tasks) if complete else None}
    summary.update(native.native_summary([read_json(output / "tasks" / (task["task_id"] + ".json")) for task in tasks]))
    summary["overall_percent"] = summary["overall_score"] * 100 if complete else None
    data.save_json(output / "summary.json", summary)
    with (output / "scores.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return summary


def run_evaluation(args) -> int:
    root, config = benchmark_root(), judge_config(args.config)
    judge = public_judge(config)
    generation = read_json(args.run / "run_config.json")
    if generation.get("benchmark") != "Kramabench":
        raise ValueError("--run must contain current KramaBench OGDA saved predictions")
    ontology_run = Path(generation["ontology"]["run"])
    if not read_json(args.run / "source-integrity.json").get("unchanged"):
        raise ValueError("Prediction generation did not verify frozen ontology and original Lake integrity")
    if input_hashes(ontology_run / "ontology") != generation["ontology"]["files_sha256"]:
        raise ValueError("Frozen ontology or Lake changed since prediction generation")
    source = Path(read_json(ontology_run / "run-info.json").get("lake", str(root / "data/environment/input"))).resolve()
    if input_hashes(source) != generation["source_files_sha256"]:
        raise ValueError("Original Lake changed since prediction generation")
    tasks = read_json(args.run / "tasks.json")
    expected = select_tasks(ontology_run / "ontology/lake", len(tasks), root=root)
    if not tasks or tasks != expected or hashlib.sha256(json.dumps(tasks, sort_keys=True).encode()).hexdigest() != generation["tasks_sha256"]:
        raise ValueError("Saved tasks differ from the selected Environment main workload")
    predictions = {}
    for task in tasks:
        path = args.run / "tasks" / (task["task_id"] + ".json")
        result = read_json(path)
        if any(result.get(key) != task[key] for key in IDENTIFIERS) or result.get("status") not in {"completed", "incomplete", "failed"}:
            raise ValueError(f"Missing or mismatched saved prediction: {task['task_id']}")
        if result["status"] == "completed":
            validate_answer(result, task["answer_type"])
        predictions[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        primary_metric(task["answer_type"], root)
    output = args.output or args.run / "evaluation/native"
    scratch = args.scratch or WORKSPACE / "exp_env/Kramabench" / (ontology_run.name + "-native-evaluation")
    for path in (output, scratch):
        if any(path.is_relative_to(source) or source.is_relative_to(path) for source in
               (ontology_run / "ontology", source, root / "data/environment/input", args.run / "tasks")):
            raise ValueError("Evaluation directories must be separate from ontology, inputs, and predictions")
    if output.is_relative_to(scratch) or scratch.is_relative_to(output):
        raise ValueError("Results and scratch must be separate trees")
    settings = {"benchmark": "Kramabench", "generation_run": str(args.run), "judge": judge, "workers": args.workers,
                "evaluation_protocol": native.EVALUATION_PROTOCOL,
                "task_timeout": args.task_timeout, "tasks_sha256": generation["tasks_sha256"], "predictions_sha256": predictions,
                "workload_sha256": hashlib.sha256((root / "workload/environment.json").read_bytes()).hexdigest()}
    if args.resume:
        if read_json(output / "run_config.json") != settings:
            raise ValueError("Resume judge, predictions, references, or evaluation settings differ; choose new directories")
    elif any(path.exists() and any(path.iterdir()) for path in (output, scratch)):
        raise FileExistsError("Evaluation output or scratch already exists; use --resume or new directories")
    print(f"Native KramaBench: {len(tasks)} main questions; judge {judge['model']}; effort {judge['reasoning_effort']}\nResults: {output}\nScratch: {scratch}", flush=True)
    if args.dry_run:
        print("Dry run passed: saved predictions, frozen ontology, native metric fixtures, and judge YAML checked; no API calls or evaluation writes.")
        return 0
    output.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    data.save_json(output / "run_config.json", settings)
    pending = [task for task in tasks if not (args.resume and scored(read_json(output / "tasks" / (task["task_id"] + ".json")), task, root,
                                                                    read_json(args.run / "tasks" / (task["task_id"] + ".json"))))]
    try:
        evaluation_summary(tasks, output, root, judge)
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(execute_evaluation, task, args, root, output, scratch): task for task in pending}
            for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
                task = futures[future]
                try:
                    result = future.result()
                except Exception as error:
                    result = {key: task[key] for key in IDENTIFIERS}
                    result.update(status="evaluator_failed", score=None, metric=primary_metric(task["answer_type"], root), error=f"{type(error).__name__}: {error}")
                    data.save_json(output / "tasks" / (task["task_id"] + ".json"), result)
                print(f"[{index}/{len(pending)}] {task['task_id']}: {result['status']}; score={result.get('score')}", flush=True)
                evaluation_summary(tasks, output, root, judge)
    finally:
        summary = evaluation_summary(tasks, output, root, judge)
        after = {task["task_id"] + ".json": hashlib.sha256((args.run / "tasks" / (task["task_id"] + ".json")).read_bytes()).hexdigest() for task in tasks}
        integrity = {"predictions_unchanged": predictions == after,
                     "frozen_ontology_unchanged": input_hashes(ontology_run / "ontology") == generation["ontology"]["files_sha256"],
                     "original_lake_unchanged": input_hashes(source) == generation["source_files_sha256"],
                     "references_unchanged": hashlib.sha256((root / "workload/environment.json").read_bytes()).hexdigest() == settings["workload_sha256"]}
        integrity["unchanged"] = all(integrity.values())
        data.save_json(output / "source-integrity.json", integrity)
        if not integrity["unchanged"]:
            summary.update(status="integrity_failed", overall_score=None, overall_percent=None)
            data.save_json(output / "summary.json", summary)
        data.export_run(args.run)
    if summary["status"] == "completed":
        print(f"Native overall: {summary['overall_percent']:.2f}% ({len(tasks)} main questions)", flush=True)
    else:
        print(f"Native evaluation unresolved for {len(summary['unresolved'])} questions; use --resume to retry.", flush=True)
    return 0 if summary["status"] == "completed" else 1


def evaluation_main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) == 2 and argv[0] == "--worker-job":
        return evaluation_worker(Path(argv[1]))
    parser = argparse.ArgumentParser(description="Evaluate saved OGDA Environment answers and executed Python with the original KramaBench evaluator.")
    parser.add_argument("--run", required=True, type=data.workspace_path)
    parser.add_argument("--config", type=data.workspace_path, default=data.workspace_path(EVALUATOR_CONFIG))
    parser.add_argument("--output", type=data.workspace_path)
    parser.add_argument("--scratch", type=data.workspace_path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--task-timeout", type=float, default=1200)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    positive_settings(parser, args)
    try:
        return run_evaluation(args)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("pipeline", "construction", "data", "evaluation"), default="pipeline")
    parser.add_argument("--ontology-run", type=data.workspace_path, help="Reuse a previously constructed frozen ontology")
    parser.add_argument("--run-id", help="Fresh pipeline/construction identifier; defaults to a UTC timestamp")
    parser.add_argument("--evaluator-config", type=data.workspace_path, default=data.workspace_path(EVALUATOR_CONFIG))
    generation_options(parser)
    args = parser.parse_args(argv)
    positive_settings(parser, args)
    try:
        root = benchmark_root()
        args.benchmark, args.benchmark_root = "Kramabench", root
        if args.stage in {"data", "evaluation"} and not args.ontology_run:
            raise ValueError("--ontology-run is required for downstream stages")
        if args.stage == "construction" and args.ontology_run:
            raise ValueError("Construction creates a fresh run; omit --ontology-run")
        if not args.ontology_run:
            from .. import launch

            if args.resume:
                raise ValueError("Construction has no resume stage; pass --ontology-run to resume downstream predictions/scoring")
            run_id = args.run_id or f"ogda-environment-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}"
            if Path(run_id).name != run_id or run_id in {".", ".."}:
                raise ValueError("--run-id must be one directory name")
            output = args.output or WORKSPACE / "results/Kramabench" / run_id
            scratch = args.scratch or WORKSPACE / "exp_env/Kramabench" / run_id
            construction = argparse.Namespace(lake=root / "data/environment/input", config=args.config, run_id=run_id,
                                              output=output, scratch=scratch, exclude=[], strip_key=[],
                                              **{name: None for name in launch.OVERRIDES})
            if args.dry_run:
                from ogda.llm import OpenAICompatibleClient

                select_tasks(construction.lake, args.limit, root=root)
                launch.public_config(args.config)
                client = OpenAICompatibleClient(args.config)
                try:
                    for role in ("proposer", "verifier", "experience", "data_agent"):
                        client.role_config(role)
                finally:
                    client.close()
                if args.stage == "pipeline":
                    public_judge(judge_config(args.evaluator_config))
                for path in (output, scratch):
                    if path.exists():
                        raise FileExistsError(f"Fresh construction directory already exists: {path}")
                print(f"Dry run passed: all Environment inputs, 20 main questions, YAML roles, and fresh paths checked.\nResults: {output}\nScratch: {scratch}")
                return 0
            code = launch.run(construction)
            if code or args.stage == "construction":
                return code
            args.ontology_run = output
            args.output, args.scratch = output / "data-agent-evaluation", scratch.parent / (scratch.name + "-data-agent-evaluation")
        if args.stage != "evaluation":
            if args.stage == "pipeline":
                public_judge(judge_config(args.evaluator_config))
            code = data.run(args)
            if args.stage == "data" or args.dry_run:
                return code
        generation_run = args.output or args.ontology_run / "data-agent-evaluation"
        evaluation = argparse.Namespace(run=generation_run, config=args.evaluator_config, output=None, scratch=None,
                                        workers=args.workers, task_timeout=args.task_timeout, resume=args.resume, dry_run=args.dry_run)
        evaluation_code = run_evaluation(evaluation)
        return evaluation_code
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
