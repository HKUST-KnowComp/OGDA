"""Run benchmark experiments: build Ontologies, answer Tasks and score answers, one stage at a time.

Run as a script: `uv run python scripts/benchmark.py --benchmark <name> --stage <stage>`. `scripts/` is on
`sys.path` when run this way, so this module imports the benchmark registry as a top-level package.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, Callable

import yaml
from openai import OpenAI

from benchmarks import REPO, ROOTS, SCORER_ROOTS, Judge, get, judge_from_config
from methods import pipeline_code, run_baseline, run_ogda
from prepare_lakes import verify
from ogda.cli import construction_summary
from ogda.construction import Construction, ConstructionConfig
from ogda.data_agent import DataAgentConfig
from ogda.llm import LLMError, OpenAICompatibleClient
from ogda.ontology import OntologyStore
from ogda.sandbox import IsolatedExecutor

STAGES = ("construct", "infer", "evaluate")
METHODS = ("ogda", "baseline")
# The model roles each stage calls; preflight checks that each resolves to a model.
ROLES = {"construct": ("proposer", "verifier", "experience"), "infer": ("data_agent",), "evaluate": ("evaluator",)}
STAGE_DIRS = {"construct": "construction", "infer": "inference", "evaluate": "inference"}
STALLED = {"verification_incomplete", "proposer_incomplete"}  # Stop reasons that wait for --resume.


class PreflightError(Exception):
    """A problem found before a stage starts; the message says how to fix it."""


@dataclass
class Plan:
    run_id: str
    lakes: list[dict]
    tasks: list[dict]
    out: Path
    env: Path


# Seams: tests replace these to run without a live model or sandbox.

def make_client(config: Path, record_path: Path) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(config, record_path=record_path)


def check_sandbox(scratch: Path) -> None:
    """Run one cell in the isolated worker on an empty Lake."""
    scratch = Path(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch, prefix="sandbox-check-") as probe:
        probe = Path(probe)
        for name in ("lake", "scratch", "artifacts"):
            (probe / name).mkdir()
        executor = IsolatedExecutor(probe / "lake", probe / "scratch", probe / "artifacts")
        try:
            executor("1")
        except RuntimeError as error:  # SandboxUnavailable is a RuntimeError.
            raise PreflightError(f"code sandbox unavailable: {error}") from None
        finally:
            executor.cleanup()


def check_judge(judge: Judge) -> None:
    """One short JSON-mode call to the judge, which both official scorers need."""
    try:
        OpenAI(api_key=judge.api_key, base_url=judge.base_url).chat.completions.create(
            model=judge.model, messages=[{"role": "user", "content": 'Reply with the JSON object {"ok": true}.'}],
            response_format={"type": "json_object"})
    except Exception as error:
        raise PreflightError(f"The judge {judge.model} at {judge.base_url} failed a JSON-mode call, which both "
                             f"scorers need: {type(error).__name__}: {error}") from None


def role_model(config: Path, role: str) -> str:
    client = OpenAICompatibleClient(config)
    try:
        return client.role_config(role)["model"]
    except LLMError as error:
        raise PreflightError(f"{config}: {error}") from None
    finally:
        client.close()


def model_slug(model: str) -> str:
    """The model name after its last `/`, safe as part of a folder name."""
    return re.sub(r"[^A-Za-z0-9.-]", "-", model.rsplit("/", 1)[-1])


def default_run_id(prefix: str, model: str, now: datetime) -> str:
    return f"{prefix}_{model_slug(model)}_{now.astimezone(timezone.utc):%Y%m%dT%H%M%SZ}"


def git_commit() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def load_config(path: Path) -> dict:
    try:
        document = yaml.safe_load(Path(path).read_text()) or {}
    except (OSError, yaml.YAMLError) as error:
        raise PreflightError(f"cannot load config {path}: {error}") from None
    if not isinstance(document, dict):
        raise PreflightError(f"config {path} must be a YAML mapping")
    return document


def select(args, module, root: Path) -> tuple[list[dict], list[dict], list[str], list[str]]:
    """The selected Lakes and Tasks, and the selected ids that do not exist."""
    lakes = {lake["lake_id"]: lake for lake in module.lakes(root)}
    tasks = module.tasks(root)
    unknown_lakes = [lake_id for lake_id in args.lakes or [] if lake_id not in lakes]
    unknown_tasks = sorted(set(args.tasks or []) - {task["task_id"] for task in tasks})
    if args.lakes:
        tasks = [task for task in tasks if task["lake_id"] in args.lakes]
    if args.tasks:
        tasks = [task for task in tasks if task["task_id"] in args.tasks]
    if args.limit is not None:
        tasks = tasks[:args.limit]
    if args.stage == "construct" and not args.tasks:
        lake_ids = [lake_id for lake_id in lakes if not args.lakes or lake_id in args.lakes]
    else:
        lake_ids = list(dict.fromkeys(task["lake_id"] for task in tasks))
    return [lakes[lake_id] for lake_id in lake_ids], tasks, unknown_lakes, unknown_tasks


def preflight(args) -> Plan:
    """Check everything the stage needs before it starts; raise PreflightError naming the first problem."""
    # 1. The config loads and the stage's roles resolve to a model.
    document = load_config(args.config)
    if args.stage == "construct":
        try:
            ConstructionConfig.from_dict(document.get("construction"))
        except (TypeError, ValueError) as error:
            raise PreflightError(f"construction settings in {args.config}: {error}") from None
    if args.stage == "infer":
        try:
            DataAgentConfig.from_dict(document.get("data_agent"))
        except (TypeError, ValueError) as error:
            raise PreflightError(f"data_agent settings in {args.config}: {error}") from None
    models = {role: role_model(args.config, role) for role in ROLES[args.stage]}
    if args.run_id:
        run_id = args.run_id
    elif args.stage == "evaluate":
        raise PreflightError("--run-id is required for evaluate: it names the infer run to score")
    else:
        prefix, role = ("ontology", "proposer") if args.stage == "construct" else (args.method, "data_agent")
        run_id = default_run_id(prefix, models[role], datetime.now(timezone.utc))

    # 2. The benchmark source directory exists; for evaluate, the scorer directory too.
    for root in (args.benchmark_root, args.scorer_root) if args.stage == "evaluate" else (args.benchmark_root,):
        if not root.is_dir():
            raise PreflightError(f"{root} is missing; follow benchmark/README.md in the enclosing project to obtain the upstream sources")
    module = get(args.benchmark)
    lakes, tasks, unknown_lakes, unknown_tasks = select(args, module, args.benchmark_root)

    # 3. The clean Lakes exist and have not changed.
    if args.stage in {"construct", "infer"}:
        hint = f"run scripts/prepare_lakes.py --benchmark {args.benchmark}"
        if not args.lake_dir.is_dir():
            raise PreflightError(f"{args.lake_dir} is missing; {hint}")
        try:
            verify(args.lake_dir, [lake["lake_id"] for lake in lakes])
        except ValueError as error:
            raise PreflightError(str(error)) from None

    # 4. The sandbox is available.
    if args.stage != "evaluate":
        check_sandbox(args.env_dir)

    # 5. ogda has a finished Ontology for every Lake its Tasks need.
    if args.stage == "infer" and args.method == "ogda":
        if not args.ontology_run:
            raise PreflightError("--method ogda needs --ontology-run: the construct run whose Ontologies it uses")
        for lake in lakes:
            summary_path = ontology_dir(args, lake["lake_id"]).parent / "summary.json"
            summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
            if not summary or summary.get("status") == "error" or summary.get("stop_reason") in STALLED:
                raise PreflightError(f"{summary_path.parent} has no finished Ontology; finish it with --stage construct "
                                     f"--run-id {args.ontology_run} --resume")

    # 6. The stage's results folder is new, unless resuming.
    out = args.results_dir / STAGE_DIRS[args.stage] / run_id
    if args.stage != "evaluate" and out.exists() and not args.resume:
        raise PreflightError(f"{out} exists; pass --resume to continue it or choose another --run-id")

    # 7. evaluate: the infer run exists and every prediction maps to a known Task.
    if args.stage == "evaluate":
        if not out.is_dir():
            raise PreflightError(f"{out} does not exist; --run-id must name an infer run")
        predictions = out / "predictions.jsonl"
        if not predictions.exists():
            raise PreflightError(f"{predictions} is missing; rerun infer with --resume")
        known = {task["task_id"] for task in module.tasks(args.benchmark_root)}
        unknown_predictions = sorted({row["task_id"] for row in read_jsonl(predictions)} - known)
        if unknown_predictions:
            raise PreflightError(f"{predictions} has unknown Task(s) for {args.benchmark}: "
                                 f"{', '.join(unknown_predictions)}")

    # 8. evaluate: the evaluator role is an openai/ model and its endpoint answers in JSON mode.
    if args.stage == "evaluate":
        try:
            judge = judge_from_config(args.config)
        except (LLMError, ValueError) as error:
            raise PreflightError(f"{args.config}: {error}") from None
        check_judge(judge)

    # 9. The selected Lakes and Tasks exist.
    if unknown_lakes:
        raise PreflightError(f"Unknown Lake(s) for {args.benchmark}: {', '.join(unknown_lakes)}")
    if unknown_tasks:
        raise PreflightError(f"Unknown Task(s) for {args.benchmark}: {', '.join(unknown_tasks)}")
    if not lakes:
        raise PreflightError("No Lakes selected")
    return Plan(run_id=run_id, lakes=lakes, tasks=tasks, out=out, env=args.env_dir / run_id)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False))


def run_each(work: Callable[[dict], Any], items: list[dict], workers: int) -> list:
    """`work(item)` for every item, in order. With one worker the items run here, one after another. Otherwise each
    item runs in a fresh process, `workers` at a time, so the host-side work of different Lakes and Tasks (data
    checks, layout fan-out, search) runs on separate CPUs rather than in one Python process. A process that stops
    abruptly, as on an out-of-memory kill, stops the items still running or waiting: each counts as failed, and
    --resume runs it again."""
    if workers <= 1 or not items:
        return [work(item) for item in items]
    results = []
    with ProcessPoolExecutor(max_workers=min(workers, len(items)), mp_context=multiprocessing.get_context("spawn"),
                             max_tasks_per_child=1) as pool:
        futures = [pool.submit(work, item) for item in items]
        for item, future in zip(items, futures):
            try:
                results.append(future.result())
            except BrokenProcessPool:
                print(f"{item.get('task_id') or item['lake_id']}: its worker process stopped before finishing",
                      file=sys.stderr)
                results.append(False)
    return results


def build_lake(args, plan: Plan, config: ConstructionConfig, lake: dict) -> bool:
    """Build or resume one Lake's Ontology; a failure is recorded in its summary.json."""
    lake_id = lake["lake_id"]
    out = plan.out / lake_id
    summary_path = out / "summary.json"
    if args.resume and summary_path.exists():
        summary = json.loads(summary_path.read_text())
        if summary.get("status") != "error" and summary.get("stop_reason") not in STALLED:
            print(f"{lake_id}: finished, skipped")
            return True
    try:
        llm = make_client(args.config, out / "logs" / "llm.jsonl")
        try:
            store = OntologyStore(out / "ontology")
            construction = Construction(store, llm, config, scratch_root=plan.env / "construction" / lake_id,
                                        log_root=out / "logs")
            if not (args.resume and store.exists()):
                construction.start(lake_id, args.lake_dir / lake_id)
            summary = construction_summary(construction.run(), llm)
        finally:
            llm.close()
    except Exception as error:  # Record the failure; the other Lakes continue.
        message = f"{type(error).__name__}: {error}"
        write_json(summary_path, {"lake_id": lake_id, "status": "error", "error": message})
        print(f"{lake_id}: {message}", file=sys.stderr)
        return False
    write_json(summary_path, summary)
    print(f"{lake_id}: {summary['stop_reason']}")
    return True


def construct(args, plan: Plan) -> int:
    """Build or resume one Ontology per selected Lake; return 1 if any Lake failed."""
    settings = load_config(args.config).get("construction") or {}
    config = ConstructionConfig.from_dict(settings)
    write_json(plan.out / "run.json", {
        "stage": "construct", "benchmark": args.benchmark, "run_id": plan.run_id, "git_commit": git_commit(),
        "models": {role: role_model(args.config, role) for role in ROLES["construct"]},
        "construction": settings, "lakes": [lake["lake_id"] for lake in plan.lakes]})
    finished = run_each(partial(build_lake, args, plan, config), plan.lakes, args.workers)
    return 0 if all(finished) else 1


def ontology_dir(args, lake_id: str) -> Path:
    return args.results_dir / STAGE_DIRS["construct"] / args.ontology_run / lake_id / "ontology"


def answer_task(args, plan: Plan, config: DataAgentConfig, task: dict) -> bool:
    """Answer one Task with the chosen method; a failure is recorded in its result.json."""
    module = get(args.benchmark)
    task_id, lake_id = task["task_id"], task["lake_id"]
    task_out = plan.out / "tasks" / task_id
    task_env = plan.env / "inference" / "tasks" / task_id
    result_path = task_out / "result.json"
    if args.resume and result_path.exists() and json.loads(result_path.read_text())["status"] != "error":
        print(f"{task_id}: finished, skipped")
        return True
    shutil.rmtree(task_out, ignore_errors=True)
    shutil.rmtree(task_env, ignore_errors=True)
    agent_task = {"id": task_id, "question": task["question"], "answer_format": module.answer_format(task)}
    check = lambda value: module.check(value, task)
    try:
        llm = make_client(args.config, task_out / "logs" / "llm.jsonl")
        try:
            if args.method == "ogda":
                result = run_ogda(agent_task, ontology_dir(args, lake_id), llm, config, task_env, task_out, check)
            else:
                result = run_baseline(agent_task, args.lake_dir / lake_id, llm, config, task_env, task_out, check)
        finally:
            llm.close()
    except Exception as error:  # Record the failure; the other Tasks continue.
        message = f"{type(error).__name__}: {error}"
        write_json(task_out / "result.json", {"task": agent_task, "status": "error", "error": message})
        print(f"{task_id}: {message}", file=sys.stderr)
        return False
    print(f"{task_id}: {result['status']}")
    return True


def infer(args, plan: Plan) -> int:
    """Answer every selected Task with the chosen method; return 1 if any Task failed."""
    module = get(args.benchmark)
    settings = load_config(args.config).get("data_agent") or {}
    config = DataAgentConfig.from_dict(settings)
    # A resumed run keeps the Tasks it answered before; both lists follow the benchmark's Task order.
    all_tasks = module.tasks(args.benchmark_root)
    run_path = plan.out / "run.json"
    selected = {task["task_id"] for task in plan.tasks}
    if args.resume and run_path.exists():
        selected |= set(json.loads(run_path.read_text()).get("tasks", []))
    write_json(run_path, {
        "stage": "infer", "benchmark": args.benchmark, "run_id": plan.run_id, "method": args.method,
        "ontology_run": args.ontology_run, "git_commit": git_commit(),
        "model": role_model(args.config, "data_agent"), "data_agent": settings,
        "tasks": [task["task_id"] for task in all_tasks if task["task_id"] in selected]})

    finished = run_each(partial(answer_task, args, plan, config), plan.tasks, args.workers)

    rows = []
    for task in all_tasks:
        task_out = plan.out / "tasks" / task["task_id"]
        if not (task_out / "result.json").exists():
            continue
        result = json.loads((task_out / "result.json").read_text())
        if result["status"] == "error":
            result = {**result, "answer": None}
        rows.append({"task_id": task["task_id"], "lake_id": task["lake_id"], "status": result["status"],
                     **module.prediction(result, task, pipeline_code(task_out / "logs"))})
    temporary = plan.out / "predictions.jsonl.tmp"
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    temporary.replace(plan.out / "predictions.jsonl")
    return 0 if all(finished) else 1


def evaluation_dir(args, plan: Plan, judge: Judge) -> Path:
    """A new `evaluation/<judge-model>_<UTC>` folder, or with --resume the newest one for this judge model."""
    prefix = model_slug(judge.model)
    if args.resume:
        existing = sorted(path for path in (plan.out / "evaluation").glob(f"{prefix}_*") if path.is_dir())
        if existing:
            return existing[-1]
    return plan.out / "evaluation" / f"{prefix}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"


def latest_scores(path: Path) -> list[dict]:
    """The last row per task_id in scores.jsonl: a retried score replaces its evaluator_failed row."""
    return list({row["task_id"]: row for row in read_jsonl(path)}.values()) if path.exists() else []


def evaluate(args, plan: Plan) -> int:
    """Score every prediction of the infer run with the official scorer; return 1 if any scorer failed."""
    module = get(args.benchmark)
    judge = judge_from_config(args.config)
    folder = evaluation_dir(args, plan, judge)
    judged_by = {"judge_model": judge.model, "judge_base_url": judge.base_url}
    write_json(folder / "run.json", {**judged_by, "infer_run": plan.run_id, "git_commit": git_commit()})
    scores_path = folder / "scores.jsonl"
    # Scored and generation_failed predictions are final; evaluator_failed ones are scored again.
    done = {row["task_id"] for row in latest_scores(scores_path) if row["status"] != "evaluator_failed"}
    tasks = {task["task_id"]: task for task in plan.tasks}
    predictions = [row for row in read_jsonl(plan.out / "predictions.jsonl")
                   if row["task_id"] in tasks and row["task_id"] not in done]
    lock = threading.Lock()

    def score(prediction: dict) -> None:
        task_id = prediction["task_id"]
        try:
            result = module.evaluate(prediction, tasks[task_id], judge, args.benchmark_root, args.scorer_root,
                                     folder)
        except Exception as error:  # Record the failure; the other predictions continue.
            message = f"{type(error).__name__}: {error}"
            result = {"task_id": task_id, "status": "evaluator_failed", "score": 0.0, "error": message}
            print(f"{task_id}: {message}", file=sys.stderr)
        with lock, scores_path.open("a") as scores:
            scores.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(f"{task_id}: {result['status']}")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(score, predictions))

    scores = latest_scores(scores_path)
    write_json(folder / "summary.json", {**judged_by, **module.summarize(scores, args.scorer_root, judge)})
    return 1 if any(row["status"] == "evaluator_failed" for row in scores) else 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, choices=sorted(ROOTS))
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    parser.add_argument("--run-id", help="default: ontology_<model>_<UTC> or <method>_<model>_<UTC>")
    parser.add_argument("--method", choices=METHODS, default="ogda", help="infer only")
    parser.add_argument("--ontology-run", help="the construct run-id whose Ontologies ogda uses")
    parser.add_argument("--lakes", nargs="+")
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    for name in ("--benchmark-root", "--scorer-root", "--lake-dir", "--results-dir", "--env-dir"):
        parser.add_argument(name, type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    args.benchmark_root = args.benchmark_root or ROOTS[args.benchmark]
    args.scorer_root = args.scorer_root or SCORER_ROOTS[args.benchmark]
    args.lake_dir = args.lake_dir or REPO / "lake" / args.benchmark
    args.results_dir = args.results_dir or REPO / "results" / args.benchmark
    args.env_dir = args.env_dir or REPO / "env" / args.benchmark
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        plan = preflight(args)
    except PreflightError as error:
        print(f"preflight: {error}", file=sys.stderr)
        return 2
    return {"construct": construct, "infer": infer, "evaluate": evaluate}[args.stage](args, plan)


if __name__ == "__main__":
    sys.exit(main())
