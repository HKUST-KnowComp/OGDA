"""Earlier DiscoveryBench runner (single data_agent or Planner/Analyst) on one frozen ontology, with full agent traces.

The suite replaced it for benchmark runs. Its single-agent INSTRUCTIONS are a DiscoveryBench-specific system prompt."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import signal
import shutil
import subprocess
import sys
import time
import uuid
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from ..artifacts import ONTOLOGY_SOURCE, WORKSPACE, export_system_prompts, input_hashes, read_json, read_jsonl, save_json
from ..benchmarks.discoverybench import ANSWER_FORMAT, validate_answer
from ..launch import public_model
from ..logging import LoggedClient
from ..viewers.dialogue import SnapshotBusy
from ..viewers.html import export_html

sys.path.insert(0, str(ONTOLOGY_SOURCE))

IDENTIFIERS = ("task_id", "split", "dataset", "metadata_id", "query_id", "question")
INSTRUCTIONS = """Answer the supplied scientific question by executing Python against the actual Lake data. Use the frozen ontology and its search_ontology/get_schema tools to locate columns, understand their meaning, and inspect recorded data quirks. The ontology provides context; compute numerical evidence from the data for your answer. LAKE_ROOT contains your private copy of the data and its documentation; SCRATCH_ROOT is your working directory. Inspect missing values and units, preserve duplicate rows unless a justified analysis requires another unit, and document any filtering, deduplication, weighting, or aggregation choices. State the discovered relationship clearly with relevant numerical comparisons and the population analyzed. Finish by calling final_answer({'hypothesis': 'your evidence-supported answer', 'workflow': 'the concrete analysis steps, columns, filters, and calculations used'}). Both fields must be nonempty strings. Do not invent measurements or return only a plan."""


class DataAgentClient(LoggedClient):
    """Logged client for downstream data-agent requests."""


def workspace_path(value: str | Path) -> Path:
    return (WORKSPACE / value).resolve()


def select_tasks(lake: Path, dataset: str, metadata_ids: list[str], limit: int | None = None, root: Path | None = None) -> list[dict]:
    from scripts.discoverybench.run_smolagents_real import REAL_ROOT, discover_tasks

    files = input_hashes(lake)
    tasks = []
    for task in discover_tasks(root or REAL_ROOT, "test"):
        if task["dataset"] != dataset or (metadata_ids and str(task["metadata_id"]) not in metadata_ids):
            continue
        if not all(source["name"] in files for source in task["inputs"]):
            continue
        for source in task["inputs"]:
            if files[source["name"]] != source["sha256"]:
                raise ValueError(f"Frozen Lake differs from benchmark input: {source['name']}")
        tasks.append({key: value for key, value in task.items() if key != "agent_metadata"})
    if not tasks:
        raise ValueError("No test questions match the selected metadata and frozen Lake inputs")
    return tasks[:limit] if limit else tasks


def task_prompt(task: dict, ontology, lake: Path) -> str:
    from ogda.retrieval import retrieve

    return json.dumps({"stage": "data_evaluation", "question": task["question"],
                       "dataset_files": [source["name"] for source in task["inputs"]],
                       "lake_files": sorted(input_hashes(lake)),
                       "ontology_context": retrieve(ontology, task["question"])}, ensure_ascii=False)


def planner_analyst_config(config: Path) -> dict:
    import yaml
    from ogda.data_agent import DataAgentConfig

    document = yaml.safe_load(config.read_text()) or {}
    try:
        return asdict(DataAgentConfig.from_dict(document.get("data_agent")))
    except TypeError as error:
        raise ValueError(f"Invalid data_agent settings: {error}") from error


def run_worker(job_path: Path) -> int:
    from .. import logging
    from ogda.agent import AgentConfig
    from ogda.construction import ontology_tools
    from ogda.ontology import OntologyStore

    job = read_json(job_path)
    task, artifact = job["task"], Path(job["artifact_root"])
    identifiers, answer_field, instructions, validator = IDENTIFIERS, "hypothesis", INSTRUCTIONS, validate_answer
    if job.get("benchmark") == "Kramabench":
        from . import kramabench

        identifiers, answer_field, instructions = kramabench.IDENTIFIERS, "answer", kramabench.INSTRUCTIONS
        validator = lambda answer: kramabench.validate_answer(answer, task["answer_type"])
    result = {key: task[key] for key in identifiers}
    result.update(status="failed", workflow="", artifact_root=str(artifact))
    result[answer_field] = None if answer_field == "answer" else ""
    started = time.monotonic()
    client = None

    def terminate(signum, frame):
        raise TimeoutError("Data-agent task deadline exceeded")

    signal.signal(signal.SIGTERM, terminate)
    try:
        logging.install_logging(Path(job["log_root"]))
        client = DataAgentClient(job["config"], artifact / "llm.jsonl")
        store = OntologyStore(job["ontology"])
        if job.get("pipeline", "single-agent") == "planner-analyst":
            from ogda.data_agent import DataAgentConfig, answer

            store.lake_root = Path(job["lake"])
            pipeline_result = answer(
                store, client, {"id": task["task_id"], "question": task["question"], "answer_format": ANSWER_FORMAT},
                DataAgentConfig.from_dict(job["data_agent_config"]), Path(job["agent_scratch"]), artifact,
                check=validate_answer,
            )
            result.update(pipeline="planner-analyst", pipeline_result=str(artifact / "result.json"),
                          steps=pipeline_result["steps"], outside_ontology=pipeline_result["graph"]["outside_ontology"],
                          explanation=pipeline_result["explanation"], stop_reason=pipeline_result["stop_reason"])
            if pipeline_result["status"] == "answered":
                validate_answer(pipeline_result["answer"])
                result.update(status="completed", hypothesis=pipeline_result["answer"]["hypothesis"],
                              workflow=pipeline_result["answer"]["workflow"])
            else:
                result["status"] = "incomplete"
        else:
            visit = logging.logged_run_agent(
                client, role="data_agent", instructions=instructions, task=job["prompt"], validate=validator,
                lake_root=Path(job["lake"]), scratch_root=Path(job["agent_scratch"]), artifact_root=artifact,
                config=AgentConfig(max_steps=job["max_steps"]), tools=ontology_tools(store.load()),
            )
            result.update(status="completed" if visit.completed else "incomplete", steps=visit.steps)
            if visit.completed:
                result.update({answer_field: visit.answer[answer_field], "workflow": visit.answer["workflow"]})
    except Exception as error:
        result.update(error=f"{type(error).__name__}: {error}")
    finally:
        usage = dict(client.usage) if client is not None else {}
        result.update(runtime=time.monotonic() - started, llm_usage=usage,
                      token_usage_input=usage.get("input_tokens"), token_usage_output=usage.get("output_tokens"),
                      token_usage=usage.get("total_tokens"))
        if job.get("benchmark") == "Kramabench":
            from ..benchmarks.kramabench import pipeline_evidence

            result.update(pipeline_evidence(result, artifact=artifact))
        logging.close_clients()
        save_json(Path(job["result_path"]), result)
    return 0 if result["status"] == "completed" else 1


def execute_worker(worker_script: Path, job_path: Path, work: Path, console_path: Path, timeout: float) -> tuple[int, bool, float]:
    """Bound one isolated worker and retain its console even if it cannot save a result."""
    started = time.monotonic()
    timed_out = False
    with console_path.open("w") as console:
        process = subprocess.Popen([sys.executable, "-B", str(worker_script), "--worker-job", str(job_path)],
                                   cwd=work, stdout=console, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    return process.returncode, timed_out, time.monotonic() - started


def execute_task(task: dict, args, ontology, lake: Path, output: Path, scratch: Path) -> dict:
    identifiers, prompt_function = IDENTIFIERS, task_prompt
    worker_script = WORKSPACE / "scripts/discoverybench/run_rsi_data_agents.py"
    benchmark = getattr(args, "benchmark", None)
    if benchmark == "Kramabench":
        from . import kramabench

        identifiers, prompt_function = kramabench.IDENTIFIERS, kramabench.task_prompt
        worker_script = WORKSPACE / "scripts/Kramabench/run_rsi_data_agents.py"
    attempt = "attempt-" + uuid.uuid4().hex[:12]
    work = scratch / task["task_id"] / attempt
    work.mkdir(parents=True)
    private_lake = work / "input"
    shutil.copytree(lake, private_lake)
    artifact = output / "logs" / "data_agent" / task["task_id"] / attempt
    artifact.mkdir(parents=True)
    job_path, result_path = work / "job.json", work / "result.json"
    save_json(job_path, {"task": {key: task[key] for key in identifiers}, "benchmark": benchmark,
                         "prompt": prompt_function(task, ontology, private_lake), "config": str(args.config),
                         "ontology": str(args.ontology_run / "ontology"), "lake": str(private_lake),
                         "agent_scratch": str(work / "agent"), "artifact_root": str(artifact),
                         "log_root": str(output / "logs"), "result_path": str(result_path),
                         "max_steps": args.max_steps, "pipeline": getattr(args, "pipeline", "single-agent"),
                         "data_agent_config": getattr(args, "data_agent_config", None)})
    exit_code, timed_out, wall_runtime = execute_worker(worker_script, job_path, work, artifact / "console.log", args.task_timeout)
    result = read_json(result_path)
    if not result:
        usage = {name: sum(row.get("usage", {}).get(name, 0) for row in read_jsonl(artifact / "llm.jsonl"))
                 for name in ("calls", "input_tokens", "output_tokens", "total_tokens", "cost_usd")}
        result = {key: task[key] for key in identifiers}
        result.update(status="failed", workflow="", llm_usage=usage,
                      token_usage_input=usage["input_tokens"], token_usage_output=usage["output_tokens"],
                      token_usage=usage["total_tokens"], error=f"Worker exited {exit_code} without a result")
        result["answer" if benchmark == "Kramabench" else "hypothesis"] = None if benchmark == "Kramabench" else ""
    if timed_out:
        result.update(status="failed", error=f"Task exceeded {args.task_timeout:g}s deadline")
    result.update(wall_runtime=wall_runtime, worker_exit_code=exit_code,
                  artifact_root=str(artifact), log_file=str(artifact / "console.log"))
    if benchmark == "Kramabench":
        from ..benchmarks.kramabench import pipeline_evidence

        result.update(pipeline_evidence(result, artifact=artifact))
    save_json(output / "tasks" / (task["task_id"] + ".json"), result)
    return result


def merge_journals(output: Path) -> list[dict]:
    calls = []
    for path in sorted((output / "logs" / "data_agent").rglob("llm.jsonl")):
        calls.extend(read_jsonl(path))
    calls.sort(key=lambda row: (row.get("time", 0), row.get("visit", ""), row.get("call_index", 0)))
    for number, row in enumerate(calls, 1):
        row["local_call_index"] = row.get("call_index")
        row["call_index"] = number
    (output / "logs" / "llm.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in calls))
    return calls


def summarize(tasks: list[dict], output: Path, frozen: dict, benchmark: str | None = None) -> dict:
    rows = [read_json(output / "tasks" / (task["task_id"] + ".json")) or {"status": "pending"} for task in tasks]
    calls = merge_journals(output)
    usage = {name: sum(row.get("usage", {}).get(name, 0) for row in calls)
             for name in ("calls", "input_tokens", "output_tokens", "total_tokens", "cost_usd")}
    counts = dict(Counter(row["status"] for row in rows))
    summary = {"tasks": len(tasks), "status": counts,
               "run_status": "running" if counts.get("pending") else "completed" if counts.get("completed") == len(tasks) else "incomplete",
               "evaluation": "not run", "llm_usage": usage, "frozen_ontology": frozen}
    if benchmark:
        summary["benchmark"] = benchmark
    save_json(output / "generation-summary.json", summary)
    return summary


def export_run(output: Path) -> None:
    from .. import logging
    from ..viewers.dialogue import export_data_agent_dialogue

    export_data_agent_dialogue(output)
    logging.export_logs(output)
    export_system_prompts(output)
    export_html(output, output / "logs" / "agent-loops.html")


def run(args) -> int:
    from ogda.ontology import OntologyStore

    store = OntologyStore(args.ontology_run / "ontology")
    ontology = store.load()
    if not store.lake_root.is_dir():
        raise FileNotFoundError(f"Frozen Lake is missing: {store.lake_root}")
    benchmark = getattr(args, "benchmark", None)
    identifiers = IDENTIFIERS
    if benchmark == "Kramabench":
        from . import kramabench

        tasks = kramabench.select_tasks(store.lake_root, args.limit, root=args.benchmark_root)
        identifiers = kramabench.IDENTIFIERS
    else:
        tasks = select_tasks(store.lake_root, args.dataset, args.metadata_id, args.limit)
    pipeline = getattr(args, "pipeline", "single-agent")
    if pipeline == "planner-analyst":
        args.data_agent_config = planner_analyst_config(args.config)
        model = {"roles": {role: public_model(args.config, role=role) for role in ("data_agent",)}}
    else:
        model = public_model(args.config, request_options=benchmark == "Kramabench")
    frozen_before = input_hashes(store.root)
    source = Path(read_json(args.ontology_run / "run-info.json").get("lake", str(store.lake_root)))
    source_before = input_hashes(source)
    frozen = {"run": str(args.ontology_run), "ontology": str(store.path),
              "ontology_sha256": frozen_before["ontology.json"], "files_sha256": frozen_before}
    output = args.output or args.ontology_run / "data-agent-evaluation"
    scratch = args.scratch or WORKSPACE / ("exp_env/Kramabench" if benchmark == "Kramabench" else "exp_env/discoverybench") / (args.ontology_run.name + "-data-agent-evaluation")
    for path in (output, scratch):
        if path.is_relative_to(store.root) or store.root.is_relative_to(path) or path.is_relative_to(source) or source.is_relative_to(path):
            raise ValueError("Evaluation directories must be separate from the frozen ontology and original Lake")
    if output.is_relative_to(scratch) or scratch.is_relative_to(output):
        raise ValueError("Results and scratch must be separate trees")
    config = {"ontology": frozen, "model": model, "max_steps": args.max_steps,
              "pipeline": pipeline, "ontology_source": str(ONTOLOGY_SOURCE),
              "task_timeout": args.task_timeout, "tasks_sha256": hashlib.sha256(json.dumps(tasks, sort_keys=True).encode()).hexdigest()}
    if pipeline == "planner-analyst":
        config.pop("max_steps")  # Planner and Analyst limits come from the YAML data_agent block.
        config["data_agent"] = args.data_agent_config
    if benchmark:
        config.update(benchmark=benchmark, workers=args.workers, source_files_sha256=source_before)
    if args.resume:
        if read_json(output / "run_config.json") != config:
            raise ValueError("Resume configuration, frozen ontology, or questions differ; choose a new output directory")
    elif any(path.exists() and any(path.iterdir()) for path in (output, scratch)):
        raise FileExistsError("Evaluation output or scratch already exists; use --resume or new directories")
    selection = "Kramabench environment main workload" if benchmark == "Kramabench" else "metadata IDs: " + ", ".join(sorted({str(t["metadata_id"]) for t in tasks}))
    print(f"Questions: {len(tasks)}; {selection}", flush=True)
    model_names = "; ".join(f"{role}: {value['settings']['model']}" for role, value in model["roles"].items()) if pipeline == "planner-analyst" else model["settings"]["model"]
    print(f"Data agent ({pipeline}): {model_names}; workers: {args.workers}\nResults: {output}\nScratch: {scratch}", flush=True)
    if args.dry_run:
        print("Dry run passed: frozen ontology, matching input hashes, selected questions, and YAML roles checked; no API calls or evaluation writes.")
        return 0
    output.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(exist_ok=True)
    save_json(output / "run_config.json", config)
    save_json(output / "tasks.json", tasks)
    save_json(output / "input-hashes-before.json", {"frozen": frozen_before, "source": source_before})
    pending = [task for task in tasks if not (args.resume and read_json(output / "tasks" / (task["task_id"] + ".json")).get("status") == "completed")]
    info = {"scope": "frozen-ontology downstream data-agent evaluation", "ontology_run": str(args.ontology_run),
            "config": str(args.config), "public_config": model, "output": str(output), "scratch": str(scratch),
            "pipeline": pipeline, "ontology_source": str(ONTOLOGY_SOURCE),
            "started": time.time(), "status": "running"}
    save_json(output / "run-info.json", info)
    try:
        summarize(tasks, output, frozen, benchmark)
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(execute_task, task, args, ontology, store.lake_root, output, scratch): task for task in pending}
            for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
                task = futures[future]
                try:
                    result = future.result()
                except Exception as error:
                    result = {key: task[key] for key in identifiers}
                    result.update(status="failed", workflow="", error=f"{type(error).__name__}: {error}")
                    result["answer" if benchmark == "Kramabench" else "hypothesis"] = None if benchmark == "Kramabench" else ""
                    save_json(output / "tasks" / (task["task_id"] + ".json"), result)
                print(f"[{index}/{len(pending)}] {task['task_id']}: {result['status']}", flush=True)
                try:
                    summarize(tasks, output, frozen, benchmark)
                    export_run(output)
                except (json.JSONDecodeError, SnapshotBusy):
                    pass  # Another worker is writing a trace; the final export reads completed files.
    finally:
        frozen_after, source_after = input_hashes(store.root), input_hashes(source)
        integrity = {"unchanged": frozen_before == frozen_after and source_before == source_after,
                     "frozen_ontology_unchanged": frozen_before == frozen_after, "original_lake_unchanged": source_before == source_after}
        save_json(output / "input-hashes-after.json", {"frozen": frozen_after, "source": source_after})
        save_json(output / "source-integrity.json", integrity)
        summary = summarize(tasks, output, frozen, benchmark)
        info.update(ended=time.time(), status=summary["run_status"], source_integrity=integrity)
        save_json(output / "run-info.json", info)
        save_json(output / "logged-run-status.json", {"status": info["status"]})
        export_run(output)
    print(f"Agent execution HTML: {output / 'logs/agent-loops.html'}", flush=True)
    return 0 if summary["run_status"] == "completed" and integrity["unchanged"] else 1


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) == 2 and argv[0] == "--worker-job":
        return run_worker(Path(argv[1]))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ontology-run", required=True, type=workspace_path)
    parser.add_argument("--config", type=workspace_path, default=workspace_path("llm_config/ds.yaml"))
    parser.add_argument("--dataset", default="meta_regression")
    parser.add_argument("--metadata-id", action="append", default=[], help="Select a metadata ID; repeat to select several (e.g. 0)")
    parser.add_argument("--output", type=workspace_path)
    parser.add_argument("--scratch", type=workspace_path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--pipeline", choices=("single-agent", "planner-analyst"), default="single-agent",
                        help="Planner/Analyst uses roles.planner, roles.analyst, and YAML data_agent settings")
    parser.add_argument("--max-steps", type=int, default=20, help="Single-agent limit; Planner/Analyst limits come from YAML data_agent")
    parser.add_argument("--task-timeout", type=float, default=600)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.workers < 1 or args.max_steps < 1 or (args.limit is not None and args.limit < 1) or not math.isfinite(args.task_timeout) or args.task_timeout <= 0:
        parser.error("Workers, steps, limit, and timeout must be positive")
    try:
        return run(args)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
