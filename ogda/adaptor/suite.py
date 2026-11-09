"""Run OGDA construction and grounded data-agent Tasks on every benchmark, with each benchmark's native scoring.

Everything a benchmark adds (its Lakes and Tasks, its answer instruction and check, its scoring) is in `benchmarks/`."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import Counter
from dataclasses import asdict

from . import benchmarks
from .artifacts import ONTOLOGY_SOURCE, WORKSPACE, file_digest, input_hashes, read_json, safe_id, save_json
from .benchmarks import BENCHMARKS, GENERATION_FAILURES, RESOLVED_SCORES, identifiers, judge_config, public_judge
from .launch import public_config, public_model

sys.path.insert(0, str(ONTOLOGY_SOURCE))

PROCESS_LOCK = threading.Lock()
LIVE_PROCESSES: dict[int, subprocess.Popen] = {}
STOPPING = threading.Event()


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def inventory(benchmark: str, root: Path, lake_root: Path, discovery_scope: str = "real-test") -> dict:
    """Whitelist generation fields and independently verify every clean source Lake."""
    if discovery_scope != "real-test":
        raise ValueError("This run selects only the DiscoveryBench real test subset")
    lakes, tasks = benchmarks.get(benchmark).inventory(root, lake_root)
    seen = set()
    for task in tasks:
        safe_id(task["task_id"])
        if task["task_id"] in seen or not isinstance(task["question"], str) or not task["question"].strip():
            raise ValueError(f"Duplicate identifier or empty question: {task['task_id']}")
        seen.add(task["task_id"])
    if not lakes or not tasks:
        raise ValueError(f"No Lakes or tasks selected for {benchmark}")
    return {"benchmark": benchmark, "lakes": lakes, "tasks": tasks}


def generation_config(path: Path, *, allow_data_agent_reasoning: bool = False) -> dict:
    import yaml
    from ogda.adaptor.llm import ModelConfig, model_request_kwargs
    from ogda.construction import ConstructionConfig
    from ogda.data_agent import DataAgentConfig
    from ogda.llm import OpenAICompatibleClient

    document = yaml.safe_load(path.read_text()) or {}
    construction = ConstructionConfig.from_dict(document.get("construction"))
    data_agent = DataAgentConfig.from_dict(document.get("data_agent"))
    if data_agent != DataAgentConfig():
        raise ValueError("The suite requires the method's default DataAgentConfig budgets")
    client = OpenAICompatibleClient(path)
    try:
        roles = {}
        for role in ("proposer", "verifier", "experience", "data_agent"):
            settings = client.role_config(role)
            model = ModelConfig(provider=settings.get("provider", "openai"), model=settings["model"],
                                reasoning_effort=settings.get("reasoning_effort"), model_kwargs=settings.get("model_kwargs", {}))
            options = model_request_kwargs(model)
            explicit_none = settings.get("reasoning_effort") in {"none", "no"}
            bodies = [settings, settings.get("model_kwargs", {}), settings.get("model_kwargs", {}).get("extra_body", {}), options, options.get("extra_body", {})]
            disabled = explicit_none or any(body.get("reasoning", {}).get("enabled") is False
                                           or body.get("thinking", {}).get("type") == "disabled" for body in bodies)
            enabled = any(body.get("reasoning", {}).get("enabled") is True
                          or body.get("thinking", {}).get("type") == "enabled" for body in bodies)
            if (not disabled or enabled) and not (role == "data_agent" and allow_data_agent_reasoning):
                raise ValueError(f"Generation reasoning must be explicitly disabled for {role}")
            roles[role] = public_model(path, role=role, request_options=True)
        return {"roles": roles, "construction": asdict(construction), "data_agent": asdict(data_agent)}
    finally:
        client.close()


def check_references(selected: dict, root: Path) -> None:
    benchmark = benchmarks.get(selected["benchmark"])
    for task in selected["tasks"]:
        benchmark.check_reference(task, root)


def construction_state(output: Path, lake: dict) -> dict:
    """A capped enrichment pass is terminal; pending verification or mapping is not."""
    from ogda.ontology import OntologyStore

    store = OntologyStore(output / "ontology")
    ontology = store.load().model_dump(mode="json") if store.exists() else {}
    metadata = ontology.get("metadata", {})
    progress = metadata.get("progress", {})
    pending = [claim_id for claim_id, claim in ontology.get("claims", {}).items() if claim.get("status") == "pending"]
    reason = metadata.get("stop_reason")
    files = input_hashes(output / "ontology" / "lake")
    unassigned = sorted(set(files) - set(progress.get("files", {})))
    terminal = reason in {"converged", "unresolved", "max_passes"}
    final_progress = progress.get("stage") == "done" or (reason == "max_passes" and progress.get("stage") == "schema_enrichment")
    ready = bool(ontology and files and terminal and final_progress and not pending and not progress.get("pending_iteration") and not unassigned)
    return {"lake_id": lake["lake_id"], "usable": ready, "status": "completed" if ready else "unfinished",
            "stop_reason": reason, "pending_claims": pending, "pending_iteration": bool(progress.get("pending_iteration")),
            "unassigned_files": unassigned, "remaining_work": progress.get("remaining_work", []),
            "unresolved_files": [name for name, value in progress.get("files", {}).items() if value.get("status") == "unresolved"]}


def assert_sources(lakes: list[dict]) -> None:
    for lake in lakes:
        if input_hashes(Path(lake["source"])) != lake["source_files_sha256"]:
            raise ValueError(f"Original clean Lake changed: {lake['lake_id']}")


def frozen_barrier(selected: dict, output: Path) -> dict:
    assert_sources(selected["lakes"])
    states = [construction_state(output / "lakes" / lake["lake_id"], lake) for lake in selected["lakes"]]
    if not all(state["usable"] for state in states):
        save_json(output / "construction-status.json", {"complete": False, "lakes": states})
        raise ValueError("Ontology barrier unfinished: " + ", ".join(state["lake_id"] for state in states if not state["usable"]))
    frozen = {lake["lake_id"]: input_hashes(output / "lakes" / lake["lake_id"] / "ontology") for lake in selected["lakes"]}
    previous = read_json(output / "frozen-harnesses.json")
    if previous and previous != frozen:
        raise ValueError("Finished Harness changed after the ontology barrier")
    save_json(output / "frozen-harnesses.json", frozen)
    save_json(output / "construction-status.json", {"complete": True, "lakes": states})
    return frozen


def verify_frozen(selected: dict, output: Path, frozen: dict) -> None:
    assert_sources(selected["lakes"])
    actual = {lake["lake_id"]: input_hashes(output / "lakes" / lake["lake_id"] / "ontology") for lake in selected["lakes"]}
    unchanged = actual == frozen
    save_json(output / "source-integrity.json", {"unchanged": unchanged, "original_lakes_unchanged": True,
                                              "changed_harnesses": [key for key in frozen if frozen[key] != actual[key]]})
    if not unchanged:
        raise ValueError("Frozen Harness changed during task execution")


def redact_error(error: BaseException, configs: list[str]) -> str:
    import yaml

    text = f"{type(error).__name__}: {error}"
    def remove(value):
        nonlocal text
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "api_key" and isinstance(item, str) and item:
                    text = text.replace(item, "[REDACTED]")
                elif key == "api_key_env" and isinstance(item, str) and os.environ.get(item):
                    text = text.replace(os.environ[item], "[REDACTED]")
                else:
                    remove(item)
        elif isinstance(value, list):
            for item in value:
                remove(item)
    for path in configs:
        remove(yaml.safe_load(Path(path).read_text()))
    return text


def build_worker(job: dict) -> dict:
    from . import logging

    output, lake = Path(job["output"]), job["lake"]
    assert_sources([lake])
    output.mkdir(parents=True, exist_ok=True)
    previous = read_json(output / "run-info.json")
    if previous and (Path(previous["lake"]).resolve() != Path(lake["source"]).resolve()
                     or previous.get("exclude") or previous.get("strip_key")):
        raise ValueError("Existing construction has incompatible inputs or settings")
    before = read_json(output / "input-hashes-before.json")
    if before and before != lake["source_files_sha256"]:
        raise ValueError("Existing construction used different source inputs")
    save_json(output / "input-hashes-before.json", lake["source_files_sha256"])
    command = ["build", "--config", job["config"], "--lake", lake["source"], "--lake-id", lake["lake_id"],
               "--output", str(output), "--scratch", job["scratch"]]
    if job.get("construction_max_tokens") is not None:
        command.extend(["--max-tokens", str(job["construction_max_tokens"])])
    if (output / "ontology" / "ontology.json").exists():
        command.append("--resume")
    try:
        logging.main(command)
    finally:
        logging.close_clients()
        assert_sources([lake])
        save_json(output / "input-hashes-after.json", input_hashes(Path(lake["source"])))
        save_json(output / "source-integrity.json", {"unchanged": True})
    result = construction_state(output, lake)
    result.update(config=job["config"], config_sha256=file_digest(Path(job["config"])),
                  construction_max_tokens=job.get("construction_max_tokens"))
    save_json(output / "suite-construction.json", result)
    return result


def ask_worker(job: dict) -> dict:
    import yaml
    from . import logging
    from ogda.data_agent import WORKSPACE as WORKSPACE_NAME, DataAgentConfig, answer
    from ogda.ontology import OntologyStore

    task, benchmark = job["task"], benchmarks.get(job["benchmark"])
    result = dict(identifiers(task), status="failed", workflow="")
    artifact, scratch = Path(job["artifact"]), Path(job["scratch"])
    logging.install_logging(artifact / "logs")
    client = logging.LoggedClient(job["config"], artifact / "logs" / "llm.jsonl")
    try:
        store = OntologyStore(job["ontology"])
        if input_hashes(Path(job["ontology"])) != job["frozen_hashes"]:
            raise ValueError("Harness changed before this Task")
        # The answer format is the only benchmark text an agent sees.
        method_task = {"id": task["task_id"], "question": task["question"], "answer_format": benchmark.answer_format(task)}
        workspace = scratch / WORKSPACE_NAME
        # The agent sees a failed check and can repair its answer before the Task ends.
        document = yaml.safe_load(Path(job["config"]).read_text()) or {}
        config = DataAgentConfig.from_dict(document.get("data_agent"))
        pipeline = answer(store, client, method_task, config, scratch, artifact,
                          check=lambda value: benchmark.check(value, task, workspace))
        result.update(status="incomplete", stop_reason=pipeline["stop_reason"],
                      explanation=pipeline["explanation"], steps=pipeline["steps"],
                      outside_ontology=pipeline["graph"]["outside_ontology"], pipeline_result=str(artifact / "result.json"))
        if pipeline["status"] == "answered":
            result.update(benchmark.prediction(pipeline["answer"], task, workspace, artifact, pipeline["explanation"]))
            result["status"] = "completed"
    finally:
        result["llm_usage"] = dict(client.usage)
        result["token_usage"] = client.usage["total_tokens"]
        if job["benchmark"] == "Kramabench":
            result.update(benchmark.pipeline_evidence({}, artifact=artifact), artifact_root=str(artifact))
        logging.close_clients()
        if input_hashes(Path(job["ontology"])) != job["frozen_hashes"]:
            raise ValueError("Task modified the frozen Harness")
    return result


def evaluation_worker(job: dict) -> dict:
    prediction = read_json(Path(job["prediction"]))
    if any(prediction.get(key) != value for key, value in identifiers(job["task"]).items()):
        raise ValueError("Saved prediction does not match the selected task")
    if prediction.get("status") not in {*GENERATION_FAILURES, "completed"}:
        raise ValueError("Generation is not terminal; an absent prediction is not a scored failure")
    return benchmarks.get(job["benchmark"]).evaluate(job, prediction)


def run_worker(path: Path) -> int:
    job = read_json(path)
    started = time.monotonic()
    result = {"status": "evaluator_failed" if job["kind"] == "evaluation" else "failed"}
    def terminate(signum, frame):
        raise TimeoutError("Worker terminated by the suite deadline or controller")
    signal.signal(signal.SIGTERM, terminate)
    try:
        result = {"construction": build_worker, "tasks": ask_worker, "evaluation": evaluation_worker}[job["kind"]](job)
    except Exception as error:
        result.update(identifiers(job.get("task", {})), error=redact_error(error, [job["config"], job["evaluator_config"]]))
        if job["kind"] == "evaluation":
            result["score"] = None
        print(result["error"], flush=True)
    finally:
        result["runtime"] = time.monotonic() - started
        save_json(Path(job["result_path"]), result)
    return 0 if result["status"] in {"completed", *RESOLVED_SCORES} else 1


def process_identity(pid: int) -> str | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        fields = stat[stat.rfind(")") + 2:].split()
        return None if fields[0] == "Z" else fields[19]
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None


def live_external_build(output: Path) -> list[int]:
    """Find current process handles; a stale running-status file is not evidence."""
    result = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            args = (proc / "cmdline").read_bytes().decode(errors="replace").split("\0")
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if "--output" in args and args[args.index("--output") + 1] == str(output) and process_identity(int(proc.name)):
            result.append(int(proc.name))
    return result


def stop_process(process: subprocess.Popen) -> None:
    descendants = descendant_processes(process.pid)
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    # The sandbox creates its own session. Its parent normally cleans it up on SIGTERM.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()
    for pid, identity in descendants.items():
        if process_identity(pid) == identity:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def descendant_processes(parent: int) -> dict[int, str]:
    """Snapshot descendants so sandbox workers in their own sessions are also cleaned."""
    parents = {}
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            stat = (path / "stat").read_text()
            fields = stat[stat.rfind(")") + 2:].split()
            if fields[0] != "Z":
                parents[int(path.name)] = (int(fields[1]), fields[19])
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    descendants = {}
    generation = {parent}
    while generation:
        children = {pid for pid, (ppid, _) in parents.items() if ppid in generation and pid not in descendants}
        descendants.update({pid: parents[pid][1] for pid in children})
        generation = children
    return descendants


def stop_controller(signum, frame) -> None:
    STOPPING.set()
    with PROCESS_LOCK:
        running = list(LIVE_PROCESSES.values())
    # Signal all workers first so their sandbox cleanup can run concurrently.
    for process in running:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    raise KeyboardInterrupt("Suite controller interrupted")


def execute_job(job: dict, output_path: Path, work: Path, artifact: Path, timeout: float | None) -> dict:
    if STOPPING.is_set():
        raise KeyboardInterrupt("Suite controller is stopping")
    attempt = "attempt-" + uuid.uuid4().hex[:12]
    work, artifact = work / attempt, artifact / attempt
    work.mkdir(parents=True)
    artifact.mkdir(parents=True)
    job = {**job, "artifact": str(artifact), "scratch": str(work / "agent"), "result_path": str(work / "result.json")}
    if job["kind"] == "construction":
        job["scratch"] = str(work / "construction")
    job_path = work / "job.json"
    save_json(job_path, job)
    console = artifact / "console.log"
    state_path = work / "worker-process.json"
    state = {"kind": job["kind"], "id": job.get("task", job.get("lake", {})).get("task_id", job.get("lake", {}).get("lake_id")),
             "job": str(job_path), "result": str(output_path), "console": str(console), "started": time.time(), "status": "starting"}
    started = time.monotonic()
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, [str(Path(__file__).resolve().parents[2]),
                                                                str(ONTOLOGY_SOURCE), str(WORKSPACE),
                                                                environment.get("PYTHONPATH")]))
    with console.open("w") as stream:
        process = subprocess.Popen([sys.executable, "-B", "-m", "ogda.adaptor.suite", "--worker-job", str(job_path)],
                                   cwd=work, env=environment, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        with PROCESS_LOCK:
            LIVE_PROCESSES[process.pid] = process
        state.update(pid=process.pid, process_start=process_identity(process.pid), status="running")
        save_json(state_path, state)
        timed_out = False
        try:
            if timeout is None:
                while process.poll() is None:
                    if STOPPING.wait(.5):
                        stop_process(process)
                        break
            else:
                deadline = time.monotonic() + timeout
                while process.poll() is None:
                    if STOPPING.wait(min(.5, max(0, deadline - time.monotonic()))):
                        stop_process(process)
                        break
                    if time.monotonic() >= deadline:
                        raise subprocess.TimeoutExpired(process.args, timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            stop_process(process)
        except BaseException:
            stop_process(process)
            raise
        finally:
            with PROCESS_LOCK:
                LIVE_PROCESSES.pop(process.pid, None)
    result = read_json(Path(job["result_path"]))
    if not result or timed_out:
        result = {**identifiers(job.get("task", {})), "status": "evaluator_failed" if job["kind"] == "evaluation" else "failed",
                  "error": f"Worker exceeded {timeout:g}s" if timed_out else f"Worker exited {process.returncode} without saving a result"}
        if job["kind"] == "evaluation":
            result["score"] = None
    result.update(worker_exit_code=process.returncode, wall_runtime=time.monotonic() - started, log_file=str(console))
    if job["kind"] == "tasks" and job.get("benchmark") == "Kramabench":
        result.update(benchmarks.get("Kramabench").pipeline_evidence(result, artifact=artifact), artifact_root=str(artifact))
    save_json(output_path, result)
    state.update(status=result["status"], ended=time.time(), worker_exit_code=process.returncode, timed_out=timed_out)
    save_json(state_path, state)
    return result


def reconcile_workers(scratch: Path) -> None:
    for path in scratch.glob("**/worker-process.json"):
        state = read_json(path)
        if state.get("status") != "running":
            continue
        if state.get("process_start") and process_identity(state["pid"]) == state["process_start"]:
            raise ValueError(f"A previous worker is still live: PID {state['pid']}; wait for it before resuming")
        result_path = Path(state["result"])
        saved = read_json(Path(state["job"]).parent / "result.json")
        if saved and not result_path.exists():
            save_json(result_path, saved)
        state.update(status="recovered" if saved else "interrupted", ended=time.time())
        save_json(path, state)


def bounded_jobs(jobs: list[tuple], workers: int) -> list[dict]:
    if not 1 <= workers <= 64:
        raise ValueError("Worker count must be between 1 and 64")
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(execute_job, *arguments): arguments[0] for arguments in jobs}
        try:
            for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
                result = future.result()
                results.append(result)
                task = futures[future].get("task", futures[future].get("lake", {}))
                name = task.get("task_id", task.get("lake_id"))
                print(f"[{index}/{len(jobs)}] {futures[future]['kind']} {name}: {result['status']}", flush=True)
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    return results


def accepted_prediction(result: dict, task: dict, benchmark: str) -> bool:
    if result.get("status") != "completed" or any(result.get(key) != value for key, value in identifiers(task).items()):
        return False
    return benchmarks.get(benchmark).accepted(result, task)


def accepted_score(result: dict, task: dict, prediction_hash: str, prediction: dict | None = None) -> bool:
    score = result.get("score")
    return (all(result.get(key) == value for key, value in identifiers(task).items())
            and result.get("prediction_sha256") == prediction_hash and result.get("status") in RESOLVED_SCORES
            and isinstance(score, (int, float)) and math.isfinite(score) and 0 <= score <= 1
            and (benchmarks.get("Kramabench").accepted_evaluation(result, prediction) if "answer_type" in task
                 else benchmarks.get("discoverybench").accepted_evaluation(result, prediction) if "query_id" in task
                 else result["status"] != "generation_failed" or score == 0))


def summarize(selected: dict, output: Path, evaluation: bool = False, evaluation_label: str | None = None) -> dict:
    folder = output / "evaluation" if evaluation else output
    if evaluation and evaluation_label is not None:
        folder /= safe_id(evaluation_label)
    rows = [read_json(folder / "tasks" / (task["task_id"] + ".json")) or dict(identifiers(task), status="pending") for task in selected["tasks"]]
    summary = {"benchmark": selected["benchmark"], "denominator": len(rows), "status_counts": dict(Counter(row["status"] for row in rows))}
    if evaluation:
        resolved = [row for row in rows if row["status"] in RESOLVED_SCORES and isinstance(row.get("score"), (float, int))
                    and math.isfinite(row["score"]) and 0 <= row["score"] <= 1]
        if selected["benchmark"] == "Kramabench":
            resolved = [row for row in resolved if benchmarks.get("Kramabench").accepted_evaluation(row)]
        elif selected["benchmark"] == "discoverybench":
            resolved = [row for row in resolved if benchmarks.get("discoverybench").accepted_evaluation(row)]
        complete = len(resolved) == len(rows)
        summary.update(status="completed" if complete else "incomplete", scored=len(resolved),
                       overall_score=sum(row["score"] for row in resolved) / len(rows) if complete else None,
                       unresolved=[row["task_id"] for row in rows if row not in resolved],
                       generation_failure_policy="zero in the full selected task denominator",
                       evaluator_error_policy="unscored and retryable")
        if selected["benchmark"] == "Kramabench":
            summary.update(benchmarks.get("Kramabench").native_summary(rows),
                           generation_failure_policy="native evaluator missing-answer handling; pipeline judged from saved code")
        elif selected["benchmark"] == "discoverybench":
            summary.update(benchmarks.get("discoverybench").native_summary())
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / "scores.csv").open("w", newline="") as stream:
            fields = ["task_id", "status", "generation_status", "metric", "score", "runtime", "error"]
            if selected["benchmark"] == "Kramabench":
                fields.insert(5, "pipeline_score")
            writer = csv.DictWriter(stream, fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    else:
        summary.update(status="completed" if all(row["status"] in {"completed", *GENERATION_FAILURES} for row in rows) else "incomplete")
    save_json(folder / "summary.json", summary)
    return summary


def prepare_run(selected: dict, args, root: Path, models: dict, judge: dict | None) -> tuple[Path, Path]:
    benchmark = selected["benchmark"]
    output, scratch = WORKSPACE / "results" / benchmark / args.run_id, WORKSPACE / "exp_env" / benchmark / args.run_id
    config = {"benchmark": benchmark, "run_id": args.run_id, "config": str(args.config), "config_sha256": file_digest(args.config),
              "evaluator_config": str(args.evaluator_config), "evaluator_config_sha256": file_digest(args.evaluator_config),
              "construction_config": str(args.construction_config), "construction_config_sha256": file_digest(args.construction_config),
              "construction_max_tokens": args.construction_max_tokens,
              "workers": args.workers, "lake_workers": args.lake_workers, "task_timeout": args.task_timeout,
              "evaluation_timeout": args.evaluation_timeout, "construction_timeout": args.construction_timeout,
              "discovery_scope": args.discovery_scope if benchmark == "discoverybench" else None,
              "inventory_sha256": digest(selected), "models": models, "judge": judge,
              "method_source_sha256": digest({path.relative_to(ONTOLOGY_SOURCE).as_posix(): file_digest(path)
                                              for path in sorted((ONTOLOGY_SOURCE / "ogda").rglob("*"))
                                              if path.is_file() and path.suffix in {".py", ".yaml"}
                                              and path.relative_to(ONTOLOGY_SOURCE / "ogda").parts[0] != "adaptor"}),
              "benchmark_root": str(root)}
    previous = read_json(output / "run_config.json")
    if previous and (not args.resume or previous != config):
        allowed = {"construction_config", "construction_config_sha256", "construction_max_tokens"}
        changed = {key for key in previous.keys() | config.keys() if previous.get(key) != config.get(key)}
        if not args.resume or not changed <= allowed or (output / "frozen-harnesses.json").exists() or (output / "tasks").exists():
            raise ValueError(f"Run exists or resume configuration/manifest differs: {output}")
        save_json(output / "construction-config-history" / (digest(previous) + ".json"), previous)
    if not previous and output.exists() and any(output.iterdir()):
        # The parent may have already built Lakes with the normal construction launcher.
        if not args.resume or any(path.name != "lakes" for path in output.iterdir()):
            raise ValueError(f"Existing result directory is not an adoptable construction-only run: {output}")
        for lake in selected["lakes"]:
            build = output / "lakes" / lake["lake_id"]
            info = read_json(build / "run-info.json")
            if build.exists() and (not info
                                   or Path(info["lake"]).resolve() != Path(lake["source"])
                                   or info.get("exclude") or info.get("strip_key")
                                   or read_json(build / "input-hashes-before.json") != lake["source_files_sha256"]
                                   or info.get("public_config", {}).get("roles") != public_config(args.config)["roles"]):
                raise ValueError(f"Existing Lake construction differs from the requested inputs or models: {build}")
            continuation = read_json(build / "construction-continuation.json")
            if continuation:
                continuation_config = Path(continuation["config"])
                if generation_config(continuation_config)["roles"] != models["roles"]:
                    raise ValueError(f"Construction continuation changed model or reasoning roles: {build}")
    output.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    if args.resume:
        reconcile_workers(scratch)
    save_json(output / "run_config.json", config)
    save_json(output / "tasks.json", selected["tasks"])
    save_json(output / "lakes.json", selected["lakes"])
    return output, scratch


def run_benchmark(selected: dict, args, root: Path, models: dict, judge: dict | None) -> int:
    output, scratch = prepare_run(selected, args, root, models, judge)
    common = {"benchmark": selected["benchmark"], "benchmark_root": str(root), "config": str(args.config),
              "evaluator_config": str(args.evaluator_config)}
    if args.stage in {"construction", "pipeline"}:
        jobs = []
        for lake in selected["lakes"]:
            build = output / "lakes" / lake["lake_id"]
            handles = live_external_build(build)
            if handles:
                print(f"Waiting for existing construction {lake['lake_id']} (PIDs {handles})", flush=True)
            while handles:
                time.sleep(5)
                handles = live_external_build(build)
            state = construction_state(build, lake)
            if state["usable"]:
                continue
            continuation = read_json(build / "construction-continuation.json")
            construction_config = continuation.get("config", str(args.construction_config))
            jobs.append(({**common, "kind": "construction", "config": construction_config,
                          "construction_max_tokens": args.construction_max_tokens, "lake": lake, "output": str(build)},
                         build / "suite-construction.json", scratch / "lakes" / lake["lake_id"] / "suite",
                         build / "logs" / "suite", args.construction_timeout))
        bounded_jobs(jobs, args.lake_workers)
        frozen = frozen_barrier(selected, output)
        if args.stage == "construction":
            return 0
    else:
        frozen = frozen_barrier(selected, output)
    if args.stage in {"tasks", "pipeline"}:
        jobs = []
        for task in selected["tasks"]:
            prediction = output / "tasks" / (task["task_id"] + ".json")
            if args.resume and accepted_prediction(read_json(prediction), task, selected["benchmark"]):
                continue
            job = {**common, "kind": "tasks", "task": identifiers(task),
                   "ontology": str(output / "lakes" / task["lake_id"] / "ontology"), "frozen_hashes": frozen[task["lake_id"]]}
            jobs.append((job, prediction, scratch / "tasks" / task["task_id"], output / "logs" / "tasks" / task["task_id"], args.task_timeout))
        bounded_jobs(jobs, args.workers)
        verify_frozen(selected, output, frozen)
        summary = summarize(selected, output)
        if summary["status"] != "completed" or args.stage == "tasks":
            return 0 if summary["status"] == "completed" else 1
    if args.stage in {"evaluation", "pipeline"}:
        jobs = []
        for task in selected["tasks"]:
            prediction_path = output / "tasks" / (task["task_id"] + ".json")
            prediction = read_json(prediction_path)
            if prediction.get("status") not in {"completed", *GENERATION_FAILURES}:
                raise ValueError(f"Generation has not finished: {task['task_id']}")
            evaluated = output / "evaluation" / "tasks" / (task["task_id"] + ".json")
            previous = read_json(evaluated)
            prediction_hash = file_digest(prediction_path)
            if args.resume and accepted_score(previous, task, prediction_hash, prediction):
                continue
            job = {**common, "kind": "evaluation", "task": task, "prediction": str(prediction_path)}
            jobs.append((job, evaluated, scratch / "evaluation" / task["task_id"],
                         output / "logs" / "evaluation" / task["task_id"], args.evaluation_timeout))
        bounded_jobs(jobs, args.workers)
        for task in selected["tasks"]:
            path = output / "evaluation" / "tasks" / (task["task_id"] + ".json")
            result = read_json(path)
            result["prediction_sha256"] = file_digest(output / "tasks" / (task["task_id"] + ".json"))
            save_json(path, result)
        verify_frozen(selected, output, frozen)
        summary = summarize(selected, output, evaluation=True)
        print(json.dumps(summary, indent=2), flush=True)
        return 0 if summary["status"] == "completed" else 1
    return 0


def canonical_benchmark(value: str) -> str:
    aliases = {"all": "all", "kramabench": "Kramabench", "discoverybench": "discoverybench"}
    try:
        return aliases[value.lower()]
    except KeyError:
        raise argparse.ArgumentTypeError("Choose Kramabench, discoverybench, or all") from None


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) == 2 and argv[0] == "--worker-job":
        return run_worker(Path(argv[1]))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", action="append", type=canonical_benchmark, default=[])
    parser.add_argument("--config", required=True, type=lambda path: (WORKSPACE / path).resolve())
    parser.add_argument("--evaluator-config", required=True, type=lambda path: (WORKSPACE / path).resolve())
    parser.add_argument("--run-id", required=True, type=safe_id)
    parser.add_argument("--stage", choices=("construction", "tasks", "evaluation", "pipeline"), default="pipeline")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--lake-workers", type=int, default=16)
    parser.add_argument("--discovery-scope", choices=("real-test",), default="real-test")
    parser.add_argument("--construction-config", type=lambda path: (WORKSPACE / path).resolve(), help="Explicit construction-only continuation YAML; defaults to --config")
    parser.add_argument("--construction-max-tokens", type=int, help="Explicit construction budget override; other method settings come from YAML")
    parser.add_argument("--task-timeout", type=float, default=3600)
    parser.add_argument("--evaluation-timeout", type=float, default=3600)
    parser.add_argument("--construction-timeout", type=float, help="Default: no construction wall deadline")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    args.construction_config = args.construction_config or args.config
    if not 1 <= args.workers <= 64 or not 1 <= args.lake_workers <= 64:
        parser.error("--workers and --lake-workers must be between 1 and 64")
    if args.construction_max_tokens is not None and args.construction_max_tokens < 1:
        parser.error("--construction-max-tokens must be positive")
    if any(not math.isfinite(value) or value <= 0 for value in [args.task_timeout, args.evaluation_timeout, *([args.construction_timeout] if args.construction_timeout is not None else [])]):
        parser.error("Worker deadlines must be positive and finite")
    selected_benchmarks = BENCHMARKS if not args.benchmark or "all" in args.benchmark else tuple(benchmark for benchmark in BENCHMARKS if benchmark in args.benchmark)
    from ogda.adaptor.runtime import prepare

    runtime = prepare()
    models = generation_config(args.config)
    construction_models = generation_config(args.construction_config)
    if construction_models["roles"] != models["roles"]:
        raise ValueError("Construction-only YAML must retain the generation model/reasoning roles")
    judge = public_judge(judge_config(args.evaluator_config))
    manifests = []
    for benchmark in selected_benchmarks:
        root = runtime / "benchmark" / benchmark
        selected = inventory(benchmark, root, WORKSPACE / "datalake" / benchmark, args.discovery_scope)
        check_references(selected, root)
        manifests.append((selected, root))
        print(f"{benchmark}: {len(selected['lakes'])} Lakes, {len(selected['tasks'])} tasks; OGDA grounded data agent and native evaluation", flush=True)
    if args.dry_run:
        save_json(WORKSPACE / "exp_env" / "setup" / args.run_id / "dry-run-manifest.json",
                  {"benchmarks": [{"benchmark": selected["benchmark"], "lakes": len(selected["lakes"]), "tasks": len(selected["tasks"]),
                                   "inventory_sha256": digest(selected)} for selected, _ in manifests],
                   "models": models, "judge": judge,
                   "workers": args.workers, "lake_workers": args.lake_workers, "discovery_scope": args.discovery_scope})
        print("Dry run passed: clean Lake hashes, sanitized documentation, full task/reference inventory, generation reasoning, and evaluator configuration checked. No model calls.", flush=True)
        return 0
    setup = WORKSPACE / "exp_env" / "setup"
    setup.mkdir(parents=True, exist_ok=True)
    settings = models["roles"]["proposer"]["settings"]
    endpoint = settings.get("base_url", settings.get("api_base", "")).rstrip("/")
    with (setup / f".ogda-suite-{digest(endpoint)}.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another OGDA benchmark suite controller is live on this generation endpoint") from None
        signal.signal(signal.SIGTERM, stop_controller)
        signal.signal(signal.SIGINT, stop_controller)
        save_json(setup / args.run_id / "suite-controller.json", {"pid": os.getpid(), "process_start": process_identity(os.getpid()),
                                                                 "benchmarks": list(selected_benchmarks), "started": time.time(), "stage": args.stage})
        for selected, root in manifests:
            print(f"Starting {selected['benchmark']} {args.stage}; all phases finish before the next benchmark", flush=True)
            code = run_benchmark(selected, args, root, models, judge)
            if code:
                return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
