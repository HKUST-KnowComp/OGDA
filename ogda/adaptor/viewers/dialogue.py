"""Export recorded data-agent conversations to a compact, standalone HTML file."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from string import Template
from urllib.parse import quote

from ..artifacts import WORKSPACE, key, pretty, read_json

JOURNALS = ("llm-inputs.jsonl", "provider-inputs.jsonl", "llm.jsonl", "steps-full.jsonl", "execution.jsonl")
AGENT_ROLES = {"data_agent", "planner", "analyst"}


class SnapshotBusy(ValueError):
    """A visit snapshot was still being written after several short retries."""


def read_journal(path: Path) -> list[dict]:
    """Keep complete records while an active writer finishes its last JSONL line."""
    if not path.exists():
        return []
    lines = path.read_bytes().splitlines(keepends=True)
    records = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except (ValueError, UnicodeDecodeError) as error:
            if number == len(lines) and not line.endswith((b"\n", b"\r")):
                break
            raise ValueError(f"Malformed journal record: {path}:{number}: {error}") from error
        if not isinstance(record, dict):
            raise ValueError(f"Expected a JSON object: {path}:{number}")
        records.append(record)
    return records


def visit_snapshot(path: Path) -> dict:
    for attempt in range(3):
        try:
            return read_json(path)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            if attempt == 2:
                raise SnapshotBusy(f"Visit snapshot is not valid JSON: {path}: {error}") from error
            time.sleep(0.02)
    return {}


def evaluation_run(run: Path) -> Path:
    run = run.resolve()
    return run / "data-agent-evaluation" if (run / "data-agent-evaluation").is_dir() or (run / "ontology").is_dir() else run


def raw_link(path: Path, output: Path) -> str:
    return quote(os.path.relpath(path.resolve(), output.parent.resolve()), safe="/.")


def call_key(record: dict) -> tuple:
    return record.get("visit"), record.get("visit_attempt", 1), record.get("local_call_index", record.get("call_index"))


def messages(record: dict) -> list[dict]:
    if record.get("provider_messages") is not None:
        return record["provider_messages"]
    try:
        request = json.loads(record.get("prompt", ""))
    except (ValueError, TypeError):
        return []
    return request.get("messages", []) if isinstance(request, dict) else []


def parse_task(value) -> dict:
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(value)
    except (ValueError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def score_folder(run: Path) -> Path | None:
    summary = read_json(run / "saved-ontology-summary.json")
    if "evaluations" in summary:
        for evaluation in summary["evaluations"]:
            label = evaluation.get("evaluator", {}).get("label")
            if evaluation.get("status") == "completed" and label:
                return run / "evaluation" / label
        return None
    summaries = sorted((run / "evaluation").glob("*/summary.json"), key=lambda path: path.stat().st_mtime_ns)
    return summaries[-1].parent if summaries else None


def build_dialogues(run: Path, output: Path) -> list[dict]:
    """Read each worker independently; aggregate logs are an older-run fallback."""
    logs = run / "logs"
    tasks = {row["task_id"]: row for row in read_json(run / "tasks.json") or []}
    results = {path.stem: read_json(path) for path in sorted((run / "tasks").glob("*.json"))}
    evaluation = score_folder(run)
    scores = {path.stem: read_json(path) for path in (evaluation / "tasks").glob("*.json")} if evaluation else {}
    groups, calls = {}, {}
    has_request_journals = False

    def group_for(record: dict, folder: Path | None = None, artifact: Path | None = None) -> dict:
        visit, attempt = record.get("visit", ""), record.get("visit_attempt", 1)
        identity = visit, attempt
        if identity not in groups:
            parts = visit.split("/")
            task_id = folder.relative_to(logs / "data_agent").parts[0] if folder and folder.is_relative_to(logs / "data_agent") else (
                parts[1] if len(parts) >= 3 and parts[0] == "data_agent" else parts[-2] if len(parts) >= 3 else visit)
            groups[identity] = {"visit": visit, "visit_attempt": attempt, "task_id": task_id,
                                "metadata": {}, "calls": [], "steps": [], "events": [], "links": {}, "folder": folder}
        group = groups[identity]
        if record.get("role") in AGENT_ROLES:
            group["role"] = record["role"]
        if folder:
            group["folder"] = folder
            if artifact is None and folder.is_relative_to(logs / "data_agent"):
                artifact = logs / "data_agent" / Path(*folder.relative_to(logs / "data_agent").parts[:2])
            for name in ("visit.json", *JOURNALS, "steps-full.json", "console.log"):
                if (folder / name).exists():
                    group["links"][name] = raw_link(folder / name, output)
        if artifact:
            group["artifact"] = artifact
        return group

    def collect(row: dict, name: str, position: int, folder: Path, artifact: Path | None = None) -> None:
        group = group_for(row, folder, artifact)
        if name in ("steps-full.jsonl", "execution.jsonl"):
            group["steps" if name == "steps-full.jsonl" else "events"].append(row)
            return
        row.setdefault("call_index", position)
        saved = calls.setdefault(call_key(row), {})
        if name == "provider-inputs.jsonl":
            saved["provider"] = row
            saved["provider_attempts"] = saved.get("provider_attempts", 0) + 1
        else:
            saved["completed" if name == "llm.jsonl" else "started"] = row

    folders = sorted({path.parent for name in ("visit.json", *JOURNALS) for path in (logs / "data_agent").rglob(name)})
    for folder in folders:
        metadata = visit_snapshot(folder / "visit.json")
        defaults = {"visit": folder.relative_to(logs).as_posix(), "visit_attempt": metadata.get("visit_attempt", 1)}
        if metadata:
            group_for({**defaults, **metadata}, folder)["metadata"] = metadata
        for name in JOURNALS:
            if name in JOURNALS[:3] and (folder / name).exists():
                has_request_journals = True
            for position, row in enumerate(read_journal(folder / name), 1):
                collect({**defaults, **row}, name, position, folder)
    for artifact in sorted((logs / "tasks").glob("*/attempt-*")):
        worker_logs = artifact / "logs"
        prefix = f"data_agent/{artifact.parent.name}/{artifact.name}/"

        def scoped(row: dict, local_visit: str = "") -> dict:
            local_visit = row.get("visit") or local_visit or row.get("role") or row.get("phase", "")
            defaults = {"visit_attempt": 1, "role": local_visit} if local_visit in AGENT_ROLES else {"visit_attempt": 1}
            return {**defaults, **row, "visit": prefix + local_visit}

        role_folders = sorted({path.parent for name in ("visit.json", *JOURNALS[3:]) for path in worker_logs.rglob(name)})
        for folder in role_folders:
            metadata = visit_snapshot(folder / "visit.json")
            if metadata:
                row = scoped(metadata, folder.relative_to(worker_logs).as_posix())
                group_for(row, folder, artifact)["metadata"] = row
            for name in JOURNALS[3:]:
                for position, row in enumerate(read_journal(folder / name), 1):
                    collect(scoped(row, folder.relative_to(worker_logs).as_posix()), name, position, folder, artifact)
        for name in JOURNALS[:3]:
            if (worker_logs / name).exists():
                has_request_journals = True
            for position, row in enumerate(read_journal(worker_logs / name), 1):
                row = scoped(row)
                folder = worker_logs / row["visit"][len(prefix):]
                collect(row, name, position, folder, artifact)
                group_for(row)["links"][name] = raw_link(worker_logs / name, output)
    aggregate = read_journal(logs / "llm.jsonl") if not has_request_journals else []
    for position, row in enumerate(aggregate, 1):
        if row.get("role") not in AGENT_ROLES:
            continue
        row.setdefault("call_index", position)
        identity = call_key(row)
        if identity not in calls:
            calls[identity] = {"completed": row}
            group_for(row)["links"]["aggregate llm.jsonl"] = raw_link(logs / "llm.jsonl", output)
    for saved in calls.values():
        record = {**saved.get("started", {}), **saved.get("completed", {})}
        provider = saved.get("provider", {})
        if not record:
            record = provider.copy()
        if record.get("provider_messages") is None and "provider_messages" in provider:
            record["provider_messages"] = provider["provider_messages"]
        record["pending"] = "completed" not in saved
        record["provider_attempts"] = saved.get("provider_attempts", 0)
        group_for(record)["calls"].append(record)

    task_groups = {}
    for group in groups.values():
        task_groups.setdefault(group["task_id"], []).append(group)
    for task_id in dict.fromkeys([*tasks, *results]):
        if task_id not in task_groups:
            group = {"visit": "", "visit_attempt": 1, "task_id": task_id, "metadata": {}, "calls": [],
                     "steps": [], "events": [], "links": {}, "folder": None}
            groups[("pending:" + task_id, 1)] = group
            task_groups[task_id] = [group]
    for task_id, attempts in task_groups.items():
        result = results.get(task_id, {})
        artifact = result.get("artifact_root")
        if not artifact and result.get("log_file"):
            artifact = str(Path(result["log_file"]).parent)
        matching = [group for group in attempts if artifact and group["folder"] and
                    group["folder"].resolve().is_relative_to(Path(artifact).resolve())]
        candidates = matching or attempts
        primary = [group for group in candidates if group.get("role", "data_agent") in {"data_agent", "planner"}]
        latest = max(primary or candidates, key=lambda group: (group["metadata"].get("started", 0), str(group["visit"]), group["visit_attempt"]))
        latest["result"], latest["score"] = result, scores.get(task_id, {})
        for name, path in (("generation prediction", run / "tasks" / f"{task_id}.json"),
                           ("evaluation result", evaluation / "tasks" / f"{task_id}.json" if evaluation else None),
                           ("prediction CSV", Path(result["prediction_file"]) if result.get("prediction_file") else None)):
            if path and path.exists():
                latest["links"][name] = raw_link(path, output)
    for group in groups.values():
        artifact = group.get("artifact")
        if artifact:
            if group.get("role") == "planner" and (group["metadata"] or not (group["folder"] / "visit.json").exists()):
                group["pipeline"] = visit_snapshot(artifact / "result.json")
            elif group.get("role") == "data_agent" and "graph" in (snapshot := visit_snapshot(artifact / "result.json")):
                group["pipeline"] = snapshot  # The grounded Data agent.
            for name in ("result.json", "logs/data_agent.jsonl", "console.log", *(f"logs/{name}" for name in JOURNALS[:3])):
                if (artifact / name).exists():
                    label = name.removeprefix("logs/") if name.removeprefix("logs/") in JOURNALS[:3] else name
                    group["links"].setdefault(label, raw_link(artifact / name, output))
    order = {task_id: number for number, task_id in enumerate(tasks)}
    return [compact_dialogue(group, tasks.get(group["task_id"], {})) for group in sorted(groups.values(),
            key=lambda group: (order.get(group["task_id"], len(order)), group["task_id"],
                               group["metadata"].get("started", 0), group["visit"], group["visit_attempt"]))]


def compact_dialogue(group: dict, task: dict) -> dict:
    metadata, result = group["metadata"], group.get("result", {})
    pipeline = group.get("pipeline", {})
    records = sorted(group["calls"], key=lambda row: (row.get("time", 0), call_key(row)[2]))
    initial = metadata.get("task")
    if initial is None and records:
        initial = next((message.get("content") for message in messages(records[0]) if message.get("role") == "user"), "")
        if isinstance(initial, str) and initial.startswith("New task:\n"):
            initial = initial[len("New task:\n"):]
    parsed = parse_task(initial)
    role = group.get("role", metadata.get("role", "data_agent"))
    answer = metadata.get("answer")
    plan = answer.get("plan", pipeline.get("plan")) if role == "planner" and isinstance(answer, dict) else pipeline.get("plan")
    if role == "planner" and plan is not None:
        answer = None
    explanation = result.get("explanation", pipeline.get("explanation", ""))
    grounded = role == "data_agent" and "graph" in pipeline
    if (role in {"planner", "analyst"} or grounded) and isinstance(answer, dict) and "answer" in answer:
        explanation = answer.get("explanation", explanation)
        answer = answer["answer"]
    if (role == "planner" or grounded) and "answer" in pipeline:
        answer = pipeline["answer"]
    if result.get("status") == "completed" and result.get("prediction_file"):
        answer = {"prediction_file": result["prediction_file"]}
    if answer is None and result.get("status") == "completed" and any(name in result for name in ("answer", "hypothesis")):
        answer = {name: result[name] for name in ("answer", "hypothesis", "workflow") if name in result}
    status = metadata.get("status", result.get("status", "running" if records else "pending"))
    if (role == "planner" or grounded) and (result.get("status") in {"failed", "incomplete"} or pipeline.get("status") in {"failed", "incomplete"}):
        status = result.get("status") if result.get("status") in {"failed", "incomplete"} else pipeline["status"]
    parts = group["visit"].split("/")
    dialogue = {"id": hashlib.sha256(f'{group["task_id"]}|{group["visit"]}|{group["visit_attempt"]}'.encode()).hexdigest()[:16],
                "task_id": group["task_id"], "visit": group["visit"], "visit_attempt": group["visit_attempt"],
                "role": role, "step_id": parsed.get("step", {}).get("id", ""),
                "attempt": parts[2] if len(parts) >= 3 and parts[0] == "data_agent" else parts[-1] or "Awaiting worker",
                "question": task.get("question", result.get("question", parsed.get("question", parsed.get("task", {}).get("question", "No question was recorded.")))),
                "answer_type": task.get("answer_type", result.get("answer_type", parsed.get("answer_type", ""))),
                "status": status, "task": initial, "instructions": metadata.get("instructions", ""),
                "systems": [], "calls": [], "unassigned_events": [], "links": group["links"],
                "answer": answer, "explanation": explanation, "plan": plan, "pipeline": pipeline,
                "score": group.get("score", {}),
                "error": {name: metadata[name] for name in ("error", "message") if name in metadata} or
                         {name: result[name] for name in ("error",) if name in result}}
    calls_by_step = {}
    previous = []
    for record in records:
        full = messages(record)
        shared = 0
        while shared < min(len(previous), len(full)) and previous[shared] == full[shared]:
            shared += 1
        additions = []
        for position, message in enumerate(full[shared:], shared + 1):
            if message.get("role") == "system":
                if message.get("content") not in dialogue["systems"]:
                    dialogue["systems"].append(message.get("content"))
                additions.append({"position": position, "role": "system", "system": dialogue["systems"].index(message.get("content"))})
            elif not previous and message.get("role") == "user" and message.get("content") in (initial, "New task:\n" + pretty(initial)):
                additions.append({"position": position, "role": "user", "initial_task": True})
            else:
                additions.append({"position": position, **message})
        call = {"number": call_key(record)[2], "step": record.get("step", record.get("step_number")),
                "time": record.get("time"), "pending": record["pending"], "shared": shared,
                "input_count": len(full), "inputs": additions, "events": [], "functions": [],
                "usage": {name: record[name] for name in ("model", "wall_seconds", "finish_reason", "usage", "provider_attempts") if name in record},
                "error": {name: record[name] for name in ("error", "error_message", "provider_error") if name in record}}
        if "response_text" in record:
            call["response"] = record["response_text"]
        if not full and "prompt" in record:
            call["prompt"] = record["prompt"]
        for function in record.get("tool_calls") or []:
            function = function.get("function", function)
            arguments = parse_task(function.get("arguments"))
            call["functions"].append({"name": function.get("name", "unknown"),
                                      "arguments": arguments if arguments else function.get("arguments")})
        dialogue["calls"].append(call)
        calls_by_step.setdefault(key(record)[2], []).append(call)
        previous = full
    for event in group["events"]:
        candidates = calls_by_step.get(key(event)[2], [])
        preceding = [call for call in candidates if not event.get("time") or not call["time"] or call["time"] <= event["time"]]
        target = preceding[-1] if preceding else candidates[-1] if candidates else None
        fields = {name: value for name, value in event.items() if name not in {"visit", "visit_attempt", "stage", "step"}}
        (target["events"] if target else dialogue["unassigned_events"]).append(fields)
    for step in group["steps"]:
        candidates = calls_by_step.get(key(step)[2], [])
        if not candidates:
            continue
        call = candidates[-1]
        call["step_error"] = step.get("error")
        if not any(event["event"] == "execution_start" for event in call["events"]) and step.get("code_action") is not None:
            call["events"].append({"event": "execution_start", "code": step["code_action"]})
        if not any(event["event"] in ("execution_end", "execution_error") for event in call["events"]) and step.get("observations") is not None:
            call["observation"] = step["observations"]
        if step.get("action_output") is not None:
            call["submission" if step.get("is_final_answer") else "action_output"] = step["action_output"]
    for call in dialogue["calls"]:
        compact_events(call["events"])
        executed = [event.get("code") for event in call["events"] if event["event"] == "execution_start"]
        for function in call["functions"]:
            arguments = function["arguments"]
            if isinstance(arguments, dict) and isinstance(arguments.get("code"), str) and any(
                    arguments["code"] == code or arguments["code"].replace("\\n", "\n") == code for code in executed):
                function["arguments"] = {name: value for name, value in arguments.items() if name != "code"}
                function["executed_code"] = True
        if "submission" in call and dialogue["answer"] is not None and call["submission"] == dialogue["answer"]:
            call.pop("submission")
            call["accepted_submission"] = True
    return dialogue


def compact_events(events: list[dict]) -> None:
    """Keep worker event order while displaying repeated result payloads once."""
    for index, event in enumerate(events):
        if event["event"] != "worker_message" or not isinstance(event.get("message"), dict):
            continue
        message = event["message"]
        following = events[index + 1:]
        if message.get("event") == "result" and any(end["event"] == "execution_end" and
                all(message.get(name) == end.get(name) for name in ("output", "stdout", "is_final_answer")) for end in following):
            event["message"] = {"event": "result", "note": "Output is shown in the following execution result."}


def export_data_agent_dialogue(run: Path, output: Path | None = None, *, refresh_seconds: float = 0) -> dict:
    run = evaluation_run(run)
    output = output.resolve() if output else run / "logs" / "data-agent-dialogue.html"
    dialogues = build_dialogues(run, output)
    counts = Counter(dialogue["status"] for dialogue in dialogues)
    calls = sum(len(dialogue["calls"]) for dialogue in dialogues)
    pending = sum(call["pending"] for dialogue in dialogues for call in dialogue["calls"])
    payload = {"run": run.parent.name + "/" + run.name if run.name == "data-agent-evaluation" else run.name,
               "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "refresh_seconds": refresh_seconds, "dialogues": dialogues}
    embedded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c").replace(
        ">", "\\u003e").replace("&", "\\u0026").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    document = Template(TEMPLATE).substitute(TITLE=html.escape(run.name, quote=True), STYLE=STYLE, DATA=embedded, SCRIPT=SCRIPT)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", prefix=f".{output.name}.", suffix=".tmp",
                                     dir=output.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(document)
    try:
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return {"html": str(output), "bytes": output.stat().st_size, "tasks": len({d["task_id"] for d in dialogues}),
            "attempts": len(dialogues), "calls": calls, "pending_calls": pending, "status": dict(counts)}


def source_signature(run: Path) -> tuple:
    """Only journal and result changes trigger watch exports, never the HTML itself."""
    paths = {run / name for name in ("tasks.json", "generation-summary.json", "saved-ontology-summary.json") if (run / name).exists()}
    paths.update((run / "tasks").glob("*.json"))
    paths.update((run / "evaluation").glob("*/summary.json"))
    paths.update((run / "evaluation").glob("*/tasks/*.json"))
    for name in ("visit.json", *JOURNALS, "result.json", "data_agent.jsonl"):
        paths.update((run / "logs" / "data_agent").rglob(name))
        paths.update((run / "logs" / "tasks").rglob(name))
    if (run / "logs/llm.jsonl").exists():
        paths.add(run / "logs/llm.jsonl")
    return tuple((str(path), path.stat().st_mtime_ns, path.stat().st_size) for path in sorted(paths))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Ontology parent run or data-agent-evaluation directory")
    parser.add_argument("--output", type=Path, help="Defaults to RUN/data-agent-evaluation/logs/data-agent-dialogue.html")
    parser.add_argument("--watch", action="store_true", help="Rebuild the HTML as request and execution journals change")
    parser.add_argument("--interval", type=float, default=5, help="Watch and browser refresh interval in seconds (default: 5)")
    args = parser.parse_args(argv)
    if not math.isfinite(args.interval) or args.interval <= 0:
        parser.error("--interval must be positive and finite")
    requested_run = (WORKSPACE / args.run).resolve()
    custom_output = (WORKSPACE / args.output).resolve() if args.output else None
    previous, last_output, busy, waiting = None, None, 0, False
    try:
        while True:
            run = evaluation_run(requested_run)
            output = custom_output or run / "logs/data-agent-dialogue.html"
            signature = source_signature(run)
            if args.watch and custom_output is None and not signature:
                if not waiting:
                    print(f"Waiting for data-agent recordings in {run}; no evaluation directory has been created.", flush=True)
                    waiting = True
                time.sleep(args.interval)
                continue
            output_stat = output.stat() if output.exists() else None
            output_version = (output_stat.st_mtime_ns, output_stat.st_size) if output_stat else None
            if signature != previous or (args.watch and output_version != last_output):
                try:
                    summary = export_data_agent_dialogue(run, output, refresh_seconds=args.interval if args.watch else 0)
                except SnapshotBusy as error:
                    busy += 1
                    if not args.watch or busy >= 3:
                        raise
                    print(f"Waiting for snapshot writer: {error}", flush=True)
                else:
                    previous, busy = signature, 0
                    output_stat = output.stat()
                    last_output = output_stat.st_mtime_ns, output_stat.st_size
                    print(json.dumps(summary, ensure_ascii=False), flush=True)
            if not args.watch:
                return
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("Stopped watching; the last exported HTML is saved.", flush=True)


TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Data-agent dialogue · $TITLE</title><style>$STYLE</style></head>
<body><header class="run-header"><p class="eyebrow">RECORDED AGENT DIALOGUE</p><h1>Data-agent questions and actions</h1><p id="run-info"></p><div class="stats" id="stats"></div><p>Read each question as a conversation: input, agent response, Python actions, tool results, and final answer. Recorded requests and attempts stay separate.</p><div class="refresh"><label><input id="auto-refresh" type="checkbox"> Refresh this view while watching</label><button id="refresh-now" type="button">Refresh now</button><span id="generated"></span></div></header>
<div class="layout"><aside aria-label="Question and attempt navigation"><label for="dialogue-search">Search questions and recorded dialogue</label><input id="dialogue-search" type="search" placeholder="Question, code, tool, or error" autocomplete="off"><label for="role-filter">Agent role</label><select id="role-filter"><option value="">All roles</option></select><label for="status-filter">Status</label><select id="status-filter"><option value="">All statuses</option></select><p id="visible-count" aria-live="polite"></p><nav id="dialogue-list"></nav></aside><main><div class="pager"><button id="prev-dialogue" type="button">← Previous</button><span id="position" aria-live="polite"></span><button id="next-dialogue" type="button">Next →</button></div><div id="conversation"></div><p id="empty" hidden>No recorded data-agent conversations were found. Request inputs will appear when workers start.</p></main></div>
<script id="dialogue-data" type="application/json">$DATA</script><script>$SCRIPT</script></body></html>"""

STYLE = """:root { color-scheme: light; font-family: system-ui, sans-serif; color: #233248; background: #f3f6fb; }
* { box-sizing: border-box; } body { margin: 0; } [hidden] { display: none !important; }
.run-header { padding: 24px 32px; color: #f4f8ff; background: #142640; } h1 { margin: 4px 0 12px; font-size: 26px; }
p { line-height: 1.6; } .eyebrow { font-size: 11px; letter-spacing: .12em; font-weight: 700; margin: 0; } .stats, .refresh { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
.stats span { border: 1px solid #6b7f98; border-radius: 20px; padding: 5px 12px; } .refresh { font-size: 13px; } #generated { color: #c1cede; }
.layout { display: grid; grid-template-columns: 310px minmax(0, 1fr); max-width: 1540px; margin: auto; }
aside { background: #fff; padding: 24px 16px; align-self: start; position: sticky; top: 0; max-height: 100vh; overflow: auto; border-right: 1px solid #dde5ef; }
label { font-size: 13px; font-weight: 600; } aside label { display: block; margin: 0 0 7px; } input[type=search], select { width: 100%; border: 1px solid #b9c7da; border-radius: 7px; background: white; padding: 9px; font: inherit; margin-bottom: 14px; }
button { font: inherit; cursor: pointer; border: 1px solid #b9c7da; border-radius: 7px; padding: 8px 12px; background: #fff; color: #24364f; } button:disabled { opacity: .45; cursor: default; }
button:focus-visible, input:focus-visible, select:focus-visible, summary:focus-visible, a:focus-visible { outline: 3px solid #4187df; outline-offset: 2px; }
.nav-button { display: block; text-align: left; width: 100%; margin: 0 0 8px; padding: 12px; overflow-wrap: anywhere; } .nav-button span, .nav-button small { display: block; margin-top: 4px; } .nav-button small { color: #63738a; font-size: 11px; overflow-wrap: anywhere; }
.nav-button[aria-current=true] { border-color: #386ec5; background: #eef5ff; box-shadow: inset 3px 0 #386ec5; }
main { min-width: 0; padding: 24px 32px 64px; } .pager { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 24px; font-size: 13px; }
.question { background: #fff; border: 1px solid #d9e2ef; border-radius: 12px; padding: 22px; margin-bottom: 26px; } .question h2 { font-size: 21px; margin: 8px 0 14px; overflow-wrap: anywhere; } .question .question-text { font-size: 18px; white-space: pre-wrap; margin: 12px 0; }
.meta { display: flex; align-items: center; flex-wrap: wrap; gap: 8px; font-size: 12px; color: #5f7088; } .badge { border-radius: 12px; padding: 3px 9px; background: #e6edf7; color: #385270; } .badge.completed { color: #166345; background: #def4e8; } .badge.failed, .badge.incomplete { color: #94391d; background: #fff0e6; }
.call { padding-top: 8px; margin-bottom: 28px; border-top: 1px solid #d6e0ee; } .call-heading { font-size: 13px; font-weight: 700; color: #566b85; margin: 18px 0 15px; display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
.bubble { max-width: 96%; margin: 12px 0; padding: 16px 19px; background: white; border: 1px solid #dbe4ef; border-radius: 14px 14px 14px 3px; }
.bubble.input { margin-left: auto; border-radius: 14px 14px 3px 14px; background: #eaf2ff; border-color: #c7daf7; } .bubble.agent { background: white; } .bubble.execution { background: #f8fbff; border-left: 3px solid #6485b8; } .bubble.tool { background: #f0f8f5; border-color: #cde3da; } .bubble.error { background: #fff2ed; border-color: #eec7b8; } .bubble.final { background: #edf9f1; border-color: #aed4bd; }
.bubble h3 { font-size: 12px; margin: 0 0 10px; text-transform: uppercase; letter-spacing: .04em; color: #425873; } .bubble h4 { font-size: 13px; margin: 16px 0 7px; } .prose { white-space: pre-wrap; overflow-wrap: anywhere; line-height: 1.6; margin: 8px 0; }
pre { white-space: pre-wrap; overflow-wrap: anywhere; word-break: break-word; font: 12px/1.6 ui-monospace, SFMono-Regular, Consolas, monospace; margin: 8px 0; padding: 12px; border-radius: 7px; background: #f0f3f8; max-height: 600px; overflow: auto; } pre.code { color: #e7f0ff; background: #1d2d44; tab-size: 4; }
details { margin: 11px 0; } summary { cursor: pointer; font-size: 13px; font-weight: 600; color: #3d5a80; padding: 6px 0; } .note { font-size: 12px; color: #63758c; line-height: 1.6; } .raw-links { display: flex; flex-wrap: wrap; gap: 8px 16px; font-size: 12px; margin: 13px 0; } a { color: #275f9e; }
@media (max-width: 820px) { .layout { display: block; } aside { position: static; max-height: 320px; border-right: 0; border-bottom: 1px solid #dde5ef; } .run-header { padding: 20px; } main { padding: 20px 14px 40px; } .bubble { max-width: 100%; } .pager { flex-wrap: wrap; } }
"""

SCRIPT = r"""const data = JSON.parse(document.getElementById('dialogue-data').textContent);
const all = data.dialogues, list = document.getElementById('dialogue-list'), conversation = document.getElementById('conversation');
const search = document.getElementById('dialogue-search'), statusFilter = document.getElementById('status-filter'), roleFilter = document.getElementById('role-filter');
let visible = [], selected = null;
const own = (value, name) => Object.prototype.hasOwnProperty.call(value, name);
const format = value => typeof value === 'string' ? value : JSON.stringify(value, null, 2);
function node(tag, text, cls) { const element = document.createElement(tag); if (text !== undefined && text !== null) element.textContent = text; if (cls) element.className = cls; return element; }
function text(parent, value, cls='prose') { parent.append(node('div', format(value), cls)); }
function pre(parent, value, code=false) { parent.append(node('pre', format(value), code ? 'code' : '')); }
function heading(parent, title) { parent.append(node('h4', title)); }
function expand(parent, title, callback) { const details = node('details'), summary = node('summary', title); details.append(summary); let loaded = false; details.addEventListener('toggle', () => { if (details.open && !loaded) { callback(details); loaded = true; } }); parent.append(details); return details; }
function bubble(parent, title, cls) { const result = node('section', null, 'bubble ' + cls); result.append(node('h3', title)); parent.append(result); return result; }
function links(parent, values) { const row = node('div', null, 'raw-links'); Object.entries(values).forEach(([name, href]) => { const link = node('a', name); link.setAttribute('href', href); link.setAttribute('target', '_blank'); link.setAttribute('rel', 'noopener'); row.append(link); }); parent.append(row); }
function eventView(parent, event) {
  const kind = event.event;
  if (kind === 'execution_start') { pre(bubble(parent, 'Action · executed Python', 'execution'), event.code, true); return; }
  if (kind === 'execution_end') {
    const box = bubble(parent, event.is_final_answer ? 'Python returned a final submission' : 'Python result', 'tool');
    if (event.stdout !== undefined && event.stdout !== '') { heading(box, 'Standard output'); pre(box, event.stdout); }
    if (event.output !== undefined && event.output !== null) { heading(box, 'Return value'); pre(box, event.output); }
    if (!event.stdout && (event.output === null || event.output === undefined)) text(box, 'Execution finished without printed output.', 'note');
    return;
  }
  if (kind === 'tool_start' || kind === 'tool_result') { const box = bubble(parent, (kind === 'tool_start' ? 'Tool call · ' : 'Tool result · ') + (event.name || 'unknown'), 'tool'); if (kind === 'tool_start') pre(box, event.arguments); else pre(box, event.result); return; }
  if (kind === 'execution_error') { const box = bubble(parent, 'Python execution error', 'error'); pre(box, {error:event.error, message:event.message}); return; }
  const box = bubble(parent, 'Worker · ' + ((event.message || {}).event || kind), 'execution');
  const message = event.message;
  if (message && message.note) text(box, message.note, 'note');
  else expand(box, 'Recorded worker event', details => pre(details, event));
}
function callView(parent, call, dialogue) {
  const article = node('article', null, 'call'); article.id = 'call-' + call.number;
  const label = node('div', null, 'call-heading'); label.append(node('span', 'Call ' + call.number + ' · step ' + (call.step ?? '?')));
  label.append(node('span', call.pending ? 'Waiting for model response' : Object.keys(call.error).length ? 'Request failed' : 'Response recorded', 'badge ' + (call.pending ? 'pending' : Object.keys(call.error).length ? 'failed' : 'completed'))); article.append(label);
  const input = bubble(article, 'Input to the agent', 'input');
  text(input, call.shared + ' earlier messages reused; ' + call.inputs.length + ' new or changed messages. Exact request histories remain in the raw traces.', 'note');
  call.inputs.forEach(message => {
    heading(input, 'Message ' + message.position + ' · ' + message.role);
    if (message.initial_task) text(input, 'The initial task and ontology context are shown above.', 'note');
    else if (own(message, 'system')) expand(input, 'System instructions', details => pre(details, dialogue.systems[message.system]));
    else { if (own(message, 'content')) text(input, message.content); const extra = Object.fromEntries(Object.entries(message).filter(([name]) => !['role','position','content'].includes(name))); if (Object.keys(extra).length) expand(input, 'Other recorded message fields', details => pre(details, extra)); }
  });
  if (own(call, 'prompt')) expand(input, 'Recorded unstructured prompt', details => pre(details, call.prompt));
  const agent = bubble(article, 'Agent response', 'agent');
  if (own(call, 'response') && call.response !== '') text(agent, call.response);
  call.functions.forEach(fn => {
    heading(agent, 'Function request · ' + fn.name);
    const args = fn.arguments;
    if (args && typeof args === 'object' && !Array.isArray(args)) {
      if (own(args, 'thought')) { heading(agent, 'Recorded thought'); text(agent, args.thought); }
      if (own(args, 'code')) { heading(agent, 'Generated code'); pre(agent, args.code, true); }
      const rest = Object.fromEntries(Object.entries(args).filter(([name]) => !['thought','code'].includes(name)));
      if (Object.keys(rest).length) pre(agent, rest);
    } else pre(agent, args);
    if (fn.executed_code) text(agent, 'Generated code is shown once in the executed Python action below. Exact function arguments are in llm.jsonl.', 'note');
  });
  if (!call.functions.length && (!own(call, 'response') || call.response === '')) text(agent, call.pending ? 'The request input has been recorded. No response has arrived yet.' : 'No response text was recorded.', 'note');
  if (Object.keys(call.error).length) pre(bubble(article, 'Model request error', 'error'), call.error);
  call.events.forEach(event => eventView(article, event));
  if (own(call, 'observation')) pre(bubble(article, 'Observation returned to the agent', 'tool'), call.observation);
  if (call.step_error !== undefined && call.step_error !== null) pre(bubble(article, 'Action or answer-validation error', 'error'), call.step_error);
  ['submission','action_output'].forEach(name => { if (own(call, name)) pre(bubble(article, name === 'submission' ? 'Final submission' : 'Action return value', 'tool'), call[name]); });
  if (call.accepted_submission) text(bubble(article, 'Final submission accepted', 'tool'), 'The accepted answer and workflow appear below.', 'note');
  expand(article, 'Usage and timing', details => pre(details, call.usage));
  parent.append(article);
}
function show(dialogue, restore=false) {
  selected = dialogue.id; conversation.replaceChildren();
  document.querySelectorAll('.nav-button').forEach(button => button.setAttribute('aria-current', String(button.dataset.id === selected)));
  const question = node('section', null, 'question'); question.append(node('div', 'QUESTION · ' + dialogue.task_id, 'eyebrow'));
  question.append(node('h2', dialogue.role + ' · ' + dialogue.task_id + (dialogue.step_id ? ' · ' + dialogue.step_id : ''))); text(question, dialogue.question, 'question-text');
  const meta = node('div', null, 'meta'); meta.append(node('span', dialogue.status, 'badge ' + dialogue.status));
  if (dialogue.answer_type) meta.append(node('span', 'Answer type: ' + dialogue.answer_type)); meta.append(node('span', dialogue.attempt + ' · visit attempt ' + dialogue.visit_attempt)); question.append(meta);
  expand(question, 'Initial task, ontology context, and system instructions', details => {
    let parsed = dialogue.task; if (typeof parsed === 'string') { try { parsed = JSON.parse(parsed); } catch {} }
    if (parsed && typeof parsed === 'object' && own(parsed, 'ontology_context')) expand(details, 'Ontology context supplied to this question', context => pre(context, parsed.ontology_context));
    expand(details, 'Exact initial task', initial => pre(initial, dialogue.task ?? 'No initial task was recorded.'));
    if (dialogue.instructions) expand(details, 'Saved role instructions', instructions => pre(instructions, dialogue.instructions));
    dialogue.systems.forEach((system, index) => expand(details, 'Exact system message ' + (index + 1), instructions => pre(instructions, system)));
  });
  expand(question, 'Raw trace files · exact inputs, outputs, and worker events', details => { text(details, dialogue.visit || 'No worker visit has started.', 'note'); links(details, dialogue.links); });
  conversation.append(question);
  if (!dialogue.calls.length) text(bubble(conversation, 'Waiting for requests', 'input'), dialogue.status === 'pending' ? 'This question has not started yet.' : 'No model requests were recorded for this attempt.', 'note');
  dialogue.calls.forEach(call => callView(conversation, call, dialogue));
  if (dialogue.unassigned_events.length) { heading(conversation, 'Execution events without a recorded model call'); dialogue.unassigned_events.forEach(event => eventView(conversation, event)); }
  if (dialogue.plan !== undefined && dialogue.plan !== null) pre(bubble(conversation, 'Planner output · Plan', 'final'), dialogue.plan);
  const final = bubble(conversation, dialogue.role === 'analyst' ? 'Analyst output' : 'Task answer and explanation', 'final');
  if (dialogue.answer !== undefined && dialogue.answer !== null) {
    const answer = dialogue.answer;
    if (answer && typeof answer === 'object' && (own(answer, 'answer') || own(answer, 'hypothesis'))) { heading(final, own(answer, 'answer') ? 'Answer' : 'Hypothesis'); pre(final, own(answer, 'answer') ? answer.answer : answer.hypothesis); if (own(answer, 'workflow')) { heading(final, 'Analysis workflow'); text(final, answer.workflow); } }
    else pre(final, answer);
  } else text(final, 'No accepted final answer has been recorded for this attempt.', 'note');
  if (dialogue.links['prediction CSV']) links(final, {'Open prediction CSV':dialogue.links['prediction CSV']});
  if (dialogue.explanation) { heading(final, 'Explanation'); text(final, dialogue.explanation); }
  if (Object.keys(dialogue.pipeline).length) expand(final, 'Recorded method result and Plan', details => pre(details, dialogue.pipeline));
  if (Object.keys(dialogue.error).length) pre(bubble(conversation, 'Attempt error', 'error'), dialogue.error);
  if (Object.keys(dialogue.score).length) { const score = bubble(conversation, 'Recorded evaluation' + (dialogue.score.evaluator_label ? ' · ' + dialogue.score.evaluator_label : ''), 'tool'); if (own(dialogue.score, 'score')) text(score, 'Score: ' + dialogue.score.score + (dialogue.score.metric ? ' · ' + dialogue.score.metric : '')); expand(score, 'Recorded score', details => pre(details, dialogue.score)); }
  else text(final, 'No evaluation score was recorded for this attempt.', 'note');
  history.replaceState(null, '', '#' + dialogue.id); updatePager();
  if (!restore) conversation.scrollIntoView({block:'start'});
}
function updatePager() { const index = visible.findIndex(dialogue => dialogue.id === selected); document.getElementById('prev-dialogue').disabled = index <= 0; document.getElementById('next-dialogue').disabled = index < 0 || index >= visible.length - 1; document.getElementById('position').textContent = index < 0 ? '' : (index + 1) + ' / ' + visible.length + ' attempts'; }
function filter(restore=false) {
  const query = search.value.toLocaleLowerCase().trim();
  visible = all.filter(dialogue => (!roleFilter.value || dialogue.role === roleFilter.value) && (!statusFilter.value || dialogue.status === statusFilter.value) && (!query || JSON.stringify(dialogue).toLocaleLowerCase().includes(query)));
  list.replaceChildren(); visible.forEach(dialogue => { const button = node('button', null, 'nav-button'); button.type = 'button'; button.dataset.id = dialogue.id; button.append(node('strong', dialogue.role + ' · ' + dialogue.task_id), node('span', dialogue.status + ' · ' + dialogue.calls.length + ' calls'), node('small', dialogue.attempt + (dialogue.step_id ? ' · Step ' + dialogue.step_id : '') + ' · visit ' + dialogue.visit_attempt)); button.addEventListener('click', () => show(dialogue)); list.append(button); });
  document.getElementById('visible-count').textContent = visible.length + ' / ' + all.length + ' attempts';
  const empty = document.getElementById('empty'); empty.hidden = visible.length > 0; empty.textContent = all.length ? 'No questions or attempts match these filters.' : 'No recorded data-agent conversations were found. Request inputs will appear when workers start.';
  if (visible.length) show(visible.find(dialogue => dialogue.id === selected) || visible[0], restore); else { conversation.replaceChildren(); updatePager(); }
}
document.getElementById('run-info').textContent = data.run;
document.getElementById('generated').textContent = 'Saved ' + data.generated;
const stats = document.getElementById('stats'), callCount = all.reduce((n,d) => n + d.calls.length,0), pending = all.reduce((n,d) => n + d.calls.filter(call => call.pending).length,0);
[new Set(all.map(d => d.task_id)).size + ' questions', all.length + ' attempts', callCount + ' recorded requests', pending + ' requests waiting'].forEach(label => stats.append(node('span', label)));
[...new Set(all.map(d => d.status))].sort().forEach(status => { const option = node('option', status); option.value = status; statusFilter.append(option); });
[...new Set(all.map(d => d.role))].sort().forEach(role => { const option = node('option', role); option.value = role; roleFilter.append(option); });
search.addEventListener('input', () => filter(true)); statusFilter.addEventListener('change', () => filter(true)); roleFilter.addEventListener('change', () => filter(true));
function move(direction) { const index = visible.findIndex(dialogue => dialogue.id === selected); if (visible[index + direction]) show(visible[index + direction]); }
document.getElementById('prev-dialogue').addEventListener('click', () => move(-1)); document.getElementById('next-dialogue').addEventListener('click', () => move(1));
document.addEventListener('keydown', event => { if (['INPUT','SELECT','TEXTAREA'].includes(event.target.tagName)) return; if (event.altKey && event.key === 'ArrowRight') { event.preventDefault(); move(1); } if (event.altKey && event.key === 'ArrowLeft') { event.preventDefault(); move(-1); } });
window.addEventListener('hashchange', () => { const dialogue = visible.find(item => item.id === location.hash.slice(1)); if (dialogue) show(dialogue); });
const stateKey = 'data-agent-dialogue:' + location.pathname, refresh = document.getElementById('auto-refresh'); refresh.checked = data.refresh_seconds > 0;
let saved = null; try { saved = JSON.parse(sessionStorage.getItem(stateKey)); } catch {}
if (saved) { search.value = saved.search || ''; statusFilter.value = saved.status || ''; roleFilter.value = saved.role || ''; refresh.checked = saved.refresh; }
selected = location.hash.slice(1); filter(true); if (saved && saved.scroll !== undefined) window.scrollTo(0, saved.scroll);
function saveState() { try { sessionStorage.setItem(stateKey, JSON.stringify({search:search.value,status:statusFilter.value,role:roleFilter.value,refresh:refresh.checked,scroll:window.scrollY})); } catch {} }
function reload() { saveState(); location.reload(); }
document.getElementById('refresh-now').addEventListener('click', reload); window.addEventListener('beforeunload', saveState);
setInterval(() => { if (refresh.checked && !document.hidden) reload(); }, Math.max(data.refresh_seconds || 5, 1) * 1000);
"""


if __name__ == "__main__":
    main()
