"""Export saved agent logs to a standalone HTML viewer using only the stdlib."""

from __future__ import annotations

import argparse
import html
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import quote

from ..artifacts import WORKSPACE, key, pretty, read_json, read_jsonl, system_messages


def escaped(value) -> str:
    return html.escape(pretty(value), quote=True)


def block(value, css_class: str = "") -> str:
    return f'<pre class="{css_class}">{escaped(value)}</pre>'


def details(title: str, content: str, css_class: str = "") -> str:
    return f'<details class="{css_class}"><summary>{escaped(title)}</summary>{content}</details>'


def section(title: str, content: str, css_class: str = "") -> str:
    return f'<section class="card {css_class}"><h3>{escaped(title)}</h3>{content}</section>'


def data_results(run: Path, groups: dict, output: Path | None = None) -> tuple[str, dict]:
    summaries = sorted((run / "evaluation").glob("*/summary.json"), key=lambda path: path.stat().st_mtime)
    evaluation = read_json(summaries[-1]) if summaries else {}
    evaluation_root = summaries[-1].parent if summaries else None
    tasks = json.loads((run / "tasks.json").read_text()) if (run / "tasks.json").exists() else []
    benchmark = read_json(run / "generation-summary.json").get("benchmark", "DiscoveryBench")
    rows = []
    for task in tasks:
        result = read_json(run / "tasks" / (task["task_id"] + ".json"))
        score = read_json(evaluation_root / "tasks" / (task["task_id"] + ".json")) if evaluation_root else {}
        dialogue = " ".join(f'<a href="#visit-{number}">{escaped(group["role"])} · attempt {escaped(group["attempt"])} dialogue</a>'
                            for number, group in enumerate(groups.values(), 1)
                            if (group["visit"] or "").startswith(f'data_agent/{task["task_id"]}/'))
        answer = details("Answer", block(result["answer"] if "answer" in result else "No answer recorded.")) if benchmark == "Kramabench" else details("Hypothesis", block(result.get("hypothesis") or "No hypothesis recorded."))
        answer += details("Analysis workflow", block(result.get("workflow") or "No workflow recorded."))
        if result.get("pipeline") == "planner-analyst":
            pipeline_result = read_json(Path(result["artifact_root"]) / "result.json")
            answer += details("Data agent explanation, task graph and steps", block({
                key: pipeline_result[key] for key in ("status", "stop_reason", "explanation", "graph", "steps", "plan",
                                                      "analyst_steps")
                if key in pipeline_result}))
        if result.get("error"):
            answer += details("Execution error", block(result["error"]))
        score_text = f'{score["score"]:.4f}' if isinstance(score.get("score"), (int, float)) else score.get("status", "Not scored")
        if benchmark == "Kramabench" and score.get("metric"):
            score_text += f' · {score["metric"]}'
        rows.append(f'<tr><td><strong>{escaped(task["task_id"])}</strong><p>{escaped(task["question"])}</p>{dialogue}</td>'
                    f'<td>{escaped(result.get("status", "pending"))}</td><td>{escaped(score_text)}</td><td>{answer}</td></tr>')
    note = f"Native {benchmark} scoring has not run."
    if evaluation:
        percent = evaluation.get("overall_percent")
        note = f'Native {benchmark} overall score: {percent:.2f}%.' if isinstance(percent, (int, float)) else f"Native {benchmark} scoring is incomplete."
        note += f' Judge: {evaluation.get("judge_model", "unknown")}. Metric: {evaluation.get("metric", "final_score")}.'
    reasoning = read_json(run / "generation-summary.json").get("reasoning_settings", {})
    if reasoning:
        note += f' Data-agent reasoning effort: {reasoning.get("data_agent", "unknown")}. Evaluator reasoning effort: {reasoning.get("evaluator", "unknown")}.'
    table = '<div class="result-table"><table><thead><tr><th>Question and execution</th><th>Generation</th><th>Native score</th><th>Answer and workflow</th></tr></thead><tbody>' + "".join(rows) + '</tbody></table></div>'
    dialogue_path = run / "logs" / "data-agent-dialogue.html"
    dialogue_link = ""
    if output is not None and dialogue_path.exists():
        href = quote(os.path.relpath(dialogue_path.resolve(), output.parent.resolve()), safe="/.")
        dialogue_link = f'<p><a href="{escaped(href)}">Open the data-agent conversation viewer</a></p>'
    return '<div class="results-summary">' + section("Question results", f'<p>{escaped(note)}</p>' + dialogue_link + table) + '</div>', evaluation


def model_input(record: dict) -> tuple[object, list[dict]]:
    try:
        request = json.loads(record.get("prompt", ""))
    except (ValueError, TypeError):
        request = None
    messages = record.get("provider_messages", request.get("messages", []) if isinstance(request, dict) else [])
    return request, messages


def render_messages(messages: list[dict], start: int = 1) -> str:
    return "".join(f'<h4>Message {position} · {escaped(message.get("role", "unknown"))}</h4>' + block(message.get("content", message))
                   for position, message in enumerate(messages, start))


def render_call(record: dict, step: dict, events: list[dict], visit: dict,
                previous_messages: list[dict] | None) -> str:
    number = int(record["call_index"])
    role = record.get("role", "unknown")
    local_step = record.get("step", "?")
    request, messages = model_input(record)
    context = [message for message in messages if message.get("role") != "system"]
    previous = [message for message in previous_messages or [] if message.get("role") != "system"]
    shared = 0
    while shared < min(len(previous), len(context)) and previous[shared] == context[shared]:
        shared += 1
    if previous_messages is None and context and context[0].get("role") == "user":
        input_content = '<p>The initial request includes the task shown above.'
        input_content += f' {len(context)} non-system context messages were supplied.</p>'
        input_content += render_messages(context[1:], 2)
    else:
        added = context[shared:]
        input_content = f'<p>Reuses {shared} earlier context messages; {len(added)} new or changed messages follow. Historical assistant messages are earlier outputs supplied as context.</p>'
        input_content += render_messages(added, shared + 1)
    if not messages:
        input_content = '<p>No structured input messages were recorded; expand the logged request below.</p>'
    input_content += f'<p class="input-context"><a href="#request-{number}">Expand the exact request and full message history below</a> · {len(messages)} messages.</p>'
    error = record.get("error")
    generated = "response_text" in record or bool(record.get("tool_calls"))
    blocked = error == "BudgetExceeded" and not generated and record.get("usage", {}).get("calls", 0) == 0
    status = "Budget blocked · no provider request" if blocked else "Model request failed" if error else "Model response received"
    chunks = [f'<header class="call-heading"><p class="eyebrow">{escaped(role)} · step {escaped(local_step)}</p><h2><a href="#call-{number}">Call {number}</a></h2><span class="badge {"error" if error else "success"}">{escaped(status)}</span></header>']
    chunks.append(section("Model input", input_content, "model-input"))
    if generated:
        provider_text = record.get("response_text", "")
        output = '<h4>Raw provider response text</h4>' + block(provider_text, "model-text")
        if not provider_text:
            output += '<p class="empty">The provider returned empty text. Its native function calls are shown below.</p>' if record.get("tool_calls") else '<p class="empty">The provider returned empty text.</p>'
        if record.get("tool_calls"):
            output += '<div class="model-tool-calls"><h4>Native function calls — raw model output</h4>' + block(record["tool_calls"]) + '</div>'
            for position, call in enumerate(record["tool_calls"], 1):
                try:
                    arguments = json.loads(call.get("arguments", ""))
                except (ValueError, TypeError):
                    arguments = None
                if isinstance(arguments, dict):
                    output += f'<div class="model-call-arguments"><h4>Function call {position} · {escaped(call.get("name", "unknown"))} · decoded model arguments</h4>'
                    if "thought" in arguments:
                        output += '<h4>Thought generated by the model</h4>' + block(arguments["thought"])
                    if "code" in arguments:
                        output += '<h4>Code generated by the model</h4>' + block(arguments["code"])
                    output += details("All decoded arguments", block(arguments)) + '</div>'
    else:
        message = "No model response. The budget check blocked this request before contacting the provider." if blocked else "No model response was recorded."
        output = f'<p class="empty">{message}</p>'
    chunks.append(section("Model output — response generated for this call", output, "model-output"))
    if step.get("model_output") is not None and step["model_output"] != record.get("response_text"):
        chunks.append(details("Agent message — formatted from the model output for CodeAgent parsing", block(step["model_output"]), "agent-output card"))
    if error:
        chunks.append(section("Request error", block({"error": error, "message": record.get("error_message", ""),
                                                       "provider_error": record.get("provider_error")}), "request-error error-card"))
    actions = [event["code"] for event in events if event.get("event") == "execution_start"]
    if not actions and step.get("code_action"):
        actions = [step["code_action"]]
    action_content = "".join(block(code) for code in actions) if actions else '<p class="empty">No code was executed for this call.</p>'
    chunks.append(section("Executed action — Python sent to the isolated worker", action_content, "executed-action"))
    observation = step.get("observations")
    chunks.append(section("Observation — execution result returned to the agent", block(observation) if observation is not None else '<p class="empty">No execution observation was recorded for this call.</p>', "observation"))
    errors = []
    if step.get("error"):
        errors.append(block(step["error"]))
    for event in events:
        if event.get("event") == "execution_error":
            errors.append(block({"error": event.get("error"), "message": event.get("message")}))
    if errors:
        chunks.append(section("Action / validation error — correction returned to the agent", "".join(errors), "step-error error-card"))
    if step.get("action_output") is not None:
        if step.get("is_final_answer") and visit.get("completed") and visit.get("answer") == step["action_output"]:
            result = '<p>This call submitted the accepted answer shown at the end of the dialogue.</p>'
        else:
            result = block(step["action_output"])
        chunks.append(section("Final submission" if step.get("is_final_answer") else "Action result value", result, "action-result"))
    event_content = "".join(f'<h4>Event {i} · {escaped(event.get("event", "unknown"))}</h4>' + block(event) for i, event in enumerate(events, 1))
    chunks.append(details(f"Tool and worker events — {len(events)} records, including arguments, results, stdout and stderr", event_content or '<p>No execution events were recorded.</p>', "worker-events card"))
    usage = {"model": record.get("model"), "finish_reason": record.get("finish_reason"),
             "wall_seconds": record.get("wall_seconds"), **record.get("usage", {})}
    chunks.append(details("Usage and timing", block(usage), "usage card"))
    chunks.append(f'<div id="request-{number}">' + details("Logged request — full messages, native tool definitions, tool choice and stop sequences", block(request if request is not None else record.get("prompt", "")), "raw-request card") + '</div>')
    if record.get("provider_messages") is not None and (not isinstance(request, dict) or record["provider_messages"] != request.get("messages")):
        chunks.append(details("Exact provider messages — includes JSON mode instructions", block(record["provider_messages"]), "provider-messages card"))
    return f'<article id="call-{number}" class="call-panel" data-call="{number}">' + "".join(chunks) + "</article>"


def render_visit(number: int, group: dict, steps: dict, events: dict, first: bool) -> tuple[str, str]:
    records, visit = group["calls"], group["metadata"]
    role, path, attempt = group["role"], group["visit"], group["attempt"]
    title = f'{role} · {path.rsplit("/", 1)[-1]}' if path else f'{role} · unassigned call {records[0]["call_index"]}'
    if role in {"data_agent", "planner", "analyst"} and path and path.startswith("data_agent/"):
        title = f'{role} · {path.split("/")[1]}'
        if role == "analyst":
            title += " · " + path.rsplit("/", 1)[-1]
    if visit.get("stage"):
        title += " · " + visit["stage"].replace("_", " ")
    task = visit.get("task")
    if task is None and records:
        _, messages = model_input(records[0])
        task = next((message.get("content", "") for message in messages if message.get("role") == "user"), records[0].get("prompt", "No task was recorded."))
    prompts = {}
    for record in records:
        for prompt in system_messages(record):
            prompts.setdefault(pretty(prompt), []).append(record["call_index"])
    system_content = "".join(f'<h4>Exact system message · calls {escaped(", ".join(map(str, calls)))}</h4>' + block(prompt) for prompt, calls in prompts.items())
    if visit.get("instructions"):
        system_content += details("Saved role instructions", block(visit["instructions"]))
    chunks = [f'<header class="visit-heading"><p class="eyebrow">Role iteration · attempt {escaped(attempt)}</p><h2>{escaped(title)}</h2><p class="visit-path">{escaped(path or "No visit identifier recorded")}</p><p>{len(records)} logged requests · {escaped(visit.get("status", "Outcome not recorded"))}</p></header>',
              section("Task for this visit", block(task if task is not None else "No task was recorded."), "visit-task"),
              details("System instructions", system_content or '<p>No system instructions were recorded.</p>', "system-prompt card")]
    previous = None
    for record in records:
        chunks.append(render_call(record, steps.get(key(record), {}), events.get(key(record), []), visit, previous))
        _, previous = model_input(record)
    if not records:
        chunks.append(section("Dialogue", '<p class="empty">No model requests were logged for this visit.</p>'))
    if visit.get("completed") and visit.get("answer") is not None:
        outcome = '<h4>Accepted final answer — output of this visit</h4>' + block(visit["answer"], "accepted-answer")
    else:
        outcome = '<p class="empty">No accepted final answer was recorded for this visit.</p>'
    if visit.get("error"):
        outcome += '<h4>Visit error</h4>' + block({"error": visit["error"], "message": visit.get("message", "")})
    if visit:
        outcome += details("Full saved visit metadata", block(visit))
    chunks.append(section("Visit outcome", outcome, "visit-outcome"))
    links = "".join(f'<a href="#call-{record["call_index"]}" data-call="{record["call_index"]}">Call {record["call_index"]} · step {escaped(record.get("step", "?"))}{" · error" if record.get("error") else ""}</a>' for record in records)
    nav = f'<div class="visit-nav" data-group="{number}"><button type="button" class="visit-button" data-group="{number}" data-role="{escaped(role)}" aria-controls="visit-{number}"><strong>{escaped(title)}</strong><span>Attempt {escaped(attempt)} · {len(records)} calls</span><small>{escaped(visit.get("status", "Outcome not recorded"))}</small></button><div class="call-links">{links}</div></div>'
    hidden = "" if first else " hidden"
    panel = f'<article id="visit-{number}" class="visit-panel" data-group="{number}"{hidden}>' + "".join(chunks) + "</article>"
    return nav, panel


STYLE = """:root { color-scheme: light; font-family: system-ui, sans-serif; color: #243044; background: #f4f6fa; }
* { box-sizing: border-box; } body { margin: 0; } [hidden] { display: none !important; }
.run-header { padding: 24px 32px; background: #12223d; color: #f5f8ff; }
h1 { font-size: 25px; margin: 0 0 8px; } h2 { font-size: 29px; margin: 4px 0; } h3 { font-size: 17px; margin: 0 0 14px; } h4 { font-size: 14px; margin: 16px 0 8px; }
p { line-height: 1.55; } .run-header p { margin: 8px 0; max-width: 1100px; } .stats { display: flex; gap: 10px; flex-wrap: wrap; margin: 14px 0; }
.stats span { background: #243652; border-radius: 6px; padding: 7px 11px; font-size: 14px; }
.layout { display: grid; grid-template-columns: 300px minmax(0, 1fr); gap: 24px; padding: 24px; max-width: 1700px; margin: auto; }
aside { position: sticky; top: 16px; align-self: start; max-height: calc(100vh - 32px); display: flex; flex-direction: column; }
.filters { background: white; padding: 16px; border-radius: 9px; border: 1px solid #dde4ee; }
label { display: block; font-size: 13px; font-weight: 650; margin: 0 0 6px; } select, input { width: 100%; padding: 9px; border: 1px solid #b8c5d6; border-radius: 5px; font: inherit; margin-bottom: 12px; }
#visible-count { color: #54657b; font-size: 13px; } #visit-list { overflow-y: auto; padding-top: 12px; }
.visit-nav { margin-bottom: 8px; } .visit-button { display: block; width: 100%; padding: 12px; text-align: left; border: 1px solid #dbe2ec; background: white; border-radius: 7px; cursor: pointer; color: inherit; }
.visit-button:hover { border-color: #507cbb; } .visit-button[aria-current="true"] { background: #e5efff; border-color: #316ac0; box-shadow: inset 4px 0 #316ac0; }
.visit-button strong, .visit-button span, .visit-button small { display: block; } .visit-button span { margin-top: 5px; font-size: 12px; } .visit-button small { font-size: 11px; color: #62728a; margin-top: 6px; overflow-wrap: anywhere; }
.call-links { display: none; padding: 8px 12px; } .visit-button[aria-current="true"] + .call-links { display: grid; gap: 6px; } .call-links a { font-size: 12px; color: #316ac0; }
main { min-width: 0; } .pager { display: flex; gap: 10px; align-items: center; margin-bottom: 18px; } .pager button { border: 1px solid #bbc8db; background: white; padding: 8px 12px; border-radius: 5px; font: inherit; cursor: pointer; } .pager button:disabled { opacity: .4; cursor: default; }
.visit-heading { margin: 0 0 20px; } .call-panel { border-top: 2px solid #c9d6e9; padding-top: 24px; margin: 26px 0; scroll-margin-top: 20px; } .call-panel:target { border-top-color: #316ac0; } .call-heading { margin: 0 0 18px; } .call-heading h2 { font-size: 23px; } .call-heading a { color: inherit; text-decoration: none; } .eyebrow { text-transform: uppercase; letter-spacing: .08em; font-size: 12px; font-weight: 700; color: #456186; margin: 0; } .visit-path { overflow-wrap: anywhere; font-size: 13px; color: #53657e; margin: 6px 0 12px; }
.badge { display: inline-block; padding: 5px 9px; border-radius: 5px; font-size: 12px; font-weight: 650; } .success { background: #e0f0e9; color: #185939; } .error { background: #ffe7e2; color: #922d24; }
.card { border: 1px solid #dce3ed; background: white; border-radius: 9px; padding: 18px; margin-bottom: 16px; } .visit-task, .visit-outcome { border-left: 4px solid #d99a25; } .model-input { border-left: 4px solid #8096b7; } .model-output { border-left: 4px solid #346bc1; } .executed-action { border-left: 4px solid #7177bd; } .observation { border-left: 4px solid #298365; } .error-card { border-left: 4px solid #b63b31; }
details > summary { cursor: pointer; font-size: 14px; font-weight: 650; line-height: 1.5; } details[open] > summary { margin-bottom: 14px; } details details { margin-top: 14px; }
pre { white-space: pre-wrap; overflow-wrap: anywhere; margin: 0; padding: 14px; background: #f5f7fa; border: 1px solid #e4e9f0; border-radius: 5px; font: 12.5px/1.6 ui-monospace, SFMono-Regular, Consolas, monospace; max-height: 32rem; overflow-y: auto; }
pre + pre { margin-top: 12px; } .empty { color: #667489; margin: 0; } #no-results { padding: 28px; background: white; border-radius: 9px; }
.results-summary { padding: 24px 24px 0; max-width: 1700px; margin: auto; } .result-table { overflow-x: auto; } .result-table table { width: 100%; border-collapse: collapse; font-size: 13px; } .result-table th, .result-table td { padding: 12px; text-align: left; border-bottom: 1px solid #dce3ed; vertical-align: top; } .result-table td:first-child { width: 40%; overflow-wrap: anywhere; } .result-table td:last-child { width: 40%; } .result-table details { margin-bottom: 12px; } .result-table pre { max-height: 16rem; }
@media (max-width: 800px) { .run-header { padding: 20px; } .layout { grid-template-columns: 1fr; padding: 16px; gap: 18px; } aside { position: static; max-height: none; } #visit-list { max-height: 280px; } .card { padding: 14px; } }
"""

SCRIPT = """const buttons = Array.from(document.querySelectorAll('.visit-button'));
const panels = new Map(Array.from(document.querySelectorAll('.visit-panel')).map(panel => [panel.dataset.group, panel]));
const searchText = new Map(buttons.map(button => [button.dataset.group, panels.get(button.dataset.group).textContent.toLowerCase()]));
const roleFilter = document.getElementById('role-filter');
const search = document.getElementById('call-search');
const previous = document.getElementById('prev-visit');
const next = document.getElementById('next-visit');
let current = null;
let visible = buttons.slice();
function showVisit(group, updateHash = true) {
  current = group;
  panels.forEach((panel, id) => { panel.hidden = id !== group; });
  buttons.forEach(button => button.setAttribute('aria-current', String(button.dataset.group === group)));
  const index = visible.findIndex(button => button.dataset.group === group);
  previous.disabled = index <= 0;
  next.disabled = index < 0 || index >= visible.length - 1;
  document.getElementById('no-results').hidden = visible.length > 0;
  document.getElementById('visit-position').textContent = index >= 0 ? `${index + 1} of ${visible.length} matching role iterations` : '';
  if (group && updateHash) history.replaceState(null, '', '#visit-' + group);
}
function filterVisits() {
  const query = search.value.trim().toLowerCase();
  visible = buttons.filter(button => {
    const matches = (!roleFilter.value || button.dataset.role === roleFilter.value) && (!query || searchText.get(button.dataset.group).includes(query));
    button.parentElement.hidden = !matches;
    return matches;
  });
  document.getElementById('visible-count').textContent = `${visible.length} / ${buttons.length} role iterations shown`;
  const keep = visible.some(button => button.dataset.group === current);
  showVisit(keep ? current : visible.length ? visible[0].dataset.group : null, false);
}
function openHash() {
  const target = document.getElementById(location.hash.slice(1));
  const panel = target && target.closest('.visit-panel');
  if (panel) {
    if (!visible.some(button => button.dataset.group === panel.dataset.group)) { roleFilter.value = ''; search.value = ''; filterVisits(); }
    showVisit(panel.dataset.group, false);
    requestAnimationFrame(() => target.scrollIntoView({block: 'start'}));
  }
}
function move(offset) {
  const index = visible.findIndex(button => button.dataset.group === current);
  if (visible[index + offset]) { showVisit(visible[index + offset].dataset.group); document.querySelector('main').scrollIntoView({block: 'start'}); }
}
buttons.forEach(button => button.addEventListener('click', () => { showVisit(button.dataset.group); document.querySelector('main').scrollIntoView({block: 'start'}); }));
previous.addEventListener('click', () => move(-1));
next.addEventListener('click', () => move(1));
roleFilter.addEventListener('change', filterVisits);
search.addEventListener('input', filterVisits);
window.addEventListener('hashchange', openHash);
current = buttons.length ? buttons[0].dataset.group : null;
filterVisits();
openHash();
"""


def export_html(run: Path, output: Path) -> dict:
    logs = run / "logs"
    calls = read_jsonl(logs / "llm.jsonl")
    steps, events, visits = {}, defaultdict(list), {}
    for path in sorted(logs.rglob("steps-full.jsonl")):
        for step in read_jsonl(path):
            steps[key(step)] = step
    for path in sorted(logs.rglob("execution.jsonl")):
        for event in read_jsonl(path):
            events[key(event)].append(event)
    for path in sorted(logs.rglob("visit.json")):
        visit = read_json(path)
        if visit:
            visits[(visit.get("role", visit["visit"].split("/")[0]), visit["visit"], visit.get("visit_attempt", 1))] = visit
    generation = read_json(run / "generation-summary.json")
    summary = generation or read_json(run / "construction-summary.json")
    status = read_json(run / "logged-run-status.json")
    log_summary = read_json(logs / "log-summary.json")
    ontology = read_json(run / "ontology" / "ontology.json")
    claim_counts = Counter(claim["status"] for claim in ontology.get("claims", {}).values())
    groups = {}
    for position, record in enumerate(calls):
        record.setdefault("call_index", position + 1)
    for record in sorted(calls, key=lambda record: record["call_index"]):
        role, path, attempt = record.get("role", "unknown"), record.get("visit"), record.get("visit_attempt", 1)
        group_key = (role, path or f'unassigned-call-{record["call_index"]}', attempt)
        group = groups.setdefault(group_key, {"role": role, "visit": path, "attempt": attempt, "calls": [], "metadata": visits.get((role, path, attempt), {})})
        group["calls"].append(record)
    for group_key, visit in visits.items():
        role, path, attempt = group_key
        groups.setdefault(group_key, {"role": role, "visit": path, "attempt": attempt, "calls": [], "metadata": visit})
    nav, panels = [], []
    for number, group in enumerate(groups.values(), 1):
        group["calls"].sort(key=lambda record: record["call_index"])
        visit_nav, panel = render_visit(number, group, steps, events, number == 1)
        nav.append(visit_nav)
        panels.append(panel)
    attempts = sum(record.get("usage", {}).get("calls", 0) for record in calls)
    tokens = summary.get("llm_usage", {}).get("total_tokens", 0)
    stop = summary.get("stop_reason", status.get("status", "unknown"))
    stats = [f"Stopped: {stop}", f"{len(calls)} logged requests", f"{attempts} provider attempts", f"{len(groups)} role iterations / other calls", f"{tokens:,} tokens"]
    outcomes = f'{summary.get("claims", len(ontology.get("claims", {})))} proposed claims · {claim_counts.get("verified", 0)} verified · {claim_counts.get("pending", 0)} pending · {summary.get("experience", len(ontology.get("experience", [])))} experience entries'
    result_content, evaluation = "", {}
    if generation:
        stats[0] = f'Status: {generation.get("run_status", "unknown")}'
        counts = generation.get("status", {})
        outcomes = f'{generation.get("tasks", 0)} questions · {counts.get("completed", 0)} completed · {counts.get("incomplete", 0)} incomplete · {counts.get("failed", 0)} failed · {counts.get("pending", 0)} pending'
        result_content, evaluation = data_results(run, groups, output)
    options = '<option value="">All roles</option>' + "".join(f'<option value="{escaped(role)}">{escaped(role)}</option>' for role in sorted({group["role"] for group in groups.values()}))
    source_summary = details("Saved run summaries", block({"generation" if generation else "construction": summary, "run_status": status, "logs": log_summary, **({"native_evaluation": evaluation} if evaluation else {})}))
    document = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Agent loops · {escaped(run.name)}</title><style>{STYLE}</style></head>
<body><header class="run-header"><h1>Agent dialogue · {escaped(summary.get("lake_id", run.name))}</h1><p>{escaped(run.name)}</p><div class="stats">{"".join(f"<span>{escaped(stat)}</span>" for stat in stats)}</div><p>{escaped(outcomes)}</p><p>Choose a role iteration to read its complete dialogue: task, model inputs and outputs, actions, observations, and the accepted outcome. Attempts stay separate. New or changed context is shown at each call; exact request histories, usage and worker events are expandable. Other model calls keep their own identifiers.</p>{source_summary}</header>
{result_content}<div class="layout"><aside aria-label="Role iteration navigation"><div class="filters"><label for="role-filter">Agent role</label><select id="role-filter">{options}</select><label for="call-search">Search complete dialogue</label><input id="call-search" type="search" placeholder="Call number, code, error, or text" autocomplete="off"><div id="visible-count" aria-live="polite">{len(groups)} role iterations</div></div><nav id="visit-list">{"".join(nav)}</nav></aside><main><div class="pager"><button id="prev-visit" type="button">← Previous iteration</button><button id="next-visit" type="button">Next iteration →</button><span id="visit-position" aria-live="polite"></span></div><div id="no-results" hidden>{"No role iterations match these filters." if groups else "No model requests or visits were recorded for this run."}</div>{"".join(panels)}</main></div><script>{SCRIPT}</script></body></html>'''
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    return {"html": str(output.resolve()), "bytes": output.stat().st_size, "calls": len(calls), "provider_attempts": attempts, "visits": len(visits), "groups": len(groups)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Saved run directory")
    parser.add_argument("--output", type=Path, help="Defaults to RUN/logs/agent-loops.html")
    args = parser.parse_args()
    run = (WORKSPACE / args.run).resolve()
    output = (WORKSPACE / args.output).resolve() if args.output else run / "logs" / "agent-loops.html"
    print(json.dumps(export_html(run, output), indent=2))


if __name__ == "__main__":
    main()
