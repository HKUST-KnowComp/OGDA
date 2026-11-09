"""Run the unmodified method with full visit logs, or rebuild readable log exports."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

from .artifacts import ONTOLOGY_SOURCE, key, pretty, read_jsonl
from .llm import ModelConfig, model_request_kwargs
sys.path.insert(0, str(ONTOLOGY_SOURCE))

from ogda import agent as agent_module
from ogda import cli, construction
from ogda.llm import OpenAICompatibleClient, request_id
from ogda.prompts import CONSOLIDATION, EXPERIENCE
from smolagents.memory import ActionStep


ACTIVE_VISIT: dict | None = None
LOG_ROOT: Path | None = None
CLIENTS: list = []


def append_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")


def visit_fields() -> dict:
    if ACTIVE_VISIT is None:
        return {}
    current_agent = ACTIVE_VISIT.get("agent")
    return {"visit": ACTIVE_VISIT["visit"], "visit_attempt": ACTIVE_VISIT["visit_attempt"],
            "stage": ACTIVE_VISIT.get("stage", ""),
            "step": current_agent.step_number if current_agent is not None else 1}


class LoggedClient(OpenAICompatibleClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        CLIENTS.append(self)

    def _request_kwargs(self, settings, system, prompt, cap):
        if settings.get("api") == "responses":
            return super()._request_kwargs(settings, system, prompt, cap)
        settings = dict(settings)
        effort = settings.pop("reasoning_effort", None)
        model = settings["model"]
        provider = settings.get("provider") or (model.split("/", 1)[0] if "/" in model else "openai")
        mapped = model_request_kwargs(ModelConfig(
            provider=provider, model=model, reasoning_effort=effort,
            model_kwargs=settings.get("model_kwargs", {}) or {},
        ))
        extra_body = mapped.pop("extra_body", None)
        settings["model_kwargs"] = mapped
        request = super()._request_kwargs(settings, system, prompt, cap)
        if extra_body:
            request["extra_body"] = extra_body
        return request

    def _complete(self, role, system, prompt, **kwargs):
        self._log_call_index = getattr(self, "_log_call_index", None)
        if self._log_call_index is None:
            existing = read_jsonl(self.record_path) if self.record_path else []
            self._log_call_index = max((r.get("call_index", i + 1) for i, r in enumerate(existing)), default=0)
        self._log_call_index += 1
        phase = "consolidation" if system == CONSOLIDATION else "experience" if system == EXPERIENCE else role
        self._log_context = visit_fields() or {"visit": f"{phase}/call-{self._log_call_index:06d}",
                                              "visit_attempt": 1, "step": 1}
        self._log_context.update(call_index=self._log_call_index, phase=phase, time=time.time())
        self._log_recorded = False
        self._provider_messages = None
        self._provider_error = None
        started = time.monotonic()
        calls_before = self.usage["calls"]
        if self.record_path:
            append_json(self.record_path.with_name("llm-inputs.jsonl"), {
                "role": role, "system": system, "prompt": prompt, **self._log_context,
                "request_id": request_id(role, system, prompt), "state": "started",
            })
        try:
            return super()._complete(role, system, prompt, **kwargs)
        except BaseException as error:
            if not self._log_recorded:
                self._record((role, system, prompt), started, error=type(error).__name__,
                             error_message=str(error), usage={"calls": self.usage["calls"] - calls_before})
            raise

    async def _request(self, kwargs):
        self._provider_messages = json.loads(json.dumps(kwargs.get("input", kwargs.get("messages"))))
        if self.record_path:
            append_json(self.record_path.with_name("provider-inputs.jsonl"), {
                **self._log_context, "provider_messages": self._provider_messages,
                "tools": kwargs.get("tools"), "tool_choice": kwargs.get("tool_choice"),
                "stop": kwargs.get("stop"),
                "api": "responses" if "input" in kwargs else "chat_completions",
                "max_tokens": kwargs.get("max_output_tokens", kwargs.get("max_completion_tokens", kwargs.get("max_tokens"))),
                "reasoning": kwargs.get("reasoning"),
                "thinking": kwargs.get("thinking", (kwargs.get("extra_body") or {}).get("thinking")),
                "reasoning_effort": kwargs.get("reasoning_effort", (kwargs.get("extra_body") or {}).get("reasoning_effort")),
            })
        try:
            return await super()._request(kwargs)
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            credential = kwargs.get("api_key")
            if credential:
                message = message.replace(str(credential), "[REDACTED]")
            self._provider_error = message
            raise

    def _record(self, request, started, **fields):
        if self._provider_messages is not None:
            fields["provider_messages"] = self._provider_messages
        if self._provider_error:
            fields["provider_error"] = self._provider_error
        super()._record(request, started, system=request[1], prompt=request[2],
                        **getattr(self, "_log_context", visit_fields()), **fields)
        self._log_recorded = True


class LoggedCodeAgent(agent_module._CodeAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if ACTIVE_VISIT is not None:
            ACTIVE_VISIT["agent"] = self
            self.step_callbacks.register(ActionStep, self._log_step)

    def _log_step(self, step, agent=None):
        append_json(ACTIVE_VISIT["artifact_root"] / "steps-full.jsonl", {
            **visit_fields(), **step.dict(),
        })


class LoggedExecutor(agent_module.IsolatedExecutor):
    def _log(self, event: str, **fields) -> None:
        append_json(self.artifact_root / "execution.jsonl", {
            "event": event, "time": time.time(), **visit_fields(), **fields,
        })

    def __call__(self, code_action):
        self._log("execution_start", code=code_action)
        try:
            output = super().__call__(code_action)
        except BaseException as error:
            self._log("execution_error", error=type(error).__name__, message=str(error))
            raise
        self._log("execution_end", output=output.output, stdout=output.logs,
                  is_final_answer=output.is_final_answer)
        return output

    def _receive(self, deadline):
        message = super()._receive(deadline)
        self._log("worker_message", message=message)
        return message

    def _run_tool(self, name, arguments):
        self._log("tool_start", name=name, arguments=arguments)
        result = super()._run_tool(name, arguments)
        self._log("tool_result", name=name, arguments=arguments, result=result)
        return result


ORIGINAL_RUN_AGENT = construction.run_agent


def logged_run_agent(llm, **kwargs):
    global ACTIVE_VISIT
    artifact = Path(kwargs["artifact_root"]).resolve()
    artifact.mkdir(parents=True, exist_ok=True)
    visit = str(artifact.relative_to(LOG_ROOT)) if LOG_ROOT and artifact.is_relative_to(LOG_ROOT) else str(artifact)
    previous = ACTIVE_VISIT
    metadata_path = artifact / "visit.json"
    old = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    metadata = {"visit": visit, "visit_attempt": old.get("visit_attempt", 0) + 1,
                "role": kwargs["role"], "task": kwargs["task"], "instructions": kwargs["instructions"],
                "started": time.time(), "status": "running"}
    if previous is not None:
        metadata.update(parent_visit=previous["visit"], parent_visit_attempt=previous["visit_attempt"],
                        parent_step=visit_fields()["step"])
    try:
        task = json.loads(kwargs["task"])
    except json.JSONDecodeError:
        task = None
    if isinstance(task, dict) and task.get("stage"):
        metadata["stage"] = task["stage"]
    ACTIVE_VISIT = {**metadata, "artifact_root": artifact}
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    result = None
    try:
        result = ORIGINAL_RUN_AGENT(llm, **kwargs)
        metadata.update(status="completed" if result.completed else "incomplete",
                        completed=result.completed, answer=result.answer, steps=result.steps)
        return result
    except BaseException as error:
        metadata.update(status="failed", completed=False, error=type(error).__name__, message=str(error))
        raise
    finally:
        current_agent = ACTIVE_VISIT.get("agent")
        if current_agent is not None:
            full_steps = current_agent.memory.get_full_steps()
            (artifact / "steps-full.json").write_text(json.dumps(full_steps, ensure_ascii=False, indent=1, default=str))
            metadata.setdefault("steps", sum(isinstance(s, ActionStep) for s in current_agent.memory.steps))
        metadata["ended"] = time.time()
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str))
        ACTIVE_VISIT = previous


def install_logging(log_root: Path) -> None:
    global LOG_ROOT
    LOG_ROOT = log_root.resolve()
    cli.OpenAICompatibleClient = LoggedClient
    agent_module._CodeAgent = LoggedCodeAgent
    agent_module.IsolatedExecutor = LoggedExecutor
    construction.run_agent = logged_run_agent
    if (ONTOLOGY_SOURCE / "ogda" / "data_agent.py").is_file():
        from ogda import data_agent

        data_agent.run_agent = logged_run_agent


def close_clients() -> None:
    while CLIENTS:
        CLIENTS.pop().close()


def export_logs(output: Path) -> dict:
    log_root = output.resolve() / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    calls = read_jsonl(log_root / "llm.jsonl")
    steps, executions, visits = {}, {}, []
    for path in sorted(log_root.rglob("visit.json")):
        visits.append(json.loads(path.read_text()))
    for path in sorted(log_root.rglob("steps-full.jsonl")):
        for step in read_jsonl(path):
            steps[key(step)] = step
    for path in sorted(log_root.rglob("execution.jsonl")):
        for event in read_jsonl(path):
            executions.setdefault(key(event), []).append(event)
    rows = []
    with (log_root / "agent-loops.txt").open("w") as stream:
        stream.write("Full model inputs and outputs, agent actions, tools, observations, and visit outcomes.\n")
        stream.write("Call numbers identify application requests; usage.calls includes provider retry attempts.\n\n")
        for visit in visits:
            stream.write(f"VISIT {visit['visit']} (attempt {visit['visit_attempt']}) [{visit['status']}]\n")
            stream.write("TASK\n" + pretty(visit["task"]) + "\nINSTRUCTIONS\n" + visit["instructions"] + "\n")
            if "answer" in visit:
                stream.write("ACCEPTED FINAL ANSWER\n" + pretty(visit["answer"]) + "\n")
            if "error" in visit:
                stream.write("VISIT ERROR\n" + visit["error"] + ": " + visit.get("message", "") + "\n")
            stream.write("\n")
        for number, record in enumerate(calls, 1):
            call_index = record.get("call_index", number)
            visit = record.get("visit", "unassigned")
            step = steps.get(key(record), {})
            stream.write("=" * 80 + "\n")
            stream.write(f"CALL {call_index} | {record['role']} | {visit} | attempt {record.get('visit_attempt', 1)} | step {record.get('step', '?')}\n")
            prompt = record.get("prompt", "[unavailable]")
            try:
                request = json.loads(prompt)
                messages = request.get("messages")
            except (ValueError, AttributeError):
                request = None
                messages = None
            if record.get("provider_messages") is not None:
                messages = record["provider_messages"]
            if messages is not None:
                for message in messages:
                    stream.write(f"\nINPUT MESSAGE [{message['role']}]\n" + pretty(message.get("content")) + "\n")
                if isinstance(request, dict) and "messages" in request:
                    stream.write("\nREQUEST OPTIONS\n" + pretty({k: v for k, v in request.items() if k != "messages"}) + "\n")
            else:
                stream.write("MODEL INPUT SYSTEM\n" + record.get("system", "[unavailable]") + "\n")
                stream.write("INPUT PROMPT\n" + prompt + "\n")
            stream.write("\nMODEL RESPONSE\n" + pretty(record.get("response_text", "[no response]")) + "\n")
            if record.get("tool_calls"):
                stream.write("\nNATIVE FUNCTION CALLS — MODEL OUTPUT\n" + pretty(record["tool_calls"]) + "\n")
            if step.get("model_output") is not None and step["model_output"] != record.get("response_text"):
                stream.write("\nAGENT MESSAGE — FORMATTED FROM MODEL OUTPUT\n" + pretty(step["model_output"]) + "\n")
            if "finish_reason" in record:
                stream.write("\nFINISH REASON\n" + pretty(record["finish_reason"]) + "\n")
            if record.get("error"):
                stream.write("MODEL ERROR\n" + pretty(record["error"]) + "\n" + record.get("error_message", "") + "\n")
            if record.get("provider_error"):
                stream.write("\nPROVIDER ERROR — REDACTED\n" + record["provider_error"] + "\n")
            stream.write("USAGE\n" + pretty(record.get("usage", {})) + "\n")
            for event in executions.get(key(record), []):
                fields = {name: value for name, value in event.items()
                          if name not in {"event", "time", "visit", "visit_attempt", "step"}}
                stream.write("\n" + event["event"].upper().replace("_", " ") + "\n" + pretty(fields) + "\n")
            for field in ("code_action", "observations", "error", "action_output", "is_final_answer"):
                if field in step and step[field] is not None:
                    stream.write("\n" + field.upper().replace("_", " ") + "\n" + pretty(step[field]) + "\n")
            rows.append({"call_index": call_index, "role": record["role"], "visit": visit,
                         "visit_attempt": record.get("visit_attempt", 1), "step": record.get("step", ""),
                         "stage": record.get("stage", ""),
                         "phase": record.get("phase", ""), "model": record.get("model", ""),
                         "error": record.get("error", ""), "wall_seconds": record.get("wall_seconds", ""),
                         "input_tokens": record.get("usage", {}).get("input_tokens", 0),
                         "output_tokens": record.get("usage", {}).get("output_tokens", 0),
                         "provider_attempts": record.get("usage", {}).get("calls", 0)})
    columns = ["call_index", "role", "visit", "visit_attempt", "step", "stage", "phase", "model", "error",
               "wall_seconds", "input_tokens", "output_tokens", "provider_attempts"]
    with (log_root / "loop-index.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    summary = {"llm_requests": len(calls), "provider_attempts": sum(row["provider_attempts"] for row in rows),
               "llm_errors": sum(bool(r.get("error")) for r in calls), "visits": len(visits),
               "visit_statuses": {status: sum(v["status"] == status for v in visits)
                                  for status in sorted({v["status"] for v in visits})},
               "action_steps": len(steps),
               "tool_calls": sum(e["event"] == "tool_start" for events in executions.values() for e in events),
               "execution_errors": sum(e["event"] == "execution_error" for events in executions.values() for e in events),
               "roles": {role: sum(r["role"] == role for r in calls) for role in sorted({r["role"] for r in calls})}}
    (log_root / "log-summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "export":
        parser = argparse.ArgumentParser()
        parser.add_argument("--output", type=Path, required=True)
        args = parser.parse_args(argv[1:])
        print(json.dumps(export_logs(args.output), indent=2))
        return
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", type=Path, required=True)
    args, _ = parser.parse_known_args(argv)
    install_logging(args.output / "logs")
    status = {"status": "running", "started": time.time()}
    try:
        cli.main(argv)
        status["status"] = "completed"
    except BaseException as error:
        status.update(status="failed", error=type(error).__name__, message=str(error))
        raise
    finally:
        status["ended"] = time.time()
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "logged-run-status.json").write_text(json.dumps(status, indent=2))
        export_logs(args.output)


if __name__ == "__main__":
    main()
