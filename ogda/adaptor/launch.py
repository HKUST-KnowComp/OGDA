"""Launch one fresh Lake construction with complete agent traces."""

from __future__ import annotations

import argparse
import contextlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from .artifacts import WORKSPACE, export_system_prompts, input_hashes, read_json
from .viewers.html import export_html
from .viewers.ontology_html import export_ontology_html

OVERRIDES = ("max_passes", "max_tokens", "files_per_visit", "max_claims_per_visit")


def public_config(config: Path) -> dict:
    import yaml
    from ogda.agent import load_model_profile
    from ogda.llm import merge_settings

    document = yaml.safe_load(config.read_text()) or {}
    llm = document.get("llm", document)
    allowed = {"model", "base_url", "api_base", "api", "reasoning", "reasoning_effort", "temperature", "seed",
               "max_calls", "max_total_tokens", "max_cost_usd", "max_output_tokens", "max_tokens", "timeout_seconds"}
    base = {key: value for key, value in llm.items() if key != "roles"}
    roles = {}
    for role in ("proposer", "verifier", "experience"):
        settings = dict(base)
        if role == "experience":
            settings = merge_settings(settings, llm.get("roles", {}).get("proposer", {}) or {})
        settings = merge_settings(settings, llm.get("roles", {}).get(role, {}) or {})
        profile = load_model_profile(settings.get("model", ""))
        roles[role] = {"settings": {key: value for key, value in settings.items() if key in allowed},
                       "model_profile": {"tool_choice": profile["tool_choice"],
                                         "code_block_tags": profile.get("code_block_tags", ["<code>", "</code>"]),
                                         "custom_system_prompt": "system_prompt" in profile}}
    construction = document.get("construction") or {}
    return {"roles": roles, "construction": {key: value for key, value in construction.items()
                                              if key in {*OVERRIDES, "agent"}}}


def public_model(config: Path, *, role: str = "data_agent", request_options: bool = False) -> dict:
    from ogda.adaptor.llm import ModelConfig, model_request_kwargs
    from ogda.agent import load_model_profile
    from ogda.llm import OpenAICompatibleClient

    client = OpenAICompatibleClient(config)
    try:
        settings = client.role_config(role)
        allowed = {"model", "base_url", "api_base", "api", "reasoning", "reasoning_effort", "temperature", "seed",
                   "max_calls", "max_total_tokens", "max_cost_usd", "max_output_tokens", "max_tokens", "timeout_seconds"}
        profile = load_model_profile(settings["model"])
        if request_options:
            allowed.update({"provider", "model_kwargs"})
        result = {"role": role, "settings": {key: value for key, value in settings.items() if key in allowed},
                "tool_choice": profile["tool_choice"], "custom_system_prompt": "system_prompt" in profile}
        if request_options and settings.get("api") == "responses":
            cap = min(int(settings.get("max_output_tokens", client.max_output_tokens)), client.max_output_tokens)
            options = client._request_kwargs(settings, "", "", cap)
            result["request_options"] = {key: value for key, value in options.items()
                                         if key in {"max_output_tokens", "reasoning", "temperature", "top_p",
                                                    "parallel_tool_calls", "store"}}
        elif request_options:
            provider = settings.get("provider") or (settings["model"].split("/", 1)[0] if "/" in settings["model"] else "openai")
            result["request_options"] = model_request_kwargs(ModelConfig(provider=provider, model=settings["model"],
                                                                         reasoning_effort=settings.get("reasoning_effort"),
                                                                         model_kwargs=settings.get("model_kwargs", {}) or {}))
        return result
    finally:
        client.close()


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def run(args) -> int:
    from . import logging

    lake = (WORKSPACE / args.lake).resolve()
    config = (WORKSPACE / args.config).resolve()
    if not lake.is_dir():
        raise FileNotFoundError(f"Lake directory does not exist: {lake}")
    if not config.is_file():
        raise FileNotFoundError(f"LLM configuration does not exist: {config}")
    run_id = args.run_id or f'ogda-{lake.name.replace("_", "-")}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}'
    if Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("--run-id must be one directory name")
    output = (WORKSPACE / (args.output or Path("results/discoverybench") / run_id)).resolve()
    scratch = (WORKSPACE / (args.scratch or Path("exp_env/discoverybench") / run_id)).resolve()
    for path in (output, scratch):
        if path.exists():
            raise FileExistsError(f"Run directory already exists; choose a new run ID or directory: {path}")
        if path.is_relative_to(lake) or lake.is_relative_to(path):
            raise ValueError("Run directories and the original Lake must be separate trees")
    if output.is_relative_to(scratch) or scratch.is_relative_to(output):
        raise ValueError("Results and scratch must be separate trees")
    output.mkdir(parents=True)
    scratch.mkdir(parents=True)
    logs = output / "logs"
    logs.mkdir()
    print(f"OGDA ontology construction: {run_id}\nResults: {output}\nScratch: {scratch}\nConsole: {logs / 'console.log'}", flush=True)
    before = input_hashes(lake)
    write_json(output / "input-hashes-before.json", before)
    info = {"run_id": run_id, "scope": "task-blind ontology construction and agent-trace diagnostic",
            "lake": str(lake), "config": str(config), "output": str(output), "scratch": str(scratch),
            "started": time.time(), "status": "running", "construction_overrides": {
                name: getattr(args, name) for name in OVERRIDES if getattr(args, name) is not None},
            "exclude": args.exclude, "strip_key": args.strip_key}
    write_json(output / "run-info.json", info)
    command = ["build", "--config", str(config), "--lake", str(lake), "--output", str(output), "--scratch", str(scratch)]
    for name in OVERRIDES:
        if getattr(args, name) is not None:
            command += ["--" + name.replace("_", "-"), str(getattr(args, name))]
    for flag, values in (("--exclude", args.exclude), ("--strip-key", args.strip_key)):
        if values:
            command += [flag, *values]
    code, failure = 0, None
    with (logs / "console.log").open("w", buffering=1) as console:
        with contextlib.redirect_stdout(console), contextlib.redirect_stderr(console):
            try:
                info["public_config"] = public_config(config)
                write_json(output / "run-info.json", info)
                logging.main(command)
            except BaseException as error:
                code = 130 if isinstance(error, KeyboardInterrupt) else 1
                failure = f"{type(error).__name__}: {error}"
                print(failure, flush=True)
                info.update(status="failed", error=type(error).__name__, message=str(error))
                if not (output / "logged-run-status.json").exists():
                    write_json(output / "logged-run-status.json", {"status": "failed", "error": type(error).__name__, "message": str(error)})
            finally:
                logging.close_clients()
                after = input_hashes(lake)
                write_json(output / "input-hashes-after.json", after)
                integrity = {"unchanged": before == after, "files_before": len(before), "files_after": len(after),
                             "added": sorted(after.keys() - before.keys()), "removed": sorted(before.keys() - after.keys()),
                             "modified": sorted(name for name in before.keys() & after.keys() if before[name] != after[name])}
                write_json(output / "source-integrity.json", integrity)
                info["source_integrity"] = integrity
                logging.export_logs(output)
                info["system_prompts"] = export_system_prompts(output)
                export_html(output, logs / "agent-loops.html")
                if (output / "ontology" / "ontology.json").is_file():
                    info["ontology_graph"] = export_ontology_html(output, output / "ontology" / "ontology.html")
                info["ended"] = time.time()
                if not code:
                    info["status"] = "completed"
                    if not integrity["unchanged"]:
                        code = 1
                        failure = "Original Lake changed during the run; inspect source-integrity.json"
                        info["status"] = "source_changed"
                summary = read_json(output / "construction-summary.json")
                info["stop_reason"] = summary.get("stop_reason")
                write_json(output / "run-info.json", info)
    if failure:
        print(f"Failed: {failure}")
    else:
        print(f"Stopped: {info['stop_reason']}")
    print(f"Agent loops: {logs / 'agent-loops.html'}\nSystem prompts: {logs / 'system-prompts.txt'}\nText trace: {logs / 'agent-loops.txt'}")
    if "ontology_graph" in info:
        print(f"Ontology graph: {info['ontology_graph']['html']}")
    return code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lake", type=Path, default=Path("test_env/dis_meta"))
    parser.add_argument("--config", type=Path, default=Path("llm_config/discoverybench/config.yaml"))
    parser.add_argument("--run-id", help="Defaults to a fresh UTC timestamped identifier")
    parser.add_argument("--output", type=Path, help="Exact new results directory")
    parser.add_argument("--scratch", type=Path, help="Exact new scratch directory")
    parser.add_argument("--exclude", nargs="*", default=[], help="File globs excluded from the Lake copy")
    parser.add_argument("--strip-key", nargs="*", default=[], help="JSON/YAML keys removed from the Lake copy")
    for name in OVERRIDES:
        parser.add_argument("--" + name.replace("_", "-"), type=int, help="Override construction YAML; otherwise use method defaults")
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        print(f"Error: {error}")
        return 1
