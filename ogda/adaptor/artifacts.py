"""Small shared helpers for workspace paths, saved JSON, file digests and saved traces."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

WORKSPACE = Path(os.environ.get("OGDA_WORKSPACE", Path(__file__).resolve().parents[2])).resolve()
ONTOLOGY_SOURCE = (WORKSPACE / os.environ["OGDA_SOURCE"]).resolve() if os.environ.get("OGDA_SOURCE") else Path(__file__).resolve().parents[2]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def input_hashes(lake: Path) -> dict:
    hashes = {}
    for path in sorted(lake.rglob("*")):
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            hashes[path.relative_to(lake).as_posix()] = digest.hexdigest()
    return hashes


def safe_id(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", value) or value in {".", ".."}:
        raise ValueError(f"Invalid artifact identifier: {value!r}")
    return value


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def key(record: dict) -> tuple:
    return record.get("visit"), record.get("visit_attempt", 1), record.get("step", record.get("step_number"))


def pretty(value) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2, default=str)


def system_messages(record: dict) -> list:
    if "provider_messages" in record:
        return [message["content"] for message in record["provider_messages"] if message.get("role") == "system"]
    try:
        request = json.loads(record.get("prompt", ""))
    except ValueError:
        request = None
    if isinstance(request, dict) and "messages" in request:
        return [message["content"] for message in request["messages"] if message.get("role") == "system"]
    system = record.get("system")
    return [system] if system and system != "codeagent_v1" else []


def export_system_prompts(run: Path) -> dict:
    logs = run / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    groups = {}
    for position, record in enumerate(read_jsonl(logs / "llm.jsonl"), 1):
        for prompt in system_messages(record):
            text = pretty(prompt)
            group = groups.setdefault((record["role"], text), [])
            group.append(record.get("call_index", position))
    roles = {}
    for (role, prompt), calls in groups.items():
        entry = f'SYSTEM PROMPT | role={role} | calls={",".join(map(str, calls))}\n\n{prompt}\n\n'
        roles.setdefault(role, []).append(entry)
    combined = "".join(entry for entries in roles.values() for entry in entries)
    (logs / "system-prompts.txt").write_text(combined or "No system prompts were recorded.\n")
    for role, entries in roles.items():
        safe_role = re.sub(r"[^A-Za-z0-9_.-]", "_", role)
        (logs / f"system-prompts-{safe_role}.txt").write_text("".join(entries))
    return {"unique_prompts": len(groups), "roles": sorted(roles)}
