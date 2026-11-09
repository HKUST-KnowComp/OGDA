"""Copy a Lake for construction and list its Files. The original Lake never changes."""

from __future__ import annotations

import fnmatch
import json
import shutil
from pathlib import Path

import yaml

STRUCTURED = {".json", ".yaml", ".yml"}


def excluded(relative: str, patterns: list[str]) -> bool:
    """Match a Lake-relative path, a parent directory, or a file name against glob patterns."""
    parts = Path(relative).parts
    return any(
        fnmatch.fnmatch(relative, pattern)
        or fnmatch.fnmatch(parts[-1], pattern)
        or any(fnmatch.fnmatch("/".join(parts[:i]), pattern.rstrip("/")) for i in range(1, len(parts)))
        for pattern in patterns
    )


def strip_keys(value, keys: set[str]):
    if isinstance(value, dict):
        return {key: strip_keys(item, keys) for key, item in value.items() if key not in keys}
    if isinstance(value, list):
        return [strip_keys(item, keys) for item in value]
    return value


def copy_lake(source: str | Path, destination: str | Path, exclude: list[str] = (), stripped_keys: list[str] = ()) -> list[str]:
    """Make a task-blind copy: leave out excluded Files and remove stripped keys from JSON/YAML Files."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Lake root does not exist: {source}")
    if destination.is_relative_to(source):
        raise ValueError("The Lake copy must be outside the original Lake")
    if destination.exists():
        shutil.rmtree(destination)
    keys = set(stripped_keys)
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source).as_posix()
        if path.is_symlink() or not path.is_file() or excluded(relative, list(exclude)):
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if keys and path.suffix.lower() in STRUCTURED:
            if path.suffix.lower() == ".json":
                target.write_text(json.dumps(strip_keys(json.loads(path.read_text()), keys), indent=2, ensure_ascii=False))
            else:
                target.write_text(yaml.safe_dump(strip_keys(yaml.safe_load(path.read_text()), keys), sort_keys=False))
        else:
            shutil.copyfile(path, target)
    destination.mkdir(parents=True, exist_ok=True)
    return list_files(destination)


def list_files(root: str | Path) -> list[str]:
    root = Path(root)
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and not p.is_symlink())


def batches(files: list[str], size: int) -> list[list[str]]:
    """Group Files by directory, prioritizing Documentation candidates in each group.

    Names affect visit order only; the Proposer must establish whether a candidate is Documentation.
    """
    if size < 1:
        raise ValueError("files_per_visit must be positive")
    groups: dict[str, list[str]] = {}
    for path in files:
        groups.setdefault(str(Path(path).parent), []).append(path)
    ordered = [sorted(group, key=lambda path: (not documentation_candidate(path), path))
               for _, group in sorted(groups.items())]
    return [group[i:i + size] for group in ordered for i in range(0, len(group), size)]


def documentation_candidate(path: str) -> bool:
    """A filename hint for ordering, never proof of Documentation authority."""
    name = Path(path).stem.casefold().replace("-", "_")
    return any(hint in name for hint in ("readme", "metadata", "meta_data", "dictionary", "schema", "documentation"))
