"""Copy each benchmark Lake into its own clean folder, with a manifest beside them.

Run as a script: `uv run python scripts/prepare_lakes.py --benchmark <name>`.
`scripts/` is on `sys.path` when run this way, so this module imports the
benchmark registry as a top-level package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from benchmarks import REPO, ROOTS, get
from ogda.lake import copy_lake

MANIFEST_NAME = "manifest.json"


def sha256_files(root: Path) -> dict[str, str]:
    """Hash every File under `root`, keyed by its path relative to `root`."""
    root = Path(root)
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    }


def prepare(benchmark: str, root: Path, lake_dir: Path, force: bool = False) -> dict:
    """Copy each of `benchmark`'s Lakes into `lake_dir/<lake_id>/`; write `manifest.json`."""
    lake_dir = Path(lake_dir)
    if lake_dir.exists():
        if not force:
            raise FileExistsError(
                f"{lake_dir} already exists; pass --force to rebuild it"
            )
        shutil.rmtree(lake_dir)
    lake_dir.mkdir(parents=True)

    module = get(benchmark)
    lakes: dict[str, dict] = {}
    for lake in module.lakes(root):
        lake_id = lake["lake_id"]
        source = Path(lake["source"])
        destination = lake_dir / lake_id
        exclude = lake["exclude"]
        strip_keys = lake["strip_keys"]
        copy_lake(source, destination, exclude, strip_keys)
        lakes[lake_id] = {
            "source": str(source),
            "exclude": exclude,
            "strip_keys": strip_keys,
            "source_sha256": sha256_files(source),
            "clean_sha256": sha256_files(destination),
        }

    manifest = {
        "benchmark": benchmark,
        "created": datetime.now(timezone.utc).isoformat(),
        "lakes": lakes,
    }
    (lake_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
    return manifest


def verify(lake_dir: Path, lake_ids: list[str]) -> None:
    """Raise ValueError naming the first missing or changed Lake in `lake_dir`."""
    lake_dir = Path(lake_dir)
    manifest_path = lake_dir / MANIFEST_NAME
    if not manifest_path.exists():
        raise ValueError(
            f"{manifest_path} is missing; run scripts/prepare_lakes.py --force"
        )
    manifest = json.loads(manifest_path.read_text())
    benchmark = manifest.get("benchmark")
    hint = f"; run scripts/prepare_lakes.py --benchmark {benchmark} --force"
    lakes = manifest.get("lakes", {})
    for lake_id in lake_ids:
        if lake_id not in lakes:
            raise ValueError(f"Lake {lake_id!r} is missing from {manifest_path}{hint}")
        clean_dir = lake_dir / lake_id
        if not clean_dir.is_dir():
            raise ValueError(f"Lake {lake_id!r} folder {clean_dir} is missing{hint}")
        current = sha256_files(clean_dir)
        if current != lakes[lake_id]["clean_sha256"]:
            raise ValueError(f"Lake {lake_id!r} has changed{hint}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, choices=sorted(ROOTS))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--root", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--lake-dir", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    root = args.root if args.root is not None else ROOTS[args.benchmark]
    lake_dir = args.lake_dir if args.lake_dir is not None else REPO / "lake" / args.benchmark

    try:
        manifest = prepare(args.benchmark, root, lake_dir, force=args.force)
    except FileExistsError as exc:
        print(f"{exc}; use --force to rebuild it", file=sys.stderr)
        return 1

    for lake_id, lake in manifest["lakes"].items():
        print(f"{lake_id}: {len(lake['clean_sha256'])} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
