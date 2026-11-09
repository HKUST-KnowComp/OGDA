"""Prepare task-blind benchmark data copies and verify their contents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from .artifacts import WORKSPACE, file_digest, input_hashes
from .benchmarks import BENCHMARKS
from .benchmarks.kramabench import DOMAINS
from .discovery_datalake import SCHEMA_PATH, build_datalake, discover_datalake
from .runtime import prepare
from .suite import canonical_benchmark


def build_kramabench(source: Path, output: Path) -> tuple[int, int]:
    """Copy only the six input trees, keeping workload and scoring files separate."""
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Datalake output already exists; choose a new directory: {output}")
    source, output = source.resolve(), output.resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Datalake output and KramaBench source must be separate trees")
    expected = {}
    for domain in DOMAINS:
        inputs = source / "data" / domain / "input"
        if not inputs.is_dir():
            raise FileNotFoundError(f"KramaBench inputs are missing: {inputs}")
        if any(path.is_symlink() for path in inputs.rglob("*")):
            raise ValueError(f"Raw input trees must not contain symlinks: {inputs}")
        hashes = input_hashes(inputs)
        if not hashes:
            raise ValueError(f"KramaBench input tree is empty: {inputs}")
        expected.update({f"{domain}/{name}": digest for name, digest in hashes.items()})
    output.mkdir(parents=True)
    for domain in DOMAINS:
        shutil.copytree(source / "data" / domain / "input", output / domain)
    if input_hashes(output) != expected:
        raise ValueError(f"KramaBench copied inputs failed hash verification: {output}")
    return len(expected), len(DOMAINS)


def build_discoverybench(source: Path, output: Path) -> tuple[int, int, int]:
    """Reuse the whitelist builder and verify every raw file and sanitized schema."""
    counts = build_datalake(source, output)
    files, schemas = discover_datalake(source)
    expected = {relative.as_posix(): file_digest(source / relative) for relative in files}
    for dataset, schema in schemas.items():
        path = output / dataset / SCHEMA_PATH
        if json.loads(path.read_text()) != schema:
            raise ValueError(f"Sanitized DiscoveryBench documentation differs: {path}")
        expected[(dataset / SCHEMA_PATH).as_posix()] = file_digest(path)
    if input_hashes(output) != expected:
        raise ValueError(f"DiscoveryBench copied inputs failed hash verification: {output}")
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=canonical_benchmark, default="all")
    parser.add_argument("--output-root", type=Path, default=Path("datalake"),
                        help="Parent for new benchmark directories; relative paths use the OGDA repository")
    args = parser.parse_args(argv)
    selected = BENCHMARKS if args.benchmark == "all" else (args.benchmark,)
    output_root = (WORKSPACE / args.output_root).resolve()
    for name in selected:
        output = output_root / name
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"Datalake output already exists; choose a new directory: {output}")
    runtime = prepare()
    for name in selected:
        source, output = runtime / "benchmark" / name, output_root / name
        if name == "Kramabench":
            files, datasets = build_kramabench(source, output)
            print(f"{name}: verified {files} raw files in {datasets} domain Lakes at {output}")
        else:
            files, datasets, schemas = build_discoverybench(source / "discoverybench", output)
            print(f"{name}: verified {files} raw files and {schemas} sanitized schemas in {datasets} dataset Lakes at {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
