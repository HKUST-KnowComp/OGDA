"""Copy DiscoveryBench raw inputs and sanitized descriptions into a clean Lake."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .artifacts import WORKSPACE
SUPPLEMENTAL_INPUT = Path("real/train/immigration_offshoring_effect_on_employment/off_shoring_imputed.dta")
SCHEMA_PATH = Path("documentation/schema.json")


def _data_file(dataset: Path, name: str) -> Path:
    relative = Path(name)
    if (not name or relative.is_absolute() or relative == Path(".")
            or any(part.startswith(".") for part in relative.parts)
            or (relative.name.lower().startswith("metadata") and relative.suffix.lower() == ".json")
            or relative == SCHEMA_PATH.parent or relative.is_relative_to(SCHEMA_PATH)):
        raise ValueError(f"Unsafe data filename in {dataset}: {name!r}")
    path = dataset
    for part in relative.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError(f"Data inputs must not use symlinks: {path}")
    if not path.resolve().is_relative_to(dataset.resolve()):
        raise ValueError(f"Data input escapes its dataset directory: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Declared data input is not a regular file: {path}")
    return path


def _metadata_datasets(path: Path) -> list[dict]:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        # Several upstream synthetic metadata files use Windows-1252.
        text = raw.decode("cp1252")
    try:
        metadata = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid metadata JSON: {path}: {error}") from error
    if not isinstance(metadata, dict) or not isinstance(metadata.get("datasets"), list):
        raise ValueError(f"Metadata must contain a datasets list: {path}")
    datasets = []
    for entry in metadata["datasets"]:
        if (not isinstance(entry, dict) or not isinstance(entry.get("name"), str)
                or not isinstance(entry.get("description", ""), str)):
            raise ValueError(f"Dataset names and descriptions must be strings: {path}")
        columns = entry.get("columns", [])
        if isinstance(columns, dict):
            columns = columns.get("raw", [])
        if not isinstance(columns, list):
            raise ValueError(f"Dataset columns must be a list or contain a raw list: {path}")
        sanitized_columns = []
        for column in columns:
            if (not isinstance(column, dict) or not isinstance(column.get("name"), str)
                    or not isinstance(column.get("description", ""), str)):
                raise ValueError(f"Column names and descriptions must be strings: {path}")
            sanitized_columns.append({"name": column["name"], "description": column.get("description", "")})
        datasets.append({"name": entry["name"], "description": entry.get("description", ""),
                         "columns": sanitized_columns})
    return datasets


def discover_datalake(source: Path) -> tuple[list[Path], dict[Path, dict]]:
    """Validate all inputs and collect raw paths and schemas relative to the source."""
    source = Path(source).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"DiscoveryBench source directory does not exist: {source}")
    files = set()
    schemas = {}
    for category in ("real", "synth"):
        category_dir = source / category
        if category_dir.is_symlink() or not category_dir.is_dir():
            continue
        for split in sorted(category_dir.iterdir()):
            if split.name.startswith(".") or split.is_symlink() or not split.is_dir():
                continue
            for dataset in sorted(split.iterdir()):
                if dataset.name.startswith(".") or dataset.is_symlink() or not dataset.is_dir():
                    continue
                metadata_files = [path for path in sorted(dataset.glob("metadata*.json"))
                                  if not path.is_symlink() and path.is_file()]
                entries = []
                for metadata in metadata_files:
                    for entry in _metadata_datasets(metadata):
                        files.add(_data_file(dataset, entry["name"]).relative_to(source))
                        if entry not in entries:
                            entries.append(entry)
                if metadata_files:
                    schemas[dataset.relative_to(source)] = {"datasets": entries}
                # This upstream raw input is omitted from datasets[].name, but is
                # part of the immigration/offshoring dataset. No general glob is used.
                if dataset.relative_to(source) == SUPPLEMENTAL_INPUT.parent:
                    supplemental = dataset / SUPPLEMENTAL_INPUT.name
                    if supplemental.exists() or supplemental.is_symlink():
                        files.add(_data_file(dataset, supplemental.name).relative_to(source))
    return sorted(files), schemas


def discover_data_files(source: Path) -> list[Path]:
    """Return validated raw input paths relative to the source root."""
    return discover_datalake(source)[0]


def build_datalake(source: Path, output: Path) -> tuple[int, int, int]:
    """Create a fresh Lake; return raw file, dataset, and documentation file counts."""
    source = Path(source).resolve()
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Datalake output already exists; choose a new directory: {output}")
    output = output.resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Datalake output and DiscoveryBench source must be separate trees")
    files, schemas = discover_datalake(source)
    if not files:
        raise ValueError(f"No raw data inputs found; source must contain real/ or synth/ split directories: {source}")
    output.mkdir(parents=True)
    for relative in files:
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, destination)
    for dataset, schema in schemas.items():
        destination = output / dataset / SCHEMA_PATH
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dataset_count = len({relative.parts[:3] for relative in files} | {dataset.parts for dataset in schemas})
    return len(files), dataset_count, len(schemas)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("benchmark/discoverybench/discoverybench"),
                        help="DiscoveryBench source root (relative paths use the workspace)")
    parser.add_argument("--output", type=Path, default=Path("datalake/discoverybench"),
                        help="New clean Lake directory (relative paths use the workspace)")
    args = parser.parse_args(argv)
    output = WORKSPACE / args.output
    try:
        file_count, dataset_count, documentation_count = build_datalake(WORKSPACE / args.source, output)
    except (OSError, ValueError) as error:
        print(f"Error: {error}")
        return 1
    print(f"Copied {file_count} raw data files and wrote {documentation_count} documentation files "
          f"for {dataset_count} dataset directories to {output.resolve()}")
    return 0
