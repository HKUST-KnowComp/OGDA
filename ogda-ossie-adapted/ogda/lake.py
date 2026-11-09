"""Copy a Lake for construction, list and read its Files, and group Files that share a layout.

The original Lake never changes.
"""

from __future__ import annotations

import csv
import fnmatch
import io
import json
import shutil
from itertools import islice
from pathlib import Path

import openpyxl
import yaml

STRUCTURED = {".json", ".yaml", ".yml"}
TABULAR = {".csv", ".tsv", ".txt", ".xlsx", ".xlsm"}
EXCEL = {".xlsx", ".xlsm"}
SNIFF_BYTES = 65536
HEAD_ROWS = 10  # Title rows a layout may have above its header row.


def excluded(relative: str, patterns: list[str]) -> bool:
    """Match a Lake-relative path, a parent directory, or a file name against glob patterns."""
    parts = Path(relative).parts
    return any(
        fnmatch.fnmatch(relative, pattern)
        or fnmatch.fnmatch(parts[-1], pattern)
        or any(fnmatch.fnmatch("/".join(parts[:i]), pattern.rstrip("/")) for i in range(1, len(parts)))
        for pattern in patterns
    )


def decode_text(path: Path) -> str:
    """Decode a File as UTF-8, falling back to cp1252 for Files like DiscoveryBench metadata."""
    data = path.read_bytes()
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("cp1252")


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
                target.write_text(
                    json.dumps(strip_keys(json.loads(decode_text(path)), keys), indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            else:
                target.write_text(
                    yaml.safe_dump(strip_keys(yaml.safe_load(decode_text(path)), keys), sort_keys=False),
                    encoding="utf-8",
                )
        else:
            shutil.copyfile(path, target)
    destination.mkdir(parents=True, exist_ok=True)
    return list_files(destination)


def list_files(root: str | Path) -> list[str]:
    root = Path(root)
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and not p.is_symlink())


def documentation_candidate(path: str) -> bool:
    """A filename hint for ordering, never proof of Documentation authority."""
    name = Path(path).stem.casefold().replace("-", "_")
    return any(hint in name for hint in ("readme", "metadata", "meta_data", "dictionary", "schema", "documentation",
                                         "description", "descrip", "codebook"))


def split_documentation(ref: str) -> tuple[str, str | None]:
    """The File and worksheet of a Documentation reference: a File path, or a workbook path and one of its
    worksheets joined by '#', such as 'tables.xlsx#README'. The worksheet is None for a whole File."""
    lower = ref.lower()
    for suffix in EXCEL:
        at = lower.find(suffix + "#")
        if at > 0 and at + len(suffix) + 1 < len(ref):
            return ref[:at + len(suffix)], ref[at + len(suffix) + 1:]
    return ref, None


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


def _text(path: Path, limit: int | None = None) -> str:
    raw = path.read_bytes() if limit is None else path.open("rb").read(limit)
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def _dialect(path: Path, text: str):
    try:
        return csv.Sniffer().sniff(text[:SNIFF_BYTES], delimiters=",\t;|")
    except csv.Error:
        return csv.excel_tab if path.suffix.lower() == ".tsv" else csv.excel


def read_rows(path: Path, sheet: str | None, count: int | None = None) -> list[list[str]]:
    """The first `count` rows of a text table or one worksheet, or all of them when `count` is None, as strings."""
    if path.suffix.lower() in EXCEL:
        workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
        try:
            rows = workbook[sheet or workbook.sheetnames[0]].iter_rows(values_only=True)
            return [["" if cell is None else str(cell) for cell in row] for row in islice(rows, count)]
        finally:
            workbook.close()
    text = _text(path)
    return list(islice(csv.reader(io.StringIO(text), _dialect(path, text)), count))


class RowReader:
    """The rows of tables for one round of checks: each File or worksheet is read once, as far as the checks ask, and
    read again only when a later check asks for more rows than were read. Use a new reader when Files may change."""

    def __init__(self) -> None:
        self.tables: dict[tuple[Path, str | None], tuple[list[list[str]], bool]] = {}  # Rows read, and whether all.

    def __call__(self, path: Path, sheet: str | None, count: int | None = None) -> list[list[str]]:
        """The first `count` rows of a table, or all of them when `count` is None; fewer when the table is shorter."""
        key = (Path(path), sheet)
        rows, whole = self.tables.get(key, ([], False))
        if not whole and (count is None or len(rows) < count):
            rows = read_rows(Path(path), sheet, count)
            whole = count is None or len(rows) < count
            self.tables[key] = rows, whole
        return rows if count is None else rows[:count]


def header_names(rows: list[list[str]], header_row: int, header_rows: int = 1) -> list[str]:
    """The column names of a Table, one per column position.

    With one header row, a name is the stripped cell. With several, a column's name joins its nonempty header cells
    from top to bottom with " / ". In every header row but the last, an empty cell repeats the nearest nonempty cell
    to its left in that row, as a merged cell would, when a header cell below it in its column is nonempty; a column
    with no header cell below ends the repetition.
    """
    block = [[cell.strip() for cell in rows[row]] for row in range(header_row, header_row + header_rows)]
    width = max(len(row) for row in block)
    block = [row + [""] * (width - len(row)) for row in block]
    for level, row in enumerate(block[:-1]):
        last = ""
        for column, cell in enumerate(row):
            if cell:
                last = cell
            elif not any(lower[column] for lower in block[level + 1:]):
                last = ""
            elif last:
                row[column] = last
    return [" / ".join(row[column] for row in block if row[column]) for column in range(width)]


def sheets(path: Path) -> tuple[list[str], str | None]:
    """The nonempty worksheets of a workbook and its first sheet name; ([], None) for other Files."""
    if path.suffix.lower() not in EXCEL:
        return [], None
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        return ([sheet.title for sheet in workbook if any(any(v is not None for v in row)
                                                          for row in sheet.iter_rows(values_only=True))],
                workbook.sheetnames[0])
    finally:
        workbook.close()


def _head(rows) -> tuple | None:
    """The rows up to and including the first header-like row (two or more nonempty cells), within HEAD_ROWS."""
    head = []
    for row in rows:
        cells = tuple("" if cell is None else str(cell).strip() for cell in row)
        head.append(cells)
        if sum(bool(cell) for cell in cells) >= 2:
            return tuple(head)
        if len(head) >= HEAD_ROWS:
            return None
    return None


def layout(path: Path) -> tuple | None:
    """A File's layout: its type and its title rows plus header row, per worksheet for a workbook. None when no
    header-like row starts the File, or it cannot be read."""
    suffix = path.suffix.lower()
    try:
        if suffix in EXCEL:
            workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
            try:
                heads = [(sheet.title, _head(sheet.iter_rows(values_only=True, max_row=HEAD_ROWS))) for sheet in workbook]
            finally:
                workbook.close()
            return (suffix, tuple(heads)) if any(head for _, head in heads) else None
        if suffix not in TABULAR:
            return None
        text = _text(path, SNIFF_BYTES)
        head = _head(csv.reader(io.StringIO(text), _dialect(path, text)))
        return (suffix, head) if head else None
    except Exception:  # An unreadable File gets its own mapping visit, where the agent can say why.
        return None


def layout_groups(root: str | Path, files: list[str]) -> dict[str, list[str]]:
    """Files that share one layout: {representative: [the other Files]}, for groups of two or more Files.

    The representative is the first File of the group in path order.
    """
    root = Path(root)
    groups: dict[tuple, list[str]] = {}
    for path in sorted(files):
        if (signature := layout(root / path)) is not None:
            groups.setdefault(signature, []).append(path)
    return {members[0]: members[1:] for members in groups.values() if len(members) > 1}
