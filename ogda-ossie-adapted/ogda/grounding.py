"""What `ground` prints and returns for Datasets, Fields, Files and patterns, with their joins and Experience.

A view prints meanings only: what the Ontology and its Experience say, never a value read from the rows apart from the
row count. A value tells the sandbox what to hand the agent: a Dataset's cached Parquet table, one Field of it, a
File's path or the paths a pattern matches. A layout group member is grounded as its own File, with its
representative's meaning; Experience anchored to a member counts for its representative.
"""

from __future__ import annotations

import csv
import difflib
import json
import re
import zipfile
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import NamedTuple

from openpyxl.utils.exceptions import InvalidFileException
import pyarrow.parquet as pq

from .binding import field_of, is_literal
from .dataset_reader import cache_dataset, column_names, file_column, read_as_text
from .lake import list_files, split_documentation
from .ontology import Harness
from .ontology_search import OntologySearch, first_sentence
from .ossie import OgdaExtension, concepts, identifiers, ogda_extension

COLUMNS = 12000  # Characters a view's column lines take at most; past it every line's meaning is cut to fit.
LESSONS = 10  # Experience lessons one view shows.
EXAMPLES = 3  # Paths a pattern's view names.
LISTED = 10   # Table labels one Entity type line of the Lake map names.
NORMALIZE = "; compare letters and digits only, lowercased"
# What reading a table's File raises when code cannot read it: a malformed text table or workbook, a missing sheet or
# header column, an undecodable File. The agent reads the File itself instead.
READ_ERRORS = (csv.Error, KeyError, IndexError, OSError, ValueError, zipfile.BadZipFile, InvalidFileException)


@dataclass
class Grounded:
    """One grounded name: the view ground prints, the value the sandbox decodes, its Files and its row count."""

    name: str
    kind: str  # "dataset", "field", "file" or "pattern"
    view: str
    value: dict
    files: list[str]
    rows: int | None


class Target(NamedTuple):
    """What a name grounds: a Dataset, one of its Fields, or Lake Files (one File, or the Files a pattern matches)."""

    kind: str
    dataset: str = ""
    field: str = ""
    paths: tuple[str, ...] = ()


class Table(NamedTuple):
    """A Dataset as code read it: its cached Parquet table, row count and the Fields read as text; with no path, code
    does not read its File's format, or could not read the File for the `failure` given."""

    path: Path | None = None
    rows: int | None = None
    text: frozenset[str] = frozenset()
    failure: str = ""


class Key(NamedTuple):
    """A column that names Entities of one Entity type by one Identifier."""

    field: str
    entity: str
    side: str  # The column as a join line shows it: its quoted header text, or `from path <field>`.
    identifier: str
    normalized: bool


def size_text(size: int) -> str:
    """A size in bytes as a view prints it: "30 bytes", "18 KB" or "2.9 MB"."""
    if size < 1000:
        return f"{size} bytes"
    if size < 1_000_000:
        return f"{round(size / 1000)} KB"
    return f"{size / 1_000_000:.1f} MB"


def as_json(value) -> str:
    """A header text in double quotes, or an object, on one line."""
    return json.dumps(value, ensure_ascii=False)


def one_line(text: str) -> str:
    """The text with each run of whitespace one space."""
    return " ".join(text.split())


def header(entry: dict) -> str:
    """Where a column's values come from: its quoted header text, trimmed as the File's header is, or `from path`."""
    return as_json(entry["column"].strip()) if "column" in entry else "from path"


def key(entry: dict, identifier: str) -> Key:
    side = header(entry) if "column" in entry else f"from path {entry['field']}"
    return Key(entry["field"], identifier.partition(".")[0], side, identifier,
               identifier in entry.get("normalized", []))


def markers(entry: dict) -> list[str]:
    """The Identifier a column gives or refers by, the Identifiers it is normalized for, and why it is unbound."""
    found = []
    if "identifies" in entry:
        found.append(f"[identifies {entry['identifies']}]")
    if "refers_to" in entry:
        found.append(f"[refers to {entry['refers_to']}]")
    if entry.get("normalized"):
        found.append(f"[normalized for {', '.join(entry['normalized'])}]")
    if "unbound" in entry:
        found.append(f"[unbound: {one_line(entry['unbound'])}]")
    return found


def datatype(entry: dict, table: Table) -> str:
    """A column's datatype, and whether its values did not all convert, so its column holds text."""
    return entry["datatype"] + (", read as text" if entry["field"] in table.text else "")


def by_header(entry: dict, column: str) -> bool:
    """Whether a Field's column in the DataFrame is named by its header text."""
    return "column" in entry and column == entry["column"].strip()


def column_line(entry: dict, table: Table, column: str, width: int | None = None) -> str:
    """A column's line: its header and Field name, header first when it names the column, datatype, meaning, markers.
    Given a width, a header that names its column stands alone, and the meaning and markers are cut at a word to fit
    it; the names and datatype stay whole."""
    if by_header(entry, column):
        names = [header(entry)] if width else [header(entry), entry["field"]]
    else:
        names = [column, header(entry)]
    stem = "  " + "  ".join([*names, datatype(entry, table)])
    rest = "  ".join(part for part in [first_sentence(entry["description"]), *markers(entry)] if part)
    if not rest:
        return stem
    if width is None or len(stem) + 2 + len(rest) <= width:
        return f"{stem}  {rest}"
    room = width - len(stem) - 3  # The two spaces before the meaning and the ellipsis after it.
    cut = rest[:max(room, 0)]
    if rest[len(cut):len(cut) + 1] != " ":
        cut = cut.rpartition(" ")[0] or cut
    return f"{stem}  {cut.rstrip()}…" if cut.strip() else stem


def fills(link: dict) -> str:
    """A fact a column fills: its sentence, the Role the column plays and the values of the other Roles, the Entity
    it is about first when the Binding names it."""
    others = ([(link["entity"], link["of"])] if "of" in link else []) + list(link["with"].items())
    line = f'Fills {link["fact"]}: "{link["says"]}" as {link["role"]}'
    if others:
        line += ", with " + ", ".join(f"{role} = {as_json(value) if isinstance(value, dict) else value}"
                                      for role, value in others)
    return line


def region(extension: OgdaExtension) -> str:
    """A Dataset's header and data rows in its File."""
    if extension.header_row is None:
        names = "no header row (columns named 0, 1, ...)"
    elif extension.header_rows > 1:
        names = f"header rows {extension.header_row} to {extension.header_row + extension.header_rows - 1}"
    else:
        names = f"header row {extension.header_row}"
    end = f" to row {extension.end_row - 1}" if extension.end_row is not None else ""
    return f"{names}, data from row {extension.start_row}{end}"


def where_in(extension: OgdaExtension) -> str:
    """Where a Dataset's rows are in its File: its sheet, as a location line writes it after the path, and region."""
    return (f"#{extension.sheet}, " if extension.sheet else "") + region(extension)


def file_format(path: str) -> str:
    return PurePosixPath(path).suffix or "extensionless"


def unread(path: str, table: Table) -> str:
    """Why ground returns a table's File path: code does not read its format, or failed to read it."""
    if table.failure:
        return f"Code could not read {path} ({table.failure}): ground returns the path; read the File yourself."
    return f"Code does not read {file_format(path)} Files: ground returns the path; read the File yourself."


def named_fields(value) -> set[str]:
    """The Fields an identify object reads, at any depth."""
    if field := field_of(value):
        return {field}
    if isinstance(value, dict) and not is_literal(value):
        return set().union(*(named_fields(inner) for inner in value.values()))
    return set()


def identifier_part(item: dict) -> str:
    source = f"column {as_json(item['column'].strip())}" if "column" in item else f"path {item['field']}"
    return (f"{item['via']} by " if "via" in item else "") + f"{item['identifies']}, from {source}"


def join_line(other: str, entity: str, new: Key, old: Key) -> str:
    line = f"  {other} on {entity}: {new.side} ({new.identifier}) ↔ {old.side} ({old.identifier})"
    if new.normalized or old.normalized:
        line += NORMALIZE
    if new.identifier != old.identifier:
        line += "; different Identifiers"
    return line


class Grounding:
    """The names one Task has grounded, the join keys of its grounded Datasets and Fields, and the Experience shown."""

    def __init__(self, harness: Harness, lake_root: Path, cache: Path, index: OntologySearch,
                 max_output_chars: int = 20000):
        self.harness, self.index = harness, index
        # The executor keeps only the head and tail of a longer cell output, so a call of several names whose views
        # pass this would lose the middle, lessons included.
        self.most = max_output_chars * 3 // 4
        self.lake_root, self.cache = Path(lake_root).absolute(), Path(cache).absolute()
        self.grounded: dict[str, Grounded] = {}
        self.keys: dict[str, list[Key]] = {}  # Each grounded Dataset and Field, in grounding order.
        self.shown: set[str] = set()          # The Experience entries shown so far.
        self.aliases: dict[str, str] = {}     # A layout group member and its Fields: the representative's names.
        for dataset in harness.semantic_model.datasets:
            if representative := ogda_extension(dataset).layout_of:
                self.aliases[dataset.name] = representative
                self.aliases.update({f"{dataset.name}.{field.name}": f"{representative}.{field.name}"
                                     for field in dataset.fields})

    def ground(self, names: list[str]) -> list[Grounded]:
        """Ground each name in order. Every name is resolved and every Dataset read before any name is grounded, so a
        bad name raises ValueError and grounds nothing; a name grounded before prints one line. Several names whose
        views together pass the output limit raise ValueError too, and ground nothing."""
        if not names:
            raise ValueError("ground needs at least one name.")
        if (bad := next((name for name in names if not isinstance(name, str)), None)) is not None:
            raise ValueError(f'ground takes each name as a string, such as ground("x", "x.y"), not {bad!r}.')
        targets = {name: self.resolve(name) for name in names if name not in self.grounded}
        tables = {target.dataset: self.read(target.dataset) for target in targets.values() if target.dataset}
        before = dict(self.grounded), dict(self.keys), set(self.shown)
        found = []
        for name in names:
            if name in self.grounded:
                found.append(replace(self.grounded[name], view=f"{name}: grounded already"))
                continue
            target = targets[name]
            if target.kind == "dataset":
                node = self.dataset_node(name, tables[target.dataset])
            elif target.kind == "field":
                node = self.field_node(name, target, tables[target.dataset])
            elif target.kind == "file":
                node = self.file_node(name, target.paths[0])
            else:
                node = self.pattern_node(name, target.paths)
            self.grounded[name] = node
            found.append(node)
        if len(names) > 1 and (size := len("\n\n".join(node.view for node in found))) > self.most:
            self.grounded, self.keys, self.shown = before
            sizes = ", ".join(f"{node.name} {len(node.view):,}" for node in found)
            raise ValueError(f"These views take {size:,} characters together, more than one call prints "
                             f"({self.most:,}): {sizes}. Ground fewer names per call.")
        return found

    def read(self, dataset: str) -> Table:
        """A Dataset's cached table, or why code returns its File's path instead."""
        try:
            cached = cache_dataset(self.harness, dataset, self.lake_root, self.cache)
        except READ_ERRORS as error:  # The view says why, and the agent reads the File itself.
            return Table(failure=f"{type(error).__name__}: {error}")
        if cached is None:
            return Table()
        return Table(cached[0], cached[1], frozenset(read_as_text(self.harness, dataset, cached[0])))

    # Names.

    def resolve(self, name: str) -> Target:
        """A Dataset, then `<dataset>.<field>`, then a Lake File, then a pattern; a name that leaves the Lake is
        refused first."""
        if PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts:
            raise ValueError(self.unknown(name))
        root = self.lake_root.resolve()
        path = (root / name).resolve()
        if not path.is_relative_to(root):  # A link out of the Lake.
            raise ValueError(self.unknown(name))
        datasets = self.harness.datasets()
        if name in datasets:
            return Target("dataset", dataset=name)
        dataset, _, field = name.partition(".")
        if dataset in datasets and any(f.name == field for f in datasets[dataset].fields):
            return Target("field", dataset=dataset, field=field)
        relative = path.relative_to(root).as_posix()
        if path.is_file():
            return Target("file", paths=(relative,))
        if any(character in name for character in "*?["):
            matches = sorted(match.relative_to(root).as_posix() for match in root.glob(relative) if match.is_file())
            if not matches:
                raise ValueError(f"No File matches '{name}'. {close_names(name, list_files(self.lake_root))}")
            return Target("pattern", paths=tuple(matches))
        raise ValueError(self.unknown(name))

    def unknown(self, name: str) -> str:
        names = [*self.harness.datasets(), *self.harness.fields(), *list_files(self.lake_root)]
        return f"No table, column or File named '{name}'. {close_names(name, names)}"

    # Views.

    def dataset_node(self, name: str, table: Table) -> Grounded:
        structure = self.structure(name)
        result, columns = structure["result"], [column["entry"] for column in structure["columns"]]
        extension = ogda_extension(self.harness.dataset(name))
        lines = [f"{name}: {result['row']}" + (f", {table.rows:,} rows" if table.path else ""),
                 self.location(name, result)]
        if extension.relation != "part":
            lines.append(f"{extension.relation.capitalize()} of {extension.relation_to}")
        if column := file_column(self.harness, name):
            lines.append(f"Column {column} holds each row's File path.")
        if table.path is None:
            lines.append(unread(extension.path, table))
        if result["description"]:
            lines.append(result["description"])
        binding = self.harness.binding_of(name)
        for entry in binding.binding if binding else []:
            fields = named_fields(entry.identify)
            if parts := [item for item in result["identified_by"] if item["field"] in fields]:
                lines.append("Identifier: " + " + ".join(identifier_part(item) for item in parts))
        if result["documentation"]:
            lines.append(f"Documentation: {', '.join(result['documentation'])}")
        lines.append(f"Columns ({len(columns)}):")
        names = column_names(self.harness, name)
        full = [column_line(entry, table, names[entry["field"]]) for entry in columns]
        if sum(len(line) + 1 for line in full) <= COLUMNS:
            lines += full
        else:
            width = COLUMNS // len(columns) - 1  # Each line and its newline.
            lines += [column_line(entry, table, names[entry["field"]], width) for entry in columns]
            lines.append(f'Meanings are cut to fit; ground "{name}.<column>" for one column\'s full meaning.')
        keys = self.dataset_keys(structure)
        lines += self.joins(name, keys)
        representative, files = self.aliases.get(name, name), self.files(name)
        # Broad lessons first; past the cap, a column's view shows its own and those left here.
        lines += self.lessons([{representative}, set(files),
                               {f"{representative}.{entry['field']}" for entry in columns},
                               {link["fact"] for entry in columns for link in entry["links"]}])
        self.keys[name] = keys
        return Grounded(name, "dataset", "\n".join(lines), self.value(name, table), files, table.rows)

    def field_node(self, name: str, target: Target, table: Table) -> Grounded:
        structure = self.structure(target.dataset)
        entry = next(column["entry"] for column in structure["columns"] if column["entry"]["field"] == target.field)
        lines = ["  ".join([name, header(entry), datatype(entry, table)])]
        if table.path is None:
            lines.append(unread(ogda_extension(self.harness.dataset(target.dataset)).path, table))
        if entry["description"]:
            lines.append(entry["description"])
        lines += [fills(link) for link in entry["links"]]
        if found := markers(entry):
            lines.append("  ".join(found))
        keys = self.dataset_keys(structure)
        own = [k for k in keys if k.field == target.field]
        keys = own + [k for k in keys if k.entity == structure["rows"] and k not in own]
        lines += self.joins(target.dataset, keys)
        representative, files = self.aliases.get(target.dataset, target.dataset), self.files(target.dataset)
        # Narrow lessons first, then those of its table and Files not shown yet.
        lines += self.lessons([{f"{representative}.{target.field}"}, {link["fact"] for link in entry["links"]},
                               {representative}, set(files)])
        self.keys[name] = keys
        value = self.value(target.dataset, table)
        if table.path:
            value["__field__"] = column_names(self.harness, target.dataset)[target.field]
        return Grounded(name, "field", "\n".join(lines), value, files, table.rows)

    def file_node(self, name: str, path: str) -> Grounded:
        datasets = self.harness.semantic_model.datasets
        tables = [f"{d.name} ({where_in(ogda_extension(d))})"
                  for d in sorted(datasets, key=lambda d: d.name) if ogda_extension(d).path == path]
        documented = sorted(d.name for d in datasets if ogda_extension(d).layout_of is None
                            and any(split_documentation(ref)[0] == path for ref in ogda_extension(d).documentation))
        lines = [f"{path}: {file_format(path)} File, {size_text((self.lake_root / path).stat().st_size)}",
                 f"Tables in it: {'; '.join(tables)}" if tables else "No table covers this File."]
        value, rows = {"__path__": str(self.lake_root / path)}, None
        # A File of a layout group's stacked table returns its own rows of that table.
        part = next((d for d in datasets if ogda_extension(d).path == path and ogda_extension(d).relation == "part"
                     and (ogda_extension(d).layout_of or self.harness.members(d.name))), None)
        if part is not None:
            group = ogda_extension(part).layout_of or part.name
            table = self.read(group)
            if table.path:
                column = file_column(self.harness, group)
                rows = pq.read_table(table.path, columns=[column]).column(column).to_pylist().count(path)
                lines.append(f"Rows of {group} from this File: {rows:,} ({region(ogda_extension(part))})")
                value = {"__table__": str(table.path), "__file__": path, "__file_column__": column}
        if documented:
            lines.append(f"Documentation of: {', '.join(documented)}")
        lines += self.lessons([{path}])
        return Grounded(name, "file", "\n".join(lines), value, [path], rows)

    def pattern_node(self, name: str, paths: tuple[str, ...]) -> Grounded:
        total = sum((self.lake_root / path).stat().st_size for path in paths)
        view = f"{name}: {len(paths)} Files, {size_text(total)}, e.g. {', '.join(paths[:EXAMPLES])}"
        return Grounded(name, "pattern", view, {"__paths__": [str(self.lake_root / path) for path in paths]},
                        list(paths), None)

    def location(self, name: str, result: dict) -> str:
        """Where a Dataset's rows are: its File and sheet, with its layout group's File count and path pattern when
        it represents one, then its header and data rows."""
        extension = ogda_extension(self.harness.dataset(name))
        file = f"{extension.path}#{extension.sheet}" if extension.sheet else extension.path
        group = result.get("files") if name not in self.aliases else None
        if group and group["count"] > 1:
            where = f'{group["count"]} Files like "{file}"'
            if group["path_pattern"]:
                where += f" (path pattern {group['path_pattern']})"
        else:
            where = f"File {file}"
        return f"{where}, {region(extension)}"

    # Data.

    def structure(self, dataset: str) -> dict:
        """What the index built for a Dataset, or for its layout group's representative."""
        return self.index.dataset(self.aliases.get(dataset, dataset))

    def files(self, dataset: str) -> list[str]:
        """The Files whose rows a Dataset's value holds: its own, and those of its layout group's other parts."""
        datasets = self.harness.datasets()
        parts = [dataset] + [member for member in self.harness.members(dataset)
                             if ogda_extension(datasets[member]).relation == "part"]
        return sorted(ogda_extension(datasets[part]).path for part in parts)

    def value(self, dataset: str, table: Table) -> dict:
        """A Dataset's cached table, or its File's path when code does not read it."""
        if table.path is None:
            return {"__path__": str(self.lake_root / ogda_extension(self.harness.dataset(dataset)).path)}
        return {"__table__": str(table.path)}

    # Joins and Experience.

    def dataset_keys(self, structure: dict) -> list[Key]:
        """The columns that name Entities by an Identifier: those that identify the Dataset's Entities, those that
        refer to other Entities, and those that fill an Identifier of their fact's Entity type."""
        entries = {column["entry"]["field"]: column["entry"] for column in structure["columns"]}
        keys = [key(entries[item["field"]], item["identifies"]) for item in structure["result"]["identified_by"]]
        for entry in entries.values():
            if "refers_to" in entry:
                keys.append(key(entry, entry["refers_to"]))
            keys += [key(entry, link["fact"]) for link in entry["links"]
                     if [link["fact"].partition(".")[2]] in identifiers(self.harness.ontology, link["entity"])]
        return list(dict.fromkeys(keys))

    def joins(self, dataset: str, keys: list[Key]) -> list[str]:
        """Per shared Entity type and grounded Dataset or Field, the column pairs that join a new node of `dataset`
        to it: those with the same Identifier when there are any, else every pair. Nodes of the new node's own
        Dataset or layout group are left out, and so is a Field whose Dataset is grounded too."""
        if not self.keys:
            return ["Joins: none yet"]
        group, others = self.aliases.get(dataset, dataset), {}
        for other, theirs in self.keys.items():
            of, _, field = other.partition(".")
            if self.aliases.get(of, of) != group and not (field and of in self.keys):
                others[other] = theirs
        lines = []
        for entity in dict.fromkeys(k.entity for k in keys):
            mine = [k for k in keys if k.entity == entity]
            for other, theirs in others.items():
                pairs = [(new, old) for new in mine for old in theirs if old.entity == entity]
                same = [(new, old) for new, old in pairs if new.identifier == old.identifier]
                lines += [join_line(other, entity, new, old) for new, old in same or pairs]
        return ["Joins:", *lines] if lines else ["Joins: none with the grounded tables"]

    def lessons(self, tiers: list[set[str]]) -> list[str]:
        """The live lessons anchored to a node, tier by tier in the order given: at most LESSONS, each shown once
        per Grounding, and a count of the rest."""
        chosen: dict[str, str] = {}
        for tier in tiers:
            for entry in self.harness.live_experience():
                if (not entry.stale and entry.id not in self.shown and entry.id not in chosen
                        and tier & {self.aliases.get(anchor, anchor) for anchor in entry.anchors}):
                    chosen[entry.id] = entry.text
        if not chosen:
            return []
        shown = list(chosen)[:LESSONS]
        self.shown.update(shown)
        lines = ["Experience:", *("  - " + "\n    ".join(chosen[i].strip().splitlines()) for i in shown)]
        if len(chosen) > LESSONS:
            lines.append(f"  ... {len(chosen) - LESSONS} more lessons; ground a column to see its own.")
        return lines


def close_names(name: str, names: list[str]) -> str:
    close = difflib.get_close_matches(name, names, n=3)
    return f"Close names: {', '.join(close)}." if close else "No close names."


def lake_map(harness: Harness, lake_root: Path, index: OntologySearch) -> str:
    """The Lake map: every table with its File and what a row is, every Entity type with its Identifier and tables,
    and the Lake Files no table covers. A subset is a table that says what it is a subset of; a copy is no table,
    and its original's line counts it."""
    datasets = [ogda_extension(dataset) for dataset in harness.semantic_model.datasets]

    def representative(name: str) -> str:
        dataset = harness.dataset(name)
        return (ogda_extension(dataset).layout_of or name) if dataset is not None else name

    copies = Counter(representative(extension.relation_to) for extension in datasets
                     if extension.relation == "copy" and extension.relation_to)
    shown = [table for table in index.tables
             if ogda_extension(harness.dataset(table["result"]["table"])).relation != "copy"]
    tables = []
    for table in shown:
        found, extension = table["result"], ogda_extension(harness.dataset(table["result"]["table"]))
        place = found["file"] + (f"#{found['sheet']}" if found.get("sheet") else "")
        where = f'{found["files"]["count"]} Files like "{found["file"]}"' if found.get("files") else place
        if count := copies[found["table"]]:
            where += f" + {count} cop{'y' if count == 1 else 'ies'}"
        sentence = first_sentence(found["description"])
        line = f"- {found['table']}: {where}; {found['row']}." + (f" {sentence}." if sentence else "")
        if extension.relation == "subset":
            line = line.removesuffix(".") + f"; subset of {extension.relation_to}"
        tables.append(line)
    entities = []
    for concept in concepts(harness.ontology).values():
        if concept.type != "EntityType":
            continue
        options = [", ".join(parts) for parts in identifiers(harness.ontology, concept.concept)]
        labels = sorted(table["label"] for table in shown if concept.concept in table["names"])
        text = f"- {concept.concept}: identified by {options[0] if options else 'no Identifier'}"
        if len(options) > 1:
            text += "; also by " + " or ".join(options[1:])
        text += "; in " + (", ".join(labels[:LISTED]) if labels else "no table")
        if len(labels) > LISTED:
            text += f" and {len(labels) - LISTED} more"
        entities.append(text)
    covered = {extension.path for extension in datasets}
    cited = {split_documentation(ref)[0] for extension in datasets for ref in extension.documentation}
    groups: dict[str, list[str]] = {}
    for path in list_files(lake_root):
        if path not in covered and not any(part.startswith(".") for part in PurePosixPath(path).parts):
            groups.setdefault(re.sub(r"\d+", "*", path), []).append(path)
    uncovered = []
    for pattern, paths in groups.items():
        size = size_text(sum((Path(lake_root) / path).stat().st_size for path in paths))
        line = (f"- {paths[0]}: {size}" if len(paths) == 1
                else f"- {pattern}: {len(paths)} Files, {size}, e.g. {paths[0]}")
        uncovered.append(line + (", Documentation" if cited.intersection(paths) else ""))
    count = sum(len(paths) for paths in groups.values())
    return "\n".join(["LAKE MAP", f"TABLES ({len(tables)})", *tables, f"ENTITY TYPES ({len(entities)})", *entities,
                      f"FILES WITH NO TABLE ({count})", *uncovered])
