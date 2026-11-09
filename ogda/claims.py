"""Claim contracts: parse a proposal, check it statically, find missing references, apply it."""

from __future__ import annotations

import csv
import io
from pathlib import Path

import openpyxl

from .ontology import Claim, Concept, Link, Ontology, Schema, Table

LINK_ENDS = {
    "field_link": ("fields", "concepts"),
    "instance_of": ("fields", "concepts"),
    "sub_concept_of": ("concepts", "concepts"),
    "about": ("tables", "concepts"),
    "mentions": ("files", "concepts"),
}
ORDER = {"schema": 0, "tables": 1, "concept": 2, "link": 3, "remove": 4}
TABULAR = {".csv", ".tsv", ".txt", ".xlsx", ".xlsm"}
JOIN_OVERLAP = 0.1  # Least share of the smaller key's distinct values that the other key must also hold.
JOIN_SAMPLE_TABLES = 3
JOIN_SAMPLE_ROWS = 10_000


def lake_path(value: str, lake_root: Path) -> str:
    """Return the Lake-relative path of an existing File. Absolute paths must be in the Lake."""
    root = Path(lake_root).resolve()
    path = Path(value)
    resolved = (path if path.is_absolute() else root / path).resolve()
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError(f"No File at {value} in the Lake")
    return resolved.relative_to(root).as_posix()


def parse_claim(item: dict, lake_root: Path) -> tuple[str, dict]:
    """Validate one proposal and normalize its paths. Raise ValueError on a bad proposal."""
    if not isinstance(item, dict) or not {"kind", "payload"} <= set(item):
        raise ValueError("A Claim needs kind and payload")
    kind, payload = item["kind"], item["payload"]
    if kind not in ORDER or not isinstance(payload, dict):
        raise ValueError(f"Unknown Claim kind {kind!r}; use one of {sorted(ORDER)}")
    if kind == "schema":
        schema = Schema.model_validate(payload.get("schema"))
        schema.documentation = [lake_path(path, lake_root) for path in schema.documentation]
        if schema.origin == "declared" and not schema.documentation:
            raise ValueError("A declared Schema needs at least one existing Documentation File")
        return kind, {"schema": schema.model_dump(mode="json"), "tables": parse_tables(payload, schema.id, lake_root)}
    if kind == "tables":
        if not isinstance(payload.get("schema_id"), str) or not payload.get("tables"):
            raise ValueError("A tables Claim needs schema_id and a non-empty tables list")
        return kind, {"schema_id": payload["schema_id"], "tables": parse_tables(payload, payload["schema_id"], lake_root)}
    if kind == "concept":
        return kind, {"concept": Concept.model_validate(payload.get("concept")).model_dump(mode="json")}
    if kind == "link":
        link = Link.model_validate(payload.get("link"))
        if link.kind == "mentions":
            link.source = lake_path(link.source, lake_root)
        return kind, {"link": link.model_dump(mode="json")}
    if not isinstance(payload.get("id"), str):
        raise ValueError("A remove Claim needs the id of an Ontology object")
    return kind, {"id": payload["id"]}


def parse_tables(payload: dict, schema_id: str, lake_root: Path) -> list[dict]:
    tables = [Table.model_validate({"schema_id": schema_id, **table}) for table in payload.get("tables", [])]
    if len({table.id for table in tables}) != len(tables):
        raise ValueError("A Claim has duplicate Table ids")
    for table in tables:
        if table.schema_id != schema_id:
            raise ValueError(f"Table {table.id} must use Schema {schema_id}")
        table.path = lake_path(table.path, lake_root)
    return [table.model_dump(mode="json") for table in tables]


def read_rows(path: Path, sheet: str | None) -> list[list[str]]:
    if path.suffix.lower() in {".xlsx", ".xlsm"}:
        workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
        try:
            rows = workbook[sheet or workbook.sheetnames[0]].iter_rows(values_only=True)
            return [["" if cell is None else str(cell) for cell in row] for row in rows]
        finally:
            workbook.close()
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    try:
        dialect = csv.Sniffer().sniff(text[:65536], delimiters=",\t;|")
    except csv.Error:
        dialect = csv.excel_tab if path.suffix.lower() == ".tsv" else csv.excel
    return list(csv.reader(io.StringIO(text), dialect))


def static_check(kind: str, payload: dict, columns: list[str] | None, lake_root: Path) -> list[str] | None:
    """Check documented structure without reinterpreting its meaning.

    Declared Schemas require existing Documentation and use all supported Table checks. Unsupported
    layouts retain path/region checks from parsing and still need no semantic Verifier. Inferred Schemas
    and tables Claims get the same checks only when every Table has a supported header layout; otherwise
    None means LLM verification. An inferred Schema always goes to the Verifier as well.
    """
    declared = kind == "schema" and payload["schema"]["origin"] == "declared"
    if kind not in ("schema", "tables") or columns is None:
        return None
    tables = payload["tables"]
    if not declared and (not tables or any(Path(t["path"]).suffix.lower() not in TABULAR
                                          or t["header_row"] is None for t in tables)):
        return None
    problems = []
    if declared:
        documentation = payload["schema"]["documentation"]
        if not documentation:
            problems.append("A declared Schema needs at least one existing Documentation File")
        for path in documentation:
            try:
                lake_path(path, lake_root)
            except ValueError as error:
                problems.append(str(error))
    for table in tables:
        if Path(table["path"]).suffix.lower() not in TABULAR:
            continue  # Existing paths and region bounds are checked by parse_claim.
        try:
            rows = read_rows(Path(lake_root) / table["path"], table["sheet"])
        except Exception as error:  # A wrong sheet name or a corrupt File: the Proposer can correct the Claim.
            problems.append(f"Table {table['id']}: cannot read {table['path']}: {type(error).__name__}: {error}")
            continue
        header, start, end = table["header_row"], table["start_row"], table["end_row"]
        if (start > len(rows) or (header is not None and (header >= len(rows) or not header < start))
                or (end is not None and not start < end <= len(rows))):
            problems.append(f"Table {table['id']}: {table['path']} has {len(rows)} rows; header_row={header}, "
                            f"start_row={start}, end_row={end} do not fit")
            continue
        if header is None:
            continue
        found = {cell.strip() for cell in rows[header]}
        absent = [column for column in columns if column.strip() not in found]
        if absent:
            problems.append(f"Table {table['id']}: header row {header} of {table['path']} has no columns {absent}")
    return problems


def field_id_conflicts(parsed: list[tuple[str, dict]], ontology: Ontology) -> list[str]:
    """Errors for schema Claims whose Field ids belong to another accepted Schema or an earlier Claim of the batch."""
    owner = {field.id: schema.id for schema in ontology.schemas.values() for field in schema.fields}
    errors = []
    for index, (kind, payload) in enumerate(parsed):
        if kind != "schema":
            continue
        schema_id = payload["schema"]["id"]
        clashes = [f["id"] for f in payload["schema"]["fields"] if owner.setdefault(f["id"], schema_id) != schema_id]
        if clashes:
            errors.append(f"claims[{index}]: Field ids {clashes} already belong to another Schema; "
                          f"Field ids must be unique across Schemas, for example {schema_id}.<column>")
    return errors


def table_id_conflicts(parsed: list[tuple[str, dict]], ontology: Ontology) -> list[str]:
    """Errors for Tables whose ids already map another File, in the Ontology or an earlier Claim of the batch."""
    owner = {table.id: table.path for table in ontology.tables.values()}
    errors = []
    for index, (kind, payload) in enumerate(parsed):
        if kind not in ("schema", "tables"):
            continue
        clashes = [t["id"] for t in payload["tables"] if owner.setdefault(t["id"], t["path"]) != t["path"]]
        if clashes:
            errors.append(f"claims[{index}]: Table ids {clashes} already map another File; "
                          f"give each File's Table its own id")
    return errors


def normalized(value: str) -> str:
    value = value.strip().lower()
    try:
        number = float(value)
    except ValueError:
        return value
    return str(int(number)) if number.is_integer() else value


def key_values(field_id: str, ontology: Ontology, lake_root: Path) -> list[list[str]]:
    """Nonempty normalized values of a Field's column, one list per sampled Table of its Schema."""
    schema = next(s for s in ontology.schemas.values() if any(f.id == field_id for f in s.fields))
    column = next(f.column for f in schema.fields if f.id == field_id).strip()
    samples = []
    tables = sorted((t for t in ontology.tables.values() if t.schema_id == schema.id), key=lambda t: t.id)
    for table in tables[:JOIN_SAMPLE_TABLES]:
        if Path(table.path).suffix.lower() not in TABULAR or table.header_row is None:
            continue
        rows = read_rows(Path(lake_root) / table.path, table.sheet)
        header = [cell.strip() for cell in rows[table.header_row]] if table.header_row < len(rows) else []
        if column not in header:
            continue
        index = header.index(column)
        end = min(len(rows) if table.end_row is None else table.end_row, table.start_row + JOIN_SAMPLE_ROWS)
        samples.append([v for v in (normalized(row[index]) for row in rows[table.start_row:end] if index < len(row)) if v])
    return samples


def join_check(parsed: list[tuple[str, dict]], ontology: Ontology, lake_root: Path) -> tuple[list[str], dict[int, str]]:
    """Check that each join_key Field linked to a Concept shares values with that Concept's other join_key Fields.

    Return errors for links whose keys share too few values, and notes with the measured overlap and
    cardinality by Claim index. Keys without a readable tabular Table are not checked.
    """
    fields = ontology.fields()
    members: dict[str, dict[str, str]] = {}
    for link in ontology.links.values():
        if link.kind == "field_link" and link.source in fields and fields[link.source].join_key:
            members.setdefault(link.target, {})[link.id] = link.source
    for kind, payload in parsed:
        if kind == "remove":
            for keys in members.values():
                keys.pop(payload["id"], None)
    cache: dict[str, list[list[str]]] = {}

    def values(field_id: str) -> list[list[str]]:
        if field_id not in cache:
            cache[field_id] = key_values(field_id, ontology, lake_root)
        return cache[field_id]

    errors, notes = [], {}
    for index, (kind, payload) in enumerate(parsed):
        if kind != "link":
            continue
        link = payload["link"]
        for keys in members.values():
            keys.pop(link["id"], None)  # A revised link leaves its old Concept.
        field = fields.get(link["source"])
        if link["kind"] != "field_link" or field is None or not field.join_key:
            continue
        measured = []
        for partner in sorted(set(members.get(link["target"], {}).values()) - {field.id}):
            mine, theirs = values(field.id), values(partner)
            ours, others = {v for t in mine for v in t}, {v for t in theirs for v in t}
            if not ours or not others:
                continue
            share = len(ours & others) / min(len(ours), len(others))
            if share < JOIN_OVERLAP:
                errors.append(f"claims[{index}]: join_key Fields {field.id} and {partner} both link to {link['target']} "
                              f"but share {share:.0%} of their values (samples {sorted(ours)[:3]} and "
                              f"{sorted(others)[:3]}), so they cannot be joined. Link {field.id} to another Concept, "
                              f"relating the two Concepts with sub_concept_of if their meanings are shared, or drop the link")
                continue
            unique = ["1" if all(len(t) == len(set(t)) for t in side) else "n" for side in (mine, theirs)]
            measured.append(f"{field.id} and {partner} share {share:.0%} of values, {unique[0]}:{unique[1]}")
        members.setdefault(link["target"], {})[link["id"]] = field.id
        if measured:
            notes[index] = "Join check on sampled Tables: " + "; ".join(measured)
    return errors, notes


def provided_ids(claims: list[Claim]) -> dict[str, set[str]]:
    """The ids that a batch of Claims would add, by collection."""
    ids: dict[str, set[str]] = {"schemas": set(), "fields": set(), "tables": set(), "concepts": set()}
    for claim in claims:
        if claim.kind == "schema":
            ids["schemas"].add(claim.payload["schema"]["id"])
            ids["fields"].update(field["id"] for field in claim.payload["schema"]["fields"])
        if claim.kind in ("schema", "tables"):
            ids["tables"].update(table["id"] for table in claim.payload["tables"])
        if claim.kind == "concept":
            ids["concepts"].add(claim.payload["concept"]["id"])
    return ids


def missing_references(claim: Claim, ontology: Ontology, batch: list[Claim] = ()) -> list[str]:
    """Return the ids that a Claim needs but that neither the Ontology nor the other batch Claims have."""
    if claim.kind == "remove":
        return [] if claim.payload["id"] in ontology.object_ids() else [claim.payload["id"]]
    extra = provided_ids([other for other in batch if other.id != claim.id])
    known = {"schemas": set(ontology.schemas) | extra["schemas"], "fields": set(ontology.fields()) | extra["fields"],
             "tables": set(ontology.tables) | extra["tables"], "concepts": set(ontology.concepts) | extra["concepts"]}
    if claim.kind == "tables":
        return [] if claim.payload["schema_id"] in known["schemas"] else [claim.payload["schema_id"]]
    if claim.kind != "link":
        return []
    link = claim.payload["link"]
    missing = []
    for end, collection in zip(("source", "target"), LINK_ENDS[link["kind"]]):
        if collection != "files" and link[end] not in known[collection]:
            missing.append(link[end])
    return missing


def is_noop(kind: str, payload: dict, ontology: Ontology) -> bool:
    """True when a proposal restates objects that the Ontology already has unchanged."""
    if kind == "schema":
        schema = ontology.schemas.get(payload["schema"]["id"])
        return (schema is not None and schema.model_dump(mode="json") == payload["schema"]
                and all(t["id"] in ontology.tables and ontology.tables[t["id"]].model_dump(mode="json") == t
                        for t in payload["tables"]))
    if kind == "tables":
        return all(t["id"] in ontology.tables and ontology.tables[t["id"]].model_dump(mode="json") == t
                   for t in payload["tables"])
    if kind in ("concept", "link"):
        current = (ontology.concepts if kind == "concept" else ontology.links).get(payload[kind]["id"])
        return current is not None and current.model_dump(mode="json") == payload[kind]
    return False


def apply_claim(claim: Claim, ontology: Ontology) -> set[str]:
    """Install a Verified Claim. Return the ids of objects that existed and changed."""
    changed: set[str] = set()
    if claim.kind == "schema":
        schema = Schema.model_validate(claim.payload["schema"])
        previous = ontology.schemas.get(schema.id)
        if previous is not None:
            changed.add(schema.id)
            changed.update(field.id for field in previous.fields)
        ontology.schemas[schema.id] = schema
    if claim.kind in ("schema", "tables"):
        for data in claim.payload["tables"]:
            table = Table.model_validate(data)
            if table.id in ontology.tables:
                changed.add(table.id)
            ontology.tables[table.id] = table
    if claim.kind == "concept":
        concept = Concept.model_validate(claim.payload["concept"])
        if concept.id in ontology.concepts:
            changed.add(concept.id)
        ontology.concepts[concept.id] = concept
    elif claim.kind == "link":
        link = Link.model_validate(claim.payload["link"])
        if link.id in ontology.links:
            changed.add(link.id)
        ontology.links[link.id] = link
    elif claim.kind == "remove":
        changed.update(remove(claim.payload["id"], ontology))
    changed.update(drop_dangling_links(ontology))
    for entry in ontology.live_experience():
        if changed & set(entry.anchors):
            entry.stale = True
    return changed


def remove(object_id: str, ontology: Ontology) -> set[str]:
    removed = {object_id}
    if object_id in ontology.schemas:
        schema = ontology.schemas.pop(object_id)
        removed.update(field.id for field in schema.fields)
        for table_id in [t.id for t in ontology.tables.values() if t.schema_id == object_id]:
            removed.add(ontology.tables.pop(table_id).id)
    for collection in (ontology.tables, ontology.concepts, ontology.links):
        collection.pop(object_id, None)
    for schema in ontology.schemas.values():
        schema.fields = [field for field in schema.fields if field.id != object_id]
    return removed


def drop_dangling_links(ontology: Ontology) -> set[str]:
    known = {"fields": ontology.fields(), "concepts": ontology.concepts, "tables": ontology.tables}
    dangling = {
        link.id for link in ontology.links.values()
        if any(collection != "files" and end not in known[collection]
               for end, collection in zip((link.source, link.target), LINK_ENDS[link.kind]))
    }
    for link_id in dangling:
        del ontology.links[link_id]
    return dangling


def anchors(claim: Claim) -> list[str]:
    """The Ontology ids and File paths that a Claim is about."""
    payload = claim.payload
    if claim.kind == "schema":
        schema = payload["schema"]
        ids = [schema["id"], *(f["id"] for f in schema["fields"]), *schema["documentation"]]
        ids += [t["id"] for t in payload["tables"]] + [t["path"] for t in payload["tables"]]
    elif claim.kind == "tables":
        ids = [payload["schema_id"]] + [t["id"] for t in payload["tables"]] + [t["path"] for t in payload["tables"]]
    elif claim.kind == "concept":
        ids = [payload["concept"]["id"]]
    elif claim.kind == "link":
        ids = [payload["link"]["id"], payload["link"]["source"], payload["link"]["target"]]
    else:
        ids = [payload["id"]]
    return list(dict.fromkeys(ids))
