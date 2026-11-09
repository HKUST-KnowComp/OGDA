"""Bindings: how a Dataset's rows fill the Ontology, the rules they follow, and the Ossie mapping compiled from them.

A Binding says which Entity type each row is (`rows`, or None when the rows are facts) and, per entry, which Concept
plays the first Role of the Relationships the entry fills, how a row identifies it (`identify`), and which values fill
each Relationship's other Roles. A value is a Field name, a literal {"value": "2018"}, a normalized Field
{"normalized": "state"} (its letters and digits, lowercased), or, for a Role played by an Entity type, an identify
object with one value per Relationship of one of its Identifiers.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from .lake import TABULAR, RowReader, header_names
from .ossie import (NUMERIC, ConceptMapping, Dataset, DatasetField, DocumentReference, LinkMapping, MappingDocument,
                    ObjectMapping, OgdaExtension, OntologyDocument, ReferentMapping, SemanticModel, base_type, concepts,
                    find_relationship, identifiers, ogda_extension, owner_of)


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BindingEntry(Model):
    concept: str
    identify: dict[str, Any]
    relationships: dict[str, list] = {}


class Binding(Model):
    rows: str | None = None
    binding: list[BindingEntry]
    unbound: dict[str, str] = {}


def is_literal(value) -> bool:
    """Whether a value at a Role played by a Value type is a literal. At a Role played by an Entity type, every object
    is an identify object, even one whose single identifying Relationship is named "value"."""
    return isinstance(value, dict) and set(value) == {"value"} and isinstance(value["value"], (str, int, float))


NORMALIZED_SQL = "lower(regexp_replace({}, '[^A-Za-z0-9]', '', 'g'))"


def is_normalized(value) -> bool:
    """Whether a value at a Role played by a Value type is a normalized Field {"normalized": "<field>"}: the Field's
    letters and digits, lowercased, so that "New Hampshire" and "NewHampshire" give the same value."""
    return isinstance(value, dict) and set(value) == {"normalized"} and isinstance(value["normalized"], str)


def normalize(text: str) -> str:
    """Letters and digits only, lowercased: what NORMALIZED_SQL computes."""
    return re.sub(r"[^A-Za-z0-9]", "", text).lower()


def field_of(value) -> str | None:
    """The Field a value reads: a Field name, or the Field of a normalized value."""
    return value if isinstance(value, str) else value["normalized"] if is_normalized(value) else None


def is_entity(document: OntologyDocument, concept: str | None) -> bool:
    return getattr(concepts(document).get(concept), "type", None) == "EntityType"


def fillings(entry: BindingEntry, name: str) -> list[list]:
    """The fillings of one Relationship: one filling, or a list of them when one row fills it several times."""
    values = entry.relationships[name]
    return values if values and isinstance(values[0], list) else [values]


def fits(datatype: str | None, base: str) -> bool:
    """Whether a Field of this datatype can fill a Role whose Value type extends `base`."""
    if base == "String":
        return True
    if base in NUMERIC:
        return datatype in NUMERIC
    return datatype == base


def quote_identifier(text: str) -> str:
    return '"' + text.replace('"', '""') + '"'


def quote_literal(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


def field_source(field: DatasetField) -> tuple[Literal["column", "literal"], str]:
    """Where a Field's values come from: a column with this header text, or one literal value for the Dataset."""
    sql = field.expression.sql
    if len(sql) >= 2 and sql[0] == sql[-1] == '"':
        return "column", sql[1:-1].replace('""', '"')
    if len(sql) >= 2 and sql[0] == sql[-1] == "'":
        return "literal", sql[1:-1].replace("''", "'")
    raise ValueError(f"Field {field.name}: {sql!r} is neither a quoted header nor a quoted literal")


def identify_relationship(document: OntologyDocument, concept: str, name: str):
    """The identifying Relationship `name` of `concept`, declared by the Concept or a supertype."""
    owner = owner_of(document, concept, name)
    return find_relationship(document, f"{owner}.{name}") if owner else None


def walk(binding: Binding, document: OntologyDocument):
    """Yield (value, role concept, where) for every value of the Binding, identify objects included, depth first."""
    def values(value, concept, where):
        yield value, concept, where
        if isinstance(value, dict) and is_entity(document, concept):
            for key, inner in value.items():
                relationship = identify_relationship(document, concept, key)
                if relationship is not None and len(relationship.roles) == 1:
                    yield from values(inner, relationship.roles[0].concept, f"{where}.{key}")

    table = concepts(document)
    for index, entry in enumerate(binding.binding):
        yield from values(entry.identify, entry.concept, f"binding[{index}].identify")
        concept = table.get(entry.concept)
        for name in entry.relationships:
            relationship = next((r for r in concept.relationships if r.name == name), None) if concept else None
            if relationship is None:
                continue
            for filling in fillings(entry, name):
                for role, value in zip(relationship.roles, filling if isinstance(filling, list) else []):
                    yield from values(value, role.concept, f"binding[{index}].relationships.{name}")


def binding_problems(binding: Binding, dataset: Dataset, document: OntologyDocument) -> list[str]:
    """Every break of the Binding rules for this Dataset, one message each."""
    table, fields = concepts(document), {f.name: f for f in dataset.fields}
    problems, used = [], set()

    def value_problems(value, role_concept: str, where: str) -> list[str]:
        if isinstance(value, str) and value in fields:
            used.add(value)  # A Field in the wrong place is reported once, not also as unbound.
        concept = table.get(role_concept)
        if concept is None:
            return [f"{where}: no Concept {role_concept}"]
        if concept.type == "ValueType":
            if is_literal(value):
                return []
            if is_normalized(value):
                name, base = value["normalized"], base_type(document, role_concept)
                if name not in fields:
                    return [f"{where}: no Field {name} in Dataset {dataset.name}"]
                used.add(name)
                if base and base != "String":
                    return [f"{where}: {role_concept} extends {base}; a normalized value is text, so it fills only "
                            "a Value type that extends String"]
                return []
            if not isinstance(value, str):
                return [f"{where}: {role_concept} is a Value type; give a Field, a literal {{'value': ...}} or a "
                        "normalized Field {'normalized': ...}"]
            if value not in fields:
                return [f"{where}: no Field {value} in Dataset {dataset.name}"]
            used.add(value)
            base = base_type(document, role_concept)
            if base and not fits(fields[value].datatype, base):
                return [f"{where}: Field {value} ({fields[value].datatype}) does not fit {role_concept}, which "
                        f"extends {base}"]
            return []
        if not isinstance(value, dict):
            return [f"{where}: {role_concept} is an Entity type; give an identify object with one value per "
                    f"Relationship of an Identifier of {role_concept}, such as {{'name': '<field>'}}"]
        return identify_problems(value, role_concept, where)

    def identify_problems(identify: dict, concept: str, where: str) -> list[str]:
        options = identifiers(document, concept)
        if sorted(identify) not in [sorted(option) for option in options]:
            return [f"{where}: {sorted(identify)} is not an Identifier of {concept}; its Identifiers are {options}"]
        found = []
        for key, value in identify.items():
            relationship = identify_relationship(document, concept, key)
            if relationship is None or len(relationship.roles) != 1:
                found.append(f"{where}.{key}: {concept} has no binary identifying Relationship {key}")
                continue
            found += value_problems(value, relationship.roles[0].concept, f"{where}.{key}")
        return found

    if not binding.binding:
        problems.append("A Binding needs at least one entry")
    for index, entry in enumerate(binding.binding):
        where = f"binding[{index}]"
        concept = table.get(entry.concept)
        if concept is None:
            problems.append(f"{where}: no Concept {entry.concept}")
            continue
        if concept.type != "EntityType":
            problems.append(f"{where}: {entry.concept} is a Value type; an entry names an Entity type")
            continue
        problems += identify_problems(entry.identify, entry.concept, f"{where}.identify")
        if entry.concept != binding.rows and not entry.relationships:
            problems.append(f"{where}: the entry for {entry.concept} fills no Relationship. To relate it to another "
                            f"Entity of the row, declare a Relationship whose Role is {entry.concept} under that "
                            f"Entity's Entity type and fill it with {entry.identify!r} in that Entity's entry; or "
                            f"fill one of {entry.concept}'s own Relationships here. Drop the entry only when the row "
                            f"is not about a {entry.concept}")
        for name in entry.relationships:
            relationship = next((r for r in concept.relationships if r.name == name), None)
            if relationship is None:
                problems.append(f"{where}: {name} is not declared under {entry.concept}")
                continue
            several = len(fillings(entry, name)) > 1
            for number, filling in enumerate(fillings(entry, name)):
                spot = f"{where}.relationships.{name}" + (f"[{number}]" if several else "")
                roles = [role.concept for role in relationship.roles]
                if not isinstance(filling, list) or len(filling) != len(roles):
                    problems.append(f"{spot}: {entry.concept}.{name} needs one value per Role after the first "
                                    f"({roles}), not {filling!r}")
                    continue
                for role, value in zip(roles, filling):
                    problems += value_problems(value, role, spot)
    if binding.rows is not None:
        entries = [entry for entry in binding.binding if entry.concept == binding.rows]
        if not entries:
            problems.append(f"rows names {binding.rows}, but no binding entry is for {binding.rows}")
        elif all(is_literal(value) for value, concept, where in walk(Binding(binding=entries[:1]), document)
                 if ".identify" in where and not is_entity(document, concept)):
            problems.append(f"rows: each row is one {binding.rows}, so its identifier needs at least one part from a "
                            "Field; literals alone name the same Entity on every row")
    for name, reason in binding.unbound.items():
        if name not in fields:
            problems.append(f"unbound names no Field {name}")
        elif not reason.strip():
            problems.append(f"unbound: Field {name} needs a reason")
    if missing := [name for name in fields if name not in used and name not in binding.unbound]:
        problems.append(f"Fields {missing} are neither bound nor in unbound with a reason")
    return problems


def primary_key(binding: Binding) -> list[str]:
    """The Fields that identify a row when the rows are Entities, flattened depth first; [] for fact rows."""
    def flat(identify: dict) -> list[str]:
        found = []
        for value in identify.values():
            if isinstance(value, str):
                found.append(value)
            elif isinstance(value, dict) and not is_literal(value):  # A nested identify object.
                found += flat(value)
        return found

    entry = next((e for e in binding.binding if e.concept == binding.rows), None) if binding.rows else None
    return flat(entry.identify) if entry else []


def named_concepts(binding: Binding, document: OntologyDocument) -> set[str]:
    """The Concepts a Binding names: entry Concepts, the Concepts of every Role it fills, identify objects included,
    and the supertypes that declare an identifying Relationship it uses."""
    named = {entry.concept for entry in binding.binding}
    for value, concept, _ in walk(binding, document):
        named.add(concept)
        if isinstance(value, dict) and is_entity(document, concept):
            named |= {owner for key in value if (owner := owner_of(document, concept, key))}
    return named



# Structure and data checks.

EXAMPLES = 5  # Row numbers quoted in one data-check message.


def header_text(rows: list[list[str]], header_row: int | None, header_rows: int) -> list[str]:
    """The header text of each column. Repeats get pandas' suffixes (Value, Value.1); without a header row, each
    column is named by its position."""
    if header_row is None:
        return [str(position) for position in range(max((len(row) for row in rows), default=0))]
    names, seen, counts = [], set(), {}
    for name in header_names(rows, header_row, header_rows):
        unique = name
        while unique in seen:
            counts[name] = counts.get(name, 0) + 1
            unique = f"{name}.{counts[name]}"
        seen.add(unique)
        names.append(unique)
    return names


def extent(region: OgdaExtension, limit: int | None) -> int | None:
    """How many rows of its File the checks of a region read: through its header and its end, or `limit` rows past
    its start when it runs to the end of the File; None reads them all."""
    if region.end_row is None and limit is None:
        return None
    header = region.header_row + region.header_rows if region.header_row is not None else 0
    return max(header, region.start_row, region.end_row if region.end_row is not None else region.start_row + limit)


def structure_problems(dataset: Dataset, lake_root: Path, limit: int | None = None,
                       reader: RowReader | None = None) -> list[str]:
    """Problems with where a Dataset's rows are: its File can be read, its region fits and its header has the
    columns its Fields name. Only CSV, TSV, TXT and Excel Files are checked, as far as `extent` reads them; a
    `reader` shared with the data checks reads each File once."""
    region = ogda_extension(dataset)
    if Path(region.path).suffix.lower() not in TABULAR:
        return []
    try:
        rows = (reader or RowReader())(Path(lake_root) / region.path, region.sheet, extent(region, limit))
    except Exception as error:  # A wrong sheet name or a corrupt File: the Proposer can correct the Claim.
        return [f"Dataset {dataset.name}: cannot read {region.path}: {type(error).__name__}: {error}"]
    header, start, end, count = region.header_row, region.start_row, region.end_row, region.header_rows
    if (start > len(rows) or (header is not None and header + count > len(rows))
            or (end is not None and not start < end <= len(rows))):
        return [f"Dataset {dataset.name}: {region.path} has {len(rows)} rows; header_row={header}, "
                f"header_rows={count}, start_row={start}, end_row={end} do not fit"]
    found = set(header_text(rows, header, count))
    columns = [text for kind, text in map(field_source, dataset.fields) if kind == "column"]
    if absent := [column for column in columns if column.strip() not in found]:
        where = ("its columns" if header is None else f"header row {header}" if count == 1
                 else f"header rows {header}-{header + count - 1}")
        return [f"Dataset {dataset.name}: {where} of {region.path} has no columns {absent}"]
    return []


def region_rows(dataset: Dataset, lake_root: Path, limit: int,
                reader: RowReader | None = None) -> tuple[list[str], list[tuple[int, list[str]]]] | None:
    """The header text and the data rows among the first `limit` rows of a Dataset's region, each with its File row
    number and its values stripped; None for Files code does not read."""
    region = ogda_extension(dataset)
    if Path(region.path).suffix.lower() not in TABULAR:
        return None
    rows = (reader or RowReader())(Path(lake_root) / region.path, region.sheet, extent(region, limit))
    header = header_text(rows, region.header_row, region.header_rows)
    end = len(rows) if region.end_row is None else region.end_row
    data = [(number, [cell.strip() for cell in rows[number]] + [""] * (len(header) - len(rows[number])))
            for number in range(region.start_row, min(end, len(rows)))]
    return header, [(number, values) for number, values in data if any(values)][:limit]  # Blank rows hold no data.


def rows_of(numbers) -> str:
    return ", ".join(map(str, sorted(numbers)[:EXAMPLES]))


def data_problems(dataset: Dataset, binding: Binding, document: OntologyDocument, lake_root: Path,
                  limit: int, reader: RowReader | None = None) -> list[str]:
    """Whether the first `limit` rows obey the Binding's constraints: unique, non-empty identifiers when the rows are
    Entities, and every ManyToOne and OneToOne filling. A Role's value is the tuple of the Field values and literals
    that give it; an empty cell is null, and a filling with a null value is skipped."""
    region = region_rows(dataset, lake_root, limit, reader)
    if region is None:
        return []
    header, rows = region
    sources = {}
    for field in dataset.fields:
        kind, text = field_source(field)
        sources[field.name] = (kind, header.index(text.strip()) if kind == "column" and text.strip() in header else text)

    def value(row: list[str], item, concept: str) -> tuple | None:
        """The values that give `item` at a Role played by `concept`, or None when one is empty."""
        if isinstance(item, str):
            kind, where = sources.get(item, ("literal", None))
            cell = (row[where] if isinstance(where, int) else None) if kind == "column" else where
            return (cell,) if cell else None
        if is_normalized(item):
            cells = value(row, item["normalized"], concept)
            return (normalize(cells[0]),) if cells and normalize(cells[0]) else None
        if not is_entity(document, concept):
            return (str(item["value"]),)
        parts = [value(row, inner, identify_relationship(document, concept, key).roles[0].concept)
                 for key, inner in item.items()]
        return None if any(part is None for part in parts) else tuple(cell for part in parts for cell in part)

    problems = []
    table = concepts(document)
    if binding.rows is not None and (entry := next((e for e in binding.binding if e.concept == binding.rows), None)):
        distinct = {tuple(row): number for number, row in reversed(rows)}  # Exact duplicate rows count once.
        owners, empty = {}, []
        for row, number in sorted(distinct.items(), key=lambda item: item[1]):
            key = value(list(row), entry.identify, entry.concept)
            if key is None:
                empty.append(number)
            else:
                owners.setdefault(key, []).append(number)
        fields = ", ".join(primary_key(binding))
        if empty:
            problems.append(f"{dataset.name}: {len(empty)} rows have an empty identifier ({fields}), "
                            f"e.g. rows {rows_of(empty)}")
        if shared := [numbers for numbers in owners.values() if len(numbers) > 1]:
            problems.append(f"{dataset.name}: {sum(map(len, shared))} rows share the identifier ({fields}) with "
                            f"another row, e.g. rows {rows_of(n for numbers in shared for n in numbers)}")
    for entry in binding.binding:
        if problems and entry.concept == binding.rows:
            continue  # A repeated or empty identifier explains these rows' other conflicts; report it alone.
        concept = table.get(entry.concept)
        for name in entry.relationships:
            relationship = next((r for r in concept.relationships if r.name == name), None) if concept else None
            if relationship is None or relationship.multiplicity is None:
                continue
            directions = [(slice(None, -1), -1)] if relationship.multiplicity == "ManyToOne" else [(0, 1), (1, 0)]
            for filling in fillings(entry, name):
                tuples = []
                for number, row in rows:
                    values = [value(row, entry.identify, entry.concept)] + [
                        value(row, item, role.concept) for item, role in zip(filling, relationship.roles)]
                    if all(v is not None for v in values):
                        tuples.append((number, values))
                for source, target in directions:
                    seen, broken = {}, set()
                    for number, values in tuples:
                        key = tuple(values[source]) if isinstance(source, slice) else values[source]
                        partner = values[target]
                        first = seen.setdefault(key, (partner, number))
                        if first[0] != partner:
                            broken |= {first[1], number}
                    if broken:
                        problems.append(f"{dataset.name}: {entry.concept}.{name} is {relationship.multiplicity}, but "
                                        f"{len(broken)} rows give one value of its other Roles different partners, "
                                        f"e.g. rows {rows_of(broken)}")
    return list(dict.fromkeys(problems))


# The Ossie mapping.

def document_names(lake_id: str) -> tuple[str, str, str]:
    """The names of a Lake's ontology, semantic model and mapping documents."""
    return f"{lake_id}_ontology", f"{lake_id}_semantic_model", f"{lake_id}_mapping"


def literal(value: dict, base: str | None) -> str:
    """A literal as SQL: a number when its Value type is numeric and the text is a number, a quoted string otherwise."""
    text = str(value["value"])
    if base in NUMERIC:
        try:
            float(text)
            return text
        except ValueError:
            pass
    return quote_literal(text)


def expression(document: OntologyDocument, dataset: str, value, role_concept: str) -> str:
    if is_literal(value):
        return literal(value, base_type(document, role_concept))
    if is_normalized(value):
        return NORMALIZED_SQL.format(f"{dataset}.{value['normalized']}")
    return f"{dataset}.{value}"


def referents(document: OntologyDocument, dataset: str, concept: str, identify: dict) -> list[ReferentMapping]:
    """Referent mappings for an identify object: qualified identifying Relationships, nested for Entity types."""
    found = []
    for key, value in identify.items():
        relationship = identify_relationship(document, concept, key)
        role = relationship.roles[0].concept
        name = f"{owner_of(document, concept, key)}.{key}"
        if concepts(document)[role].type == "ValueType":
            found.append(ReferentMapping(relationship=name, expression=expression(document, dataset, value, role)))
        else:
            found.append(ReferentMapping(relationship=name, referent_mappings=referents(document, dataset, role, value)))
    return found


def object_mapping(document: OntologyDocument, dataset: str, value, role_concept: str) -> ObjectMapping:
    if concepts(document)[role_concept].type == "ValueType":
        return ObjectMapping(concept=role_concept, expression=expression(document, dataset, value, role_concept))
    return ObjectMapping(concept=role_concept, referent_mappings=referents(document, dataset, role_concept, value))


class Node:
    """One level of a LinkMapping tree; fillings with equal leading values share nodes."""

    def __init__(self, key, mapping: ObjectMapping):
        self.key, self.mapping, self.relationship, self.children = key, mapping, None, []

    def place(self, steps: list[tuple], relationship: str) -> None:
        (key, mapping), last = steps[0], len(steps) == 1
        child = next((c for c in self.children if c.key == key and not (last and c.relationship)), None)
        if child is None:
            child = Node(key, mapping)
            self.children.append(child)
        if last:
            child.relationship = relationship
        else:
            child.place(steps[1:], relationship)

    def link(self) -> LinkMapping:
        return LinkMapping(object_mapping=self.mapping, relationship=self.relationship,
                           children=[child.link() for child in self.children])


def compile_mapping(document: OntologyDocument, model: SemanticModel, bindings: dict[str, Binding],
                    lake_id: str) -> MappingDocument:
    """The Ossie mapping of every Dataset's Binding: one ConceptMapping per Concept, in Ontology order."""
    ontology_name, model_name, mapping_name = document_names(lake_id)
    table, slots = concepts(document), {}
    for dataset in sorted(model.datasets, key=lambda d: d.name):
        binding = bindings.get(dataset.name)
        for entry in binding.binding if binding else []:
            own = ObjectMapping(referent_mappings=referents(document, dataset.name, entry.concept, entry.identify))
            objects, links = slots.setdefault(entry.concept, ([], []))
            if binding.rows == entry.concept:
                objects.append(own)
            root = Node(None, own)
            for name in entry.relationships:
                relationship = next(r for r in table[entry.concept].relationships if r.name == name)
                for filling in fillings(entry, name):
                    root.place([((role.concept, json.dumps(value, sort_keys=True)),
                                 object_mapping(document, dataset.name, value, role.concept))
                                for role, value in zip(relationship.roles, filling)], name)
            if root.children:
                links.append(root.link())
    return MappingDocument(
        name=mapping_name, ontology_ref=DocumentReference(name=ontology_name, iri="./ontology.yaml"),
        semantic_model_ref=DocumentReference(name=model_name, iri="./semantic_model.yaml"),
        concept_mappings=[ConceptMapping(concept=name, object_mappings=slots[name][0], link_mappings=slots[name][1])
                          for name in table if name in slots])


# Rewrites for the concept layer.

def rewrite(binding: Binding, document: OntologyDocument, entry_concept, identify_key, filling) -> Binding:
    """A copy of a Binding with entry Concepts renamed by `entry_concept(name)`, identify keys by
    `identify_key(concept, key)` (concept is the Role's Concept in `document`), and each filling of an entry's
    Relationship replaced by `filling(entry, name, values) -> (new name, new values)`."""
    table = concepts(document)

    def value(item, concept):
        if not isinstance(item, dict) or not is_entity(document, concept):
            return item
        renamed = {}
        for key, inner in item.items():
            relationship = identify_relationship(document, concept, key)
            role = relationship.roles[0].concept if relationship is not None and relationship.roles else None
            renamed[identify_key(concept, key)] = value(inner, role)
        return renamed

    entries = []
    for entry in binding.binding:
        declared = {r.name: r for r in table[entry.concept].relationships} if entry.concept in table else {}
        relationships = {}
        for name, values in entry.relationships.items():
            roles = [role.concept for role in declared[name].roles] if name in declared else []
            several = bool(values) and isinstance(values[0], list)
            rewritten = [[value(item, role) for item, role in zip(one, roles)] + one[len(roles):]
                         for one in fillings(entry, name)]
            new_name, rewritten = filling(entry, name, rewritten)
            relationships[new_name] = rewritten if several else rewritten[0]
        entries.append(BindingEntry(concept=entry_concept(entry.concept), identify=value(entry.identify, entry.concept),
                                    relationships=relationships))
    rows = entry_concept(binding.rows) if binding.rows else None
    return Binding(rows=rows, binding=entries, unbound=dict(binding.unbound))


def merge_binding(binding: Binding, document: OntologyDocument, source: str, target: str,
                  relationships: dict[str, str]) -> Binding:
    """The Binding after `source` is merged into `target`: its entries become `target` entries, and its Relationship
    names, in entries and identify objects, are renamed through `relationships`. `document` is the Ontology before
    the merge."""
    def entry_concept(name):
        return target if name == source else name

    def identify_key(concept, key):
        return relationships.get(key, key) if concept == source else key

    def filling(entry, name, values):
        return (relationships.get(name, name) if entry.concept == source else name), values

    return rewrite(binding, document, entry_concept, identify_key, filling)


def replace_relationship(binding: Binding, before: OntologyDocument, after: OntologyDocument, concept: str, old: str,
                         new: str) -> Binding:
    """The Binding after `concept.old` is replaced by `concept.new`. Fillings move to the new name, identify objects
    that used `old` as an identifying Relationship of `concept` follow it, and where a Role changes from a Value type V
    to an Entity type E with a one-item Identifier whose Role is V, each Field or literal value becomes an identify
    object of E. Other values stay as they are, for the Binding rules to report."""
    old_roles = [role.concept for role in getattr(find_relationship(before, f"{concept}.{old}"), "roles", [])]
    new_roles = [role.concept for role in getattr(find_relationship(after, f"{concept}.{new}"), "roles", [])]
    table = concepts(after)

    def wrap(item, previous, current):
        if getattr(concepts(before).get(previous), "type", None) != "ValueType":
            return item
        if getattr(table.get(current), "type", None) != "EntityType":
            return item
        for option in identifiers(after, current):
            relationship = identify_relationship(after, current, option[0]) if len(option) == 1 else None
            if relationship is not None and relationship.roles[0].concept == previous:
                return {option[0]: item}
        return item

    def identify_key(role_concept, key):
        return new if role_concept == concept and key == old else key

    def filling(entry, name, values):
        if entry.concept != concept or name != old:
            return name, values
        return new, [[wrap(item, previous, current) for item, previous, current in zip(one, old_roles, new_roles)]
                     + one[len(new_roles):] for one in values]

    return rewrite(binding, before, lambda name: name, identify_key, filling)
