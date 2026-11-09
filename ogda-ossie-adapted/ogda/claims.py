"""Claim contracts: parse an agent's proposal, apply it to the Harness, and check what an answer would change.

Schema induction proposes `concept`, `relationship` and `dataset` Claims; the concept layer proposes `concept`,
`relationship`, `merge`, `binding` and `remove` Claims. An answer is checked by applying its Claims to a copy of the
Harness (`trial`) and running the Ontology rules, the Binding rules and the data checks on what changed
(`answer_problems`).
"""

from __future__ import annotations

import re
from pathlib import Path

from .binding import (Binding, binding_problems, data_problems, field_source, merge_binding, named_concepts,
                      primary_key, quote_identifier, quote_literal, replace_relationship, structure_problems)
from .lake import RowReader, sheets, split_documentation
from .ontology import Claim, Harness
from .ossie import (BUILTINS, NUMERIC, SNAKE, Concept, Dataset, DatasetField, Expression, OgdaExtension, Relationship,
                    base_type, concepts, identifiers, ogda_extension, ontology_problems, role_concepts,
                    with_ogda_extension)

KINDS = {"induction": ("concept", "relationship", "dataset"),
         "concepts": ("concept", "relationship", "merge", "binding", "remove")}
ORDER = {"concept": 0, "relationship": 1, "dataset": 2, "merge": 3, "binding": 4, "remove": 5}
REGION = ("sheet", "header_row", "header_rows", "start_row", "end_row")
DATASET_KEYS = {"name", "path", *REGION, "description", "origin", "documentation", "path_pattern", "relation",
                "relation_to", "fields", "rows", "binding", "unbound", "same_as"}
SAME_AS_KEYS = {"name", "path", *REGION, "relation", "relation_to", "same_as"}


def lake_path(value: str, lake_root: Path) -> str:
    """Return the Lake-relative path of an existing File. Absolute paths must be in the Lake."""
    root = Path(lake_root).resolve()
    path = Path(value)
    resolved = (path if path.is_absolute() else root / path).resolve()
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError(f"No File at {value} in the Lake")
    return resolved.relative_to(root).as_posix()


def documentation_ref(value: str, lake_root: Path) -> str:
    """Return a Documentation reference in Lake terms: an existing File, or an existing nonempty worksheet of a
    workbook as 'path#sheet'."""
    path, sheet = split_documentation(value)
    if sheet is None:
        return lake_path(value, lake_root)
    path = lake_path(path, lake_root)
    try:
        names, _ = sheets(Path(lake_root) / path)
    except Exception as error:  # A corrupt workbook: the Proposer can cite something else.
        raise ValueError(f"cannot read the worksheets of {path}: {type(error).__name__}: {error}") from None
    if sheet not in names:
        raise ValueError(f"{path} has no nonempty worksheet {sheet!r}; its nonempty worksheets are {names}")
    return f"{path}#{sheet}"


def only(item: dict, keys: set[str], shape: str) -> None:
    if unknown := sorted(set(item) - keys):
        raise ValueError(f"Unknown keys {unknown}; {shape}")


# Parsing.

def path_values(pattern: str | None, path: str) -> dict[str, str]:
    if not pattern:
        return {}
    try:
        match = re.fullmatch(pattern, path)
    except re.error as error:
        raise ValueError(f"path_pattern is not a regular expression: {error}") from None
    if match is None:
        raise ValueError(f"{path} does not match path_pattern {pattern!r}")
    if missing := sorted(name for name, value in match.groupdict().items() if value is None):
        raise ValueError(f"{path} does not match the named groups {missing} of path_pattern {pattern!r}")
    return match.groupdict()


def build_fields(items, path: str, pattern: str | None) -> list[DatasetField]:
    """Ossie fields from the agent's Fields: a column Field quotes its header text, a path Field holds the literal
    that `pattern` reads from the path."""
    if not isinstance(items, list) or not items:
        raise ValueError("A Dataset needs fields: [{'name', 'column' or 'from_path', 'datatype', 'description'}]")
    values, fields = path_values(pattern, path), []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Each Field is an object")
        only(item, {"name", "column", "from_path", "datatype", "description"},
             "a Field is {'name', 'column' or 'from_path': True, 'datatype', 'description'}")
        name, datatype = item.get("name"), item.get("datatype")
        if not isinstance(name, str) or not SNAKE.match(name):
            raise ValueError(f"Field name {name!r} must be snake_case, such as acres_burned")
        if datatype not in BUILTINS:
            raise ValueError(f"Field {name}: datatype is one of {list(BUILTINS)}, not {datatype!r}")
        if item.get("from_path") is True:
            if "column" in item or name not in values:
                raise ValueError(f"Field {name}: a path Field has no column and its name is a named group of "
                                 "path_pattern")
            sql = quote_literal(values[name])
        elif isinstance(item.get("column"), str):
            sql = quote_identifier(item["column"])
        else:
            raise ValueError(f"Field {name} needs its header text in column, or from_path: True")
        fields.append(DatasetField(name=name, expression=Expression(sql=sql), datatype=datatype,
                                   description=item.get("description") or None))
    names = [f.name for f in fields]
    if len(set(names)) != len(names):
        raise ValueError(f"Field names repeat: {sorted({n for n in names if names.count(n) > 1})}")
    path_fields = {f.name for f in fields if field_source(f)[0] == "literal"}
    if set(values) != path_fields:
        raise ValueError(f"path_pattern's named groups {sorted(values)} must be exactly the path Fields "
                         f"{sorted(path_fields)}")
    return fields


def parse_dataset(item, harness: Harness, lake_root: Path) -> dict:
    if not isinstance(item, dict):
        raise ValueError("A dataset Claim is {'kind': 'dataset', 'dataset': {...}}")
    only(item, DATASET_KEYS, f"a Dataset has {sorted(DATASET_KEYS)}")
    name = item.get("name")
    if not isinstance(name, str) or not SNAKE.match(name):
        raise ValueError(f"Dataset name {name!r} must be snake_case, such as fires_2020")
    path = lake_path(str(item.get("path", "")), lake_root)
    payload = {}
    if same_as := item.get("same_as"):
        only(item, SAME_AS_KEYS, "a same_as Dataset gives only name, path, sheet, its region, relation and relation_to; "
                                 "same_as copies the rest")
        source = harness.dataset(same_as)
        if source is None or harness.binding_of(same_as) is None:
            raise ValueError(f"same_as names no accepted Dataset {same_as!r}")
        copied = ogda_extension(source)
        fields = [{"name": f.name, "datatype": f.datatype, "description": f.description,
                   **({"from_path": True} if field_source(f)[0] == "literal" else {"column": field_source(f)[1]})}
                  for f in source.fields]
        item = {**item, "description": source.description, "origin": copied.origin,
                "documentation": copied.documentation, "path_pattern": copied.path_pattern, "fields": fields,
                **harness.binding_of(same_as).model_dump(mode="json")}
        payload["same_as"] = same_as
    binding = Binding.model_validate({key: item[key] for key in ("rows", "binding", "unbound") if key in item})
    pattern = item.get("path_pattern")
    fields = build_fields(item.get("fields"), path, pattern)
    documentation = list(dict.fromkeys(documentation_ref(ref, lake_root) for ref in item.get("documentation") or []))
    if item.get("origin") == "declared" and not documentation:
        raise ValueError("A declared Dataset cites at least one existing Documentation File or worksheet")
    extension = OgdaExtension(path=path, documentation=documentation, path_pattern=pattern, **{
        key: item[key] for key in (*REGION, "origin", "relation", "relation_to") if key in item})
    source = f"{path}#{extension.sheet}" if extension.sheet else path
    dataset = Dataset(name=name, source=source, primary_key=primary_key(binding),
                      description=item.get("description") or None, fields=fields)
    return {"dataset": with_ogda_extension(dataset, extension).model_dump(mode="json"),
            "binding": binding.model_dump(mode="json"), **payload}


def parse_claim(item: dict, harness: Harness, lake_root: Path) -> tuple[str, dict]:
    """Check one proposal's shape and normalize it into a Claim payload. Raise ValueError on a bad proposal."""
    if not isinstance(item, dict) or "kind" not in item:
        raise ValueError(f"A Claim is an object with a kind: one of {list(ORDER)}")
    kind, rest = item["kind"], {key: value for key, value in item.items() if key != "kind"}
    if kind not in ORDER:
        raise ValueError(f"Unknown Claim kind {kind!r}; use one of {list(ORDER)}")
    if kind == "concept":
        only(rest, {"concept"}, "a concept Claim is {'kind': 'concept', 'concept': {...}}")
        concept = Concept.model_validate(rest.get("concept"))
        if concept.relationships:
            raise ValueError("Give a Concept's Relationships with relationship Claims, one each")
        return kind, {"concept": concept.model_dump(mode="json")}
    if kind == "relationship":
        only(rest, {"concept", "relationship", "replaces"},
             "a relationship Claim is {'kind': 'relationship', 'concept', 'relationship': {...}, optional 'replaces'}")
        if not isinstance(rest.get("concept"), str) or not isinstance(rest.get("replaces", ""), str):
            raise ValueError("A relationship Claim names its Concept, and replaces names a Relationship of it")
        relationship = Relationship.model_validate(rest.get("relationship"))
        return kind, {"concept": rest["concept"], "relationship": relationship.model_dump(mode="json"),
                      "replaces": rest.get("replaces") or None}
    if kind == "dataset":
        only(rest, {"dataset"}, "a dataset Claim is {'kind': 'dataset', 'dataset': {...}}")
        return kind, parse_dataset(rest.get("dataset"), harness, lake_root)
    if kind == "merge":
        only(rest, {"from", "into", "relationships"},
             "a merge Claim is {'kind': 'merge', 'from', 'into', 'relationships': {from name: into name}}")
        mapping = rest.get("relationships", {})
        if (not isinstance(rest.get("from"), str) or not isinstance(rest.get("into"), str)
                or not isinstance(mapping, dict) or any(not isinstance(v, str) for v in mapping.values())):
            raise ValueError("A merge Claim names from and into, and maps Relationship names in relationships")
        return kind, {"from": rest["from"], "into": rest["into"], "relationships": dict(mapping)}
    if kind == "binding":
        only(rest, {"dataset", "rows", "binding", "unbound"},
             "a binding Claim is {'kind': 'binding', 'dataset', 'rows', 'binding': [...], 'unbound': {...}}")
        dataset = harness.dataset(rest.get("dataset")) if isinstance(rest.get("dataset"), str) else None
        if dataset is None:
            raise ValueError(f"No Dataset {rest.get('dataset')!r}")
        if layout_of := ogda_extension(dataset).layout_of:
            raise ValueError(f"{dataset.name} shares the Binding of {layout_of}; rebind {layout_of}")
        binding = Binding.model_validate({key: rest[key] for key in ("rows", "binding", "unbound") if key in rest})
        return kind, {"dataset": dataset.name, "binding": binding.model_dump(mode="json")}
    only(rest, {"id"}, "a remove Claim is {'kind': 'remove', 'id': Concept or 'Concept.relationship'}")
    if not isinstance(rest.get("id"), str):
        raise ValueError("A remove Claim needs the id of a Concept or a Relationship")
    return kind, {"id": rest["id"]}


# Applying.

def uses(harness: Harness, name: str) -> list[str]:
    """What still refers to a Concept or Relationship: Bindings, Roles, identify_by and extends."""
    found = []
    owner, _, relationship = name.partition(".")
    for dataset, binding in sorted(harness.bindings.items()):
        if relationship:
            if any(e.concept == owner and relationship in e.relationships for e in binding.binding):
                found.append(f"the Binding of {dataset}")
        elif name in named_concepts(binding, harness.ontology):
            found.append(f"the Binding of {dataset}")
    for concept in harness.ontology.ontology:
        if relationship:
            if concept.concept == owner and relationship in concept.identify_by:
                found.append(f"{owner}.identify_by")
            continue
        if concept.concept != name and name in concept.extends:
            found.append(f"{concept.concept}.extends")
        found += [f"{concept.concept}.{r.name}" for r in concept.relationships if concept.concept != name
                  and name in role_concepts(concept.concept, r)[1:]]
    return found


def placeholder(source: str, target: str, pattern: str) -> str:
    """A verbalization with the placeholders of `source` ({Source} and {Source:name}) renamed to `target`."""
    return re.sub(r"\{" + re.escape(source) + r"(?=[:}])", "{" + target, pattern)


def merge(harness: Harness, source: str, target: str, mapping: dict[str, str]) -> set[str]:
    table = concepts(harness.ontology)
    if source not in table or target not in table or source == target:
        raise ValueError(f"merge needs two different existing Concepts, not {source!r} and {target!r}")
    old, new = table[source], table[target]
    if old.type != new.type:
        raise ValueError(f"merge: {source} is a {old.type} and {target} a {new.type}")
    if old.type == "ValueType":
        bases = {base_type(harness.ontology, source), base_type(harness.ontology, target)}
        if len(bases) > 1 and not bases <= NUMERIC:
            raise ValueError(f"merge: {source} and {target} have incompatible bases {sorted(map(str, bases))}")
    before = harness.ontology.model_copy(deep=True)

    def renamed(concept: str) -> str:
        return target if concept == source else concept

    targets = {r.name: r for r in new.relationships}
    for relationship in old.relationships:
        name = mapping.get(relationship.name, relationship.name)
        roles = [renamed(role.concept) for role in relationship.roles]
        if relationship.name in mapping:
            if name not in targets:
                raise ValueError(f"merge: {target} has no Relationship {name} for {source}.{relationship.name}")
            if [role.concept for role in targets[name].roles] != roles:
                raise ValueError(f"merge: {source}.{relationship.name} and {target}.{name} have different Roles")
        elif name in targets:
            raise ValueError(f"merge: {target} already has a Relationship {name}; map {source}.{name} onto it")
        else:
            moved = relationship.model_copy(deep=True)
            for role in moved.roles:
                role.concept = renamed(role.concept)
            moved.verbalizes = [placeholder(source, target, pattern) for pattern in moved.verbalizes]
            new.relationships.append(moved)
    harness.ontology.ontology = [c for c in harness.ontology.ontology if c.concept != source]
    for concept in harness.ontology.ontology:
        concept.extends = list(dict.fromkeys(renamed(s) for s in concept.extends if renamed(s) != concept.concept))
        for relationship in concept.relationships:
            for role in relationship.roles:
                if role.concept == source:
                    role.concept = target
                    relationship.verbalizes = [placeholder(source, target, p) for p in relationship.verbalizes]
    harness.bindings = {name: merge_binding(binding, before, source, target, mapping)
                        for name, binding in harness.bindings.items()}
    return {source, target} | {f"{source}.{r.name}" for r in old.relationships}


def apply_claim(claim: Claim, harness: Harness) -> set[str]:
    """Install an accepted or verified Claim. Return the ids of existing objects it changed, and mark Experience
    anchored to them stale. Raise ValueError when the Claim does not fit the Harness."""
    kind, payload, changed = claim.kind, claim.payload, set()
    if kind == "concept":
        concept = Concept.model_validate(payload["concept"])
        names = [c.concept for c in harness.ontology.ontology]
        if concept.concept in names:
            index = names.index(concept.concept)
            concept.relationships = harness.ontology.ontology[index].relationships
            harness.ontology.ontology[index] = concept
            changed.add(concept.concept)
        else:
            harness.ontology.ontology.append(concept)
    elif kind == "relationship":
        owner = harness.concept(payload["concept"])
        if owner is None:
            raise ValueError(f"No Concept {payload['concept']} to declare {payload['relationship']['name']} under")
        relationship = Relationship.model_validate(payload["relationship"])
        old = payload.get("replaces") or relationship.name
        names = [r.name for r in owner.relationships]
        if payload.get("replaces") and old not in names:
            raise ValueError(f"{owner.concept} has no Relationship {old} to replace")
        if old != relationship.name and relationship.name in names:
            raise ValueError(f"{owner.concept} already has a Relationship {relationship.name}")
        before = harness.ontology.model_copy(deep=True)
        if old in names:
            owner.relationships[names.index(old)] = relationship
            changed |= {f"{owner.concept}.{old}", owner.concept}
        else:
            owner.relationships.append(relationship)
        owner.identify_by = [relationship.name if part == old else part for part in owner.identify_by]
        if old in names:  # Renamed or retyped: Bindings follow, and Value-type values wrap into identify objects.
            harness.bindings = {name: replace_relationship(binding, before, harness.ontology, owner.concept, old,
                                                           relationship.name)
                                for name, binding in harness.bindings.items()}
    elif kind == "dataset":
        dataset = Dataset.model_validate(payload["dataset"])
        names = [d.name for d in harness.semantic_model.datasets]
        if dataset.name in names:
            harness.semantic_model.datasets[names.index(dataset.name)] = dataset
            changed.add(dataset.name)
        else:
            harness.semantic_model.datasets.append(dataset)
        if ogda_extension(dataset).layout_of is None and payload.get("binding") is not None:
            harness.bindings[dataset.name] = Binding.model_validate(payload["binding"])
    elif kind == "merge":
        changed |= merge(harness, payload["from"], payload["into"], payload["relationships"])
    elif kind == "binding":
        binding = Binding.model_validate(payload["binding"])
        harness.bindings[payload["dataset"]] = binding
        for name in [payload["dataset"], *harness.members(payload["dataset"])]:
            harness.dataset(name).primary_key = primary_key(binding)
        changed.add(payload["dataset"])
    elif kind == "remove":
        target = payload["id"]
        owner, _, relationship = target.partition(".")
        concept = harness.concept(owner)
        if concept is None or (relationship and relationship not in [r.name for r in concept.relationships]):
            raise ValueError(f"remove: no Concept or Relationship {target}")
        if still := uses(harness, target):
            raise ValueError(f"remove: {target} is still used by {still}")
        if relationship:
            concept.relationships = [r for r in concept.relationships if r.name != relationship]
        else:
            harness.ontology.ontology = [c for c in harness.ontology.ontology if c.concept != owner]
        changed.add(target)
    for entry in harness.live_experience():
        if changed & set(entry.anchors):
            entry.stale = True
    return changed


def trial(harness: Harness, parsed: list[tuple[str, dict]]) -> Harness:
    """A copy of the Harness with the Claims applied in ORDER (stable within one kind)."""
    copy = harness.model_copy(deep=True)
    for kind, payload in sorted(parsed, key=lambda item: ORDER[item[0]]):
        apply_claim(Claim(id="trial", kind=kind, payload=payload, status="accepted"), copy)
    return copy


# Checking.

def signature(harness: Harness, name: str):
    """What a Binding's rules and data checks depend on in one Concept; a description change leaves it unchanged."""
    concept = harness.concept(name)
    if concept is None:
        return None
    relationships = tuple((r.name, tuple(role_concepts(name, r)), r.multiplicity) for r in concept.relationships)
    return concept.type, tuple(map(tuple, identifiers(harness.ontology, name))), tuple(concept.extends), relationships


def affected_datasets(before: Harness, after: Harness) -> set[str]:
    """The Datasets whose checks may change: new ones, ones whose Dataset or Binding changed, and ones that name a
    Concept whose constraints changed, with the members of their layout groups."""
    names = {c.concept for c in before.ontology.ontology} | {c.concept for c in after.ontology.ontology}
    changed = {name for name in names if signature(before, name) != signature(after, name)}
    affected = set()
    for name, dataset in after.datasets().items():
        binding = after.binding_of(name)
        if before.dataset(name) != dataset or before.binding_of(name) != binding:
            affected.add(name)
        elif binding is not None and named_concepts(binding, after.ontology) & changed:
            affected.add(name)
    groups: dict[str, set[str]] = {}
    for dataset in after.semantic_model.datasets:
        if layout_of := ogda_extension(dataset).layout_of:
            groups.setdefault(layout_of, set()).add(dataset.name)
    return affected | {member for name in affected for member in groups.get(name, ())}


def answer_problems(before: Harness, after: Harness, new_datasets: set[str], lake_root: Path,
                    check_rows: int) -> list[str]:
    """What is wrong with the Harness an answer would produce: the Ontology rules first (the other checks assume
    them), then the structure of new Datasets, the Binding rules of every Dataset, and the data checks of every new or
    affected Dataset."""
    if problems := ontology_problems(after.ontology):
        return problems
    datasets, unreadable, reader = after.datasets(), set(), RowReader()  # Each File is read once for all checks.
    for name in sorted(new_datasets):
        if found := structure_problems(datasets[name], lake_root, check_rows, reader):
            problems += found
            unreadable.add(name)
    affected = new_datasets | affected_datasets(before, after)
    for name, dataset in sorted(datasets.items()):
        binding = after.binding_of(name)
        if binding is None or (ogda_extension(dataset).layout_of and name not in affected):
            continue  # A member shares its representative's Fields and Binding.
        found = [f"Dataset {name}: {p}" for p in binding_problems(binding, dataset, after.ontology)]
        problems += found
        if not found and name in affected and name not in unreadable:
            problems += data_problems(dataset, binding, after.ontology, lake_root, check_rows, reader)
    return list(dict.fromkeys(problems))


def is_noop(kind: str, payload: dict, harness: Harness) -> bool:
    """True when a proposal restates what the Harness already has."""
    if kind == "concept":
        current = harness.concept(payload["concept"]["concept"])
        return current is not None and current.model_dump(mode="json", exclude={"relationships"}) == {
            key: value for key, value in payload["concept"].items() if key != "relationships"}
    if kind == "relationship":
        owner = harness.concept(payload["concept"])
        current = next((r for r in owner.relationships if r.name == payload["relationship"]["name"]), None) if owner else None
        return payload.get("replaces") is None and current is not None and current.model_dump(mode="json") == payload["relationship"]
    if kind == "dataset":
        name = payload["dataset"]["name"]
        current = harness.dataset(name)
        binding = harness.bindings.get(name)
        return (current is not None and current.model_dump(mode="json") == payload["dataset"]
                and (binding.model_dump(mode="json") if binding else None) == payload.get("binding"))
    if kind == "binding":
        binding = harness.bindings.get(payload["dataset"])
        return binding is not None and binding.model_dump(mode="json") == payload["binding"]
    return False


def anchors(claim: Claim) -> list[str]:
    """The Concepts, Relationships, Datasets, Fields and Lake paths that a Claim is about."""
    payload = claim.payload
    if claim.kind == "concept":
        ids = [payload["concept"]["concept"]]
    elif claim.kind == "relationship":
        owner = payload["concept"]
        ids = [f"{owner}.{payload['relationship']['name']}"]
        ids += [f"{owner}.{payload['replaces']}"] if payload.get("replaces") else []
        ids.append(owner)
    elif claim.kind == "dataset":
        dataset = Dataset.model_validate(payload["dataset"])
        extension = ogda_extension(dataset)
        ids = [dataset.name, *(f"{dataset.name}.{f.name}" for f in dataset.fields), extension.path,
               *(split_documentation(ref)[0] for ref in extension.documentation)]
    elif claim.kind == "merge":
        ids = [payload["from"], payload["into"]]
    elif claim.kind == "binding":
        ids = [payload["dataset"]]
    else:
        ids = [payload["id"]]
    return list(dict.fromkeys(ids))
