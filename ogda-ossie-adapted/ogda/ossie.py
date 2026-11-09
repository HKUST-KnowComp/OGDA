"""Apache Ossie documents: the ontology document, the core semantic model and the mapping document.

The models mirror Ossie's JSON schemas (version 0.2.0.dev0), vendored in `ogda/ossie_schemas/`, field for field and
forbid keys Ossie does not define. `validate` checks a document against its schema; `read_yaml` and `write_yaml`
always validate. OGDA's own details of a dataset (its File, region, origin, Documentation and layout group) travel in
one custom extension whose vendor is "OGDA".
"""

from __future__ import annotations

import json
import os
import re
from functools import cache, lru_cache
from importlib import resources
from pathlib import Path
from typing import Literal, TypeVar, get_args

import yaml
from jsonschema import Draft202012Validator
from pydantic import BaseModel, ConfigDict, Field as PydanticField, model_validator
from referencing import Registry, Resource

OSSIE_VERSION = "0.2.0.dev0"
Datatype = Literal["Boolean", "Date", "DateTime", "Decimal", "Float", "Integer", "String"]
BUILTINS: tuple[str, ...] = get_args(Datatype)
NUMERIC = frozenset({"Integer", "Decimal", "Float"})
VENDOR = "OGDA"
RAW = "https://raw.githubusercontent.com/apache/ossie/main/"
CONCEPT_NAME = re.compile(r"^[A-Z][A-Za-z0-9]*$")
SNAKE = re.compile(r"^[a-z][a-z0-9_]*$")
PLACEHOLDER = re.compile(r"\{([^{}]+)\}")


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


# The ontology document.

class Role(Model):
    concept: str
    name: str | None = None


class Relationship(Model):
    name: str
    verbalizes: list[str]
    description: str | None = None
    roles: list[Role] = []
    multiplicity: Literal["ManyToOne", "OneToOne"] | None = None


class Concept(Model):
    concept: str
    type: Literal["EntityType", "ValueType"]
    description: str | None = None
    extends: list[str] = []
    identify_by: list[str] = []
    relationships: list[Relationship] = []


class OntologyDocument(Model):
    version: str = OSSIE_VERSION
    name: str
    description: str | None = None
    ontology: list[Concept] = []


# The core semantic model.

class Expression(Model):
    sql: str


class DatasetField(Model):
    name: str
    expression: Expression
    datatype: Datatype | None = None
    description: str | None = None


class CustomExtension(Model):
    vendor_name: str
    data: str


class Dataset(Model):
    name: str
    source: str
    primary_key: list[str] = []
    description: str | None = None
    fields: list[DatasetField] = []
    custom_extensions: list[CustomExtension] = []


class SemanticModel(Model):
    version: str = OSSIE_VERSION
    name: str
    description: str | None = None
    datasets: list[Dataset] = []


# The mapping document.

class ReferentMapping(Model):
    relationship: str
    expression: str | None = None
    referent_mappings: list[ReferentMapping] = []


class ObjectMapping(Model):
    concept: str | None = None
    expression: str | None = None
    referent_mappings: list[ReferentMapping] = []


class LinkMapping(Model):
    object_mapping: ObjectMapping
    relationship: str | None = None
    children: list[LinkMapping] = []


class ConceptMapping(Model):
    concept: str
    object_mappings: list[ObjectMapping] = []
    link_mappings: list[LinkMapping] = []


class DocumentReference(Model):
    name: str
    iri: str | None = None


class MappingDocument(Model):
    version: str = OSSIE_VERSION
    name: str
    description: str | None = None
    ontology_ref: DocumentReference
    semantic_model_ref: DocumentReference
    concept_mappings: list[ConceptMapping] = []


Document = OntologyDocument | SemanticModel | MappingDocument
T = TypeVar("T", OntologyDocument, SemanticModel, MappingDocument)


# The Ontology rules.

def concepts(document: OntologyDocument) -> dict[str, Concept]:
    return {concept.concept: concept for concept in document.ontology}


def find_relationship(document: OntologyDocument, relationship_id: str) -> Relationship | None:
    """The Relationship with an id like "Fire.acres_burned", or None."""
    owner, _, name = relationship_id.partition(".")
    concept = concepts(document).get(owner)
    return next((r for r in concept.relationships if r.name == name), None) if concept else None


def role_concepts(owner: str, relationship: Relationship) -> list[str]:
    """The Concepts of a Relationship's Roles in order; the first Role is played by the owner."""
    return [owner] + [role.concept for role in relationship.roles]


def role_keys(owner: str, relationship: Relationship) -> list[str]:
    """How verbalization placeholders name the Roles: the Concept, or "Concept:name" for a named Role."""
    return [owner] + [f"{r.concept}:{r.name}" if r.name else r.concept for r in relationship.roles]


def supertypes(document: OntologyDocument, name: str) -> list[str]:
    """The Concepts that `name` extends, directly or not, nearest first."""
    table, found, queue = concepts(document), [], [name]
    while queue:
        concept = table.get(queue.pop(0))
        for parent in concept.extends if concept else []:
            if parent not in found and parent != name:
                found.append(parent)
                queue.append(parent)
    return found


def owner_of(document: OntologyDocument, concept: str, relationship: str) -> str | None:
    """The Concept that declares `relationship` for `concept`: the Concept itself or its nearest supertype."""
    table = concepts(document)
    for candidate in [concept] + supertypes(document, concept):
        if candidate in table and any(r.name == relationship for r in table[candidate].relationships):
            return candidate
    return None


def base_type(document: OntologyDocument, name: str) -> str | None:
    """The built-in that a Value type extends, or None."""
    concept = concepts(document).get(name)
    if concept is None or concept.type != "ValueType" or len(concept.extends) != 1:
        return None
    return concept.extends[0] if concept.extends[0] in BUILTINS else None


def alternate(document: OntologyDocument, relationship: Relationship) -> bool:
    """A binary OneToOne Relationship to a Value type, which identifies the Entities of its owner."""
    table = concepts(document)
    return (len(relationship.roles) == 1 and relationship.multiplicity == "OneToOne"
            and getattr(table.get(relationship.roles[0].concept), "type", None) == "ValueType")


def identifiers(document: OntologyDocument, name: str) -> list[list[str]]:
    """The Identifiers of an Entity type: its preferred `identify_by` (its own, or its nearest supertype's), then
    each alternate Identifier as a one-item list."""
    table = concepts(document)
    for candidate in [name] + supertypes(document, name):
        concept = table.get(candidate)
        if concept is not None and concept.identify_by:
            preferred = list(concept.identify_by)
            return [preferred] + [[r.name] for r in concept.relationships
                                  if alternate(document, r) and [r.name] != preferred]
    return []


def extends_cycle(document: OntologyDocument) -> list[str] | None:
    table = concepts(document)
    done: set[str] = set()

    def walk(name: str, path: list[str]) -> list[str] | None:
        if name in path:
            return path[path.index(name):] + [name]
        if name in done or name not in table:
            return None
        for parent in table[name].extends:
            if cycle := walk(parent, path + [name]):
                return cycle
        done.add(name)
        return None

    for name in table:
        if cycle := walk(name, []):
            return cycle
    return None


def relationship_problems(document: OntologyDocument, owner: str, relationship: Relationship) -> list[str]:
    table, label = concepts(document), f"{owner}.{relationship.name}"
    problems = []
    if not SNAKE.match(relationship.name):
        problems.append(f"{label}: a Relationship name is snake_case, such as acres_burned")
    if not relationship.roles:
        problems.append(f"{label} needs at least one Role after its first, which {owner} plays")
    for role in relationship.roles:
        if role.concept in BUILTINS:
            problems.append(f"{label}: a Role is played by the built-in {role.concept}; declare a Value type that "
                            f"extends {role.concept} and name it instead")
        elif role.concept not in table:
            problems.append(f"{label}: no Concept {role.concept}")
        if role.name is not None and not SNAKE.match(role.name):
            problems.append(f"{label}: the Role name {role.name!r} is snake_case")
    if relationship.multiplicity == "OneToOne" and len(relationship.roles) != 1:
        problems.append(f"{label}: OneToOne is only for binary Relationships")
    keys = role_keys(owner, relationship)
    if repeated := sorted({key for key in keys if keys.count(key) > 1}):
        problems.append(f"{label}: {repeated} play more than one Role; give each repeated Role a name")
    if not relationship.verbalizes:
        problems.append(f"{label} needs at least one verbalizes pattern")
    for pattern in relationship.verbalizes:
        placeholders = PLACEHOLDER.findall(pattern)
        if unknown := [p for p in placeholders if p not in keys]:
            problems.append(f"{label}: in {pattern!r}, " + ", ".join(f"{{{p}}}" for p in unknown)
                            + f" names no Role; the Roles are {['{' + k + '}' for k in keys]}")
        if missing := [k for k in keys if k not in placeholders]:
            problems.append(f"{label}: {pattern!r} must name every Role; it misses "
                            + ", ".join("{" + k + "}" for k in missing))
    return problems


def ontology_problems(document: OntologyDocument) -> list[str]:
    """Every break of the Ontology rules, one message each, naming the Concept or Relationship."""
    table, problems = concepts(document), []
    names = [concept.concept for concept in document.ontology]
    for name in sorted({n for n in names if names.count(n) > 1}):
        problems.append(f"Concept {name} is declared more than once")
    for concept in document.ontology:
        name = concept.concept
        if not CONCEPT_NAME.match(name):
            problems.append(f"Concept {name}: a Concept name is PascalCase, such as CountyName")
        if name in BUILTINS:
            problems.append(f"Concept {name} is a built-in; choose a name that says what the values mean")
        relationship_names = [r.name for r in concept.relationships]
        for repeated in sorted({n for n in relationship_names if relationship_names.count(n) > 1}):
            problems.append(f"{name}.{repeated} is declared more than once")
        if concept.type == "ValueType":
            if len(concept.extends) != 1 or concept.extends[0] not in BUILTINS:
                problems.append(f"Value type {name} extends exactly one built-in ({', '.join(BUILTINS)}), "
                                f"not {concept.extends}")
            if concept.relationships:
                problems.append(f"Value type {name} has no Relationships; declare them under an Entity type")
            if concept.identify_by:
                problems.append(f"Value type {name} has no identify_by")
            continue
        for parent in concept.extends:
            if parent not in table:
                problems.append(f"Entity type {name} extends no Concept {parent}")
            elif table[parent].type != "EntityType":
                problems.append(f"Entity type {name} extends only Entity types, not the Value type {parent}")
        if not identifiers(document, name):
            problems.append(f"Entity type {name} needs identify_by, or extends an Entity type that has it")
        own = {r.name: r for r in concept.relationships}
        for part in concept.identify_by:
            relationship = own.get(part)
            if relationship is None:
                problems.append(f"{name}.identify_by: {part} is not a Relationship of {name}")
            elif len(relationship.roles) != 1:
                problems.append(f"{name}.{part} is in an Identifier of {name} and must be binary")
            elif len(concept.identify_by) == 1 and not alternate(document, relationship):
                problems.append(f"{name}.{part} identifies {name} alone, so it is OneToOne and its Role is a "
                                "Value type")
            elif len(concept.identify_by) > 1 and relationship.multiplicity != "ManyToOne":
                problems.append(f"{name}.{part} is part of a compound Identifier of {name} and must be ManyToOne")
        for relationship in concept.relationships:
            problems += relationship_problems(document, name, relationship)
    if cycle := extends_cycle(document):
        problems.append(f"extends forms a cycle: {' -> '.join(cycle)}")
    return list(dict.fromkeys(problems))


# OGDA's details of one dataset.

class OgdaExtension(Model):
    """Where a Dataset's rows are: a region of one File, rows counted from 0, and how the Dataset was made.

    The column names are in `header_rows` rows from `header_row` down; `header_row` is None when the region has no
    header row. `relation` says how the Dataset relates to another one: a part of one dataset, a copy of
    `relation_to`, or a subset of it.
    """

    path: str
    sheet: str | None = None
    header_row: int | None = PydanticField(default=0, ge=0)
    header_rows: int = PydanticField(default=1, ge=1)
    start_row: int = PydanticField(default=1, ge=0)
    end_row: int | None = PydanticField(default=None, ge=0)
    origin: Literal["declared", "inferred"] = "inferred"
    documentation: list[str] = []
    path_pattern: str | None = None
    layout_of: str | None = None
    relation: Literal["part", "copy", "subset"] = "part"
    relation_to: str | None = None

    @model_validator(mode="after")
    def valid_region(self):
        if self.header_row is None and self.header_rows != 1:
            raise ValueError("a region without a header row has header_rows 1")
        if self.header_row is not None and self.header_row + self.header_rows > self.start_row:
            raise ValueError("the header rows must precede start_row")
        if self.end_row is not None and self.end_row <= self.start_row:
            raise ValueError("end_row must follow start_row")
        if (self.relation == "part") != (self.relation_to is None):
            raise ValueError("a copy or subset names the Dataset it relates to in relation_to; a part has none")
        return self


@lru_cache(maxsize=65536)
def parse_extension(data: str) -> OgdaExtension:
    return OgdaExtension.model_validate_json(data)


def ogda_extension(dataset: Dataset) -> OgdaExtension:
    """A Dataset's OGDA extension. The parsed object is shared between calls: copy it before changing it."""
    for extension in dataset.custom_extensions:
        if extension.vendor_name == VENDOR:
            return parse_extension(extension.data)
    raise ValueError(f"Dataset {dataset.name} has no {VENDOR} extension")


def with_ogda_extension(dataset: Dataset, extension: OgdaExtension) -> Dataset:
    others = [e for e in dataset.custom_extensions if e.vendor_name != VENDOR]
    data = json.dumps(extension.model_dump(mode="json"), sort_keys=True)
    return dataset.model_copy(update={"custom_extensions": others + [CustomExtension(vendor_name=VENDOR, data=data)]})


# Serialization and validation.

def prune(value):
    """Drop empty lists and dicts, which Ossie's schemas reject where they set minItems."""
    if isinstance(value, dict):
        pruned = {key: prune(item) for key, item in value.items()}
        return {key: item for key, item in pruned.items() if item not in ([], {})}
    if isinstance(value, list):
        return [prune(item) for item in value]
    return value


def dump(document: BaseModel) -> dict:
    return prune(document.model_dump(mode="json", exclude_none=True))


@cache
def validators() -> dict[type, Draft202012Validator]:
    folder = resources.files("ogda") / "ossie_schemas"
    schemas = {name: json.loads((folder / name).read_text()) for name in
               ("ontology.json", "mapping.json", "ossie-schema.json")}
    registry = Registry()
    for name, path in (("ontology.json", "ontology/ontology.json"), ("mapping.json", "ontology/mapping.json"),
                       ("ossie-schema.json", "core-spec/ossie-schema.json")):
        resource = Resource.from_contents(schemas[name])
        registry = registry.with_resources([(schemas[name]["$id"], resource), (RAW + path, resource)])
    return {model: Draft202012Validator(schemas[name], registry=registry) for model, name in
            ((OntologyDocument, "ontology.json"), (MappingDocument, "mapping.json"), (SemanticModel, "ossie-schema.json"))}


def validate(document: Document) -> None:
    """Raise jsonschema.ValidationError when the document breaks its Ossie schema."""
    validators()[type(document)].validate(dump(document))


def render_yaml(document: Document) -> str:
    """The document as YAML, after checking it against its Ossie schema."""
    validate(document)
    dumper = getattr(yaml, "CSafeDumper", yaml.SafeDumper)
    return yaml.dump(dump(document), Dumper=dumper, sort_keys=False, allow_unicode=True)


def write_yaml(path: Path, document: Document) -> None:
    temporary = Path(path).with_suffix(".tmp")
    temporary.write_text(render_yaml(document))
    os.replace(temporary, path)


def read_yaml(path: Path, model: type[T]) -> T:
    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    document = model.model_validate(yaml.load(Path(path).read_text(), Loader=loader))
    validate(document)
    return document
