"""The Ontology of one Lake: its objects, Claims, Experience and persistence."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field as PydanticField, model_validator

FORMAT_VERSION = 5


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Field(Model):
    id: str
    name: str
    column: str
    description: str = ""
    dtype: str = "string"
    join_key: bool = False


class Schema(Model):
    id: str
    description: str = ""
    fields: list[Field] = PydanticField(default_factory=list)
    origin: Literal["declared", "inferred"] = "inferred"
    documentation: list[str] = PydanticField(default_factory=list)

    @model_validator(mode="after")
    def unique_fields(self):
        if len({field.id for field in self.fields}) != len(self.fields):
            raise ValueError(f"Schema {self.id} has duplicate Field ids")
        return self


class Table(Model):
    """A record region of one File, described by one Schema. Rows start at 0."""

    id: str
    schema_id: str
    path: str
    sheet: str | None = None
    header_row: int | None = PydanticField(default=0, ge=0)
    start_row: int = PydanticField(default=1, ge=0)
    end_row: int | None = PydanticField(default=None, ge=0)

    @model_validator(mode="after")
    def valid_region(self):
        if self.header_row is not None and self.header_row >= self.start_row:
            raise ValueError(f"Table {self.id}: header_row must precede start_row")
        if self.end_row is not None and self.end_row <= self.start_row:
            raise ValueError(f"Table {self.id}: end_row must follow start_row")
        return self


class Concept(Model):
    id: str
    name: str
    description: str = ""


class Link(Model):
    """field_link: Field -> Concept, its values are instances of the Concept; instance_of: Field -> Concept, the
    column itself is one instance (a wide layout); sub_concept_of: Concept -> Concept; about: Table -> Concept;
    mentions: File path -> Concept."""

    id: str
    kind: Literal["field_link", "instance_of", "sub_concept_of", "about", "mentions"]
    source: str
    target: str


ClaimKind = Literal["schema", "tables", "concept", "link", "remove"]
ClaimStatus = Literal["pending", "verified", "accepted", "refuted", "undecidable"]


class Claim(Model):
    id: str
    kind: ClaimKind
    payload: dict[str, Any]
    status: ClaimStatus = "pending"
    notes: str = ""
    evidence_refs: list[str] = PydanticField(default_factory=list)
    iteration: int = 0
    proposer_visit: str = ""
    verifier_visit: str = ""


ExperienceKind = Literal["data_quirk", "procedural_know_how", "dead_end", "documentation_discrepancy"]


class ExperienceEntry(Model):
    id: str
    kind: ExperienceKind
    text: str
    anchors: list[str] = PydanticField(default_factory=list)
    claim_ids: list[str] = PydanticField(default_factory=list)
    provisional: bool = False
    stale: bool = False
    superseded_by: str | None = None

    def labelled(self) -> dict:
        """The entry as runtime agents see it: an unverified lesson with its stale and provisional flags."""
        return {**self.model_dump(mode="json"), "verified": False}


class Ontology(Model):
    lake_id: str
    schemas: dict[str, Schema] = PydanticField(default_factory=dict)
    tables: dict[str, Table] = PydanticField(default_factory=dict)
    concepts: dict[str, Concept] = PydanticField(default_factory=dict)
    links: dict[str, Link] = PydanticField(default_factory=dict)
    claims: dict[str, Claim] = PydanticField(default_factory=dict)
    experience: list[ExperienceEntry] = PydanticField(default_factory=list)
    metadata: dict[str, Any] = PydanticField(default_factory=dict)

    def fields(self) -> dict[str, Field]:
        return {field.id: field for schema in self.schemas.values() for field in schema.fields}

    def object_ids(self) -> set[str]:
        return set(self.schemas) | set(self.tables) | set(self.concepts) | set(self.links) | set(self.fields())

    def live_experience(self) -> list[ExperienceEntry]:
        return [entry for entry in self.experience if entry.superseded_by is None]

    def serving_view(self) -> dict:
        """The Harness as the Data agent reads it: objects, File dispositions and labelled live Experience."""
        data = self.model_dump(mode="json", include={"schemas", "tables", "concepts", "links"})
        data["files"] = self.metadata.get("progress", {}).get("files", {})
        data["experience"] = [entry.labelled() for entry in self.live_experience()]
        return data


class OntologyStore:
    """One directory: `ontology.json` and the agents' writable copy of the Lake."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.path = self.root / "ontology.json"
        self.lake_root = self.root / "lake"

    def exists(self) -> bool:
        return self.path.is_file()

    def load(self) -> Ontology:
        data = json.loads(self.path.read_text())
        if data.get("metadata", {}).get("format_version") != FORMAT_VERSION:
            raise ValueError(f"Unsupported ontology format; rebuild it with format version {FORMAT_VERSION}")
        return Ontology.model_validate(data)

    def save(self, ontology: Ontology) -> None:
        ontology.metadata["format_version"] = FORMAT_VERSION
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(ontology.model_dump_json(indent=2))
        os.replace(temporary, self.path)
