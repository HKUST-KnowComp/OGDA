"""The Harness of one Lake: its Ossie documents, the Bindings of its Datasets, Claims, Experience and persistence.

`ontology.yaml` (Ossie ontology) and `semantic_model.yaml` (Ossie core semantic model) hold the Concepts and Datasets;
`state.json` holds what Ossie has no place for: Bindings, Claims, Experience and scheduler progress. `mapping.yaml`
(Ossie mapping) is compiled from the Bindings on every save and never read back.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field as PydanticField

from .binding import Binding, binding_problems, compile_mapping, document_names
from .ossie import (Concept, Dataset, DatasetField, MappingDocument, OntologyDocument, SemanticModel, ogda_extension,
                    read_yaml, render_yaml)

FORMAT_VERSION = 7
UNSUPPORTED = f"Unsupported ontology format; rebuild it with format version {FORMAT_VERSION}"


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


ClaimKind = Literal["concept", "relationship", "dataset", "merge", "binding", "remove"]
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


class Harness(Model):
    lake_id: str
    ontology: OntologyDocument
    semantic_model: SemanticModel
    bindings: dict[str, Binding] = PydanticField(default_factory=dict)  # By representative or single Dataset.
    claims: dict[str, Claim] = PydanticField(default_factory=dict)
    experience: list[ExperienceEntry] = PydanticField(default_factory=list)
    metadata: dict[str, Any] = PydanticField(default_factory=dict)

    @classmethod
    def empty(cls, lake_id: str, **data) -> Harness:
        ontology, model, _ = document_names(lake_id)
        return cls(lake_id=lake_id, ontology=OntologyDocument(name=ontology), semantic_model=SemanticModel(name=model),
                   **data)

    def concept(self, name: str) -> Concept | None:
        return next((c for c in self.ontology.ontology if c.concept == name), None)

    def datasets(self) -> dict[str, Dataset]:
        return {dataset.name: dataset for dataset in self.semantic_model.datasets}

    def dataset(self, name: str) -> Dataset | None:
        return self.datasets().get(name)

    def members(self, name: str) -> list[str]:
        """The Datasets of the layout group that `name` represents."""
        return sorted(d.name for d in self.semantic_model.datasets if ogda_extension(d).layout_of == name)

    def binding_of(self, name: str) -> Binding | None:
        """A Dataset's Binding; a layout group member shares its representative's."""
        dataset = self.dataset(name)
        if dataset is None:
            return None
        return self.bindings.get(ogda_extension(dataset).layout_of or name)

    def fields(self) -> dict[str, DatasetField]:
        return {f"{d.name}.{f.name}": f for d in self.semantic_model.datasets for f in d.fields}

    def relationship_ids(self) -> set[str]:
        return {f"{c.concept}.{r.name}" for c in self.ontology.ontology for r in c.relationships}

    def object_ids(self) -> set[str]:
        return ({c.concept for c in self.ontology.ontology} | self.relationship_ids() | set(self.datasets())
                | set(self.fields()))

    def live_experience(self) -> list[ExperienceEntry]:
        return [entry for entry in self.experience if entry.superseded_by is None]

    def mapping(self) -> MappingDocument:
        bindings = {name: binding for name in self.datasets() if (binding := self.binding_of(name)) is not None}
        return compile_mapping(self.ontology, self.semantic_model, bindings, self.lake_id)


class OntologyStore:
    """One directory: the three Ossie documents, `state.json` and the agents' writable copy of the Lake."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.state_path = self.root / "state.json"
        self.ontology_path = self.root / "ontology.yaml"
        self.semantic_model_path = self.root / "semantic_model.yaml"
        self.mapping_path = self.root / "mapping.yaml"
        self.legacy_path = self.root / "ontology.json"  # Written by format 6 and earlier.
        self.lake_root = self.root / "lake"

    def exists(self) -> bool:
        return self.state_path.is_file() or self.legacy_path.is_file()

    def documents(self) -> list[Path]:
        return [self.ontology_path, self.semantic_model_path, self.mapping_path]

    def load(self) -> Harness:
        if not self.state_path.is_file() and self.legacy_path.is_file():
            raise ValueError(UNSUPPORTED)
        state = json.loads(self.state_path.read_text())
        if state.get("format_version") != FORMAT_VERSION:
            raise ValueError(UNSUPPORTED)
        state.pop("format_version")
        expected = state.pop("documents", {})
        if torn := [path.name for path in self.documents()
                    if expected.get(path.name) != (digest(path.read_text()) if path.is_file() else None)]:
            raise ValueError(f"The last save was interrupted: {torn} do not match state.json; resume from a copy saved "
                             "before the interruption, or rebuild the Ontology")
        harness = Harness.empty(state.pop("lake_id"), **state)
        if self.ontology_path.is_file():
            harness.ontology = read_yaml(self.ontology_path, OntologyDocument)
        if self.semantic_model_path.is_file():
            harness.semantic_model = read_yaml(self.semantic_model_path, SemanticModel)
        problems = [f"{name}: no Binding" for name in harness.datasets() if harness.binding_of(name) is None]
        problems += [f"{name}: {problem}" for name, dataset in harness.datasets().items()
                     if (binding := harness.binding_of(name)) is not None
                     for problem in binding_problems(binding, dataset, harness.ontology)]
        if problems:
            raise ValueError("The saved Bindings do not fit the saved documents:\n" + "\n".join(problems))
        return harness

    def save(self, harness: Harness) -> None:
        """Write the documents that have content (Ossie requires at least one Concept and one dataset), delete the
        ones that became empty, then write state.json, which records each document's digest. Every document is
        rendered and checked before any file is replaced, and the files are swapped in by renames."""
        self.root.mkdir(parents=True, exist_ok=True)
        texts = {self.ontology_path: render_yaml(harness.ontology) if harness.ontology.ontology else None,
                 self.semantic_model_path: render_yaml(harness.semantic_model) if harness.semantic_model.datasets
                 else None}
        texts[self.mapping_path] = (render_yaml(harness.mapping())
                                    if texts[self.ontology_path] and texts[self.semantic_model_path] else None)
        state = {"format_version": FORMAT_VERSION,
                 "documents": {path.name: digest(text) if text is not None else None for path, text in texts.items()},
                 **harness.model_dump(mode="json", exclude={"ontology", "semantic_model"})}
        staged = [(path, text) for path, text in texts.items() if text is not None]
        staged.append((self.state_path, json.dumps(state, indent=2, ensure_ascii=False)))
        for path, text in staged:
            path.with_suffix(".tmp").write_text(text)
        for path, text in staged[:-1]:
            os.replace(path.with_suffix(".tmp"), path)
        for path, text in texts.items():
            if text is None and path.exists():
                path.unlink()
        os.replace(self.state_path.with_suffix(".tmp"), self.state_path)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()
