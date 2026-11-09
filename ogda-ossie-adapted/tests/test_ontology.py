"""The Harness and its store: three Ossie documents and OGDA's state file."""

import json

import pytest

from ogda.binding import Binding, BindingEntry
from ogda.ontology import Harness, OntologyStore
from ogda.ossie import (Concept, Dataset, DatasetField, Expression, MappingDocument, OgdaExtension, Relationship, Role,
                        read_yaml, with_ogda_extension)


def income_dataset(name="income", path="income.csv", **extension):
    dataset = Dataset(name=name, source=path, primary_key=["household_id"], fields=[
        DatasetField(name="household_id", expression=Expression(sql='"household_id"'), datatype="String"),
        DatasetField(name="annual_income", expression=Expression(sql='"annual_income"'), datatype="Decimal")])
    return with_ogda_extension(dataset, OgdaExtension(path=path, end_row=3, origin="declared",
                                                      documentation=["README.md"], **extension))


def income_harness_models() -> Harness:
    harness = Harness.empty("toy")
    harness.ontology.ontology = [
        Concept(concept="HouseholdId", type="ValueType", extends=["String"]),
        Concept(concept="IncomeUsd", type="ValueType", extends=["Decimal"],
                description="Annual household income in US dollars"),
        Concept(concept="Household", type="EntityType", identify_by=["household_id"], relationships=[
            Relationship(name="household_id", roles=[Role(concept="HouseholdId")], multiplicity="OneToOne",
                         verbalizes=["{Household} has id {HouseholdId}"]),
            Relationship(name="annual_income", roles=[Role(concept="IncomeUsd")], multiplicity="ManyToOne",
                         verbalizes=["{Household} earns {IncomeUsd} a year"])])]
    harness.semantic_model.datasets = [income_dataset()]
    harness.bindings["income"] = Binding(rows="Household", binding=[BindingEntry(
        concept="Household", identify={"household_id": "household_id"}, relationships={"annual_income": ["annual_income"]})])
    return harness


def test_store_round_trip_writes_three_valid_ossie_documents_and_state(tmp_path):
    store = OntologyStore(tmp_path / "ontology")
    harness = income_harness_models()
    store.save(harness)
    assert {p.name for p in store.root.iterdir()} >= {"ontology.yaml", "semantic_model.yaml", "mapping.yaml", "state.json"}
    assert json.loads(store.state_path.read_text())["format_version"] == 7
    assert store.load() == harness
    assert read_yaml(store.mapping_path, MappingDocument) == harness.mapping()
    assert harness.mapping().concept_mappings[0].concept == "Household"


def test_an_empty_harness_saves_only_state_and_loads_back(tmp_path):
    store = OntologyStore(tmp_path / "ontology")
    store.save(Harness.empty("toy"))
    assert sorted(p.name for p in store.root.iterdir()) == ["state.json"]
    assert store.load() == Harness.empty("toy")


def test_a_document_that_becomes_empty_is_deleted(tmp_path):
    store = OntologyStore(tmp_path / "ontology")
    harness = income_harness_models()
    store.save(harness)
    harness.semantic_model.datasets, harness.bindings = [], {}
    store.save(harness)
    assert not store.semantic_model_path.exists() and not store.mapping_path.exists() and store.ontology_path.exists()
    assert store.load() == harness


def test_a_format_6_folder_is_refused_with_a_rebuild_message(tmp_path):
    root = tmp_path / "ontology"
    root.mkdir()
    (root / "ontology.json").write_text('{"metadata": {"format_version": 6}}')
    assert OntologyStore(root).exists()
    with pytest.raises(ValueError, match="format version 7"):
        OntologyStore(root).load()
    (root / "state.json").write_text('{"format_version": 6, "lake_id": "toy"}')
    with pytest.raises(ValueError, match="format version 7"):
        OntologyStore(root).load()


def test_layout_members_share_their_representatives_binding():
    harness = income_harness_models()
    harness.semantic_model.datasets.append(income_dataset("income_2021", "income_2021.csv", layout_of="income"))
    assert harness.members("income") == ["income_2021"]
    assert harness.binding_of("income_2021") == harness.bindings["income"] and "income_2021" not in harness.bindings
    expressions = [o.referent_mappings[0].expression for o in harness.mapping().concept_mappings[0].object_mappings]
    assert expressions == ["income.household_id", "income_2021.household_id"]


def test_load_rejects_bindings_that_do_not_fit_the_documents(tmp_path):
    store = OntologyStore(tmp_path / "ontology")
    store.save(income_harness_models())
    state = json.loads(store.state_path.read_text())
    state["bindings"]["income"]["binding"][0]["relationships"]["annual_income"] = ["salary"]
    store.state_path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="income.*no Field salary"):
        store.load()


def test_object_ids_use_concept_relationship_dataset_and_field_ids():
    harness = income_harness_models()
    assert {"Household", "Household.annual_income", "income", "income.annual_income"} <= harness.object_ids()
    assert harness.fields()["income.annual_income"].datatype == "Decimal"
    assert harness.concept("Household").identify_by == ["household_id"] and harness.dataset("income").name == "income"


# Final review fixes.

def test_a_save_whose_mapping_does_not_validate_writes_nothing(tmp_path):
    store = OntologyStore(tmp_path / "ontology")
    harness = income_harness_models()
    store.save(harness)
    before = {p.name: p.read_bytes() for p in store.root.iterdir()}
    harness.bindings["income"] = Binding(binding=[BindingEntry(concept="Household",
                                                               identify={"household_id": "household_id"})])
    harness.ontology.ontology[2].description = "Changed"
    with pytest.raises(Exception):
        store.save(harness)
    assert {p.name: p.read_bytes() for p in store.root.iterdir() if p.suffix != ".tmp"} == before


def test_an_interrupted_save_is_reported_on_load(tmp_path):
    store = OntologyStore(tmp_path / "ontology")
    harness = income_harness_models()
    store.save(harness)
    state = store.state_path.read_text()
    harness.ontology.ontology[2].description = "Changed"
    store.save(harness)
    store.state_path.write_text(state)  # The new ontology.yaml next to the old state.json.
    with pytest.raises(ValueError, match="interrupted"):
        store.load()


def test_load_rejects_a_dataset_without_a_binding(tmp_path):
    store = OntologyStore(tmp_path / "ontology")
    harness = income_harness_models()
    harness.semantic_model.datasets.append(income_dataset("orphan", "orphan.csv"))
    harness.bindings["orphan"] = harness.bindings["income"]
    store.save(harness)
    state = json.loads(store.state_path.read_text())
    del state["bindings"]["orphan"]
    store.state_path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="orphan"):
        store.load()
