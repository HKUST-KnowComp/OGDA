"""Retrieval over the Ossie Harness: search, summaries and Concept similarity."""

import pytest

from helpers import income_harness, make_lake
from ossie_examples import valid_ontology
from ogda.ontology import ExperienceEntry, Harness
from ogda.ossie import OgdaExtension, Relationship, Role, ogda_extension, with_ogda_extension
from ogda.retrieval import (concept_detail, dataset_detail, dataset_summary, ontology_summary, retrieve,
                            similar_concepts)


@pytest.fixture
def harness(tmp_path):
    return income_harness(make_lake(tmp_path))


def add_member(harness, name="income_2021"):
    income = harness.dataset("income")
    extension = ogda_extension(income).model_copy(update={"path": f"{name}.csv", "layout_of": "income"})
    harness.semantic_model.datasets.append(
        with_ogda_extension(income.model_copy(update={"name": name, "source": f"{name}.csv"}), extension))


def test_search_matches_concepts_relationships_datasets_and_fields_and_walks_edges(harness):
    found = retrieve(harness, "annual income")
    assert "Household.annual_income" in found["matched"] and "income.annual_income" in found["matched"]
    assert "income" in [d["name"] for d in found["datasets"]]
    assert "Household" in [c["concept"] for c in found["concepts"]]
    assert {"source": "income", "kind": "fills", "target": "Household.annual_income"} in found["edges"]


def test_layout_members_appear_as_their_representative_with_a_count(harness):
    add_member(harness)
    found = retrieve(harness, "income household")
    assert [(d["name"], d["members"]) for d in found["datasets"]] == [("income", 1)]
    assert "income_2021" not in str(found)


def test_concept_detail_lists_binding_datasets_and_the_fields_per_role(harness):
    add_member(harness)
    assert concept_detail(harness, "Household")["datasets"] == [{
        "name": "income", "members": 1, "rows": True, "identify": {"household_id": "household_id"},
        "fills": {"Household.annual_income": [["annual_income"]]}}]
    assert concept_detail(harness, "IncomeUsd")["datasets"] == [{
        "name": "income", "members": 1, "rows": False, "fills": {"Household.annual_income": [["annual_income"]]}}]
    assert concept_detail(harness, "IncomeUsd")["extends"] == ["Decimal"]


def test_dataset_detail_and_summary_show_the_region_binding_and_concepts(harness):
    detail = dataset_detail(harness, "income")
    assert detail["extension"]["path"] == "income.csv" and detail["binding"]["rows"] == "Household"
    assert dataset_summary(harness, "income") == {
        "name": "income", "path": "income.csv", "sheet": None, "description": "Annual income of each household",
        "rows": "Household", "concepts": ["Household"], "members": 0}
    assert [c["concept"] for c in ontology_summary(harness)] == ["HouseholdId", "IncomeUsd", "Household"]


def test_similar_concepts_ranks_by_term_overlap():
    harness = Harness.empty("toy")
    harness.ontology = valid_ontology()
    assert similar_concepts(harness, "CountyName", ["Fire", "County", "Acres"], 2) == ["County", "Fire"]


def test_experience_anchored_to_returned_objects_is_returned(harness):
    harness.experience.append(ExperienceEntry(id="e1", kind="data_quirk", text="Top coded at 150",
                                              anchors=["income.annual_income"]))
    assert [e["id"] for e in retrieve(harness, "annual income")["experience"]] == ["e1"]


# Link candidates.

def area_harness(lake):
    from ogda.binding import Binding, BindingEntry, quote_identifier, quote_literal
    from ogda.ossie import Concept, Dataset, DatasetField, Expression, Relationship, Role

    def named(entity, value):
        return [Concept(concept=value, type="ValueType", extends=["String"]),
                Concept(concept=entity, type="EntityType", identify_by=["name"], relationships=[
                    Relationship(name="name", roles=[Role(concept=value)], multiplicity="OneToOne",
                                 verbalizes=[f"{{{entity}}} is named {{{value}}}"])])]

    def table(name, path, fields, **extension):
        return with_ogda_extension(Dataset(name=name, source=path, fields=[
            DatasetField(name=n, expression=Expression(sql=sql), datatype="String") for n, sql in fields]),
            OgdaExtension(path=path, **extension))

    for path, text in {"states.csv": "State\nNew Hampshire\nNevada\nDistrict of Columbia\n",
                       "areas/NewHampshire.csv": "Area\nManchester\n", "areas/Nevada.csv": "Area\nReno\n",
                       "areas/DistrictofColumbia.csv": "Area\nWashington\n"}.items():
        (lake / path).parent.mkdir(parents=True, exist_ok=True)
        (lake / path).write_text(text)
    harness = Harness.empty("toy")
    harness.ontology.ontology = named("State", "StateName") + named("Area", "AreaName")
    harness.semantic_model.datasets = [table("states", "states.csv", [("state", quote_identifier("State"))])] + [
        table(name, f"areas/{slug}.csv", [("area", quote_identifier("Area")), ("state", quote_literal(slug))],
              **({} if name == "areas" else {"layout_of": "areas"}))
        for name, slug in [("areas", "NewHampshire"), ("areas__nevada", "Nevada"),
                           ("areas__dc", "DistrictofColumbia")]]
    harness.bindings = {
        "states": Binding(rows="State", binding=[BindingEntry(concept="State", identify={"name": "state"})]),
        "areas": Binding(rows="Area", binding=[BindingEntry(concept="Area", identify={"name": "area"})],
                         unbound={"state": "The File name, not a full state name"})}
    return harness


def test_link_candidates_list_fields_whose_values_identify_another_entity_type(tmp_path):
    from ogda.retrieval import link_candidates

    lake = make_lake(tmp_path)
    found = link_candidates(area_harness(lake), lake, 100)
    assert found == [{"field": "areas.state", "values": 3, "target": "State.name", "identifier": ["name"],
                      "exact": 1, "normalized": 3, "unmatched": ["DistrictofColumbia", "NewHampshire"]}]


def test_a_field_that_already_identifies_the_entity_type_is_no_candidate(tmp_path):
    from ogda.binding import Binding, BindingEntry
    from ogda.retrieval import link_candidates

    lake = make_lake(tmp_path)
    harness = area_harness(lake)
    harness.ontology.ontology[3].relationships.append(Relationship(
        name="state", roles=[Role(concept="State")], verbalizes=["{Area} lies in {State}"]))
    harness.bindings["areas"] = Binding(rows="Area", binding=[BindingEntry(
        concept="Area", identify={"name": "area"}, relationships={"state": [{"name": {"normalized": "state"}}]})])
    assert link_candidates(harness, lake, 100) == []
