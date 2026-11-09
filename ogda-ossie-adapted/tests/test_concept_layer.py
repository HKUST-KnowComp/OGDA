"""The concept layer: one review per Concept, merges, References, compound Identifiers and convergence."""

import json

from helpers import HOUSEHOLD_ID, INCOME_CLAIMS, INCOME_DATASET, INCOME_USD, Script, answer, code, construction, make_lake
from test_induction import README, MAPPED, events, induction, mapped, relationship, statuses, task, value_type
from ogda.ossie import MappingDocument, read_yaml

INCOME_CONCEPTS = ["Household", "HouseholdId", "IncomeUsd"]


def reviewed(*names, reason="Distinct from the other Concepts"):
    return [{"concept": name, "reason": reason} for name in names]


def review(*names, claims=()):
    return answer({"claims": list(claims), "reviewed": reviewed(*names), "summary": "Reviewed"})


def build(tmp_path, script, lake=None, **settings):
    built = construction(tmp_path, script, **settings)
    built.start("toy", lake or make_lake(tmp_path))
    return built


def concept_tasks(script):
    return [task(p) for p in script.prompts["proposer"] if task(p)["stage"] == "concepts"]


def test_every_concept_gets_one_review_and_construction_converges(tmp_path):
    script = Script([answer(induction()), review(*INCOME_CONCEPTS)])
    built = build(tmp_path, script)
    harness = built.run()
    assert harness.metadata["stop_reason"] == "converged"
    assert sorted(harness.metadata["progress"]["reviewed"]) == INCOME_CONCEPTS
    visit = concept_tasks(script)[0]
    assert [c["concept"] for c in visit["review"]] == INCOME_CONCEPTS and visit["remaining"] == 0
    assert visit["review"][0]["datasets"][0]["name"] == "income"
    assert visit["link_candidates"] == []  # The one Field that names Households already identifies them.


def test_the_dataset_limit_does_not_cap_concept_layer_claims(tmp_path):
    described = [{"kind": "concept", "concept": {**claim["concept"], "description": "Reviewed"}}
                 for claim in (HOUSEHOLD_ID, INCOME_USD)]
    script = Script([answer(induction()), review(*INCOME_CONCEPTS, claims=described)])
    harness = build(tmp_path, script, max_datasets_per_visit=1).run()
    assert harness.metadata["stop_reason"] == "converged"
    assert harness.concept("HouseholdId").description == harness.concept("IncomeUsd").description == "Reviewed"


def test_a_batch_starts_with_the_first_unreviewed_concept_and_adds_similar_ones(tmp_path):
    script = Script([answer(induction()), review("Household", "HouseholdId"), review("IncomeUsd")])
    harness = build(tmp_path, script, concepts_per_visit=2).run()
    visits = concept_tasks(script)
    assert [[c["concept"] for c in v["review"]] for v in visits] == [["Household", "HouseholdId"], ["IncomeUsd"]]
    assert visits[0]["remaining"] == 1 and harness.metadata["stop_reason"] == "converged"


FAMILY = [
    {"kind": "concept", "concept": {"concept": "Family", "type": "EntityType", "identify_by": ["family_id"]}},
    relationship("Family", "family_id", "HouseholdId", "OneToOne", "{Family} has id {HouseholdId}"),
    relationship("Family", "annual_income", "IncomeUsd", "ManyToOne", "{Family} earns {IncomeUsd} a year"),
    {"kind": "dataset", "dataset": {
        "name": "families", "path": "families.csv", "origin": "inferred", "documentation": [], "fields": [
            {"name": "family_id", "column": "family_id", "datatype": "String"},
            {"name": "annual_income", "column": "annual_income", "datatype": "Decimal"}],
        "rows": "Family", "binding": [{"concept": "Family", "identify": {"family_id": "family_id"},
                                        "relationships": {"annual_income": ["annual_income"]}}]}},
]
MERGE = {"kind": "merge", "from": "Family", "into": "Household",
         "relationships": {"family_id": "household_id", "annual_income": "annual_income"}}


def test_a_merge_rewrites_bindings_and_the_merged_concept_leaves_the_work_list(tmp_path):
    lake = make_lake(tmp_path)
    (lake / "families.csv").write_text("family_id,annual_income\nF1,10\n")
    first = answer(induction([*INCOME_CLAIMS, INCOME_DATASET, *FAMILY],
                             dispositions=[MAPPED, README, mapped("families.csv", "families")]))
    script = Script([first, review("Family", "Household", claims=[MERGE]), review("HouseholdId", "IncomeUsd")])
    built = build(tmp_path, script, lake, files_per_visit=10, concepts_per_visit=2)
    harness = built.run()
    assert harness.concept("Family") is None and harness.bindings["families"].rows == "Household"
    assert harness.bindings["families"].binding[0].identify == {"household_id": "family_id"}
    assert harness.metadata["stop_reason"] == "converged"
    household = next(m for m in read_yaml(built.store.mapping_path, MappingDocument).concept_mappings
                     if m.concept == "Household")
    assert len(household.object_mappings) == 2


PEOPLE = [
    value_type("PersonId", "String"),
    {"kind": "concept", "concept": {"concept": "Person", "type": "EntityType", "identify_by": ["person_id"]}},
    relationship("Person", "person_id", "PersonId", "OneToOne", "{Person} has id {PersonId}"),
    relationship("Person", "household_ref", "HouseholdId", "ManyToOne", "{Person} lives in the household {HouseholdId}"),
    {"kind": "dataset", "dataset": {
        "name": "people", "path": "people.csv", "origin": "inferred", "documentation": [], "fields": [
            {"name": "person", "column": "person", "datatype": "String"},
            {"name": "household", "column": "household", "datatype": "String"}],
        "rows": "Person", "binding": [{"concept": "Person", "identify": {"person_id": "person"},
                                        "relationships": {"household_ref": ["household"]}}]}},
]
REFERENCE = {"kind": "relationship", "concept": "Person", "replaces": "household_ref", "relationship": {
    "name": "household", "roles": [{"concept": "Household"}], "multiplicity": "ManyToOne",
    "verbalizes": ["{Person} lives in {Household}"]}}


def test_an_attribute_becomes_a_reference(tmp_path):
    lake = make_lake(tmp_path)
    (lake / "people.csv").write_text("person,household\nP1,H1\n")
    first = answer(induction([*INCOME_CLAIMS, INCOME_DATASET, *PEOPLE],
                             dispositions=[MAPPED, README, mapped("people.csv", "people")]))
    script = Script([first, review(*INCOME_CONCEPTS, "Person", "PersonId", claims=[REFERENCE])])
    built = build(tmp_path, script, lake, files_per_visit=10)
    harness = built.run()
    assert harness.bindings["people"].binding[0].relationships == {"household": [{"household_id": "household"}]}
    person = next(m for m in read_yaml(built.store.mapping_path, MappingDocument).concept_mappings
                  if m.concept == "Person")
    child = person.link_mappings[0].children[0]
    assert (child.relationship, child.object_mapping.concept) == ("household", "Household")
    assert child.object_mapping.referent_mappings[0].relationship == "Household.household_id"
    assert harness.metadata["stop_reason"] == "converged"


def beach_lake(tmp_path):
    lake = tmp_path / "original"
    lake.mkdir()
    (lake / "beaches.csv").write_text("beach_name,community\nNantasket,Hull\nShort,Nahant\n")
    (lake / "samples.csv").write_text("sample_id,beach_name,count\nS1,Nantasket,3\nS2,Short,2\n")
    return lake


def beach_dataset(name, path, fields, rows, entry):
    return {"kind": "dataset", "dataset": {"name": name, "path": path, "origin": "inferred", "documentation": [],
                                           "fields": [{"name": f, "column": f, "datatype": "String"} for f in fields],
                                           "rows": rows, "binding": [entry]}}


BEACH_CLAIMS = [
    value_type("BeachName", "String"), value_type("MunicipalityName", "String"), value_type("SampleId", "String"),
    value_type("SampleCount", "String"),
    {"kind": "concept", "concept": {"concept": "Municipality", "type": "EntityType", "identify_by": ["name"]}},
    {"kind": "concept", "concept": {"concept": "Beach", "type": "EntityType", "identify_by": ["name"]}},
    {"kind": "concept", "concept": {"concept": "Sample", "type": "EntityType", "identify_by": ["sample_id"]}},
    relationship("Municipality", "name", "MunicipalityName", "OneToOne", "{Municipality} is named {MunicipalityName}"),
    relationship("Beach", "name", "BeachName", "OneToOne", "{Beach} is named {BeachName}"),
    relationship("Beach", "municipality", "Municipality", "ManyToOne", "{Beach} lies in {Municipality}"),
    relationship("Sample", "sample_id", "SampleId", "OneToOne", "{Sample} has id {SampleId}"),
    relationship("Sample", "beach", "Beach", "ManyToOne", "{Sample} was taken at {Beach}"),
    relationship("Sample", "count", "SampleCount", "ManyToOne", "{Sample} counted {SampleCount}"),
    beach_dataset("beaches", "beaches.csv", ["beach_name", "community"], "Beach", {
        "concept": "Beach", "identify": {"name": "beach_name"},
        "relationships": {"municipality": [{"name": "community"}]}}),
    beach_dataset("samples", "samples.csv", ["sample_id", "beach_name", "count"], "Sample", {
        "concept": "Sample", "identify": {"sample_id": "sample_id"},
        "relationships": {"beach": [{"name": "beach_name"}], "count": ["count"]}}),
]
COMPOUND = [
    {"kind": "concept", "concept": {"concept": "Beach", "type": "EntityType", "identify_by": ["name", "municipality"]}},
    relationship("Beach", "name", "BeachName", "ManyToOne", "{Beach} is named {BeachName}"),
]
REBIND = [
    relationship("Sample", "beach_name", "BeachName", "ManyToOne", "{Sample} names the beach {BeachName}"),
    {"kind": "binding", "dataset": "beaches", "rows": "Beach", "binding": [{
        "concept": "Beach", "identify": {"name": "beach_name", "municipality": {"name": "community"}}}]},
    {"kind": "binding", "dataset": "samples", "rows": "Sample", "binding": [{
        "concept": "Sample", "identify": {"sample_id": "sample_id"},
        "relationships": {"beach_name": ["beach_name"], "count": ["count"]}}]},
]
BEACH_CONCEPTS = ["Beach", "BeachName", "Municipality", "MunicipalityName", "Sample", "SampleCount", "SampleId"]


def test_a_compound_identifier_that_breaks_a_binding_is_rejected_unless_the_answer_rebinds_it(tmp_path):
    first = answer(induction(BEACH_CLAIMS, dispositions=[mapped("beaches.csv", "beaches"), mapped("samples.csv", "samples")]))
    script = Script([first, review(*BEACH_CONCEPTS, claims=COMPOUND),
                     review(*BEACH_CONCEPTS, claims=[*COMPOUND, *REBIND])])
    harness = build(tmp_path, script, beach_lake(tmp_path), max_steps=4).run()
    error = script.prompts["proposer"][2]
    assert "Dataset beaches" in error and "Dataset samples" in error and "not an Identifier of Beach" in error
    assert harness.concept("Beach").identify_by == ["name", "municipality"]
    assert harness.bindings["samples"].binding[0].relationships == {"beach_name": ["beach_name"], "count": ["count"]}
    assert harness.dataset("beaches").primary_key == ["beach_name", "community"]
    assert harness.metadata["stop_reason"] == "converged"


def test_a_concept_added_in_this_stage_joins_the_work_list_unless_reviewed(tmp_path):
    region = value_type("Region", "String")
    script = Script([answer(induction()), review(*INCOME_CONCEPTS, claims=[region]), review("Region")])
    harness = build(tmp_path, script).run()
    assert [[c["concept"] for c in v["review"]] for v in concept_tasks(script)] == [INCOME_CONCEPTS, ["Region"]]
    assert harness.metadata["stop_reason"] == "converged"


def test_a_visit_that_accepts_and_reviews_nothing_stops_unresolved(tmp_path):
    harness = build(tmp_path, Script([answer(induction()), answer({"claims": [], "reviewed": []})])).run()
    assert harness.metadata["stop_reason"] == "unresolved" and harness.metadata["progress"]["stage"] == "done"


def test_a_visit_without_an_answer_stops_for_an_explicit_resume(tmp_path):
    built = build(tmp_path, Script([answer(induction()), code("print('Thinking')")]))
    assert built.run().metadata["stop_reason"] == "proposer_incomplete"
    resumed = construction(tmp_path, Script([review(*INCOME_CONCEPTS)]))
    assert resumed.run().metadata["stop_reason"] == "converged"


def test_resume_after_the_pass_limit_continues_the_review(tmp_path):
    script = Script([answer(induction()), review("Household")])
    built = build(tmp_path, script, max_passes=1, concepts_per_visit=1)
    assert built.run().metadata["stop_reason"] == "max_passes"
    resumed = construction(tmp_path, Script([review("HouseholdId"), review("IncomeUsd")]), max_passes=5,
                           concepts_per_visit=1)
    harness = resumed.run()
    assert harness.metadata["stop_reason"] == "converged"
    assert sorted(harness.metadata["progress"]["reviewed"]) == INCOME_CONCEPTS


def test_induction_rejects_concept_layer_claims_and_the_concept_layer_rejects_dataset_claims(tmp_path):
    remove = {"kind": "remove", "id": "IncomeUsd"}
    script = Script([answer(induction([*INCOME_CLAIMS, INCOME_DATASET, remove])), answer(induction()),
                     review(*INCOME_CONCEPTS, claims=[INCOME_DATASET]), review(*INCOME_CONCEPTS)])
    harness = build(tmp_path, script).run()
    assert "Schema induction permits only concept, relationship and dataset Claims" in script.prompts["proposer"][1]
    assert "The concept layer permits only" in script.prompts["proposer"][3]
    assert harness.metadata["stop_reason"] == "converged"
