"""Claims: parsing the agents' shapes, applying them to the Harness, and checking an answer."""

import copy

import pytest

from helpers import INCOME_BINDING, INCOME_CLAIMS, INCOME_DATASET, INCOME_FIELDS, accept, income_harness, make_lake
from ogda.claims import affected_datasets, anchors, answer_problems, apply_claim, is_noop, parse_claim, trial
from ogda.ontology import Claim, ExperienceEntry, Harness
from ogda.ossie import ogda_extension


@pytest.fixture
def lake(tmp_path):
    return make_lake(tmp_path)


def dataset_claim(**changes):
    item = copy.deepcopy(INCOME_DATASET)
    item["dataset"].update(changes)
    return item


def parsed(harness, *items, lake_root):
    return [parse_claim(item, harness, lake_root) for item in items]


def test_a_dataset_claim_becomes_an_ossie_dataset_with_its_extension_and_binding(lake):
    harness = income_harness(lake)
    income = harness.dataset("income")
    assert income.source == "income.csv" and income.primary_key == ["household_id"]
    assert harness.fields()["income.annual_income"].expression.sql == '"annual_income"'
    extension = ogda_extension(income)
    assert (extension.origin, extension.documentation, extension.end_row) == ("declared", ["README.md"], 3)
    assert harness.bindings["income"].rows == "Household"
    assert harness.mapping().concept_mappings[0].concept == "Household"


def test_header_text_with_quotes_becomes_a_quoted_identifier(lake):
    fields = [*INCOME_FIELDS, {"name": "size", "column": 'Size "m2"', "datatype": "Decimal"}]
    _, payload = parse_claim(dataset_claim(fields=fields), Harness.empty("toy"), lake)
    assert payload["dataset"]["fields"][2]["expression"]["sql"] == '"Size ""m2"""'


def test_path_fields_take_their_literal_from_the_path_pattern(lake):
    (lake / "states").mkdir()
    (lake / "states" / "Alabama.csv").write_text("household_id,annual_income\nH1,1\n")
    (lake / "states" / "X1.csv").write_text("household_id,annual_income\nH1,1\n")
    state = {"name": "state", "from_path": True, "datatype": "String", "description": "The state of the File"}
    item = dataset_claim(name="alabama", path="states/Alabama.csv", end_row=None,
                         path_pattern=r"states/(?P<state>[A-Za-z]+)\.csv", fields=[*INCOME_FIELDS, state],
                         unbound={"state": "Kept for joins later"})
    _, payload = parse_claim(item, Harness.empty("toy"), lake)
    assert payload["dataset"]["fields"][2]["expression"]["sql"] == "'Alabama'"
    with pytest.raises(ValueError, match="does not match"):
        parse_claim(dataset_claim(**{**item["dataset"], "path": "states/X1.csv"}), Harness.empty("toy"), lake)
    with pytest.raises(ValueError, match="named group"):
        parse_claim(dataset_claim(**{**item["dataset"], "path_pattern": r"states/[A-Za-z]+\.csv"}), Harness.empty("toy"),
                    lake)


def test_same_as_copies_fields_and_binding_from_an_accepted_dataset(lake):
    (lake / "income_2021.csv").write_text("household_id,annual_income\nH1,110\n")
    harness = income_harness(lake)
    kind, payload = parse_claim({"kind": "dataset", "dataset": {"name": "income_2021", "path": "income_2021.csv",
                                                                "same_as": "income"}}, harness, lake)
    assert kind == "dataset" and payload["same_as"] == "income"
    assert payload["binding"] == harness.bindings["income"].model_dump(mode="json")
    assert [f["name"] for f in payload["dataset"]["fields"]] == ["household_id", "annual_income"]
    with pytest.raises(ValueError, match="same_as"):
        parse_claim({"kind": "dataset", "dataset": {"name": "x", "path": "income_2021.csv", "same_as": "income",
                                                     "fields": []}}, harness, lake)


def test_a_declared_dataset_needs_existing_documentation(lake):
    with pytest.raises(ValueError, match="declared"):
        parse_claim(dataset_claim(documentation=[]), Harness.empty("toy"), lake)
    with pytest.raises(ValueError, match="No File"):
        parse_claim(dataset_claim(documentation=["MISSING.md"]), Harness.empty("toy"), lake)


def test_concept_claims_keep_existing_relationships(lake):
    harness = income_harness(lake)
    accept(harness, [{"kind": "concept", "concept": {"concept": "Household", "type": "EntityType",
                                                     "identify_by": ["household_id"], "description": "A home"}}], lake)
    assert harness.concept("Household").description == "A home"
    assert [r.name for r in harness.concept("Household").relationships] == ["household_id", "annual_income"]
    with pytest.raises(ValueError, match="relationship Claims"):
        parse_claim({"kind": "concept", "concept": {"concept": "X", "type": "EntityType", "relationships": [
            {"name": "a", "verbalizes": ["{X}"]}]}}, harness, lake)


FAMILY = [
    {"kind": "concept", "concept": {"concept": "Family", "type": "EntityType", "identify_by": ["family_id"]}},
    {"kind": "relationship", "concept": "Family", "relationship": {
        "name": "family_id", "roles": [{"concept": "HouseholdId"}], "multiplicity": "OneToOne",
        "verbalizes": ["{Family} has id {HouseholdId}"]}},
    {"kind": "relationship", "concept": "Family", "relationship": {
        "name": "annual_income", "roles": [{"concept": "IncomeUsd"}], "multiplicity": "ManyToOne",
        "verbalizes": ["{Family} earns {IncomeUsd} a year"]}},
    {"kind": "dataset", "dataset": {
        "name": "families", "path": "families.csv", "fields": [
            {"name": "family_id", "column": "family_id", "datatype": "String"},
            {"name": "annual_income", "column": "annual_income", "datatype": "Decimal"}],
        "rows": "Family", "binding": [{"concept": "Family", "identify": {"family_id": "family_id"},
                                        "relationships": {"annual_income": ["annual_income"]}}]}},
]


def test_merge_moves_relationships_rewrites_roles_and_bindings_and_removes_the_source(lake):
    (lake / "families.csv").write_text("family_id,annual_income\nF1,10\n")
    before = accept(income_harness(lake), FAMILY, lake)
    merge = {"kind": "merge", "from": "Family", "into": "Household",
             "relationships": {"family_id": "household_id", "annual_income": "annual_income"}}
    after = trial(before, parsed(before, merge, lake_root=lake))
    assert after.concept("Family") is None
    assert after.bindings["families"].rows == "Household"
    assert after.bindings["families"].binding[0].identify == {"household_id": "family_id"}
    assert answer_problems(before, after, set(), lake, 100) == []
    with pytest.raises(ValueError, match="no Relationship"):
        trial(before, parsed(before, {**merge, "relationships": {"family_id": "id"}}, lake_root=lake))


PEOPLE = [
    {"kind": "concept", "concept": {"concept": "PersonId", "type": "ValueType", "extends": ["String"]}},
    {"kind": "concept", "concept": {"concept": "Person", "type": "EntityType", "identify_by": ["person_id"]}},
    {"kind": "relationship", "concept": "Person", "relationship": {
        "name": "person_id", "roles": [{"concept": "PersonId"}], "multiplicity": "OneToOne",
        "verbalizes": ["{Person} has id {PersonId}"]}},
    {"kind": "relationship", "concept": "Person", "relationship": {
        "name": "household_ref", "roles": [{"concept": "HouseholdId"}], "multiplicity": "ManyToOne",
        "verbalizes": ["{Person} lives in the household {HouseholdId}"]}},
    {"kind": "dataset", "dataset": {
        "name": "people", "path": "people.csv", "fields": [
            {"name": "person", "column": "person", "datatype": "String"},
            {"name": "household", "column": "household", "datatype": "String"}],
        "rows": "Person", "binding": [{"concept": "Person", "identify": {"person_id": "person"},
                                        "relationships": {"household_ref": ["household"]}}]}},
]


def test_replacing_an_attribute_with_a_reference_rewrites_bindings(lake):
    (lake / "people.csv").write_text("person,household\nP1,H1\n")
    before = accept(income_harness(lake), PEOPLE, lake)
    reference = {"kind": "relationship", "concept": "Person", "replaces": "household_ref", "relationship": {
        "name": "household", "roles": [{"concept": "Household"}], "multiplicity": "ManyToOne",
        "verbalizes": ["{Person} lives in {Household}"]}}
    after = trial(before, parsed(before, reference, lake_root=lake))
    assert after.bindings["people"].binding[0].relationships == {"household": [{"household_id": "household"}]}
    assert [r.name for r in after.concept("Person").relationships] == ["person_id", "household"]
    assert answer_problems(before, after, set(), lake, 100) == []


def test_remove_rejects_a_used_concept_and_removes_an_unused_one(lake):
    harness = income_harness(lake)
    with pytest.raises(ValueError, match="still"):
        trial(harness, parsed(harness, {"kind": "remove", "id": "IncomeUsd"}, lake_root=lake))
    with pytest.raises(ValueError, match="still"):
        trial(harness, parsed(harness, {"kind": "remove", "id": "Household.annual_income"}, lake_root=lake))
    accept(harness, [{"kind": "concept", "concept": {"concept": "Unused", "type": "ValueType", "extends": ["String"]}}], lake)
    after = trial(harness, parsed(harness, {"kind": "remove", "id": "Unused"}, lake_root=lake))
    assert after.concept("Unused") is None


def test_a_description_change_affects_no_dataset_but_a_base_change_does(lake):
    before = income_harness(lake)
    described = {"kind": "concept", "concept": {"concept": "Household", "type": "EntityType",
                                                "identify_by": ["household_id"], "description": "A home"}}
    assert affected_datasets(before, trial(before, parsed(before, described, lake_root=lake))) == set()
    rebased = {"kind": "concept", "concept": {"concept": "IncomeUsd", "type": "ValueType", "extends": ["String"]}}
    assert affected_datasets(before, trial(before, parsed(before, rebased, lake_root=lake))) == {"income"}


def test_answer_problems_report_a_duplicate_identifier_in_a_new_dataset(lake):
    (lake / "dup.csv").write_text("household_id,annual_income\nH1,100\nH1,200\n")
    before = income_harness(lake)
    item = dataset_claim(name="dup", path="dup.csv", end_row=None, origin="inferred", documentation=[])
    after = trial(before, parsed(before, item, lake_root=lake))
    assert any("share the identifier" in p for p in answer_problems(before, after, {"dup"}, lake, 100))


def test_answer_problems_check_an_accepted_dataset_whose_constraint_changed(lake):
    (lake / "spending.csv").write_text("household_id,spent\nH1,100\nH1,150\n")
    spent = [{"kind": "concept", "concept": {"concept": "SpentUsd", "type": "ValueType", "extends": ["Decimal"]}},
             {"kind": "relationship", "concept": "Household", "relationship": {
                 "name": "spent", "roles": [{"concept": "SpentUsd"}], "verbalizes": ["{Household} spent {SpentUsd}"]}},
             {"kind": "dataset", "dataset": {"name": "spending", "path": "spending.csv", "fields": [
                 {"name": "household_id", "column": "household_id", "datatype": "String"},
                 {"name": "spent", "column": "spent", "datatype": "Decimal"}], "binding": [
                 {"concept": "Household", "identify": {"household_id": "household_id"},
                  "relationships": {"spent": ["spent"]}}]}}]
    before = accept(income_harness(lake), spent, lake)
    stricter = copy.deepcopy(spent[1])
    stricter["relationship"]["multiplicity"] = "ManyToOne"
    after = trial(before, parsed(before, stricter, lake_root=lake))
    assert any("spending" in p and "ManyToOne" in p for p in answer_problems(before, after, set(), lake, 100))


def test_repeating_an_accepted_dataset_is_a_noop(lake):
    harness = income_harness(lake)
    kind, payload = parse_claim(INCOME_DATASET, harness, lake)
    assert is_noop(kind, payload, harness)
    assert not is_noop(*parse_claim(dataset_claim(description="Other"), harness, lake), harness)


def test_applying_a_claim_marks_anchored_experience_stale(lake):
    harness = income_harness(lake)
    harness.experience.append(ExperienceEntry(id="e1", kind="data_quirk", text="Top coded", anchors=["Household"]))
    kind, payload = parse_claim({"kind": "concept", "concept": {"concept": "Household", "type": "EntityType",
                                                                 "identify_by": ["household_id"], "description": "A home"}},
                                harness, lake)
    apply_claim(Claim(id="claim-000099", kind=kind, payload=payload, status="accepted"), harness)
    assert harness.experience[0].stale


def test_anchors_name_concepts_relationships_datasets_fields_and_paths(lake):
    harness = Harness.empty("toy")
    kind, payload = parse_claim(INCOME_DATASET, harness, lake)
    assert {"income", "income.household_id", "income.csv", "README.md"} <= set(
        anchors(Claim(id="c", kind=kind, payload=payload)))
    kind, payload = parse_claim(INCOME_CLAIMS[4], harness, lake)
    assert anchors(Claim(id="c", kind=kind, payload=payload)) == ["Household.annual_income", "Household"]


# Final review fixes.

STATES = [
    {"kind": "concept", "concept": {"concept": "StateName", "type": "ValueType", "extends": ["String"]}},
    {"kind": "concept", "concept": {"concept": "StateNumber", "type": "ValueType", "extends": ["Integer"]}},
    {"kind": "concept", "concept": {"concept": "State", "type": "EntityType", "identify_by": ["name"]}},
    {"kind": "relationship", "concept": "State", "relationship": {
        "name": "name", "roles": [{"concept": "StateName"}], "multiplicity": "OneToOne",
        "verbalizes": ["{State} is named {StateName}"]}},
    {"kind": "concept", "concept": {"concept": "Capital", "type": "EntityType", "extends": ["State"]}},
    {"kind": "dataset", "dataset": {"name": "capitals", "path": "capitals.csv", "fields": [
        {"name": "capital", "column": "capital", "datatype": "String"}],
        "rows": "Capital", "binding": [{"concept": "Capital", "identify": {"name": "capital"}}]}},
]


def test_an_inherited_identifier_change_is_checked_in_datasets_of_the_subtype(lake):
    (lake / "capitals.csv").write_text("capital\nSacramento\n")
    before = accept(Harness.empty("toy"), STATES, lake)
    renumbered = {"kind": "relationship", "concept": "State", "relationship": {
        "name": "name", "roles": [{"concept": "StateNumber"}], "multiplicity": "OneToOne",
        "verbalizes": ["{State} is numbered {StateNumber}"]}}
    after = trial(before, parsed(before, renumbered, lake_root=lake))
    assert "capitals" in affected_datasets(before, after)
    assert any("capitals" in p and "does not fit" in p for p in answer_problems(before, after, set(), lake, 100))


def test_answer_problems_stop_at_ontology_problems_instead_of_crashing(lake):
    harness = income_harness(lake)
    broken = {"kind": "concept", "concept": {"concept": "Household", "type": "EntityType", "identify_by": ["code"]}}
    rebound = {"kind": "binding", "dataset": "income", "rows": "Household", "binding": [
        {"concept": "Household", "identify": {"code": "household_id"}, "relationships": {"annual_income": ["annual_income"]}}]}
    after = trial(harness, parsed(harness, broken, rebound, lake_root=lake))
    problems = answer_problems(harness, after, set(), lake, 100)
    assert any("not a Relationship of Household" in p for p in problems)


def test_a_merge_renames_only_whole_placeholders(lake):
    (lake / "sites.csv").write_text("site,elevation\nA,10\n")
    claims = [
        {"kind": "concept", "concept": {"concept": "StationElevation", "type": "ValueType", "extends": ["Decimal"]}},
        {"kind": "concept", "concept": {"concept": "SiteId", "type": "ValueType", "extends": ["String"]}},
        *[{"kind": "concept", "concept": {"concept": name, "type": "EntityType", "identify_by": ["code"]}}
          for name in ("Site", "Station")],
        *[{"kind": "relationship", "concept": name, "relationship": {
            "name": "code", "roles": [{"concept": "SiteId"}], "multiplicity": "OneToOne",
            "verbalizes": [f"{{{name}}} has code {{SiteId}}"]}} for name in ("Site", "Station")],
        {"kind": "relationship", "concept": "Station", "relationship": {
            "name": "elevation", "roles": [{"concept": "StationElevation"}], "multiplicity": "ManyToOne",
            "verbalizes": ["{Station} has elevation {StationElevation}"]}},
    ]
    before = accept(Harness.empty("toy"), claims, lake)
    merge = {"kind": "merge", "from": "Station", "into": "Site", "relationships": {"code": "code"}}
    after = trial(before, parsed(before, merge, lake_root=lake))
    elevation = next(r for r in after.concept("Site").relationships if r.name == "elevation")
    assert elevation.verbalizes == ["{Site} has elevation {StationElevation}"]
    assert answer_problems(before, after, set(), lake, 100) == []


def test_replacing_a_relationship_under_its_own_name_retypes_its_bindings(lake):
    (lake / "people.csv").write_text("person,household\nP1,H1\n")
    before = accept(income_harness(lake), PEOPLE, lake)
    retyped = {"kind": "relationship", "concept": "Person", "replaces": "household_ref", "relationship": {
        "name": "household_ref", "roles": [{"concept": "Household"}], "multiplicity": "ManyToOne",
        "verbalizes": ["{Person} lives in {Household}"]}}
    after = trial(before, parsed(before, retyped, lake_root=lake))
    assert after.bindings["people"].binding[0].relationships == {"household_ref": [{"household_id": "household"}]}
    assert answer_problems(before, after, set(), lake, 100) == []


def test_an_optional_path_group_that_does_not_match_is_an_error(lake):
    (lake / "states").mkdir()
    (lake / "states" / "Alabama.csv").write_text("household_id,annual_income\nH1,1\n")
    state = {"name": "year", "from_path": True, "datatype": "String"}
    item = dataset_claim(name="alabama", path="states/Alabama.csv", end_row=None,
                         path_pattern=r"states/(?P<state_name>[A-Za-z]+)(?P<year>\d{4})?\.csv",
                         fields=[*INCOME_FIELDS, state, {"name": "state_name", "from_path": True, "datatype": "String"}],
                         unbound={"year": "Kept", "state_name": "Kept"})
    with pytest.raises(ValueError, match="year"):
        parse_claim(item, Harness.empty("toy"), lake)


def test_answer_problems_read_each_file_once(lake, monkeypatch):
    import ogda.lake as lake_module

    (lake / "dup.csv").write_text("household_id,annual_income\nH1,100\nH2,200\n")
    before = income_harness(lake)
    item = dataset_claim(name="dup", path="dup.csv", end_row=None, origin="inferred", documentation=[])
    after = trial(before, parsed(before, item, lake_root=lake))
    reads, real = [], lake_module.read_rows
    monkeypatch.setattr(lake_module, "read_rows",
                        lambda path, sheet, count=None: reads.append(path.name) or real(path, sheet, count))
    assert answer_problems(before, after, {"dup"}, lake, 100) == []
    assert reads == ["dup.csv"]
