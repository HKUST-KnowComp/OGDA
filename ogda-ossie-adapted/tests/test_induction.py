"""Schema induction: Datasets and their Bindings, the checks on an answer, verification, File dispositions, fan-out
and resume. Runs stop with max_passes=0 once induction is done, before the concept layer."""

import copy
import json

import openpyxl
import pytest
from smolagents.utils import AgentGenerationError

from helpers import ANNUAL_INCOME_REL, INCOME_CLAIMS, INCOME_DATASET, Script, answer, code, construction, make_lake, verdict
from ogda.induction import locate
from ogda.ossie import MappingDocument, ogda_extension, read_yaml

README = {"path": "README.md", "status": "documentation", "reason": "Defines the income Fields"}
MAPPED = {"path": "income.csv", "status": "mapped", "complete": True, "datasets": ["income"]}


def induction(claims=None, *, dispositions=None, summary="Income table", **extra):
    claims = [*INCOME_CLAIMS, INCOME_DATASET] if claims is None else claims
    dispositions = [MAPPED, README] if dispositions is None else dispositions
    return {"claims": claims, "summary": summary, "unresolved": [], "file_dispositions": dispositions, **extra}


def inferred(item=INCOME_DATASET, **changes):
    claim = copy.deepcopy(item)
    claim["dataset"].update(origin="inferred", documentation=[], **changes)
    return claim


def dataset(name, path, fields, *, binding=None, rows=None, unbound=None, **region):
    return {"kind": "dataset", "dataset": {
        "name": name, "path": path, "origin": "inferred", "documentation": [], "description": f"Rows of {path}",
        "fields": [{"name": f, "column": f, "datatype": "Decimal" if f == "annual_income" else "String"} for f in fields],
        "rows": rows, "binding": binding or [{"concept": "Household", "identify": {"household_id": "household_id"},
                                              "relationships": {"annual_income": ["annual_income"]}}],
        "unbound": unbound or {}, **region}}


def mapped(path, *names):
    return {"path": path, "status": "mapped", "complete": True, "datasets": list(names)}


def events(built, name):
    return [e for e in map(json.loads, (built.log_root / "construction.jsonl").read_text().splitlines())
            if e["event"] == name]


def task(prompt):
    # ScriptedClient serializes the user task after smolagents' role preamble.
    for message in json.loads(prompt):
        content = message["content"]
        if '{"stage":' in content:
            return json.JSONDecoder().raw_decode(content[content.index('{"stage":'):])[0]
    raise AssertionError("No explicit construction task found")


def statuses(harness):
    return {path: item["status"] for path, item in harness.metadata["progress"]["files"].items()}


def build(tmp_path, script, lake=None, **settings):
    built = construction(tmp_path, script, max_passes=0, **settings)
    built.start("toy", lake or make_lake(tmp_path))
    return built


def routes(built):
    return [e["acceptance_route"] for e in events(built, "claim_outcome")]


def test_a_dataset_answer_is_checked_verified_and_saved_as_ossie_documents(tmp_path):
    script = Script([answer(induction([*INCOME_CLAIMS, inferred()]))])
    built = build(tmp_path, script)
    harness = built.run()
    assert harness.metadata["stop_reason"] == "max_passes" and list(harness.datasets()) == ["income"]
    assert statuses(harness) == {"README.md": "documentation", "income.csv": "mapped"}
    assert read_yaml(built.store.mapping_path, MappingDocument).concept_mappings[0].concept == "Household"
    assert len(script.prompts["verifier"]) == 1 and task(script.prompts["verifier"][0])["stage"] == "induction"
    assert routes(built) == ["ontology"] * 5 + ["verifier"]
    assert task(script.prompts["proposer"][0])["ontology"] == []


def test_concepts_are_accepted_at_once_and_datasets_wait_for_the_verifier(tmp_path):
    script = Script([answer(induction([*INCOME_CLAIMS, inferred()]))], verifier=[verdict("refuted", "Not households")])
    harness = build(tmp_path, script).run()
    assert harness.concept("Household") is not None and harness.datasets() == {}
    assert statuses(harness)["income.csv"] == "unresolved"


def test_a_declared_dataset_skips_the_verifier(tmp_path):
    script = Script([answer(induction())])
    built = build(tmp_path, script)
    assert list(built.run().datasets()) == ["income"]
    assert script.prompts["verifier"] == [] and routes(built)[-1] == "documentation"


SAME_AS = {"kind": "dataset", "dataset": {"name": "income_2021", "path": "income_2021.csv", "header_row": 1,
                                          "start_row": 2, "same_as": "income"}}


def test_a_same_as_dataset_skips_the_verifier(tmp_path):
    lake = make_lake(tmp_path)
    (lake / "income_2021.csv").write_text("Income 2021\nhousehold_id,annual_income\nH1,110\n")
    script = Script([answer(induction()), answer({"claims": [SAME_AS],
                                                  "file_dispositions": [mapped("income_2021.csv", "income_2021")]})])
    built = build(tmp_path, script, lake, files_per_visit=2)
    harness = built.run()
    assert harness.binding_of("income_2021") == harness.binding_of("income")
    assert script.prompts["verifier"] == [] and routes(built)[-1] == "structural"


def test_a_json_dataset_goes_to_the_verifier_even_with_rows_or_same_as(tmp_path):
    lake = make_lake(tmp_path)
    (lake / "records.json").write_text('[{"household_id": "H3", "annual_income": 300}, '
                                       '{"household_id": "H3", "annual_income": 300}]')
    records = {"kind": "dataset", "dataset": {"name": "records", "path": "records.json", "header_row": None,
                                              "start_row": 0, "same_as": "income"}}
    script = Script([answer(induction()), answer({"claims": [records],
                                                  "file_dispositions": [mapped("records.json", "records")]})])
    built = build(tmp_path, script, lake, files_per_visit=2)
    harness = built.run()
    assert "records" in harness.datasets() and len(script.prompts["verifier"]) == 1
    assert routes(built)[-1] == "verifier"


BUILTIN_ROLE = copy.deepcopy(ANNUAL_INCOME_REL)
BUILTIN_ROLE["relationship"]["roles"] = [{"concept": "Decimal"}]
UNBOUND = inferred(binding=[{"concept": "Household", "identify": {"household_id": "household_id"}}])
MISSING_COLUMN = inferred(fields=[{"name": "household_id", "column": "household_id", "datatype": "String"},
                                  {"name": "annual_income", "column": "income", "datatype": "Decimal"}])


@pytest.mark.parametrize("claims, message", [
    ([*INCOME_CLAIMS, MISSING_COLUMN], "has no columns ['income']"),
    ([*INCOME_CLAIMS, UNBOUND], "neither bound nor in unbound"),
    ([*INCOME_CLAIMS[:4], BUILTIN_ROLE, inferred()], "built-in"),
])
def test_answer_errors_go_back_to_the_proposer_and_a_corrected_answer_is_saved(tmp_path, claims, message):
    script = Script([answer(induction(claims)), answer(induction([*INCOME_CLAIMS, inferred()]))])
    harness = build(tmp_path, script).run()
    assert message in script.prompts["proposer"][1] and "earlier answers stay saved" in script.prompts["proposer"][1]
    assert list(harness.datasets()) == ["income"] and statuses(harness)["income.csv"] == "mapped"


def test_a_duplicate_identifier_goes_back_and_a_weaker_binding_is_saved(tmp_path):
    lake = make_lake(tmp_path)
    (lake / "income.csv").write_text("household_id,annual_income\nH1,100\nH1,150\n")
    weaker = copy.deepcopy(ANNUAL_INCOME_REL)
    del weaker["relationship"]["multiplicity"]
    script = Script([answer(induction([*INCOME_CLAIMS, inferred(end_row=None)])),
                     answer(induction([*INCOME_CLAIMS[:4], weaker, inferred(end_row=None, rows=None)]))])
    harness = build(tmp_path, script, lake).run()
    assert "share the identifier (household_id)" in script.prompts["proposer"][1]
    assert harness.bindings["income"].rows is None and statuses(harness)["income.csv"] == "mapped"


def other_lake(tmp_path, rows="H3,300,2\n"):
    lake = make_lake(tmp_path)
    (lake / "other.csv").write_text("household_id,annual_income,size\n" + rows)
    return lake


OTHER = dataset("other", "other.csv", ["household_id", "annual_income", "size"],
                unbound={"size": "Household size; no Concept yet"})


def test_a_later_dataset_that_breaks_an_accepted_many_to_one_is_rejected(tmp_path):
    weaker = copy.deepcopy(ANNUAL_INCOME_REL)
    del weaker["relationship"]["multiplicity"]
    script = Script([answer(induction()), answer({"claims": [OTHER], "file_dispositions": [mapped("other.csv", "other")]}),
                     answer({"claims": [weaker, OTHER], "file_dispositions": [mapped("other.csv", "other")]})])
    harness = build(tmp_path, script, other_lake(tmp_path, "H3,300,2\nH3,310,2\n"), files_per_visit=2).run()
    assert "Household.annual_income is ManyToOne" in script.prompts["proposer"][2]
    assert harness.concept("Household").relationships[1].multiplicity is None
    assert set(harness.datasets()) == {"income", "other"}


def test_a_dataset_name_that_maps_another_file_is_rejected(tmp_path):
    clash = copy.deepcopy(OTHER)
    clash["dataset"]["name"] = "income"
    script = Script([answer(induction()), answer({"claims": [clash], "file_dispositions": [mapped("other.csv", "income")]}),
                     answer({"claims": [OTHER], "file_dispositions": [mapped("other.csv", "other")]})])
    harness = build(tmp_path, script, other_lake(tmp_path), files_per_visit=2).run()
    assert "Dataset name income already maps another File" in script.prompts["proposer"][2]
    assert set(harness.datasets()) == {"income", "other"}


def test_a_refuted_dataset_file_is_retried_once_with_the_notes(tmp_path):
    notes = "household_id does not identify rows"
    script = Script([answer(induction([*INCOME_CLAIMS, inferred()])),
                     answer(induction([*INCOME_CLAIMS, inferred(description="Second try")]))],
                    verifier=[verdict("refuted", notes), verdict("verified")])
    built = build(tmp_path, script)
    harness = built.run()
    retry = task(script.prompts["proposer"][1])
    path = str(built.store.lake_root / "income.csv")
    assert retry["files"] == [path] and retry["retry"] == {path: [f"dataset Claim refuted: {notes}"]}
    assert retry["decided_proposals"][0]["notes"] == notes
    assert harness.dataset("income").description == "Second try" and statuses(harness)["income.csv"] == "mapped"


def test_a_second_refutation_is_final(tmp_path):
    script = Script([answer(induction([*INCOME_CLAIMS, inferred()])),
                     answer(induction([*INCOME_CLAIMS, inferred(description="Second try")]))],
                    verifier=[verdict("refuted", "Wrong rows")])
    built = build(tmp_path, script)
    harness = built.run()
    assert statuses(harness)["income.csv"] == "unresolved" and len(events(built, "file_retry")) == 1
    assert harness.metadata["stop_reason"] == "max_passes"


def judged(status, **corrections):
    return answer({"verdicts": [{"claim_id": "claim-000006", "status": status, "notes": "Rows hold", **corrections}]})


def test_the_verifier_corrects_descriptions_and_datatypes(tmp_path):
    fixed = "Annual household income in US dollars, e.g. 100, 200"
    script = Script([answer(induction([*INCOME_CLAIMS, inferred()]))],
                    verifier=[judged("refuted", descriptions={"income.annual_income": fixed}),
                              judged("verified", descriptions={"income.nope": "x"}),
                              judged("verified", descriptions={"income.annual_income": fixed},
                                     datatypes={"income.annual_income": "Integer"})])
    built = build(tmp_path, script)
    harness = built.run()
    assert "corrected descriptions go only with a verified verdict" in script.prompts["verifier"][1]
    assert "['income.nope'] are not Fields of this Dataset" in script.prompts["verifier"][2]
    assert harness.fields()["income.annual_income"].description == fixed
    assert harness.fields()["income.annual_income"].datatype == "Integer"
    assert [(e["field_id"], e["new"]) for e in events(built, "description_corrected")] == [("income.annual_income", fixed)]
    assert [(e["field_id"], e["old"], e["new"]) for e in events(built, "datatype_corrected")] == [
        ("income.annual_income", "Decimal", "Integer")]


def test_a_datatype_correction_that_breaks_the_binding_is_rejected(tmp_path):
    script = Script([answer(induction([*INCOME_CLAIMS, inferred()]))],
                    verifier=[judged("verified", datatypes={"income.annual_income": "Boolean"}), judged("verified")])
    harness = build(tmp_path, script).run()
    assert "does not fit IncomeUsd" in script.prompts["verifier"][1]
    assert harness.fields()["income.annual_income"].datatype == "Decimal"


def test_several_answers_in_one_visit_each_save_and_a_rejected_one_leaves_earlier_ones(tmp_path):
    other = {"claims": [OTHER], "file_dispositions": [mapped("other.csv", "other")]}
    wrong = copy.deepcopy(other)
    wrong["claims"][0]["dataset"]["end_row"] = 9  # other.csv has two rows.
    again = {"file_dispositions": [{"path": "income.csv", "status": "unresolved", "reason": "Second thoughts"}]}
    script = Script([code(f"submit({induction()!r})"), code(f"submit({wrong!r})"), code(f"submit({again!r})"),
                     answer(other)])
    built = build(tmp_path, script, other_lake(tmp_path), max_steps=5)
    harness = built.run()
    prompts = script.prompts["proposer"]
    assert "earlier answers stay saved" in prompts[2] and "do not fit" in prompts[2]
    assert "['income.csv'] already have a different disposition in this visit" in prompts[3]
    assert statuses(harness) == {"README.md": "documentation", "income.csv": "mapped", "other.csv": "mapped"}
    assert sum(v["stage"] == "induction" for v in events(built, "iteration_start")) == 1


def test_a_mapped_file_must_cover_every_worksheet(tmp_path):
    lake = make_lake(tmp_path)
    (lake / "income.csv").unlink()
    workbook = openpyxl.Workbook()
    workbook.active.title = "z-first"
    for sheet in (workbook.active, workbook.create_sheet("a-second")):
        sheet.append(["household_id", "annual_income"])
        sheet.append(["H1", 100])
    workbook.save(lake / "income.xlsx")
    workbook_dataset = copy.deepcopy(INCOME_DATASET)
    workbook_dataset["dataset"].update(path="income.xlsx", end_row=None)
    script = Script([answer(induction([*INCOME_CLAIMS, workbook_dataset],
                                      dispositions=[mapped("income.xlsx", "income"), README]))])
    harness = build(tmp_path, script, lake).run()
    assert "worksheets ['a-second'] are in no Dataset" in script.prompts["proposer"][1]
    assert "income.xlsx" not in statuses(harness) and harness.datasets() == {}
    assert harness.metadata["stop_reason"] == "proposer_incomplete"


def test_a_file_without_a_submitted_disposition_stays_in_the_queue(tmp_path):
    script = Script([answer({"claims": [*INCOME_CLAIMS, INCOME_DATASET]}), answer({"file_dispositions": [MAPPED]})])
    built = build(tmp_path, script)
    harness = built.run()
    assert [v["files"] for v in events(built, "iteration_start")] == [["README.md", "income.csv"], ["income.csv"]]
    assert statuses(harness) == {"README.md": "documentation", "income.csv": "mapped"}


def test_a_visit_that_submits_nothing_stops_for_an_explicit_resume(tmp_path):
    script = Script([code("print('Still considering')")])
    built = build(tmp_path, script)
    harness = built.run()
    assert harness.metadata["stop_reason"] == "proposer_incomplete" and statuses(harness) == {}
    assert events(built, "proposer_incomplete")[0]["remaining_files"] == ["README.md", "income.csv"]
    resumed = construction(tmp_path, Script([answer(induction())]), max_passes=0)
    assert statuses(resumed.run()) == {"README.md": "documentation", "income.csv": "mapped"}


def test_a_visit_that_accepts_claims_but_finishes_no_file_is_progress(tmp_path):
    script = Script([answer({"claims": INCOME_CLAIMS, "summary": "Concepts first"}), answer(induction([INCOME_DATASET]))])
    built = build(tmp_path, script)
    harness = built.run()
    assert [v["files"] for v in events(built, "iteration_start")] == [["README.md", "income.csv"]] * 2
    assert harness.metadata["stop_reason"] == "max_passes"
    assert statuses(harness) == {"README.md": "documentation", "income.csv": "mapped"}


def test_the_dataset_limit_counts_only_dataset_claims(tmp_path):
    both = induction([*INCOME_CLAIMS, INCOME_DATASET, OTHER], dispositions=[MAPPED, README, mapped("other.csv", "other")])
    later = induction([OTHER], dispositions=[mapped("other.csv", "other")])
    script = Script([answer(both), answer(induction()), answer(later)])
    built = build(tmp_path, script, other_lake(tmp_path), files_per_visit=10, max_datasets_per_visit=1)
    harness = built.run()
    assert any("Submit at most 1 Datasets in this visit" in prompt for prompt in script.prompts["proposer"])
    assert [v["files"] for v in events(built, "iteration_start")] == [["README.md", "income.csv", "other.csv"],
                                                                      ["other.csv"]]
    assert statuses(harness) == {"README.md": "documentation", "income.csv": "mapped", "other.csv": "mapped"}


def test_resume_verifies_only_outstanding_claims_without_rerunning_the_proposer(tmp_path):
    def crash(prompt):
        raise RuntimeError("worker host failed")

    lake = make_lake(tmp_path)
    (lake / "survey.csv").write_text("household_id,size\nH1,3\n")
    survey = dataset("survey", "survey.csv", ["household_id", "size"], rows="Household",
                     binding=[{"concept": "Household", "identify": {"household_id": "household_id"}}],
                     unbound={"size": "Household size; no Concept yet"})
    partial = code("submit_verdicts([{'claim_id': 'claim-000006', 'status': 'verified', "
                   "'notes': 'Columns match income.csv', 'evidence_refs': ['income.csv']}])")
    claims = [*INCOME_CLAIMS, inferred(), survey]
    script = Script([answer(induction(claims, dispositions=[MAPPED, README, mapped("survey.csv", "survey")]))],
                    verifier=[partial, crash])
    built = build(tmp_path, script, lake, files_per_visit=10)
    with pytest.raises(AgentGenerationError):
        built.run()
    saved = built.store.load()
    assert saved.claims["claim-000006"].status == "verified" and saved.claims["claim-000007"].status == "pending"
    resume_script = Script([code("print('unused')")], verifier=[verdict("verified")])
    harness = construction(tmp_path, resume_script, max_passes=0).run()
    assert resume_script.prompts["proposer"] == []
    assert "claim-000007" in resume_script.prompts["verifier"][0]
    assert "claim-000006" not in resume_script.prompts["verifier"][0]
    assert statuses(harness)["survey.csv"] == "mapped"


def test_budget_stops_construction_and_saves_progress(tmp_path):
    built = build(tmp_path, Script([code("print('inspection')")]), max_tokens=1)
    assert built.run().metadata["stop_reason"] == "budget"
    assert built.store.load().metadata["progress"]["pending_iteration"]["phase"] == "propose"
    resumed = construction(tmp_path, Script([answer(induction())]), max_passes=0)
    assert resumed.run().metadata["stop_reason"] == "max_passes"
    assert len(events(resumed, "iteration_start")) == 1  # Resume finishes the interrupted Iteration.


def test_an_answer_with_unknown_keys_is_rejected(tmp_path):
    script = Script([answer(INCOME_DATASET), answer(induction())])
    harness = build(tmp_path, script).run()
    assert "Unknown answer keys ['dataset', 'kind']" in script.prompts["proposer"][1]
    assert list(harness.datasets()) == ["income"]


# Layout groups: fan-out and locators.

def state_lake(tmp_path, texas=True):
    lake = tmp_path / "original"
    (lake / "states").mkdir(parents=True)
    (lake / "states" / "README.md").write_text("Reports per Area; each File is one State.\n")
    (lake / "states" / "Alabama.csv").write_text("Area,Reports\nX,1\n")
    (lake / "states" / "Alaska.csv").write_text("Area,Reports\nY,2\n")
    (lake / "states" / "Arizona.csv").write_text("Area,Reports\nX,1\n")
    if texas:
        (lake / "texas").mkdir()
        (lake / "texas" / "Texas.csv").write_text("Area,Reports\nZ,3\n")
    return lake


def value_type(name, base):
    return {"kind": "concept", "concept": {"concept": name, "type": "ValueType", "extends": [base]}}


def relationship(owner, name, role, multiplicity, sentence):
    return {"kind": "relationship", "concept": owner, "relationship": {
        "name": name, "roles": [{"concept": role}], "multiplicity": multiplicity, "verbalizes": [sentence]}}


REPORT_CLAIMS = [
    value_type("StateName", "String"), value_type("AreaName", "String"), value_type("ReportCount", "Integer"),
    {"kind": "concept", "concept": {"concept": "State", "type": "EntityType", "identify_by": ["name"]}},
    {"kind": "concept", "concept": {"concept": "Area", "type": "EntityType", "identify_by": ["name", "state"]}},
    relationship("State", "name", "StateName", "OneToOne", "{State} is named {StateName}"),
    relationship("Area", "name", "AreaName", "ManyToOne", "{Area} is named {AreaName}"),
    relationship("Area", "state", "State", "ManyToOne", "{Area} lies in {State}"),
    relationship("Area", "reports", "ReportCount", "ManyToOne", "{Area} had {ReportCount} reports"),
]
REPORTS = {"kind": "dataset", "dataset": {
    "name": "reports_alabama", "path": "states/Alabama.csv", "origin": "declared", "documentation": ["states/README.md"],
    "description": "Reports per Area of one State", "path_pattern": r"states/(?P<state>[^/]+)\.csv",
    "fields": [{"name": "state", "from_path": True, "datatype": "String", "description": "The State of the File"},
               {"name": "area", "column": "Area", "datatype": "String", "description": "Area name, e.g. X"},
               {"name": "reports", "column": "Reports", "datatype": "Integer", "description": "Reports, e.g. 1"}],
    "rows": "Area", "binding": [{"concept": "Area", "identify": {"name": "area", "state": {"name": "state"}},
                                 "relationships": {"reports": ["reports"]}}]}}
STATE_DOC = {"path": "states/README.md", "status": "documentation", "reason": "Defines the reports"}
ALABAMA = mapped("states/Alabama.csv", "reports_alabama")


def reports_answer(*extra, **options):
    return answer(induction([*REPORT_CLAIMS, REPORTS, *extra], dispositions=[ALABAMA, STATE_DOC], **options))


def same_as(name, path, **region):
    return {"kind": "dataset", "dataset": {"name": name, "path": path, "same_as": "reports_alabama", **region}}


def test_a_layout_group_is_induced_from_its_representative(tmp_path):
    texas = {"path": "texas/Texas.csv", "status": "unresolved", "reason": "Outside the states path pattern"}
    script = Script([reports_answer(), answer({"file_dispositions": [texas]})])
    built = build(tmp_path, script, state_lake(tmp_path))
    harness = built.run()
    root = built.store.lake_root
    assert task(script.prompts["proposer"][0])["layout_groups"] == {str(root / "states/Alabama.csv"): {
        "files": 4, "members": ["states/Alaska.csv", "states/Arizona.csv", "texas/Texas.csv"], "more": 0}}
    assert set(harness.datasets()) == {"reports_alabama", "reports_alabama__alaska", "reports_alabama__arizona"}
    assert harness.fields()["reports_alabama__alaska.state"].expression.sql == "'Alaska'"
    assert harness.binding_of("reports_alabama__alaska") == harness.bindings["reports_alabama"]
    assert "reports_alabama__alaska" not in harness.bindings
    arizona = harness.dataset("reports_alabama__arizona")
    assert (ogda_extension(arizona).relation, ogda_extension(arizona).relation_to) == ("copy", "reports_alabama")
    assert ogda_extension(arizona).layout_of == "reports_alabama"
    assert "layout" in routes(built)
    released = events(built, "layout_group_mapped")[0]["released"]
    assert list(released) == ["texas/Texas.csv"] and "does not match path_pattern" in released["texas/Texas.csv"][0]
    assert [v["files"] for v in events(built, "iteration_start")][1] == ["texas/Texas.csv"]
    assert statuses(harness) == {"states/README.md": "documentation", "states/Alabama.csv": "mapped",
                                 "states/Alaska.csv": "mapped", "states/Arizona.csv": "mapped",
                                 "texas/Texas.csv": "unresolved"}
    area = next(m for m in read_yaml(built.store.mapping_path, MappingDocument).concept_mappings if m.concept == "Area")
    assert len(area.object_mappings) == 3 and harness.metadata["progress"]["groups"] == {}


def test_a_member_that_fails_a_data_check_joins_the_queue(tmp_path):
    lake = state_lake(tmp_path, texas=False)
    (lake / "states" / "Alaska.csv").write_text("Area,Reports\nY,2\nY,3\n")
    alaska = {"path": "states/Alaska.csv", "status": "unresolved", "reason": "Areas repeat"}
    built = build(tmp_path, Script([reports_answer(), answer({"file_dispositions": [alaska]})]), lake)
    harness = built.run()
    released = events(built, "layout_group_mapped")[0]["released"]
    assert "share the identifier" in released["states/Alaska.csv"][0]
    assert [v["files"] for v in events(built, "iteration_start")][1] == ["states/Alaska.csv"]
    assert statuses(harness)["states/Alaska.csv"] == "unresolved"


def test_separate_members_are_induced_on_their_own(tmp_path):
    script = Script([reports_answer(separate=["states/Alaska.csv"]),
                     answer({"claims": [same_as("alaska_part", "states/Alaska.csv")],
                             "file_dispositions": [mapped("states/Alaska.csv", "alaska_part")]})])
    built = build(tmp_path, script, state_lake(tmp_path, texas=False))
    harness = built.run()
    assert [v["files"] for v in events(built, "iteration_start")][1] == ["states/Alaska.csv"]
    assert harness.fields()["alaska_part.state"].expression.sql == "'Alaska'"
    assert routes(built)[-1] == "structural"


def test_members_of_an_unmapped_representative_are_unresolved(tmp_path):
    failed = {"path": "states/Alabama.csv", "status": "unresolved", "reason": "Unclear columns"}
    built = build(tmp_path, Script([answer(induction([], dispositions=[failed, STATE_DOC]))]),
                  state_lake(tmp_path, texas=False))
    harness = built.run()
    assert "Same layout as states/Alabama.csv, which was not mapped: Unclear columns" in (
        harness.metadata["progress"]["files"]["states/Alaska.csv"]["reason"])
    assert len(events(built, "iteration_start")) == 1


def test_regions_that_end_early_fan_out_to_members_with_the_same_row_count(tmp_path):
    lake = state_lake(tmp_path, texas=False)
    (lake / "states" / "Alabama.csv").write_text("Area,Reports\nX,1\nTotal,1\n")
    (lake / "states" / "Alaska.csv").write_text("Area,Reports\nY,2\nTotal,2\n")
    (lake / "states" / "Arizona.csv").write_text("Area,Reports\nX,1\nZ,4\nTotal,5\n")
    ending = copy.deepcopy(REPORTS)
    ending["dataset"]["end_row"] = 2
    arizona = answer({"claims": [same_as("reports_arizona", "states/Arizona.csv", end_row=3)],
                      "file_dispositions": [mapped("states/Arizona.csv", "reports_arizona")]})
    script = Script([answer(induction([*REPORT_CLAIMS, ending], dispositions=[ALABAMA, STATE_DOC])), arizona])
    built = build(tmp_path, script, lake)
    harness = built.run()
    assert ogda_extension(harness.dataset("reports_alabama__alaska")).end_row == 2
    released = events(built, "layout_group_mapped")[0]["released"]
    assert list(released) == ["states/Arizona.csv"] and "row count differs" in released["states/Arizona.csv"][0]
    assert set(statuses(harness).values()) == {"mapped", "documentation"}


def test_members_the_proposer_mapped_keep_their_own_datasets(tmp_path):
    alaska = copy.deepcopy(REPORTS)
    alaska["dataset"].update(name="reports_alaska", path="states/Alaska.csv")
    built = build(tmp_path, Script([reports_answer(alaska)]), state_lake(tmp_path, texas=False))
    harness = built.run()
    assert set(harness.datasets()) == {"reports_alabama", "reports_alaska", "reports_alabama__arizona"}
    assert harness.metadata["progress"]["files"]["states/Alaska.csv"] == {
        "status": "mapped", "complete": True, "datasets": ["reports_alaska"], "reason": "Mapped by its own Datasets"}
    assert events(built, "layout_group_mapped")[0]["proposed"] == 1


LOCATOR = ("def find_region(rows):\n"
           "    header = next(i for i, row in enumerate(rows) if row[:1] == ['Area'])\n"
           "    blank = [i for i in range(header + 1, len(rows)) if not any(cell.strip() for cell in rows[i])]\n"
           "    if not blank:\n"
           "        raise ValueError('no blank row after the data')\n"
           "    return header, header + 1, blank[0]\n")


def report_lake(tmp_path):
    """State Files with one layout: a title, a header, a different number of rows, a blank row and a footnote."""
    lake = tmp_path / "original"
    (lake / "states").mkdir(parents=True)
    (lake / "states" / "README.md").write_text("Reports per Area; each File is one State.\n")
    for name, areas, footnote in [("Alabama", "XY", True), ("Alaska", "Z", True), ("Arizona", "XWVU", True),
                                  ("Arkansas", "T", False)]:
        rows = "".join(f"{area},{n}\n" for n, area in enumerate(areas, 1))
        (lake / "states" / f"{name}.csv").write_text(
            "Reports by Area,\n,\nArea,Reports\n" + rows + (",\nSource: FTC,\n" if footnote else ""))
    ending = copy.deepcopy(REPORTS)
    ending["dataset"].update(header_row=2, start_row=3, end_row=5)
    arkansas = answer({"claims": [same_as("reports_arkansas", "states/Arkansas.csv", header_row=2, start_row=3)],
                       "file_dispositions": [mapped("states/Arkansas.csv", "reports_arkansas")]})
    return lake, ending, arkansas


def test_a_locator_places_each_members_region_and_releases_members_it_fails_on(tmp_path):
    lake, ending, arkansas = report_lake(tmp_path)
    first = answer(induction([*REPORT_CLAIMS, ending], dispositions=[ALABAMA, STATE_DOC],
                             locators={"reports_alabama": LOCATOR}))
    built = build(tmp_path, Script([first, arkansas]), lake)
    harness = built.run()
    regions = {ogda_extension(d).path: (ogda_extension(d).header_row, ogda_extension(d).start_row,
                                        ogda_extension(d).end_row) for d in harness.datasets().values()}
    assert regions == {"states/Alabama.csv": (2, 3, 5), "states/Alaska.csv": (2, 3, 4),
                       "states/Arizona.csv": (2, 3, 7), "states/Arkansas.csv": (2, 3, None)}
    fanned = events(built, "layout_group_mapped")[0]
    assert fanned["located"] == ["reports_alabama"] and fanned["mapped"] == 2
    assert "no blank row after the data" in fanned["released"]["states/Arkansas.csv"][0]
    assert [e["source"] for e in events(built, "locator")] == [LOCATOR]


def test_a_locator_that_does_not_reproduce_its_region_goes_back_to_the_proposer(tmp_path):
    lake, ending, arkansas = report_lake(tmp_path)
    wrong = {"reports_alabama": "def find_region(rows):\n    return 2, 3, None\n", "nope": LOCATOR}
    script = Script([answer(induction([*REPORT_CLAIMS, ending], dispositions=[ALABAMA, STATE_DOC], locators=wrong)),
                     answer(induction([*REPORT_CLAIMS, ending], dispositions=[ALABAMA, STATE_DOC],
                                      locators={"reports_alabama": LOCATOR})), arkansas])
    harness = build(tmp_path, script, lake).run()
    error = script.prompts["proposer"][1]
    assert "find_region returned (2, 3, None); it must reproduce" in error and "(2, 3, 5)" in error
    assert "'nope' is not a Dataset of a layout group representative in this visit" in error
    assert ogda_extension(harness.dataset("reports_alabama__arizona")).end_row == 7


def test_locators_run_without_lake_access(tmp_path):
    built = build(tmp_path, Script([]))
    path = str(built.store.lake_root / "income.csv")
    reads = f"def find_region(rows):\n    open({path!r}).read()\n    return 0, 1, None\n"
    fine = "def find_region(rows):\n    return 0, 1, len(rows)\n"
    blocked = locate(built, reads, [("income.csv", None)], "probe")[0]
    assert isinstance(blocked, str) and "PermissionError" in blocked
    assert locate(built, fine, [("income.csv", None)], "ok")[0] == (0, 1, 3)


def test_only_layout_group_members_can_be_separate(tmp_path):
    script = Script([answer(induction(separate=["income.csv"])), answer(induction())])
    build(tmp_path, script).run()
    assert "separate lists Files that are not members of a layout group" in script.prompts["proposer"][1]


def test_fan_out_saves_the_store_once_per_group(tmp_path, monkeypatch):
    import ogda.induction as induction_module
    from ogda.ontology import OntologyStore
    saves, during = [0], []
    original_save, original_fan_out = OntologyStore.save, induction_module.fan_out

    def save(self, harness):
        saves[0] += 1
        original_save(self, harness)

    def fan_out(*args):
        before = saves[0]
        original_fan_out(*args)
        during.append(saves[0] - before)

    monkeypatch.setattr(OntologyStore, "save", save)
    monkeypatch.setattr(induction_module, "fan_out", fan_out)
    build(tmp_path, Script([reports_answer()]), state_lake(tmp_path, texas=False)).run()
    assert during == [1]


# Final review fixes.

def test_start_refuses_a_folder_that_already_holds_a_harness(tmp_path):
    built = build(tmp_path, Script([answer(induction())]))
    (tmp_path / "again").mkdir()
    with pytest.raises(FileExistsError, match="resume it"):
        built.start("toy", make_lake(tmp_path / "again"))
    (tmp_path / "old").mkdir()
    legacy = construction(tmp_path / "old", Script([]))
    legacy.store.root.mkdir(parents=True)
    (legacy.store.root / "ontology.json").write_text('{"metadata": {"format_version": 6}}')
    with pytest.raises(FileExistsError):
        legacy.start("toy", make_lake(tmp_path / "old"))


def test_an_answer_is_saved_once_however_many_claims_it_has(tmp_path, monkeypatch):
    import ogda.induction as induction_module
    from ogda.ontology import OntologyStore
    saves, during = [0], []
    original_save, original_propose = OntologyStore.save, induction_module.propose

    def save(self, harness):
        saves[0] += 1
        original_save(self, harness)

    def propose(*args):
        before = saves[0]
        original_propose(*args)
        during.append(saves[0] - before)

    monkeypatch.setattr(OntologyStore, "save", save)
    monkeypatch.setattr(induction_module, "propose", propose)
    build(tmp_path, Script([answer(induction())])).run()
    assert during == [2]  # The accepted answer, then the end of the visit.
