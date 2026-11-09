"""The Data agent's Ontology search: BM25 over columns and tables, a line per matching Field, and the structure of a Dataset."""

from collections import Counter

import pytest
from county_lake import FIRES_PATTERN, column, county_harness, entity, path_field, table

from ogda.binding import Binding, BindingEntry
from ogda.ontology_search import NO_MATCH, OntologySearch, bm25, first_sentence


@pytest.fixture
def search(tmp_path):
    return OntologySearch(county_harness(tmp_path / "lake"))


def test_bm25_scores_a_document_with_a_rarer_matching_word_higher():
    docs = [Counter(["acre", "county"]), Counter(["county", "river"]), Counter(["county", "state"])]
    scores = bm25(docs, Counter(["acre", "county"]))
    assert scores[0] > scores[2] > 0 and scores[1] == scores[2]


def col(search, dataset, field):
    return next(c["entry"] for c in search.dataset(dataset)["columns"] if c["entry"]["field"] == field)


def test_search_prints_one_line_per_matching_field_best_first(search):
    assert search.search("acres burned").splitlines()[0] == (
        'burned.acres_2020  "2020"  Decimal  Acres burned in the county in 2020')


def test_a_path_field_shows_from_path(search):
    assert any(line.startswith("fires.state  from path  String") for line in search.search("state").splitlines())


def test_limit_caps_the_lines(search):
    assert len(search.search("county", limit=2).splitlines()) == 2


@pytest.mark.parametrize("words", ["zebra", ""])
def test_words_that_match_nothing_print_the_note(search, words):
    assert search.search(words) == NO_MATCH


@pytest.mark.parametrize("text, first", [("County name, e.g. Coos", "County name, e.g. Coos"),
                                         ("One row per state. Sorted by name.", "One row per state"),
                                         ("Rain in\ninches. Daily.", "Rain in inches"),
                                         ("Losses in the U.S. by state. Yearly.", "Losses in the U.S. by state"),
                                         ("Fees, taxes etc. per year. More.", "Fees, taxes etc. per year"),
                                         ("Station No. 5 readings. Daily.", "Station No. 5 readings"),
                                         ("", "")])
def test_first_sentence_keeps_abbreviations(text, first):
    assert first_sentence(text) == first


def test_a_dataset_gives_the_region_the_row_and_the_identifying_columns(search):
    burned = search.dataset("burned")["result"]
    keys = ("file", "sheet", "header_row", "header_rows", "start_row", "end_row", "documentation", "row")
    assert {k: burned[k] for k in keys} == {"file": "burned.csv", "sheet": None, "header_row": 0, "header_rows": 1,
                                            "start_row": 1, "end_row": None, "documentation": [], "row": "facts"}
    assert burned["identified_by"] == [{"column": "County", "field": "county", "identifies": "County.name"}]


def test_a_dataset_that_is_not_in_the_index_is_none(search):
    assert search.dataset("nope") is None


def test_a_column_links_to_its_entity_fact_and_other_roles(search):
    column = col(search, "burned", "acres_2020")
    assert column["links"] == [{"entity": "County", "fact": "County.burned_in", "says": "{County} burned {Acres} in {Year}",
                                "role": "Acres", "with": {"Year": "2020"}}]


def test_a_reference_column_and_a_normalized_path_field_are_marked(search):
    state_name = col(search, "counties", "state_name")
    assert state_name["refers_to"] == "State.name" and state_name["links"][0]["fact"] == "County.state"
    path = col(search, "fires", "state")
    assert path["from_path"] is True and path["refers_to"] == "State.key" and path["normalized"] == ["State.key"]


def test_an_unbound_field_carries_its_reason(search):
    notes = col(search, "counties", "notes")
    assert notes["unbound"] == "Free text" and notes["links"] == []


def test_entities_give_identifiers_tables_and_references(search):
    state = search.entity("State")
    assert state["identifier"] == ["name"] and state["other_identifiers"] == [["key"]]
    assert state["tables"] == ["counties", "fires (2 Files)", "states"] and state["referenced_by"] == ["County.state"]


def test_a_layout_group_appears_once_with_its_files(search):
    fires = search.dataset("fires")["result"]
    assert search.dataset("fires__nevada") is None
    assert fires["files"] == {"count": 2, "path_pattern": FIRES_PATTERN,
                              "examples": ["fires/Nevada.csv", "fires/NewHampshire.csv"]}


def test_a_worksheet_dataset_gives_its_sheet_and_nested_identifier(search):
    beaches = search.dataset("beaches")["result"]
    assert beaches["sheet"] == "Malibu"
    assert {"column": "community", "field": "community", "identifies": "Municipality.name",
            "via": "Beach.municipality"} in beaches["identified_by"]


@pytest.mark.parametrize("limit", [None, "many", 2.5])
def test_a_limit_that_is_not_a_whole_number_says_so(search, limit):
    with pytest.raises(ValueError, match="whole number"):
        search.search("state", limit=limit)


def test_a_limit_given_as_text_of_a_number_works(search):
    assert len(search.search("state", limit="1").splitlines()) == 1


def test_a_table_word_lifts_its_columns(tmp_path):
    from ogda.ontology import Harness

    harness = Harness.empty("toy")
    harness.semantic_model.datasets = [
        table("a_sites", "a.csv", [column("level", "Level", "Decimal", "Measured level")], "Sites"),
        table("b_gauges", "b.csv", [column("level", "Level", "Decimal", "Measured level")], "Rainfall gauges")]
    lines = OntologySearch(harness).search("level rainfall").splitlines()
    assert [line.split("  ")[0] for line in lines] == ["b_gauges.level", "a_sites.level"]


def test_a_header_with_a_line_break_stays_on_one_line(tmp_path):
    harness = county_harness(tmp_path / "lake")
    harness.semantic_model.datasets.append(table("t", "t.csv", [column("total", 'Total\n"acres"', "Decimal", "Total")],
                                                 "Totals"))
    [line] = [line for line in OntologySearch(harness).search("total").splitlines() if line.startswith("t.")]
    assert line == 't.total  "Total\\n\\"acres\\""  Decimal  Total'


def test_a_path_field_outside_a_group_is_from_path(tmp_path):
    harness = county_harness(tmp_path / "lake")
    harness.semantic_model.datasets.append(table(
        "burned_2021", "burned/2021.csv", [column("county", "County"), path_field("year", "2021", "Year of the File"),
                                           column("acres", "Acres", "Decimal", "Acres burned")],
        "Acres burned per county in one year", path_pattern=r"burned/(?P<year>\d+)\.csv"))
    harness.bindings["burned_2021"] = Binding(binding=[BindingEntry(
        concept="County", identify={"name": "county"}, relationships={"burned_in": [["year", "acres"]]})])
    search = OntologySearch(harness)
    year = col(search, "burned_2021", "year")
    assert year["from_path"] is True and "column" not in year and "files" not in search.dataset("burned_2021")["result"]
    acres = col(search, "burned_2021", "acres")
    assert acres["links"][0]["with"] == {"Year": "2021"}


def test_a_path_field_of_a_layout_group_shows_as_from_path_in_with(tmp_path):
    harness = county_harness(tmp_path / "lake")
    for year, member in (("2021", {}), ("2022", {"layout_of": "burned_2021"})):
        harness.semantic_model.datasets.append(table(
            f"burned_{year}", f"burned/{year}.csv",
            [column("county", "County"), path_field("year", year, "Year of the File"),
             column("acres", "Acres", "Decimal", "Acres burned")],
            "Acres burned per county in one year", path_pattern=r"burned/(?P<year>\d+)\.csv", **member))
    harness.bindings["burned_2021"] = Binding(binding=[BindingEntry(
        concept="County", identify={"name": "county"}, relationships={"burned_in": [["year", "acres"]]})])
    acres = col(OntologySearch(harness), "burned_2021", "acres")
    assert acres["links"][0]["with"] == {"Year": {"from_path": "year"}}


def test_copies_and_subsets_are_marked_and_not_counted_as_files(tmp_path):
    harness = county_harness(tmp_path / "lake")
    harness.semantic_model.datasets += [
        table("fires__copy", "fires/NewHampshire_copy.csv",
              [column("county", "County", description="County with fires, e.g. Coos"),
               path_field("state", "NewHampshire_copy", "The state the File covers, from its name")],
              "Counties with fires, one File per state", path_pattern=FIRES_PATTERN, layout_of="fires",
              relation="copy", relation_to="fires"),
        table("states_subset", "states_subset.csv", [column("state", "State", description="State name, e.g. Nevada")],
              "Some states, one row per state", relation="subset", relation_to="states")]
    harness.bindings["states_subset"] = harness.bindings["states"]
    search = OntologySearch(harness)
    fires = search.dataset("fires")["result"]
    assert fires["files"] == {"count": 2, "path_pattern": FIRES_PATTERN,
                              "examples": ["fires/Nevada.csv", "fires/NewHampshire.csv"],
                              "copies": [{"file": "fires/NewHampshire_copy.csv", "copy_of": "fires/NewHampshire.csv"}]}
    assert search.dataset("states_subset")["result"]["subset_of"] == "states.csv"
    assert "fires (2 Files)" in search.entity("State")["tables"]


def test_a_link_of_an_entry_identified_by_a_literal_names_its_entity(tmp_path):
    harness = county_harness(tmp_path / "lake")
    harness.semantic_model.datasets.append(table(
        "burned_by_county", "burned_by_county.csv",
        [column("year", "Year", "Integer", "Year of the fires"), column("coos", "Coos", "Decimal", "Acres burned in Coos"),
         column("washoe", "Washoe", "Decimal", "Acres burned in Washoe")],
        "Acres burned per year, one column per county"))
    harness.bindings["burned_by_county"] = Binding(binding=[
        BindingEntry(concept="County", identify={"name": {"value": name}}, relationships={"burned_in": [["year", field]]})
        for name, field in (("Coos", "coos"), ("Washoe", "washoe"))])
    coos = col(OntologySearch(harness), "burned_by_county", "coos")
    assert coos["links"] == [{"entity": "County", "of": {"name": "Coos"}, "fact": "County.burned_in",
                              "says": "{County} burned {Acres} in {Year}", "role": "Acres", "with": {"Year": "Year"}}]


def test_normalized_names_only_the_identifier_that_uses_it(search):
    state = col(search, "states", "state")
    assert state["identifies"] == "State.name" and state["normalized"] == ["State.key"]
