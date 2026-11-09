"""Grounding: what ground prints and returns for Datasets, Fields, Files and patterns, joins and Experience."""

import pytest

from county_lake import FIRES_PATTERN, column, county_harness, path_field, table
from ogda.binding import Binding, BindingEntry
from ogda.grounding import Grounding, lake_map, size_text
from ogda.ontology import ExperienceEntry
from ogda.ontology_search import OntologySearch


def grounding_of(harness, tmp_path):
    return Grounding(harness, tmp_path / "lake", tmp_path / "cache", OntologySearch(harness))


@pytest.fixture
def grounding(tmp_path):
    harness = county_harness(tmp_path / "lake")
    harness.experience = [
        ExperienceEntry(id="e1", kind="data_quirk", text="Coos burned twice in 2020", anchors=["burned.acres_2020"]),
        ExperienceEntry(id="e2", kind="data_quirk", text="Old lesson", anchors=["burned"], stale=True)]
    return grounding_of(harness, tmp_path)


def view(grounding, name):
    return grounding.ground([name])[0].view


def test_a_dataset_view_gives_rows_region_identifier_and_columns(grounding):
    lines = view(grounding, "burned").splitlines()
    assert lines[:3] == ["burned: facts, 2 rows", "File burned.csv, header row 0, data from row 1",
                         "Acres burned per county and year"]
    assert 'Identifier: County.name, from column "County"' in lines
    assert '  "County"  county  String  County name, e.g. Coos  [identifies County.name]' in lines
    assert '  "2020"  acres_2020  Decimal  Acres burned in the county in 2020' in lines


def test_a_view_shows_no_value_from_the_rows(grounding):
    text = view(grounding, "counties")
    assert "north" not in text and "west" not in text


def test_a_dataset_returns_its_cached_table(grounding, tmp_path):
    grounded = grounding.ground(["burned"])[0]
    assert grounded.value == {"__table__": str(tmp_path / "cache" / "burned.parquet")}
    assert (grounded.kind, grounded.files, grounded.rows) == ("dataset", ["burned.csv"], 2)


def many_columns(tmp_path, count, meaning):
    harness = county_harness(tmp_path / "lake")
    (tmp_path / "lake/many.csv").write_text(",".join(f"C{i}" for i in range(count)) + "\n" + ",".join("1" * count) + "\n")
    harness.semantic_model.datasets.append(table(
        "many", "many.csv", [column(f"c{i}", f"C{i}", "Integer", meaning(i)) for i in range(count)], "Many measures"))
    return grounding_of(harness, tmp_path).ground(["many"])[0].view.splitlines()


def test_a_view_of_many_columns_that_fits_shows_every_meaning(tmp_path):
    lines = many_columns(tmp_path, 41, lambda i: f"Measure {i}")
    assert '  "C0"  c0  Integer  Measure 0' in lines and '  "C40"  c40  Integer  Measure 40' in lines
    assert not any("for one column's" in line for line in lines)


def test_a_view_too_long_for_its_columns_cuts_every_meaning_at_a_word(tmp_path):
    lines = many_columns(tmp_path, 200, lambda i: f"Share of samples that passed quality control in batch {i} of the run")
    columns = [line for line in lines if line.startswith('  "C')]
    assert '  "C0"  Integer  Share of samples that passed quality…' in columns
    assert len(columns) == 200 and sum(len(line) + 1 for line in columns) <= 12000
    assert 'Meanings are cut to fit; ground "many.<column>" for one column\'s full meaning.' in lines


def test_a_cut_line_keeps_its_header_and_datatype_whole(tmp_path):
    lines = many_columns(tmp_path, 1200, lambda i: f"Measure {i}")
    assert '  "C0"  Integer' in lines and '  "C1199"  Integer' in lines


def test_a_field_view_gives_its_meaning_and_the_fact_it_fills(grounding, tmp_path):
    grounded = grounding.ground(["burned.acres_2020"])[0]
    lines = grounded.view.splitlines()
    assert lines[:2] == ['burned.acres_2020  "2020"  Decimal', "Acres burned in the county in 2020"]
    assert 'Fills County.burned_in: "{County} burned {Acres} in {Year}" as Acres, with Year = 2020' in lines
    assert grounded.kind == "field"
    assert grounded.value == {"__table__": str(tmp_path / "cache" / "burned.parquet"), "__field__": "2020"}


def test_field_markers_name_references_and_unbound_reasons(grounding):
    assert "[refers to State.name]" in view(grounding, "counties.state_name")
    assert "[unbound: Free text]" in view(grounding, "counties.notes")


def test_a_file_and_a_pattern(grounding, tmp_path):
    grounded = grounding.ground(["burned.csv"])[0]
    assert grounded.view.splitlines()[:2] == ["burned.csv: .csv File, 30 bytes",
                                              "Tables in it: burned (header row 0, data from row 1)"]
    assert grounded.value == {"__path__": str(tmp_path / "lake" / "burned.csv")} and grounded.kind == "file"
    pattern = grounding.ground(["fires/*.csv"])[0]
    assert pattern.kind == "pattern" and pattern.view.startswith("fires/*.csv: 2 Files, ")
    assert pattern.value == {"__paths__": [str(tmp_path / "lake/fires/Nevada.csv"),
                                           str(tmp_path / "lake/fires/NewHampshire.csv")]}


def test_grounding_a_name_again_prints_one_line(grounding):
    first = grounding.ground(["burned"])[0]
    again = grounding.ground(["burned"])[0]
    assert again.view == "burned: grounded already" and again.value == first.value


def test_a_name_with_a_dot_is_a_file_when_no_field_matches(tmp_path):
    harness = county_harness(tmp_path / "lake")
    (tmp_path / "lake/a.csv").write_text("x\n1\n")
    harness.semantic_model.datasets.append(table("a", "a.csv", [column("x", "x")], "A"))
    grounding = grounding_of(harness, tmp_path)
    assert grounding.ground(["a.csv"])[0].kind == "file" and grounding.ground(["a.x"])[0].kind == "field"


@pytest.mark.parametrize("names, message", [
    (["burned", "Burnd"], r"No table, column or File named 'Burnd'\. Close names: burned"),
    (["nothing/*.csv"], r"No File matches 'nothing/\*\.csv'"),
    (["../secret.txt"], r"No table, column or File named '\.\./secret\.txt'"),
    ([], "ground needs at least one name")])
def test_a_bad_name_grounds_nothing(grounding, tmp_path, names, message):
    (tmp_path / "secret.txt").write_text("SECRET")
    with pytest.raises(ValueError, match=message):
        grounding.ground(names)
    assert grounding.grounded == {}


def test_the_first_grounding_has_no_joins(grounding):
    assert "Joins: none yet" in view(grounding, "burned")


def test_joins_name_the_columns_and_identifier_on_each_side(grounding):
    grounding.ground(["counties"])
    assert '  counties on County: "County" (County.name) ↔ "County" (County.name)' in view(grounding, "burned")


def test_joins_say_to_normalize_when_one_side_is_normalized(grounding):
    grounding.ground(["states"])
    assert ('  states on State: from path state (State.key) ↔ "State" (State.key); compare letters and digits only, '
            'lowercased') in view(grounding, "fires")


def test_a_lesson_shows_once_and_a_stale_lesson_never(grounding):
    first = view(grounding, "burned")
    assert "Experience:\n  - Coos burned twice in 2020" in first and "Old lesson" not in first
    assert "Coos burned twice" not in view(grounding, "burned.acres_2020")


def test_a_view_shows_at_most_ten_lessons(tmp_path):
    harness = county_harness(tmp_path / "lake")
    harness.experience = [ExperienceEntry(id=f"e{i}", kind="data_quirk", text=f"Lesson {i}", anchors=["burned"])
                          for i in range(12)]
    text = grounding_of(harness, tmp_path).ground(["burned"])[0].view
    assert text.count("  - Lesson") == 10 and "  ... 2 more lessons; ground a column to see its own." in text


def test_a_dataset_code_cannot_read_returns_its_path(tmp_path):
    harness = county_harness(tmp_path / "lake")
    (tmp_path / "lake/grid.npz").write_bytes(b"x")
    harness.semantic_model.datasets.append(table("grid", "grid.npz", [column("x", "x")], "A grid"))
    grounded = grounding_of(harness, tmp_path).ground(["grid"])[0]
    assert grounded.value == {"__path__": str(tmp_path / "lake/grid.npz")} and grounded.rows is None
    assert grounded.view.splitlines()[0] == "grid: facts"
    assert "Code does not read .npz Files: ground returns the path; read the File yourself." in grounded.view


@pytest.mark.parametrize("size, text", [(30, "30 bytes"), (18_234, "18 KB"), (2_900_000, "2.9 MB")])
def test_size_text_gives_bytes_kilobytes_or_megabytes(size, text):
    assert size_text(size) == text


@pytest.mark.parametrize("region, rows, header, location", [
    ({"header_row": None, "start_row": 0}, "1,2\n", "0",
     "File region.csv, no header row (columns named 0, 1, ...), data from row 0"),
    ({"header_rows": 2, "start_row": 2}, "A\nB\n1\n", "A / B", "File region.csv, header rows 0 to 1, data from row 2"),
    ({"end_row": 2}, "x\n1\n2\n", "x", "File region.csv, header row 0, data from row 1 to row 1")])
def test_the_location_gives_the_header_and_the_data_rows(tmp_path, region, rows, header, location):
    harness = county_harness(tmp_path / "lake")
    (tmp_path / "lake/region.csv").write_text(rows)
    harness.semantic_model.datasets.append(table("region", "region.csv", [column("x", header)], "A region", **region))
    assert view(grounding_of(harness, tmp_path), "region").splitlines()[1] == location


def test_a_layout_group_gives_its_file_count_and_path_pattern(grounding):
    grounded = grounding.ground(["fires"])[0]
    assert grounded.view.splitlines()[:2] == [
        "fires: one County, 2 rows",
        f'2 Files like "fires/NewHampshire.csv" (path pattern {FIRES_PATTERN}), header row 0, data from row 1']
    assert grounded.files == ["fires/Nevada.csv", "fires/NewHampshire.csv"]


def test_a_member_is_its_own_file_and_its_lessons_belong_to_its_group(tmp_path):
    harness = county_harness(tmp_path / "lake")
    harness.experience = [ExperienceEntry(id="e1", kind="data_quirk", text="Nevada lists one county",
                                          anchors=["fires__nevada"])]
    grounding = grounding_of(harness, tmp_path)
    assert "  - Nevada lists one county" in view(grounding, "fires")
    member = grounding.ground(["fires__nevada"])[0]
    assert member.view.splitlines()[1] == "File fires/Nevada.csv, header row 0, data from row 1"
    assert (member.files, member.rows) == (["fires/Nevada.csv"], 1)


def test_joins_name_different_identifiers_only_when_none_match(grounding):
    grounding.ground(["counties"])
    lines = view(grounding, "fires").splitlines()
    assert '  counties on County: "County" (County.name) ↔ "County" (County.name)' in lines
    assert ('  counties on State: from path state (State.key) ↔ "State" (State.name); compare letters and digits '
            'only, lowercased; different Identifiers') in lines


def test_joins_say_when_no_entity_type_is_shared(grounding):
    grounding.ground(["beaches"])
    assert "Joins: none with the grounded tables" in view(grounding, "burned")


def test_a_file_names_the_tables_it_documents(tmp_path):
    harness = county_harness(tmp_path / "lake")
    (tmp_path / "lake/README.md").write_text("# states\n")
    harness.semantic_model.datasets[0] = table("states", "states.csv", [column("state", "State")], "One row per state",
                                               documentation=["README.md"])
    grounding = grounding_of(harness, tmp_path)
    assert view(grounding, "README.md").splitlines() == [
        "README.md: .md File, 9 bytes", "No table covers this File.", "Documentation of: states"]
    assert "Documentation: README.md" in view(grounding, "states").splitlines()


def test_a_field_of_one_entity_names_it_among_the_other_roles(tmp_path):
    harness = county_harness(tmp_path / "lake")
    (tmp_path / "lake/by_county.csv").write_text("Year,Coos\n2020,10\n")
    harness.semantic_model.datasets.append(table(
        "by_county", "by_county.csv", [column("year", "Year", "Integer", "Year of the fires"),
                                       column("coos", "Coos", "Decimal", "Acres burned in Coos")],
        "Acres burned per year, one column per county"))
    harness.bindings["by_county"] = Binding(binding=[BindingEntry(
        concept="County", identify={"name": {"value": "Coos"}}, relationships={"burned_in": [["year", "coos"]]})])
    assert ('Fills County.burned_in: "{County} burned {Acres} in {Year}" as Acres, with County = {"name": "Coos"}, '
            'Year = Year') in view(grounding_of(harness, tmp_path), "by_county.coos").splitlines()


def test_an_unbound_reason_shows_whole_on_one_line(tmp_path):
    harness = county_harness(tmp_path / "lake")
    harness.bindings["counties"].unbound["notes"] = "Free text.\nNot a fact about the county."
    grounding = grounding_of(harness, tmp_path)
    reason = "[unbound: Free text. Not a fact about the county.]"
    assert f'  "Notes"  notes  String  Free-text notes about the county  {reason}' in view(
        grounding, "counties").splitlines()
    assert reason in view(grounding, "counties.notes").splitlines()


def test_a_column_view_shows_its_tables_lessons_past_the_cap(tmp_path):
    harness = county_harness(tmp_path / "lake")
    harness.experience = [ExperienceEntry(id=f"e{i}", kind="data_quirk", text=f"Lesson {i}", anchors=["burned"])
                          for i in range(12)]
    grounding = grounding_of(harness, tmp_path)
    assert view(grounding, "burned").count("  - Lesson") == 10
    column = view(grounding, "burned.acres_2020")
    assert column.count("  - Lesson") == 2 and "Experience:\n  - Lesson 10\n  - Lesson 11" in column


def test_a_table_view_shows_broad_lessons_first_and_a_column_view_narrow_ones(tmp_path):
    harness = county_harness(tmp_path / "lake")
    harness.experience = [
        ExperienceEntry(id="e1", kind="data_quirk", text="On the column", anchors=["burned.acres_2020"]),
        ExperienceEntry(id="e2", kind="data_quirk", text="On the table", anchors=["burned"])]
    table_view = view(grounding_of(harness, tmp_path), "burned")
    column_view = view(grounding_of(harness, tmp_path), "burned.acres_2020")
    assert table_view.index("On the table") < table_view.index("On the column")
    assert column_view.index("On the column") < column_view.index("On the table")


def test_a_file_gives_each_table_with_its_sheet_and_region(tmp_path):
    harness = county_harness(tmp_path / "lake")
    harness.semantic_model.datasets.append(table("beaches_top", "beaches.xlsx", [column("beach_name", "Beach")],
                                                 "The first beach", sheet="Malibu", end_row=2))
    assert view(grounding_of(harness, tmp_path), "beaches.xlsx").splitlines()[1] == (
        "Tables in it: beaches (#Malibu, header row 0, data from row 1); "
        "beaches_top (#Malibu, header row 0, data from row 1 to row 1)")


@pytest.mark.parametrize("first, then", [("burned", "burned.county"), ("fires", "fires__nevada")])
def test_joins_leave_out_the_nodes_own_table_and_layout_group(grounding, first, then):
    grounding.ground([first])
    assert "Joins: none with the grounded tables" in view(grounding, then).splitlines()


def test_joins_leave_out_a_column_whose_table_is_grounded(grounding):
    grounding.ground(["counties", "counties.state_name"])
    assert [line for line in view(grounding, "burned").splitlines() if "↔" in line] == [
        '  counties on County: "County" (County.name) ↔ "County" (County.name)']


def test_a_field_of_a_dataset_code_cannot_read_returns_the_path(tmp_path):
    harness = county_harness(tmp_path / "lake")
    (tmp_path / "lake/grid.npz").write_bytes(b"x")
    harness.semantic_model.datasets.append(table("grid", "grid.npz", [column("x", "x")], "A grid"))
    grounded = grounding_of(harness, tmp_path).ground(["grid.x"])[0]
    assert grounded.value == {"__path__": str(tmp_path / "lake/grid.npz")} and grounded.rows is None
    assert "Code does not read .npz Files: ground returns the path; read the File yourself." in grounded.view


def test_the_map_lists_tables_entity_types_and_files_with_no_table(tmp_path):
    lake = tmp_path / "lake"
    harness = county_harness(lake)
    harness.semantic_model.datasets[0] = table("states", "states.csv",
                                               [column("state", "State", description="State name, e.g. Nevada")],
                                               "One row per state", documentation=["README.md"])
    (lake / "notes").mkdir()
    for month in ("01", "02"):
        (lake / f"notes/2023-{month}.txt").write_text("free text\n")
    (lake / "README.md").write_text("# states\n")
    (lake / ".DS_Store").write_bytes(b"x")
    text = lake_map(harness, lake, OntologySearch(harness))
    lines = text.splitlines()
    assert lines[:2] == ["LAKE MAP", "TABLES (6)"]
    assert "- burned: burned.csv; facts. Acres burned per county and year." in lines
    assert '- fires: 2 Files like "fires/NewHampshire.csv"; one County. Counties with fires, one File per state.' in lines
    assert "- beaches: beaches.xlsx#Malibu; one Beach. Beaches of one town." in lines
    assert "- State: identified by name; also by key; in counties, fires (2 Files), states" in lines
    assert "FILES WITH NO TABLE (3)" in lines
    assert "- notes/*-*.txt: 2 Files, 20 bytes, e.g. notes/2023-01.txt" in lines
    assert "- README.md: 9 bytes, Documentation" in lines and ".DS_Store" not in text


def test_a_copy_is_not_a_table(tmp_path):
    harness = county_harness(tmp_path / "lake")
    (tmp_path / "lake/burned_copy.csv").write_text("County,2020\nCoos,10\nWashoe,20\n")
    harness.semantic_model.datasets.append(table("burned_copy", "burned_copy.csv", [column("county", "County")],
                                                 "A copy", relation="copy", relation_to="burned"))
    text = lake_map(harness, tmp_path / "lake", OntologySearch(harness))
    assert "TABLES (6)" in text and "burned_copy" not in text


def test_a_copy_is_not_listed_under_its_entity_types(tmp_path):
    harness = county_harness(tmp_path / "lake")
    (tmp_path / "lake/burned_copy.csv").write_text("County,2020\nCoos,10\nWashoe,20\n")
    harness.semantic_model.datasets.append(table("burned_copy", "burned_copy.csv", [column("county", "County")],
                                                 "A copy", relation="copy", relation_to="burned"))
    harness.bindings["burned_copy"] = Binding(rows="County", binding=[
        BindingEntry(concept="County", identify={"name": "county"})])
    lines = lake_map(harness, tmp_path / "lake", OntologySearch(harness)).splitlines()
    assert "- County: identified by name; in burned, counties, fires (2 Files), wide" in lines


def test_views_too_long_for_one_call_ground_nothing(tmp_path):
    harness = county_harness(tmp_path / "lake")
    harness.experience = [ExperienceEntry(id="e1", kind="data_quirk", text="Coos burned twice", anchors=["burned"])]
    grounding = Grounding(harness, tmp_path / "lake", tmp_path / "cache", OntologySearch(harness),
                          max_output_chars=1200)
    with pytest.raises(ValueError, match=r"^These views take [\d,]+ characters together, more than one call prints "
                                         r"\(900\): burned \d+, wide [\d,]+\. Ground fewer names per call\.$"):
        grounding.ground(["burned", "wide"])
    assert grounding.grounded == {} and grounding.shown == set() and grounding.keys == {}
    assert "Coos burned twice" in view(grounding, "burned")
    wide = view(grounding, "wide")
    assert len(wide) > 900 and "Columns (21):" in wide


@pytest.mark.parametrize("names", [[3], [["burned"]]])
def test_names_that_are_not_strings_ground_nothing(grounding, names):
    with pytest.raises(ValueError, match=r"ground takes each name as a string"):
        grounding.ground(names)
    assert grounding.grounded == {}


def test_an_absolute_path_is_refused_even_inside_the_lake(grounding, tmp_path):
    with pytest.raises(ValueError, match="No table, column or File named"):
        grounding.ground([str(tmp_path / "lake" / "burned.csv")])


def test_a_pattern_that_matches_no_file_names_close_names(grounding):
    with pytest.raises(ValueError, match=r"No File matches 'fires/\*\.cvs'\. Close names: fires/"):
        grounding.ground(["fires/*.cvs"])


def test_a_composite_identifier_is_one_line(grounding):
    lines = [line for line in view(grounding, "beaches").splitlines() if line.startswith("Identifier")]
    assert lines == ['Identifier: Beach.name, from column "Beach" + Beach.municipality by Municipality.name, '
                     'from column "community"']


def test_a_column_read_as_text_says_so(tmp_path):
    harness = county_harness(tmp_path / "lake")
    (tmp_path / "lake/gauges.csv").write_text("Rain\n1.5\n-\n")
    harness.semantic_model.datasets.append(table("gauges", "gauges.csv", [column("rain", "Rain", "Decimal", "Rain")],
                                                 "Rain gauges"))
    grounding = grounding_of(harness, tmp_path)
    assert '  "Rain"  rain  Decimal, read as text  Rain' in view(grounding, "gauges").splitlines()
    assert view(grounding, "gauges.rain").splitlines()[0] == 'gauges.rain  "Rain"  Decimal, read as text'
    assert '  "2020"  acres_2020  Decimal  Acres burned in the county in 2020' in view(grounding, "burned").splitlines()


@pytest.mark.parametrize("content, sheet, reason", [
    (b"not a workbook", "S", "BadZipFile: File is not a zip file"),
    (None, "Nowhere", "KeyError: 'Worksheet Nowhere does not exist.'")])
def test_a_dataset_whose_file_fails_to_read_is_grounded_as_its_file(tmp_path, content, sheet, reason):
    harness = county_harness(tmp_path / "lake")
    if content is not None:
        (tmp_path / "lake/bad.xlsx").write_bytes(content)
    path = "bad.xlsx" if content is not None else "beaches.xlsx"
    harness.semantic_model.datasets.append(table("bad", path, [column("x", "x")], "A sheet", sheet=sheet))
    grounded = grounding_of(harness, tmp_path).ground(["bad"])[0]
    assert grounded.value == {"__path__": str(tmp_path / "lake" / path)} and grounded.rows is None
    assert grounded.view.splitlines()[0] == "bad: facts"
    assert (f"Code could not read {path} ({reason}): ground returns the path; read the File yourself."
            in grounded.view.splitlines())


def test_a_layout_group_view_names_its_file_column(tmp_path):
    harness = county_harness(tmp_path / "lake")
    assert "Column file holds each row's File path." in view(grounding_of(harness, tmp_path), "fires").splitlines()
    for name in ("fires", "fires__nevada"):
        harness.dataset(name).fields.append(path_field("file", "x", "A Field named file"))
    grounding = Grounding(harness, tmp_path / "lake", tmp_path / "cache2", OntologySearch(harness))
    assert "Column file_ holds each row's File path." in view(grounding, "fires").splitlines()


def with_copies(tmp_path):
    harness = county_harness(tmp_path / "lake")
    for path in ("burned_copy.csv", "burned_top.csv"):
        (tmp_path / "lake" / path).write_text("County,2020\nCoos,10\n")
    (tmp_path / "lake/fires/NevadaCopy.csv").write_text("County\nWashoe\n")
    harness.semantic_model.datasets += [
        table("burned_copy", "burned_copy.csv", [column("county", "County")], "A copy", relation="copy",
              relation_to="burned"),
        table("burned_top", "burned_top.csv", [column("county", "County")], "The first row of burned",
              relation="subset", relation_to="burned"),
        table("fires__copy", "fires/NevadaCopy.csv", [column("county", "County"), path_field("state", "Nevada", "")],
              "A copy", path_pattern=FIRES_PATTERN, layout_of="fires", relation="copy", relation_to="fires__nevada")]
    return harness


def test_the_map_lists_subsets_and_counts_copies(tmp_path):
    harness = with_copies(tmp_path)
    lines = lake_map(harness, tmp_path / "lake", OntologySearch(harness)).splitlines()
    assert "TABLES (7)" in lines
    assert "- burned: burned.csv + 1 copy; facts. Acres burned per county and year." in lines
    assert "- burned_top: burned_top.csv; facts. The first row of burned; subset of burned" in lines
    assert ('- fires: 2 Files like "fires/NewHampshire.csv" + 1 copy; one County. Counties with fires, one File per '
            'state.') in lines
    assert not any(line.startswith("- burned_copy") for line in lines)


def test_a_copy_and_a_subset_say_what_they_copy(tmp_path):
    grounding = grounding_of(with_copies(tmp_path), tmp_path)
    assert "Copy of burned" in view(grounding, "burned_copy").splitlines()
    assert "Subset of burned" in view(grounding, "burned_top").splitlines()
    assert "Copy of fires__nevada" in view(grounding, "fires__copy").splitlines()


def test_a_column_named_by_its_field_keeps_the_field_first(grounding):
    lines = view(grounding, "fires").splitlines()
    assert any(line.startswith("  state  from path  String") for line in lines)
    assert any(line.startswith('  "County"  county  String') for line in lines)


def test_a_file_of_a_layout_group_returns_its_rows(grounding, tmp_path):
    grounded = grounding.ground(["fires/Nevada.csv"])[0]
    assert "Rows of fires from this File: 1 (header row 0, data from row 1)" in grounded.view.splitlines()
    assert grounded.value == {"__table__": str(tmp_path / "cache" / "fires.parquet"),
                              "__file__": "fires/Nevada.csv", "__file_column__": "file"}
    assert (grounded.kind, grounded.rows) == ("file", 1)
    assert grounding.ground(["fires/NewHampshire.csv"])[0].value["__file__"] == "fires/NewHampshire.csv"


def test_a_file_of_no_group_or_a_copy_returns_its_path(tmp_path):
    harness = county_harness(tmp_path / "lake")
    (tmp_path / "lake/fires/NevadaCopy.csv").write_text("County\nWashoe\n")
    harness.semantic_model.datasets.append(table(
        "fires__copy", "fires/NevadaCopy.csv", [column("county", "County"), path_field("state", "Nevada", "")],
        "A copy", layout_of="fires", relation="copy", relation_to="fires__nevada"))
    grounding = grounding_of(harness, tmp_path)
    assert "__path__" in grounding.ground(["fires/NevadaCopy.csv"])[0].value
    assert "__path__" in grounding.ground(["burned.csv"])[0].value


def test_a_file_of_a_group_code_fails_to_read_returns_its_path(tmp_path):
    harness = county_harness(tmp_path / "lake")
    for name in ("fires", "fires__nevada"):  # A header the Files lack: reading raises KeyError, one of READ_ERRORS.
        harness.dataset(name).fields[0] = column("county", "Missing")
    grounded = grounding_of(harness, tmp_path).ground(["fires/Nevada.csv"])[0]
    assert "__path__" in grounded.value and grounded.rows is None


def test_grounding_a_member_file_then_its_table_still_shows_the_table(grounding):
    grounding.ground(["fires/Nevada.csv"])
    lines = view(grounding, "fires").splitlines()
    assert lines[0] == "fires: one County, 2 rows" and "Joins: none yet" in lines


def test_a_padded_header_shows_trimmed_as_its_column_is_named(tmp_path):
    harness = county_harness(tmp_path / "lake")
    (tmp_path / "lake/reports.csv").write_text("Fraud ,Year\n5,2001\n")
    harness.semantic_model.datasets.append(table(
        "reports", "reports.csv", [column("fraud", "Fraud ", "Integer", "Fraud reports"),
                                   column("year", "Year", "Integer", "Year")], "Reports per year"))
    grounding = grounding_of(harness, tmp_path)
    assert '  "Fraud"  fraud  Integer  Fraud reports' in view(grounding, "reports").splitlines()
    field = grounding.ground(["reports.fraud"])[0]
    assert field.view.splitlines()[0] == 'reports.fraud  "Fraud"  Integer' and field.value["__field__"] == "Fraud"


def test_a_column_given_a_suffixed_name_shows_that_name(tmp_path):
    harness = county_harness(tmp_path / "lake")
    (tmp_path / "lake/pair.csv").write_text("b\n1\n")
    harness.semantic_model.datasets.append(table(
        "pair", "pair.csv", [column("x", "b", "Integer", "A measure"), path_field("b", "k", "A constant")], "Pairs"))
    lines = view(grounding_of(harness, tmp_path), "pair").splitlines()
    assert any(line.startswith("  b_  from path  String") for line in lines)
