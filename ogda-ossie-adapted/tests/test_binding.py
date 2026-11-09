"""Bindings: their rules, the mapping compiler, the structure and data checks, and Stage 2 rewrites."""

import openpyxl
import pytest

from ossie_examples import rel, valid_data, valid_ontology, value_type
from ogda.binding import (Binding, BindingEntry, binding_problems, compile_mapping, data_problems, document_names,
                          field_source, fits, header_text, merge_binding, named_concepts, normalize, primary_key,
                          quote_identifier, quote_literal, region_rows, replace_relationship, structure_problems)
from ogda.lake import RowReader
from ogda.ossie import (Dataset, DatasetField, Expression, OgdaExtension, OntologyDocument, SemanticModel, dump, validate,
                        with_ogda_extension)


def field(name, column=None, datatype="String"):
    return DatasetField(name=name, expression=Expression(sql=quote_identifier(column or name)), datatype=datatype)


def dataset(name, path, *fields):
    return with_ogda_extension(Dataset(name=name, source=path, fields=list(fields)), OgdaExtension(path=path))


def fires_dataset():
    return dataset("fires_2020", "fires/2020.csv", field("incident_id", "Incident ID"),
                   field("acres", "Acres Burned", "Decimal"), field("county", "County"))


def fires_by_year():
    return dataset("fires_by_year", "fires_by_year.csv", field("county", "County"),
                   field("y2018", "2018", "Decimal"), field("y2019", "2019", "Decimal"))


def fire_entry(**relationships):
    return BindingEntry(concept="Fire", identify={"incident_number": "incident_id"},
                        relationships={"acres_burned": ["acres"], "county": [{"name": "county"}], **relationships})


def fire_binding(**relationships):
    return Binding(rows="Fire", binding=[fire_entry(**relationships)])


def test_a_complete_entity_binding_has_no_problems_and_names_its_primary_key():
    assert binding_problems(fire_binding(), fires_dataset(), valid_ontology()) == []
    assert primary_key(fire_binding()) == ["incident_id"]


def test_fact_rows_of_a_wide_layout_have_no_problems():
    binding = Binding(binding=[BindingEntry(concept="County", identify={"name": "county"}, relationships={
        "burned_in": [[{"value": "2018"}, "y2018"], [{"value": "2019"}, "y2019"]]})])
    assert binding_problems(binding, fires_by_year(), valid_ontology()) == []
    assert primary_key(binding) == []


def test_an_alternate_identifier_identifies_an_entity():
    states = dataset("states", "states.csv", field("st"), field("state_name"))
    binding = Binding(rows="State", binding=[BindingEntry(concept="State", identify={"code": "st"},
                                                          relationships={"name": ["state_name"]})])
    assert binding_problems(binding, states, valid_ontology()) == []
    assert primary_key(binding) == ["st"]


@pytest.mark.parametrize("binding, message", [
    (fire_binding(acres_burned=[]), "one value per Role"),
    (fire_binding(acres_burned=["acreage"]), "no Field acreage"),
    (fire_binding(acres_burned=[{"name": "acres"}]), "give a Field, a literal"),
    (fire_binding(county=["county"]), "an identify object"),
    (fire_binding(county=[{"fips": "county"}]), "an Identifier of County"),
    (fire_binding(burned_in=["acres"]), "not declared under Fire"),
    (Binding(binding=[BindingEntry(concept="Acres", identify={})]), "an Entity type"),
    (Binding(binding=[]), "at least one entry"),
])
def test_binding_rules(binding, message):
    problems = binding_problems(binding, fires_dataset(), valid_ontology())
    assert any(message in problem for problem in problems), problems


def test_every_field_is_bound_or_unbound_with_a_reason():
    binding = Binding(rows="Fire", binding=[BindingEntry(concept="Fire", identify={"incident_number": "incident_id"},
                                                         relationships={"acres_burned": ["acres"]})])
    problems = binding_problems(binding, fires_dataset(), valid_ontology())
    assert any("neither bound nor in unbound" in p and "county" in p for p in problems)
    unbound = binding.model_copy(update={"unbound": {"county": "A free-text place name"}})
    assert binding_problems(unbound, fires_dataset(), valid_ontology()) == []


def test_a_field_must_fit_its_value_type():
    flags = dataset("fires_2020", "fires/2020.csv", field("incident_id", "Incident ID"),
                    field("acres", "Acres Burned", "Boolean"), field("county", "County"))
    assert any("does not fit" in p for p in binding_problems(fire_binding(), flags, valid_ontology()))
    assert fits("Integer", "Decimal") and fits("Boolean", "String") and not fits("String", "Integer")


def test_entity_rows_need_a_field_in_their_identifier():
    binding = Binding(rows="Fire", binding=[fire_entry().model_copy(update={"identify": {"incident_number": {"value": "7"}}})],
                      unbound={"incident_id": "Shown in the identify literal"})
    assert any("at least one part from a Field" in p for p in binding_problems(binding, fires_dataset(), valid_ontology()))


def malibu_beaches():
    return (dataset("beaches", "beaches.csv", field("beach_name", "Beach"), field("length", "Length")),
            Binding(rows="Beach", binding=[BindingEntry(concept="Beach", identify={
                "name": "beach_name", "municipality": {"name": {"value": "Malibu"}}})],
                    unbound={"length": "No Concept for beach length yet"}))


def test_entity_rows_may_fix_one_part_of_a_compound_identifier(tmp_path):
    beaches, binding = malibu_beaches()
    assert binding_problems(binding, beaches, valid_ontology()) == []
    assert primary_key(binding) == ["beach_name"]
    write(tmp_path, "beaches.csv", "Beach,Length\nZuma,3\nPoint Dume,1\n")
    assert data_problems(beaches, binding, valid_ontology(), tmp_path, 100) == []
    write(tmp_path, "beaches.csv", "Beach,Length\nZuma,3\nPoint Dume,1\nZuma,4\n")
    problems = data_problems(beaches, binding, valid_ontology(), tmp_path, 100)
    assert len(problems) == 1 and "share the identifier (beach_name)" in problems[0]


def test_rows_names_an_entity_type_of_an_entry():
    binding = fire_binding().model_copy(update={"rows": "County"})
    assert any("no binding entry" in p for p in binding_problems(binding, fires_dataset(), valid_ontology()))


def test_named_concepts_include_role_and_identify_concepts():
    assert named_concepts(fire_binding(), valid_ontology()) == {"Fire", "IncidentNumber", "Acres", "County", "CountyName"}


def test_field_source_reads_back_quoted_headers_and_literals():
    header = DatasetField(name="size", expression=Expression(sql=quote_identifier('Size "m2"')))
    literal = DatasetField(name="owner", expression=Expression(sql=quote_literal("O'Brien")))
    assert header.expression.sql == '"Size ""m2"""'
    assert field_source(header) == ("column", 'Size "m2"') and field_source(literal) == ("literal", "O'Brien")


def test_entity_rows_may_fill_relationships_with_literals():
    binding = fire_binding(acres_burned=[{"value": "3"}]).model_copy(update={"unbound": {"acres": "Rounded copy"}})
    assert binding_problems(binding, fires_dataset(), valid_ontology()) == []


# Compiling the mapping.

def compiled(datasets_and_bindings, document=None):
    model = SemanticModel(name="toy_semantic_model", datasets=[d for d, _ in datasets_and_bindings])
    bindings = {d.name: b for d, b in datasets_and_bindings}
    return dump(compile_mapping(document or valid_ontology(), model, bindings, "toy"))["concept_mappings"]


def county_burned_in():
    return Binding(binding=[BindingEntry(concept="County", identify={"name": "county"}, relationships={
        "burned_in": [[{"value": "2018"}, "y2018"], [{"value": "2019"}, "y2019"]]})])


FIRE = {"referent_mappings": [{"relationship": "Fire.incident_number", "expression": "fires_2020.incident_id"}]}


def test_document_names_follow_the_lake():
    assert document_names("toy") == ("toy_ontology", "toy_semantic_model", "toy_mapping")


def test_entity_rows_compile_to_object_and_link_mappings():
    assert compiled([(fires_dataset(), fire_binding())]) == [{
        "concept": "Fire", "object_mappings": [FIRE], "link_mappings": [{"object_mapping": FIRE, "children": [
            {"object_mapping": {"concept": "Acres", "expression": "fires_2020.acres"}, "relationship": "acres_burned"},
            {"object_mapping": {"concept": "County", "referent_mappings": [
                {"relationship": "County.name", "expression": "fires_2020.county"}]}, "relationship": "county"}]}]}]


def test_a_wide_layout_shares_the_first_role_and_writes_literals():
    county = {"referent_mappings": [{"relationship": "County.name", "expression": "fires_by_year.county"}]}
    assert compiled([(fires_by_year(), county_burned_in())]) == [{"concept": "County", "link_mappings": [
        {"object_mapping": county, "children": [
            {"object_mapping": {"concept": "Year", "expression": year}, "children": [
                {"object_mapping": {"concept": "Acres", "expression": f"fires_by_year.y{year}"},
                 "relationship": "burned_in"}]} for year in ("2018", "2019")]}]}]


def test_a_string_based_literal_is_quoted():
    data = valid_data()
    next(c for c in data["ontology"] if c["concept"] == "Year")["extends"] = ["String"]
    mappings = compiled([(fires_by_year(), county_burned_in())], OntologyDocument.model_validate(data))
    years = [child["object_mapping"]["expression"] for child in mappings[0]["link_mappings"][0]["children"]]
    assert years == ["'2018'", "'2019'"]


def test_a_compound_identifier_nests_referent_mappings():
    samples = dataset("samples", "samples.csv", field("beach_name"), field("community"))
    binding = Binding(rows="Beach", binding=[BindingEntry(
        concept="Beach", identify={"name": "beach_name", "municipality": {"name": "community"}})])
    assert compiled([(samples, binding)]) == [{"concept": "Beach", "object_mappings": [{"referent_mappings": [
        {"relationship": "Beach.name", "expression": "samples.beach_name"},
        {"relationship": "Beach.municipality", "referent_mappings": [
            {"relationship": "Municipality.name", "expression": "samples.community"}]}]}]}]


def test_an_alternate_identifier_is_the_referent_relationship():
    states = dataset("states", "states.csv", field("st"), field("state_name"))
    binding = Binding(rows="State", binding=[BindingEntry(concept="State", identify={"code": "st"},
                                                          relationships={"name": ["state_name"]})])
    mapping = compiled([(states, binding)])[0]
    assert mapping["object_mappings"] == [{"referent_mappings": [{"relationship": "State.code", "expression": "states.st"}]}]
    assert mapping["link_mappings"][0]["children"] == [
        {"object_mapping": {"concept": "StateName", "expression": "states.state_name"}, "relationship": "name"}]


def test_layout_members_get_their_own_mappings():
    member = fires_dataset().model_copy(update={"name": "fires_2021", "source": "fires/2021.csv"})
    mapping = compiled([(member, fire_binding()), (fires_dataset(), fire_binding())])[0]
    expressions = [o["referent_mappings"][0]["expression"] for o in mapping["object_mappings"]]
    assert expressions == ["fires_2020.incident_id", "fires_2021.incident_id"]
    assert len(mapping["link_mappings"]) == 2


def test_the_compiled_mapping_validates_against_the_ossie_schema():
    model = SemanticModel(name="toy_semantic_model", datasets=[fires_dataset(), fires_by_year()])
    validate(compile_mapping(valid_ontology(), model, {"fires_2020": fire_binding(), "fires_by_year": county_burned_in()},
                             "toy"))


# Structure and data checks.

def write(root, path, text):
    (root / path).parent.mkdir(parents=True, exist_ok=True)
    (root / path).write_text(text)


FIRES_HEADER = "Incident ID,Acres Burned,County\n"


def fire_problems(root, rows, limit=100):
    write(root, "fires/2020.csv", FIRES_HEADER + rows)
    return data_problems(fires_dataset(), fire_binding(), valid_ontology(), root, limit)


def state_problems(root, rows):
    write(root, "states.csv", "st,state_name\n" + rows)
    states = dataset("states", "states.csv", field("st"), field("state_name"))
    binding = Binding(binding=[BindingEntry(concept="State", identify={"code": "st"}, relationships={"name": ["state_name"]})])
    return data_problems(states, binding, valid_ontology(), root, 100)


def test_unique_identifiers_and_functional_relationships_pass(tmp_path):
    assert fire_problems(tmp_path, "7,10,Kern\n8,5,Kern\n") == []


def test_a_duplicate_identifier_is_reported_with_its_rows(tmp_path):
    problems = fire_problems(tmp_path, "7,10,Kern\n8,5,Kern\n7,3,Inyo\n")
    assert len(problems) == 1 and "incident_id" in problems[0] and "1, 3" in problems[0]


def test_exact_duplicate_rows_do_not_break_an_identifier(tmp_path):
    assert fire_problems(tmp_path, "7,10,Kern\n7,10,Kern\n") == []


def test_a_many_to_one_violation_is_reported(tmp_path):
    write(tmp_path, "fires_by_year.csv", "County,2018,2019\nKern,10,20\nKern,11,20\n")
    problems = data_problems(fires_by_year(), county_burned_in(), valid_ontology(), tmp_path, 100)
    assert len(problems) == 1 and "County.burned_in is ManyToOne" in problems[0] and "1, 2" in problems[0]


def test_a_one_to_one_violation_is_reported_in_either_direction(tmp_path):
    assert any("State.name is OneToOne" in p for p in state_problems(tmp_path, "CA,California\nCA,Calif\n"))
    assert any("State.name is OneToOne" in p for p in state_problems(tmp_path, "CA,California\nCB,California\n"))
    assert state_problems(tmp_path, "CA,California\nNV,Nevada\n") == []


def test_rows_with_a_null_value_are_skipped(tmp_path):
    write(tmp_path, "fires_by_year.csv", "County,2018,2019\nKern,10,20\nKern,,20\n")
    assert data_problems(fires_by_year(), county_burned_in(), valid_ontology(), tmp_path, 100) == []


def test_an_empty_identifier_is_reported(tmp_path):
    assert any("empty identifier" in p for p in fire_problems(tmp_path, "7,10,Kern\n,5,Kern\n"))


def test_only_the_first_check_rows_rows_are_checked(tmp_path):
    assert fire_problems(tmp_path, "7,10,Kern\n8,5,Kern\n7,3,Inyo\n", limit=2) == []
    assert fire_problems(tmp_path, "7,10,Kern\n8,5,Kern\n7,3,Inyo\n", limit=10) != []


def test_repeated_header_names_get_suffixes_like_pandas(tmp_path):
    assert header_text([["Value", "Value", "Other", "Value"]], 0, 1) == ["Value", "Value.1", "Other", "Value.2"]
    write(tmp_path, "values.csv", "Value,Value,Other\n1,2,3\n")
    values = dataset("values", "values.csv", field("first", "Value"), field("second", "Value.1"), field("other", "Other"))
    assert structure_problems(values, tmp_path) == []


def test_a_region_without_a_header_row_names_columns_by_position(tmp_path):
    assert header_text([["a", "b"], ["c", "d", "e"]], None, 1) == ["0", "1", "2"]


def test_json_datasets_are_not_read(tmp_path):
    write(tmp_path, "records.json", '[{"incident_id": "7"}, {"incident_id": "7"}]')
    records = with_ogda_extension(dataset("records", "records.json", field("incident_id")),
                                  OgdaExtension(path="records.json", header_row=None, start_row=0))
    assert region_rows(records, tmp_path, 10) is None and structure_problems(records, tmp_path) == []
    binding = Binding(rows="Fire", binding=[BindingEntry(concept="Fire", identify={"incident_number": "incident_id"})])
    assert data_problems(records, binding, valid_ontology(), tmp_path, 10) == []


def test_structure_problems_check_the_region_and_header_text(tmp_path):
    write(tmp_path, "fires/2020.csv", FIRES_HEADER + "7,10,Kern\n")
    long = with_ogda_extension(fires_dataset(), OgdaExtension(path="fires/2020.csv", end_row=10))
    assert any("do not fit" in p for p in structure_problems(long, tmp_path))
    renamed = dataset("fires_2020", "fires/2020.csv", field("incident_id", "Incident ID"), field("acres", "Acreage"))
    assert any("has no columns ['Acreage']" in p for p in structure_problems(renamed, tmp_path))
    assert structure_problems(fires_dataset(), tmp_path) == []


def test_an_excel_region_below_a_title_row_is_read(tmp_path):
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Data"
    for row in (["Fire report 2020"], ["Incident ID", "Acres Burned", "County"], [7, 10, "Kern"], [8, 5, "Inyo"]):
        sheet.append(row)
    workbook.save(tmp_path / "fires.xlsx")
    fires = with_ogda_extension(dataset("fires", "fires.xlsx", *fires_dataset().fields),
                                OgdaExtension(path="fires.xlsx", sheet="Data", header_row=1, start_row=2))
    header, rows = region_rows(fires, tmp_path, 10)
    assert header == ["Incident ID", "Acres Burned", "County"] and rows == [(2, ["7", "10", "Kern"]), (3, ["8", "5", "Inyo"])]
    assert structure_problems(fires, tmp_path) == []
    assert data_problems(fires, fire_binding(), valid_ontology(), tmp_path, 10) == []


# Rewrites for the concept layer.

def ontology_with(change) -> OntologyDocument:
    data = valid_data()
    change({c["concept"]: c for c in data["ontology"]}, data)
    return OntologyDocument.model_validate(data)


def with_counties(table, data):
    data["ontology"].append({"concept": "Counties", "type": "EntityType", "identify_by": ["county_name"], "relationships": [
        rel("county_name", "CountyName", multiplicity="OneToOne", verbalizes="{Counties} is named {CountyName}")]})
    table["Fire"]["relationships"][2]["roles"] = [{"concept": "Counties"}]


def with_county_name(table, data):
    table["Fire"]["relationships"][2] = rel("county_name", "CountyName", multiplicity="ManyToOne",
                                            verbalizes="{Fire} started in {CountyName}")


def test_merge_renames_entries_and_identify_keys():
    before = ontology_with(with_counties)
    counties = Binding(rows="Counties", binding=[BindingEntry(concept="Counties", identify={"county_name": "county"})])
    merged = merge_binding(counties, before, "Counties", "County", {"county_name": "name"})
    assert merged.rows == "County" and merged.binding == [BindingEntry(concept="County", identify={"name": "county"})]


def test_merge_renames_nested_identify_objects():
    before = ontology_with(with_counties)
    merged = merge_binding(fire_binding(county=[{"county_name": "county"}]), before, "Counties", "County",
                           {"county_name": "name"})
    assert merged.binding[0].relationships["county"] == [{"name": "county"}]
    assert binding_problems(merged, fires_dataset(), valid_ontology()) == []


def test_replacing_an_attribute_with_a_reference_wraps_its_values():
    before = ontology_with(with_county_name)
    binding = Binding(rows="Fire", binding=[BindingEntry(concept="Fire", identify={"incident_number": "incident_id"},
                      relationships={"acres_burned": ["acres"], "county_name": ["county"]})])
    rebound = replace_relationship(binding, before, valid_ontology(), "Fire", "county_name", "county")
    assert rebound.binding[0].relationships == {"acres_burned": ["acres"], "county": [{"name": "county"}]}
    assert binding_problems(rebound, fires_dataset(), valid_ontology()) == []


def test_replacing_without_a_matching_identifier_leaves_values_for_the_rules():
    before = ontology_with(with_county_name)

    def to_beach(table, data):
        table["Fire"]["relationships"][2] = rel("county", "Beach", multiplicity="ManyToOne", verbalizes="{Fire} is at {Beach}")
    after = ontology_with(to_beach)
    binding = Binding(rows="Fire", binding=[BindingEntry(concept="Fire", identify={"incident_number": "incident_id"},
                      relationships={"acres_burned": ["acres"], "county_name": ["county"]})])
    rebound = replace_relationship(binding, before, after, "Fire", "county_name", "county")
    assert rebound.binding[0].relationships["county"] == ["county"]
    assert any("an identify object" in p for p in binding_problems(rebound, fires_dataset(), after))


def test_renaming_an_identifying_relationship_renames_identify_keys():
    def renamed(table, data):
        county = table["County"]
        county["relationships"][0]["name"], county["identify_by"] = "county_name", ["county_name"]
    after = ontology_with(renamed)
    rebound = replace_relationship(fire_binding(), valid_ontology(), after, "County", "name", "county_name")
    assert rebound.binding[0].relationships["county"] == [{"county_name": "county"}]
    assert binding_problems(rebound, fires_dataset(), after) == []


def test_an_identifying_relationship_named_value_is_not_taken_for_a_literal(tmp_path):
    data = valid_data()
    data["ontology"] += [
        {"concept": "ReadingValue", "type": "ValueType", "extends": ["Integer"]},
        {"concept": "Reading", "type": "EntityType", "identify_by": ["value"], "relationships": [
            rel("value", "ReadingValue", multiplicity="OneToOne", verbalizes="{Reading} has value {ReadingValue}")]},
        {"concept": "Visit", "type": "EntityType", "identify_by": ["code"], "relationships": [
            rel("code", "StateCode", multiplicity="OneToOne", verbalizes="{Visit} has code {StateCode}"),
            rel("reading", "Reading", multiplicity="ManyToOne", verbalizes="{Visit} took {Reading}")]}]
    document = OntologyDocument.model_validate(data)
    readings = dataset("a", "a.csv", field("col", "col", "Integer"))
    binding = Binding(rows="Reading", binding=[BindingEntry(concept="Reading", identify={"value": "col"})])
    write(tmp_path, "a.csv", "col\n1\n2\n")
    assert binding_problems(binding, readings, document) == [] and primary_key(binding) == ["col"]
    assert data_problems(readings, binding, document, tmp_path, 10) == []
    model = SemanticModel(name="toy_semantic_model", datasets=[readings])
    mapping = dump(compile_mapping(document, model, {"a": binding}, "toy"))["concept_mappings"]
    assert mapping[0]["object_mappings"] == [{"referent_mappings": [{"relationship": "Reading.value", "expression": "a.col"}]}]
    visits = dataset("visits", "visits.csv", field("code"), field("reading", "reading", "Integer"))
    visit = Binding(rows="Visit", binding=[BindingEntry(concept="Visit", identify={"code": "code"},
                                                        relationships={"reading": [{"value": "reading"}]})])
    assert binding_problems(visit, visits, document) == []
    assert named_concepts(visit, document) >= {"Reading", "ReadingValue"}


# Final review fixes.

def test_an_entry_that_is_not_the_rows_entry_must_fill_a_relationship():
    binding = Binding(rows="Fire", binding=[fire_entry(), BindingEntry(concept="County", identify={"name": "county"})])
    assert any("fills no Relationship" in p for p in binding_problems(binding, fires_dataset(), valid_ontology()))


def test_an_entry_that_fills_nothing_is_told_how_to_link_it():
    binding = Binding(rows="Fire", binding=[fire_entry(), BindingEntry(concept="County", identify={"name": "county"})])
    problem = next(p for p in binding_problems(binding, fires_dataset(), valid_ontology()) if "fills no" in p)
    assert "a Relationship whose Role is County" in problem and "{'name': 'county'}" in problem
    assert "Drop the entry only when the row is not about a County" in problem


def test_binding_checks_report_a_missing_identifying_relationship_instead_of_crashing():
    data = valid_data()
    next(c for c in data["ontology"] if c["concept"] == "County")["identify_by"] = ["code"]
    problems = binding_problems(fire_binding(county=[{"code": "county"}]), fires_dataset(),
                                OntologyDocument.model_validate(data))
    assert any("no binary identifying Relationship code" in p for p in problems)


def test_a_field_given_where_an_identify_object_belongs_is_not_also_reported_unbound():
    problems = binding_problems(fire_binding(county=["county"]), fires_dataset(), valid_ontology())
    assert any("an identify object" in p for p in problems)
    assert not any("neither bound nor in unbound" in p for p in problems)


def test_fully_blank_rows_are_skipped_by_the_data_checks(tmp_path):
    assert fire_problems(tmp_path, "7,10,Kern\n8,5,Kern\n,,\n,,\n") == []


# Normalized values: a Field compared by its letters and digits only.

def keyed_ontology():
    data = valid_data()
    data["ontology"].append(value_type("StateKey", "String"))
    for concept in data["ontology"]:
        if concept["concept"] == "State":
            concept["relationships"].append(rel("key", "StateKey", multiplicity="OneToOne",
                                                verbalizes="{State} has key {StateKey}"))
        if concept["concept"] == "County":
            concept["relationships"].append(rel("state", "State", multiplicity="ManyToOne",
                                                verbalizes="{County} lies in {State}"))
    return OntologyDocument.model_validate(data)


def keyed_states():
    states = dataset("states", "states.csv", field("st"), field("state_name"))
    binding = Binding(rows="State", binding=[BindingEntry(concept="State", identify={"name": "state_name"},
                                                          relationships={"code": ["st"],
                                                                         "key": [{"normalized": "state_name"}]})])
    return states, binding


def counties_of_a_state():
    counties = dataset("counties_nh", "counties/NewHampshire.csv", field("county"),
                       DatasetField(name="state", expression=Expression(sql=quote_literal("NewHampshire")),
                                    datatype="String"))
    binding = Binding(rows="County", binding=[BindingEntry(concept="County", identify={"name": "county"},
                                                           relationships={"state": [{"key": {"normalized": "state"}}]})])
    return counties, binding


def test_normalize_keeps_only_lowercase_letters_and_digits():
    assert normalize("District of Columbia") == normalize("DistrictofColumbia") == "districtofcolumbia"
    assert normalize("Credit Cards§") == "creditcards" and normalize("NY-NJ-PA") == "nynjpa"


def test_a_normalized_field_fills_a_text_value_type_and_identifies_through_a_key():
    assert binding_problems(*keyed_states()[::-1], keyed_ontology()) == []
    assert binding_problems(*counties_of_a_state()[::-1], keyed_ontology()) == []
    states, binding = keyed_states()
    binding.binding[0].relationships["key"] = [{"normalized": "state"}]
    assert any("no Field state" in p for p in binding_problems(binding, states, keyed_ontology()))
    problems = binding_problems(fire_binding(acres_burned=[{"normalized": "acres"}]), fires_dataset(), keyed_ontology())
    assert any("extends Decimal" in p and "normalized" in p for p in problems)


def test_the_data_checks_compare_normalized_values(tmp_path):
    states, binding = keyed_states()
    write(tmp_path, "states.csv", "st,state_name\nNH,New Hampshire\nNV,Nevada\n")
    assert data_problems(states, binding, keyed_ontology(), tmp_path, 100) == []
    write(tmp_path, "states.csv", "st,state_name\nNH,New Hampshire\nNX,NEW HAMPSHIRE\n")
    assert any("State.key is OneToOne" in p for p in data_problems(states, binding, keyed_ontology(), tmp_path, 100))


def test_a_normalized_value_compiles_to_a_portable_sql_expression():
    counties, binding = counties_of_a_state()
    mapping = compiled([(counties, binding)], keyed_ontology())[0]
    state = mapping["link_mappings"][0]["children"][0]["object_mapping"]
    assert state == {"concept": "State", "referent_mappings": [
        {"relationship": "State.key",
         "expression": "lower(regexp_replace(counties_nh.state, '[^A-Za-z0-9]', '', 'g'))"}]}
    model = SemanticModel(name="toy_semantic_model", datasets=[counties])
    validate(compile_mapping(keyed_ontology(), model, {"counties_nh": binding}, "toy"))


def test_the_checks_of_a_region_share_one_read_of_the_rows_they_need(tmp_path, monkeypatch):
    import ogda.lake as lake

    beaches, binding = malibu_beaches()
    write(tmp_path, "beaches.csv", "Beach,Length\n" + "".join(f"B{i},{i}\n" for i in range(1000)))
    reads, real = [], lake.read_rows
    monkeypatch.setattr(lake, "read_rows",
                        lambda path, sheet, count=None: reads.append((path.name, sheet, count)) or real(path, sheet, count))
    reader = RowReader()
    assert structure_problems(beaches, tmp_path, 10, reader) == []
    assert data_problems(beaches, binding, valid_ontology(), tmp_path, 10, reader) == []
    assert reads == [("beaches.csv", None, 11)]


def test_a_headerless_region_is_checked_on_its_data_rows(tmp_path):
    write(tmp_path, "pairs.csv", "Zuma,3\nPoint Dume,1\n")
    pairs = with_ogda_extension(dataset("pairs", "pairs.csv", field("beach_name", "0"), field("length", "1")),
                                OgdaExtension(path="pairs.csv", header_row=None, start_row=0))
    assert structure_problems(pairs, tmp_path, 10) == []
    assert region_rows(pairs, tmp_path, 10) == (["0", "1"], [(0, ["Zuma", "3"]), (1, ["Point Dume", "1"])])
