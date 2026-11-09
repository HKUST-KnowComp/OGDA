"""County Lake fixtures: a toy Ontology and Harness over a small Lake, shared by tests."""

import openpyxl

from ogda.binding import Binding, BindingEntry, quote_identifier, quote_literal
from ogda.ontology import Harness
from ogda.ossie import Concept, Dataset, DatasetField, Expression, OgdaExtension, Relationship, Role, with_ogda_extension

FIRES_PATTERN = r"fires/(?P<state>[^/]+)\.csv"


def value_type(name, base="String", description=None):
    return Concept(concept=name, type="ValueType", extends=[base], description=description)


def rel(name, *roles, multiplicity=None, says, description=None):
    return Relationship(name=name, roles=[Role(concept=role) for role in roles], multiplicity=multiplicity,
                        verbalizes=[says], description=description)


def entity(name, identify_by, *relationships, description):
    return Concept(concept=name, type="EntityType", identify_by=identify_by, relationships=list(relationships),
                   description=description)


def column(name, header, datatype="String", description=""):
    return DatasetField(name=name, expression=Expression(sql=quote_identifier(header)), datatype=datatype,
                        description=description)


def path_field(name, value, description):
    return DatasetField(name=name, expression=Expression(sql=quote_literal(value)), datatype="String",
                        description=description)


def table(name, path, fields, description, **extension):
    return with_ogda_extension(Dataset(name=name, source=path, description=description, fields=fields),
                               OgdaExtension(path=path, **extension))


def write_lake(lake):
    files = {"states.csv": "State\nNew Hampshire\nNevada\n",
             "counties.csv": "County,State,Notes\nCoos,New Hampshire,north\nWashoe,Nevada,west\n",
             "burned.csv": "County,2020\nCoos,10\nWashoe,20\n",
             "fires/NewHampshire.csv": "County\nCoos\n", "fires/Nevada.csv": "County\nWashoe\n",
             "wide.csv": "County," + ",".join(f"C{i:02d}" for i in range(20)) + "\nCoos," + ",".join("1" * 20) + "\n"}
    for path, text in files.items():
        (lake / path).parent.mkdir(parents=True, exist_ok=True)
        (lake / path).write_text(text)
    workbook = openpyxl.Workbook()
    workbook.active.title = "Malibu"
    workbook.active.append(["Beach", "community"])
    workbook.active.append(["Zuma", "Malibu"])
    workbook.save(lake / "beaches.xlsx")


def county_harness(lake) -> Harness:
    write_lake(lake)
    harness = Harness.empty("toy")
    harness.ontology.ontology = [
        value_type("StateName", description="The name of a state"),
        value_type("StateKey", description="A state name with only its letters and digits, lowercased"),
        value_type("CountyName", description="The name of a county"),
        value_type("Year", "Integer", "A calendar year"),
        value_type("Acres", "Decimal", "An area in acres"),
        value_type("BeachName", description="The name of a beach"),
        value_type("MunicipalityName", description="The name of a town"),
        entity("State", ["name"], rel("name", "StateName", multiplicity="OneToOne", says="{State} is named {StateName}"),
               rel("key", "StateKey", multiplicity="OneToOne", says="{State} has key {StateKey}"),
               description="A state of the United States"),
        entity("County", ["name"], rel("name", "CountyName", multiplicity="OneToOne", says="{County} is named {CountyName}"),
               rel("state", "State", multiplicity="ManyToOne", says="{County} lies in {State}",
                   description="The state the county lies in"),
               rel("burned_in", "Year", "Acres", multiplicity="ManyToOne", says="{County} burned {Acres} in {Year}",
                   description="Acres burned by wildfires in the county in a year"),
               description="A county"),
        entity("Municipality", ["name"], rel("name", "MunicipalityName", multiplicity="OneToOne",
                                             says="{Municipality} is named {MunicipalityName}"), description="A town"),
        entity("Beach", ["name", "municipality"],
               rel("name", "BeachName", multiplicity="ManyToOne", says="{Beach} is named {BeachName}"),
               rel("municipality", "Municipality", multiplicity="ManyToOne", says="{Beach} lies in {Municipality}"),
               description="A public beach"),
    ]
    fires = [column("county", "County", description="County with fires, e.g. Coos")]
    harness.semantic_model.datasets = [
        table("states", "states.csv", [column("state", "State", description="State name, e.g. Nevada")],
              "One row per state"),
        table("counties", "counties.csv", [column("county", "County", description="County name, e.g. Coos"),
                                           column("state_name", "State", description="State of the county, e.g. Nevada"),
                                           column("notes", "Notes", description="Free-text notes about the county")],
              "One row per county with its state"),
        table("burned", "burned.csv", [column("county", "County", description="County name, e.g. Coos"),
                                       column("acres_2020", "2020", "Decimal", "Acres burned in the county in 2020")],
              "Acres burned per county and year"),
        table("fires", "fires/NewHampshire.csv",
              fires + [path_field("state", "NewHampshire", "The state the File covers, from its name")],
              "Counties with fires, one File per state", path_pattern=FIRES_PATTERN),
        table("fires__nevada", "fires/Nevada.csv",
              fires + [path_field("state", "Nevada", "The state the File covers, from its name")],
              "Counties with fires, one File per state", path_pattern=FIRES_PATTERN, layout_of="fires"),
        table("beaches", "beaches.xlsx", [column("beach_name", "Beach", description="Beach name, e.g. Zuma"),
                                          column("community", "community", description="Town of the beach, e.g. Malibu")],
              "Beaches of one town", sheet="Malibu"),
        table("wide", "wide.csv", [column("county", "County", description="County name, e.g. Coos")]
              + [column(f"c{i:02d}", f"C{i:02d}", description=f"Measure number {i}") for i in range(20)],
              "Twenty measures per county"),
    ]
    county = BindingEntry(concept="County", identify={"name": "county"})
    harness.bindings = {
        "states": Binding(rows="State", binding=[BindingEntry(concept="State", identify={"name": "state"},
                                                              relationships={"key": [{"normalized": "state"}]})]),
        "counties": Binding(rows="County", binding=[county.model_copy(update={
            "relationships": {"state": [{"name": "state_name"}]}})], unbound={"notes": "Free text"}),
        "burned": Binding(binding=[county.model_copy(update={
            "relationships": {"burned_in": [[{"value": "2020"}, "acres_2020"]]}})]),
        "fires": Binding(rows="County", binding=[county.model_copy(update={
            "relationships": {"state": [{"key": {"normalized": "state"}}]}})]),
        "beaches": Binding(rows="Beach", binding=[BindingEntry(concept="Beach", identify={
            "name": "beach_name", "municipality": {"name": "community"}})]),
        "wide": Binding(rows="County", binding=[county],
                        unbound={f"c{i:02d}": "No Concept for this measure yet" for i in range(20)}),
    }
    return harness
