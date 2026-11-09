"""An example Ontology shared by the Ossie and Binding tests: fires, counties, beaches and states."""

from ogda.ossie import OntologyDocument


def value_type(name, base):
    return {"concept": name, "type": "ValueType", "extends": [base]}


def rel(name, *roles, multiplicity=None, verbalizes):
    data = {"name": name, "roles": [{"concept": role} for role in roles], "verbalizes": [verbalizes]}
    return {**data, "multiplicity": multiplicity} if multiplicity else data


def valid_data() -> dict:
    return {"name": "toy_ontology", "ontology": [
        *(value_type(name, base) for name, base in [
            ("IncidentNumber", "String"), ("Acres", "Decimal"), ("CountyName", "String"), ("BeachName", "String"),
            ("MunicipalityName", "String"), ("StateName", "String"), ("StateCode", "String"), ("Year", "Integer")]),
        {"concept": "Fire", "type": "EntityType", "identify_by": ["incident_number"], "relationships": [
            rel("incident_number", "IncidentNumber", multiplicity="OneToOne",
                verbalizes="{Fire} has incident number {IncidentNumber}"),
            rel("acres_burned", "Acres", multiplicity="ManyToOne", verbalizes="{Fire} burned {Acres}"),
            rel("county", "County", multiplicity="ManyToOne", verbalizes="{Fire} started in {County}")]},
        {"concept": "County", "type": "EntityType", "identify_by": ["name"], "relationships": [
            rel("name", "CountyName", multiplicity="OneToOne", verbalizes="{County} is named {CountyName}"),
            rel("burned_in", "Year", "Acres", multiplicity="ManyToOne", verbalizes="{County} burned {Acres} in {Year}")]},
        {"concept": "Municipality", "type": "EntityType", "identify_by": ["name"], "relationships": [
            rel("name", "MunicipalityName", multiplicity="OneToOne",
                verbalizes="{Municipality} is named {MunicipalityName}")]},
        {"concept": "Beach", "type": "EntityType", "identify_by": ["name", "municipality"], "relationships": [
            rel("name", "BeachName", multiplicity="ManyToOne", verbalizes="{Beach} is named {BeachName}"),
            rel("municipality", "Municipality", multiplicity="ManyToOne", verbalizes="{Beach} lies in {Municipality}")]},
        {"concept": "State", "type": "EntityType", "identify_by": ["name"], "relationships": [
            rel("name", "StateName", multiplicity="OneToOne", verbalizes="{State} is named {StateName}"),
            rel("code", "StateCode", multiplicity="OneToOne", verbalizes="{State} has code {StateCode}")]},
        {"concept": "Capital", "type": "EntityType", "extends": ["State"]},
    ]}


def valid_ontology() -> OntologyDocument:
    return OntologyDocument.model_validate(valid_data())
