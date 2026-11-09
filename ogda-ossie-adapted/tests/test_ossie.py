"""Ossie documents: the models, YAML, schema validation, the OGDA extension and the Ontology rules."""

import jsonschema
import pydantic
import pytest

from ossie_examples import rel, valid_data, valid_ontology
from ogda.ossie import (Concept, ConceptMapping, Dataset, DatasetField, DocumentReference, Expression, LinkMapping,
                        MappingDocument, ObjectMapping, OgdaExtension, OntologyDocument, ReferentMapping, Relationship,
                        Role, SemanticModel, base_type, dump, find_relationship, identifiers, ogda_extension,
                        ontology_problems, owner_of, read_yaml, role_concepts, validate, with_ogda_extension, write_yaml)


def fire_ontology() -> OntologyDocument:
    return OntologyDocument(name="toy_ontology", ontology=[
        Concept(concept="IncidentNumber", type="ValueType", extends=["String"]),
        Concept(concept="Acres", type="ValueType", extends=["Decimal"], description="An area in acres"),
        Concept(concept="Fire", type="EntityType", identify_by=["incident_number"], relationships=[
            Relationship(name="incident_number", roles=[Role(concept="IncidentNumber")], multiplicity="OneToOne",
                         verbalizes=["{Fire} has incident number {IncidentNumber}"]),
            Relationship(name="acres_burned", roles=[Role(concept="Acres")], multiplicity="ManyToOne",
                         verbalizes=["{Fire} burned {Acres}"]),
        ]),
    ])


def fires_model() -> SemanticModel:
    dataset = Dataset(name="fires_2020", source="fires/2020.csv", primary_key=["incident_id"], fields=[
        DatasetField(name="incident_id", expression=Expression(sql='"Incident ID"'), datatype="String"),
        DatasetField(name="acres", expression=Expression(sql='"Acres Burned"'), datatype="Decimal",
                     description="Acres burned"),
    ])
    return SemanticModel(name="toy_semantic_model", datasets=[
        with_ogda_extension(dataset, OgdaExtension(path="fires/2020.csv", documentation=["fires/README.md"]))])


def fire_mapping() -> MappingDocument:
    fire = ObjectMapping(referent_mappings=[ReferentMapping(relationship="Fire.incident_number",
                                                             expression="fires_2020.incident_id")])
    return MappingDocument(
        name="toy_mapping", ontology_ref=DocumentReference(name="toy_ontology", iri="./ontology.yaml"),
        semantic_model_ref=DocumentReference(name="toy_semantic_model", iri="./semantic_model.yaml"),
        concept_mappings=[ConceptMapping(concept="Fire", object_mappings=[fire], link_mappings=[LinkMapping(
            object_mapping=fire, children=[LinkMapping(
                object_mapping=ObjectMapping(concept="Acres", expression="fires_2020.acres"),
                relationship="acres_burned")])])])


def test_documents_round_trip_through_yaml_and_validate(tmp_path):
    for document in (fire_ontology(), fires_model(), fire_mapping()):
        path = tmp_path / f"{document.name}.yaml"
        write_yaml(path, document)
        assert read_yaml(path, type(document)) == document


def test_dump_drops_empty_lists_and_nones():
    data = dump(Dataset(name="fires", source="fires.csv",
                        fields=[DatasetField(name="acres", expression=Expression(sql='"Acres"'))]))
    assert data == {"name": "fires", "source": "fires.csv",
                    "fields": [{"name": "acres", "expression": {"sql": '"Acres"'}}]}


def test_validate_applies_the_ossie_schemas():
    with pytest.raises(jsonschema.ValidationError):
        validate(OntologyDocument(name="x"))  # An ontology needs at least one Concept.
    with pytest.raises(jsonschema.ValidationError):
        validate(SemanticModel(name="x"))  # A semantic model needs at least one dataset.
    validate(fire_mapping())


def test_models_forbid_keys_ossie_does_not_define():
    with pytest.raises(pydantic.ValidationError):
        Dataset.model_validate({"name": "a", "source": "a.csv", "sheet": "S"})


def test_the_ogda_extension_round_trips_inside_custom_extensions():
    dataset = with_ogda_extension(Dataset(name="a", source="a.csv"), OgdaExtension(path="a.csv", end_row=3))
    assert [e.vendor_name for e in dataset.custom_extensions] == ["OGDA"]
    assert ogda_extension(dataset).end_row == 3
    assert ogda_extension(with_ogda_extension(dataset, OgdaExtension(path="a.csv"))).end_row is None
    assert len(with_ogda_extension(dataset, OgdaExtension(path="a.csv")).custom_extensions) == 1


def test_the_ogda_extension_checks_its_region():
    with pytest.raises(pydantic.ValidationError, match="header rows must precede start_row"):
        OgdaExtension(path="a.csv", header_row=2, start_row=1)


# The Ontology rules.

def concept_in(data, name):
    return next(c for c in data["ontology"] if c["concept"] == name)


def rel_in(data, relationship_id):
    owner, name = relationship_id.split(".")
    return next(r for r in concept_in(data, owner)["relationships"] if r["name"] == name)


def change(function):
    return [function]


def two_bases(name):
    return change(lambda d: concept_in(d, name).update(extends=["Decimal", "Integer"]))


def value_type_relationship(name):
    return change(lambda d: concept_in(d, name).update(
        relationships=[rel("unit", "CountyName", verbalizes=f"{{{name}}} has unit {{CountyName}}")]))


def role_played_by(relationship_id, concept):
    return change(lambda d: rel_in(d, relationship_id)["roles"][0].update(concept=concept))


def no_roles(relationship_id):
    return change(lambda d: rel_in(d, relationship_id).update(roles=[]))


def drop_identify_by(name):
    return change(lambda d: concept_in(d, name).update(identify_by=[]))


def multiplicity(relationship_id, value):
    return change(lambda d: rel_in(d, relationship_id).update(multiplicity=value))


def identify_by(name, relationships):
    return change(lambda d: concept_in(d, name).update(identify_by=relationships))


def verbalizes(relationship_id, patterns):
    return change(lambda d: rel_in(d, relationship_id).update(verbalizes=patterns))


def extends(name, supertypes):
    return change(lambda d: concept_in(d, name).update(extends=supertypes))


def duplicate_concept(name):
    return change(lambda d: d["ontology"].append(dict(concept_in(d, name))))


def rename_concept(old, new):
    return change(lambda d: concept_in(d, old).update(concept=new))


def rename_relationship(relationship_id, new):
    return change(lambda d: rel_in(d, relationship_id).update(name=new))


def apply_change(changes) -> OntologyDocument:
    data = valid_data()
    for function in changes:
        function(data)
    return OntologyDocument.model_validate(data)


@pytest.mark.parametrize("changes, message", [
    (two_bases("Acres"), "extends exactly one built-in"),
    (value_type_relationship("Acres"), "has no Relationships"),
    (role_played_by("Fire.acres_burned", "Decimal"), "built-in"),
    (no_roles("Fire.acres_burned"), "at least one Role after"),
    (drop_identify_by("County"), "needs identify_by"),
    (multiplicity("County.name", "ManyToOne"), "OneToOne"),
    (multiplicity("Beach.municipality", "OneToOne"), "ManyToOne"),
    (identify_by("County", ["population"]), "not a Relationship of County"),
    (verbalizes("Fire.acres_burned", ["{Fire} burned"]), "every Role"),
    (verbalizes("Fire.acres_burned", ["{Fire} burned {Acre}"]), "names no Role"),
    (extends("County", ["Fire"]) + extends("Fire", ["County"]), "cycle"),
    (extends("County", ["Acres"]), "extends only Entity types"),
    (duplicate_concept("County"), "more than once"),
    (rename_concept("County", "county"), "PascalCase"),
    (rename_relationship("Fire.acres_burned", "AcresBurned"), "snake_case"),
    (role_played_by("Fire.county", "Town"), "no Concept Town"),
])
def test_ontology_rules(changes, message):
    problems = ontology_problems(apply_change(changes))
    assert any(message in problem for problem in problems), problems


def test_a_valid_ontology_has_no_problems():
    assert ontology_problems(valid_ontology()) == []


def test_identifiers_list_the_preferred_one_then_alternates_and_inherit():
    document = valid_ontology()
    assert identifiers(document, "State") == [["name"], ["code"]]
    assert identifiers(document, "Capital") == [["name"], ["code"]]
    assert identifiers(document, "Beach") == [["name", "municipality"]]
    assert base_type(document, "Acres") == "Decimal" and base_type(document, "Fire") is None


def test_relationships_are_found_by_id_with_their_role_concepts():
    document = valid_ontology()
    burned_in = find_relationship(document, "County.burned_in")
    assert role_concepts("County", burned_in) == ["County", "Year", "Acres"]
    assert find_relationship(document, "County.population") is None
    assert owner_of(document, "Capital", "code") == "State"
