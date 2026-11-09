"""The Data agent's search over the Ontology: BM25 over the columns and tables of a Harness.

A column is a Field of a representative Dataset; its document holds its name, header text and description, the
Relationships it fills with their sentences and the Concepts of their Roles, and the Entity type it identifies or
refers to. A table is a representative Dataset; its document holds its name, description, File path and the Entity
type its rows are. A column scores its own BM25 plus half its table's. A search prints one line per column whose own
words match; `dataset` gives the structure the index built for a Dataset and `entity` that of an Entity type. No Lake
File is read.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter

from .binding import (field_of, field_source, fillings, identify_relationship, is_entity, is_literal,
                      is_normalized)
from .ontology import Harness
from .ossie import concepts, identifiers, ogda_extension, owner_of
from .retrieval import representatives, tokens

K1, B = 1.5, 0.75
TABLE_WEIGHT = 0.5  # A table's score adds to each of its columns at this weight.
LIMIT = 15          # Lines a search prints.
LISTED = 10         # Tables and References listed per Entity type.
EXAMPLES = 3        # Example paths of a layout group.
NO_MATCH = "No column matches these words; try others."
ABBREVIATIONS = {"e.g.", "i.e.", "etc.", "No.", "U.S."}  # Their periods end no sentence.


def bm25(documents: list[Counter], query: Counter, k1: float = K1, b: float = B) -> list[float]:
    """The BM25 score of each document for the query's words."""
    if not documents:
        return []
    count = len(documents)
    average = sum(sum(document.values()) for document in documents) / count or 1.0
    frequency = Counter(word for document in documents for word in document)
    idf = {word: math.log((count - frequency[word] + 0.5) / (frequency[word] + 0.5) + 1) for word in query}
    scores = []
    for document in documents:
        length, score = sum(document.values()), 0.0
        for word in query:
            if found := document.get(word, 0):
                score += idf[word] * found * (k1 + 1) / (found + k1 * (1 - b + b * length / average))
        scores.append(score)
    return scores


def first_sentence(text: str) -> str:
    """The text on one line, each run of whitespace one space, up to its first period followed by whitespace (not the
    one ending an abbreviation such as "e.g." or "U.S."); the whole text without a final period when there is none."""
    text = " ".join(text.split())
    for found in re.finditer(r"\.(?=\s)", text):
        if text[:found.end()].split()[-1].lstrip("([\"'") not in ABBREVIATIONS:
            return text[:found.start()]
    return text.removesuffix(".")


def has_literal(value) -> bool:
    """Whether an identify object names its Entity by a literal anywhere, not only by Fields."""
    return is_literal(value) or (isinstance(value, dict) and any(has_literal(inner) for inner in value.values()))


def role_keys(roles) -> list[str]:
    """Keys for a Relationship's Roles: the Role's name or its Concept, with .2, .3 for repeats."""
    seen, keys = Counter(), []
    for role in roles:
        base = role.name or role.concept
        seen[base] += 1
        keys.append(base if seen[base] == 1 else f"{base}.{seen[base]}")
    return keys


class OntologySearch:
    """BM25 over a Harness's columns and tables, built once."""

    def __init__(self, harness: Harness):
        self.harness, self.document = harness, harness.ontology
        self.concepts = concepts(self.document)
        self.tables = [self.table(name) for name in representatives(harness)]
        self.column_documents = [column["document"] for table in self.tables for column in table["columns"]]
        self.table_documents = [table["document"] for table in self.tables]
        self.where = [(t, c) for t, table in enumerate(self.tables) for c in range(len(table["columns"]))]
        self.entity_tables = {}
        for table in self.tables:
            for name in table["names"]:
                self.entity_tables.setdefault(name, []).append(table["label"])

    # Building the index.

    def describe(self, name: str | None) -> list[str]:
        concept = self.concepts.get(name)
        return [name or "", concept.description or ""] if concept else [name or ""]

    def table(self, name: str) -> dict:
        dataset, binding = self.harness.dataset(name), self.harness.binding_of(name)
        extension, members = ogda_extension(dataset), self.harness.members(name)
        columns = {field.name: {"field": field, "links": [], "text": [field.name.replace("_", " "), field.description or ""]}
                   for field in dataset.fields}
        for field in dataset.fields:
            kind, text = field_source(field)
            if kind == "column":
                columns[field.name]["text"].append(text)
        identified_by, named = [], set()

        def header(field_name: str) -> dict:
            field = columns[field_name]["field"]
            kind, text = field_source(field)
            return {"column": text} if kind == "column" else {"from_path": True}

        def shown(value, concept: str):
            """A value as the agent reads it: a column header, a literal, a path Field that differs per File of a
            layout group, or an identify object of them."""
            if (field := field_of(value)) and field in columns:
                kind, text = field_source(columns[field]["field"])
                return text if kind == "column" or not members else {"from_path": field}
            if is_literal(value):
                return str(value["value"])
            if isinstance(value, dict) and is_entity(self.document, concept):
                return {key: shown(inner, self.role_of(concept, key)) for key, inner in value.items()}
            return value

        def identify(value: dict, concept: str, via: str | None, link: dict | None, text: tuple = ()) -> None:
            """Record the Fields of an identify object for `concept`: as identifying the row's Entities when `link`
            is None, else as referring to `concept` through the fact in `link`, whose words `text` they get."""
            named.add(concept)
            for key, inner in value.items():
                relationship = identify_relationship(self.document, concept, key)
                if relationship is None or len(relationship.roles) != 1:
                    continue
                role = relationship.roles[0].concept
                qualified = f"{owner_of(self.document, concept, key)}.{key}"
                if is_entity(self.document, role) and isinstance(inner, dict):
                    identify(inner, role, qualified, link, text)
                    continue
                field = field_of(inner)
                if field not in columns:
                    continue
                column = columns[field]
                column["text"] += [key.replace("_", " "), *self.describe(concept), *text]
                if is_normalized(inner) and qualified not in column.setdefault("normalized", []):
                    column["normalized"].append(qualified)
                if link is None:
                    column["identifies"] = qualified
                    if via:
                        column["via"] = via
                    identified_by.append({**header(field), "field": field, "identifies": qualified,
                                          **({"via": via} if via else {})})
                else:
                    column["refers_to"] = qualified
                    if link not in column["links"]:
                        column["links"].append(link)

        for entry in binding.binding if binding else []:
            identify(entry.identify, entry.concept, None, None)
            of = {"of": shown(entry.identify, entry.concept)} if has_literal(entry.identify) else {}
            declared = {r.name: r for r in self.concepts[entry.concept].relationships} if entry.concept in self.concepts else {}
            for fact in entry.relationships:
                relationship = declared.get(fact)
                if relationship is None:
                    continue
                keys = role_keys(relationship.roles)
                says = (relationship.verbalizes or [""])[0]
                text = (fact.replace("_", " "), relationship.description or "", says.replace("{", "").replace("}", ""),
                        *self.describe(entry.concept),
                        *(word for role in relationship.roles for word in self.describe(role.concept)))
                for filling in fillings(entry, fact):
                    for index, (role, value) in enumerate(zip(relationship.roles, filling)):
                        link = {"entity": entry.concept, **of, "fact": f"{entry.concept}.{fact}", "says": says,
                                "role": role.concept,
                                "with": {keys[other]: shown(item, relationship.roles[other].concept)
                                         for other, item in enumerate(filling) if other != index}}
                        if is_entity(self.document, role.concept) and isinstance(value, dict):
                            identify(value, role.concept, None, link, text)
                        elif (field := field_of(value)) in columns:
                            columns[field]["links"].append(link)
                            columns[field]["text"] += text
                            if is_normalized(value) and link["fact"] not in columns[field].setdefault("normalized", []):
                                columns[field]["normalized"].append(link["fact"])
        rows = binding.rows if binding else None
        table_text = [name.replace("_", " "), dataset.description or "", extension.path, *self.describe(rows)]
        entries = []
        for field in dataset.fields:
            column = columns[field.name]
            entry = {**header(field.name), "field": field.name, "datatype": field.datatype,
                     "description": field.description or "", "links": column["links"]}
            for key in ("identifies", "via", "refers_to", "normalized"):
                if key in column:
                    entry[key] = column[key]
            if binding and field.name in binding.unbound:
                entry["unbound"] = binding.unbound[field.name]
            entries.append({"entry": entry, "document": tokens(" ".join(column["text"])),
                            "identifying": "identifies" in column})
        others = [ogda_extension(self.harness.dataset(member)) for member in members]
        paths = sorted([extension.path] + [other.path for other in others if other.relation == "part"])
        copies = sorted(({"file": other.path, f"{other.relation}_of": self.path_of(other.relation_to)}
                         for other in others if other.relation != "part"), key=lambda copy: copy["file"])
        label = f"{name} ({len(paths)} Files)" if len(paths) > 1 else name
        result = {"table": name, "description": dataset.description or "", "file": extension.path,
                  "sheet": extension.sheet, "header_row": extension.header_row, "header_rows": extension.header_rows,
                  "start_row": extension.start_row, "end_row": extension.end_row,
                  "documentation": list(extension.documentation)}
        if extension.relation != "part":
            result[f"{extension.relation}_of"] = self.path_of(extension.relation_to)
        if members:
            result["files"] = {"count": len(paths), "path_pattern": extension.path_pattern, "examples": paths[:EXAMPLES],
                               **({"copies": copies} if copies else {})}
        result.update({"row": f"one {rows}" if rows else "facts", "identified_by": identified_by})
        return {"result": result, "columns": entries, "document": tokens(" ".join(table_text)), "rows": rows,
                "label": label, "names": sorted(named | ({rows} if rows else set()))}

    def path_of(self, name: str) -> str:
        """The File of the Dataset a copy or subset relates to."""
        dataset = self.harness.dataset(name)
        return ogda_extension(dataset).path if dataset is not None else name

    def role_of(self, concept: str, key: str) -> str | None:
        relationship = identify_relationship(self.document, concept, key)
        return relationship.roles[0].concept if relationship is not None and relationship.roles else None

    # Searching.

    def dataset(self, name: str) -> dict | None:
        """The structure the index built for a representative Dataset, or None."""
        return next((table for table in self.tables if table["result"]["table"] == name), None)

    def search(self, words: str, limit: int = LIMIT) -> str:
        """One line per Field whose own words match, best first: `<dataset>.<field>  <header>  <datatype>  <first
        sentence>`. A Field ranks by its own BM25 plus TABLE_WEIGHT times its Dataset's."""
        try:
            whole = float(limit)
        except (TypeError, ValueError):
            whole = None
        if whole is None or isinstance(limit, bool) or not whole.is_integer():
            raise ValueError(f"limit must be a whole number of columns, such as 15, not {limit!r}")
        limit = int(whole)
        query = Counter(tokens(words))
        own = bm25(self.column_documents, query)
        tables = bm25(self.table_documents, query)
        ranked = sorted((-(score + TABLE_WEIGHT * tables[t]), self.tables[t]["result"]["table"], c, t)
                        for (t, c), score in zip(self.where, own) if score > 0)
        if not ranked:
            return NO_MATCH
        lines = []
        for _, name, c, t in ranked[:max(1, limit)]:
            entry = self.tables[t]["columns"][c]["entry"]
            header = json.dumps(entry["column"], ensure_ascii=False) if "column" in entry else "from path"
            parts = [f'{name}.{entry["field"]}', header, entry["datatype"], first_sentence(entry["description"])]
            lines.append("  ".join(part for part in parts if part))
        return "\n".join(lines)

    def entity(self, name: str) -> dict:
        concept = self.concepts.get(name)
        options = identifiers(self.document, name)
        tables = sorted(self.entity_tables.get(name, []))
        referenced = [f"{other.concept}.{r.name}" for other in self.document.ontology
                      if other.type == "EntityType" and other.concept != name
                      for r in other.relationships if any(role.concept == name for role in r.roles)]
        found = {"description": (concept.description or "") if concept else "",
                 "identifier": options[0] if options else [], "other_identifiers": options[1:],
                 "tables": tables[:LISTED], "referenced_by": referenced[:LISTED]}
        if len(tables) > LISTED:
            found["more_tables"] = len(tables) - LISTED
        if len(referenced) > LISTED:
            found["more_references"] = len(referenced) - LISTED
        return found
