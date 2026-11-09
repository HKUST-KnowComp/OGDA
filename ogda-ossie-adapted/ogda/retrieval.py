"""Deterministic lexical search over the Harness, then an n-hop walk over its graph, and the summaries agents read.

Nodes are Concepts, Relationships (`Concept.name`), Datasets, Fields (`dataset.field`) and Files. Edges join a
Relationship to the Concepts of its Roles (`role`), a Concept to its supertypes (`extends`), a Field to its Dataset
(`field_of`), a Dataset to the Concepts its Binding has entries for (`binds`) and the Relationships it fills (`fills`),
and a Dataset to its File (`file_of`). Members of a layout group are not nodes: their representative stands for them
with a member count. A query scores Concepts, Relationships, Datasets and Fields by term overlap; the top matches seed
a breadth-first walk that ignores edge direction, passing at most `NEIGHBORS` neighbours on per hop.

Link candidates are Fields whose values identify Entities of an Entity type the Field is not bound to identify; the
concept layer decides whether each one is a link.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

from smolagents import tool

from .binding import (field_of, field_source, fillings, identify_relationship, is_entity, is_literal, named_concepts,
                      normalize, region_rows, walk)
from .ontology import Harness
from .ossie import concepts, dump, identifiers, ogda_extension, owner_of, role_concepts

LINK_SHARE = 0.8  # Share of a Field's distinct values that must identify Entities for a link candidate.
LINK_VALUES = 3   # Distinct values a Field needs before its matches mean anything.
LINK_LIMIT = 40   # Link candidates in one concept-layer task.
STOP = {"the", "a", "an", "of", "and", "in", "to", "for", "is", "are", "with", "what", "which",
        "how", "from", "by", "on", "at", "as", "be", "or", "that", "this"}
NEIGHBORS = 10


def tokens(text: str) -> Counter:
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(text))
    words = re.findall(r"[a-z0-9]+", text.lower())
    return Counter(w[:-1] if len(w) > 4 and w.endswith("s") else w for w in words if w not in STOP)


def similarity(query: str, text: str) -> float:
    a, b = tokens(query), tokens(text)
    if not a or not b:
        return 0.0
    dot = sum(count * b[word] for word, count in a.items())
    return dot / math.sqrt(sum(v * v for v in a.values()) * sum(v * v for v in b.values()))


def top(scored: list[tuple[float, str]], limit: int) -> list[str]:
    return [key for score, key in sorted(scored, key=lambda pair: (-pair[0], pair[1]))[:limit] if score > 0]


def representatives(harness: Harness) -> list[str]:
    """The Datasets that are not layout group members, by name."""
    return sorted(d.name for d in harness.semantic_model.datasets if ogda_extension(d).layout_of is None)


def ontology_summary(harness: Harness) -> list[dict]:
    """Every Concept with its Relationships, without verbalizations."""
    return [{"concept": c.concept, "type": c.type, "description": c.description, "extends": c.extends,
             "identify_by": c.identify_by,
             "relationships": [{"name": r.name, "roles": [role.concept for role in r.roles],
                                "multiplicity": r.multiplicity, "description": r.description}
                               for r in c.relationships]}
            for c in harness.ontology.ontology]


def dataset_summary(harness: Harness, name: str) -> dict:
    dataset, binding = harness.dataset(name), harness.binding_of(name)
    extension = ogda_extension(dataset)
    return {"name": name, "path": extension.path, "sheet": extension.sheet, "description": dataset.description,
            "rows": binding.rows if binding else None,
            "concepts": list(dict.fromkeys(e.concept for e in binding.binding)) if binding else [],
            "members": len(harness.members(name))}


def concept_detail(harness: Harness, name: str) -> dict:
    """An Ossie Concept with the Datasets that bind it: whether their rows are its Entities, how they identify it, and
    the values that fill each Relationship one of whose Roles it plays."""
    concept = harness.concept(name)
    if concept is None:
        raise ValueError(f"No Concept {name!r}; search_ontology finds Concepts")
    datasets = []
    for dataset in representatives(harness):
        binding = harness.binding_of(dataset)
        if binding is None or name not in named_concepts(binding, harness.ontology):
            continue
        table, item = concepts(harness.ontology), {"name": dataset, "members": len(harness.members(dataset)),
                                                   "rows": binding.rows == name}
        if own := next((e for e in binding.binding if e.concept == name), None):
            item["identify"] = own.identify
        fills = {}
        for entry in binding.binding:
            for relationship in table[entry.concept].relationships if entry.concept in table else []:
                if relationship.name in entry.relationships and name in role_concepts(entry.concept, relationship):
                    fills[f"{entry.concept}.{relationship.name}"] = fillings(entry, relationship.name)
        datasets.append({**item, "fills": fills})
    return {**dump(concept), "datasets": datasets}


def dataset_detail(harness: Harness, name: str) -> dict:
    """An Ossie Dataset with its region and origin, its Binding and how many layout group members share it."""
    dataset = harness.dataset(name)
    if dataset is None:
        raise ValueError(f"No Dataset {name!r}; search_ontology finds Datasets")
    binding = harness.binding_of(name)
    data = dump(dataset)
    data.pop("custom_extensions", None)
    return {**data, "extension": ogda_extension(dataset).model_dump(mode="json"),
            "binding": binding.model_dump(mode="json") if binding else None, "members": len(harness.members(name))}


def texts(harness: Harness) -> dict[str, str]:
    """The text each searchable node is matched by."""
    found = {c.concept: f"{c.concept} {c.type} {c.description or ''}" for c in harness.ontology.ontology}
    found.update({f"{c.concept}.{r.name}": f"{c.concept} {r.name} {r.description or ''} {' '.join(r.verbalizes)}"
                  for c in harness.ontology.ontology for r in c.relationships})
    for name in representatives(harness):
        dataset = harness.dataset(name)
        found[name] = f"{name} {ogda_extension(dataset).path} {dataset.description or ''}"
        found.update({f"{name}.{f.name}": f"{f.name} {field_source(f)[1]} {f.description or ''}" for f in dataset.fields})
    return found


def graph(harness: Harness) -> dict[str, list[tuple[str, tuple[str, str, str]]]]:
    """Each node's neighbours, with the directed edge (source, kind, target) that joins them."""
    edges = []
    for concept in harness.ontology.ontology:
        edges += [(concept.concept, "extends", parent) for parent in concept.extends]
        for relationship in concept.relationships:
            edges += [(f"{concept.concept}.{relationship.name}", "role", role)
                      for role in dict.fromkeys(role_concepts(concept.concept, relationship))]
    for name in representatives(harness):
        dataset, binding = harness.dataset(name), harness.binding_of(name)
        edges += [(f"{name}.{f.name}", "field_of", name) for f in dataset.fields]
        edges.append((name, "file_of", ogda_extension(dataset).path))
        for entry in binding.binding if binding else []:
            edges.append((name, "binds", entry.concept))
            edges += [(name, "fills", f"{entry.concept}.{relationship}") for relationship in entry.relationships]
    neighbours: dict[str, list[tuple[str, tuple[str, str, str]]]] = {}
    for edge in dict.fromkeys(edges):
        source, _, target = edge
        neighbours.setdefault(source, []).append((target, edge))
        neighbours.setdefault(target, []).append((source, edge))
    return neighbours


def retrieve(harness: Harness, query: str, limit: int = 10, hops: int = 2, experience: bool = True) -> dict:
    searchable = texts(harness)
    score = {node: similarity(query, text) for node, text in searchable.items()}
    seeds = top([(value, node) for node, value in score.items()], limit)

    neighbours = graph(harness)
    distance, paths, walked = {node: 0 for node in seeds}, {node: [node] for node in seeds}, {}
    frontier = list(seeds)
    for hop in range(1, hops + 1):
        reached = []
        for node in frontier:
            for neighbour, edge in sorted(neighbours.get(node, []), key=lambda n: (-score.get(n[0], 0.0), n[0]))[:NEIGHBORS]:
                walked.setdefault(edge, None)
                if neighbour not in distance:
                    distance[neighbour], paths[neighbour] = hop, [*paths[node], edge[1], neighbour]
                    reached.append(neighbour)
        frontier = reached

    order = sorted(distance, key=lambda node: (distance[node], -score.get(node, 0.0), node))
    table, datasets, fields = concepts(harness.ontology), set(representatives(harness)), harness.fields()
    relationships = {f"{c.concept}.{r.name}": (c.concept, r) for c in harness.ontology.ontology for r in c.relationships}
    summary = {c["concept"]: c for c in ontology_summary(harness)}
    result = {
        "matched": seeds,
        "concepts": [summary[n] for n in order if n in table],
        "relationships": [{"id": n, "roles": role_concepts(owner, r), "multiplicity": r.multiplicity,
                           "verbalizes": r.verbalizes, "description": r.description}
                          for n in order if n in relationships for owner, r in [relationships[n]]],
        "datasets": [dataset_summary(harness, n) for n in order if n in datasets],
        "fields": [{"id": n, **dump(fields[n]), "dataset": n.split(".")[0]} for n in order if n in fields],
        "files": [n for n in order if n not in searchable and n not in table and n not in datasets],
        "edges": [{"source": s, "kind": k, "target": t} for s, k, t in walked],
        "paths": [paths[n] for n in order if distance[n] > 0],
    }
    if experience:
        anchors = set(distance)
        live = harness.live_experience()
        anchored = {e.id for e in live if anchors & set(e.anchors)}
        anchored |= set(top([(similarity(query, e.text), e.id) for e in live if e.id not in anchored], limit))
        result["experience"] = [e.labelled() for e in live if e.id in anchored]
    return result


def similar_concepts(harness: Harness, name: str, candidates: list[str], limit: int) -> list[str]:
    """The candidates most like `name` by term overlap over names, descriptions and verbalizations; ties by name."""
    def text(concept: str) -> str:
        found = harness.concept(concept)
        verbalized = " ".join(p for r in found.relationships for p in r.verbalizes) if found else ""
        return f"{concept} {getattr(found, 'description', '') or ''} {verbalized}"

    reference = text(name)
    ranked = sorted(candidates, key=lambda candidate: (-similarity(reference, text(candidate)), candidate))
    return ranked[:max(0, limit)]


def ontology_tools(harness: Harness) -> list:
    """The read-only search tools that construction agents get."""

    @tool
    def search_ontology(query: str, hops: int = 2) -> dict:
        """Return the Concepts, Relationships, Datasets and Fields that match a text query and their neighbours, with
        the edges and paths walked.

        Args:
            query: Words to match against Concept, Relationship, Dataset and Field names and descriptions.
            hops: How many edges to walk from the matched objects.
        """
        return retrieve(harness, query, hops=hops)

    @tool
    def get_concept(name: str) -> dict:
        """Return one Concept as Ossie defines it, with its Relationships and the Datasets that bind it.

        Args:
            name: The name of a Concept, such as County.
        """
        return concept_detail(harness, name)

    @tool
    def get_dataset(name: str) -> dict:
        """Return one Dataset with its Fields, region, Documentation, Binding and layout group member count.

        Args:
            name: The name of a Dataset.
        """
        return dataset_detail(harness, name)

    return [search_ontology, get_concept, get_dataset]


def field_values(harness: Harness, lake_root: Path, limit: int, cache: dict) -> dict[tuple[str, str], set[str]]:
    """The distinct non-empty values of each Field on the first `limit` rows, per representative Dataset: a layout
    group's members add theirs to the representative's Fields. `cache` keeps each Dataset's values between calls."""
    merged: dict[tuple[str, str], set[str]] = {}
    for name, dataset in harness.datasets().items():
        if name not in cache:
            region = region_rows(dataset, lake_root, limit)
            values = {}
            for field in dataset.fields:
                kind, text = field_source(field)
                if kind == "literal":
                    values[field.name] = {text.strip()} - {""}
                elif region is not None and text.strip() in region[0]:
                    column = region[0].index(text.strip())
                    values[field.name] = {row[column] for _, row in region[1]} - {""}
            cache[name] = values
        representative = ogda_extension(dataset).layout_of or name
        for field, values in cache[name].items():
            merged.setdefault((representative, field), set()).update(values)
    return merged


def link_candidates(harness: Harness, lake_root: Path, limit: int, cache: dict | None = None) -> list[dict]:
    """Fields whose values identify Entities of an Entity type that the Field does not identify in its Binding. For
    each part of each Identifier, the values that identify Entities in any Dataset are gathered; a Field is listed
    against a part when at least LINK_SHARE of its distinct values are among them, compared exactly or normalized
    (letters and digits only, lowercased). Fields with fewer than LINK_VALUES distinct values, or only numbers, are
    skipped."""
    document, values = harness.ontology, field_values(harness, lake_root, limit, {} if cache is None else cache)
    known: dict[tuple[str, str], set[str]] = {}  # (owner, identifying Relationship) -> identifying values
    identifies: dict[tuple[str, str], set[str]] = {}  # (Dataset, Field) -> Entity types it identifies
    for name in representatives(harness):
        binding = harness.binding_of(name)
        for value, concept, _ in walk(binding, document) if binding else []:
            if not (isinstance(value, dict) and is_entity(document, concept)):
                continue
            for key, part in value.items():
                relationship = identify_relationship(document, concept, key)
                if relationship is None or len(relationship.roles) != 1 or is_entity(document, relationship.roles[0].concept):
                    continue
                slot = known.setdefault((owner_of(document, concept, key), key), set())
                if field := field_of(part):
                    slot |= values.get((name, field), set())
                    identifies.setdefault((name, field), set()).add(concept)
                elif is_literal(part):
                    slot.add(str(part["value"]))
    normalized = {slot: {normalize(v) for v in found} for slot, found in known.items()}
    candidates = []
    for (name, field), found in sorted(values.items()):
        if len(found) < LINK_VALUES or all(number(v) for v in found):
            continue
        for (owner, key), targets in sorted(known.items()):
            if owner in identifies.get((name, field), set()):
                continue
            exact = found & targets
            close = {v for v in found if normalize(v) in normalized[owner, key]}
            if max(len(exact), len(close)) < LINK_SHARE * len(found):
                continue
            candidates.append({"field": f"{name}.{field}", "values": len(found), "target": f"{owner}.{key}",
                               "identifier": next((o for o in identifiers(document, owner) if key in o), [key]),
                               "exact": len(exact), "normalized": len(close),
                               "unmatched": sorted(found - exact)[:3]})
    candidates.sort(key=lambda c: (-max(c["exact"], c["normalized"]) / c["values"], -c["values"], c["field"]))
    return candidates[:LINK_LIMIT]


def number(text: str) -> bool:
    try:
        float(text.replace(",", ""))
        return True
    except ValueError:
        return False
