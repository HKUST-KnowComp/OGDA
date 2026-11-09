"""Deterministic lexical search over every Ontology object, then an n-hop walk over the Ontology graph.

Nodes are Concepts, Schemas, Fields, Tables and Files. Edges join a Field or a Table to its Schema
(`field_of`, `table_of`), a Table to its File (`file_of`), and the two ends of every Link. A query scores each
Concept, Schema, Field and Table by term overlap; the top matches of any type seed a breadth-first walk that
ignores edge direction. A node passes at most `NEIGHBORS` neighbours to the next hop, ranked by the query, so a
Schema with many Fields or Tables keeps the result small. Schemas come back as summaries; `schema_detail` opens
one. Experience is returned when it is anchored to a returned object or when its text matches.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from .ontology import Ontology

STOP = {"the", "a", "an", "of", "and", "in", "to", "for", "is", "are", "with", "what", "which",
        "how", "from", "by", "on", "at", "as", "be", "or", "that", "this"}
NEIGHBORS = 10
TABLES_PER_SCHEMA = 10


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


def table_counts(ontology: Ontology) -> Counter:
    return Counter(table.schema_id for table in ontology.tables.values())


def schema_summary(ontology: Ontology, schema_id: str, counts: Counter | None = None) -> dict:
    """A Schema without its Fields: what one record is, its Documentation and how many Fields and Tables it has."""
    schema = ontology.schemas[schema_id]
    counts = table_counts(ontology) if counts is None else counts
    return {"id": schema.id, "description": schema.description, "origin": schema.origin,
            "documentation": schema.documentation, "field_count": len(schema.fields), "table_count": counts[schema.id]}


def schema_detail(ontology: Ontology, schema_id: str, tables: int = TABLES_PER_SCHEMA) -> dict:
    """A Schema with all its Fields and its first Tables; table_count says how many Tables it has."""
    if schema_id not in ontology.schemas:
        raise ValueError(f"No Schema {schema_id!r}; search_ontology finds Schema ids")
    own = sorted((t for t in ontology.tables.values() if t.schema_id == schema_id), key=lambda t: t.id)
    return {**ontology.schemas[schema_id].model_dump(mode="json"), "table_count": len(own),
            "tables": [t.model_dump(mode="json") for t in own[:tables]]}


def graph(ontology: Ontology) -> dict[str, list[tuple[str, tuple[str, str, str]]]]:
    """Each node's neighbours, with the directed edge (source, kind, target) that joins them."""
    edges = [(f.id, "field_of", s.id) for s in ontology.schemas.values() for f in s.fields]
    edges += [(t.id, "table_of", t.schema_id) for t in ontology.tables.values()]
    edges += [(t.id, "file_of", t.path) for t in ontology.tables.values()]
    edges += [(link.source, link.kind, link.target) for link in ontology.links.values()]
    neighbours: dict[str, list[tuple[str, tuple[str, str, str]]]] = {}
    for edge in edges:
        source, _, target = edge
        neighbours.setdefault(source, []).append((target, edge))
        neighbours.setdefault(target, []).append((source, edge))
    return neighbours


def retrieve(ontology: Ontology, query: str, limit: int = 10, hops: int = 2) -> dict:
    fields = ontology.fields()
    texts = {c.id: f"{c.name} {c.description}" for c in ontology.concepts.values()}
    texts.update({f.id: f"{f.name} {f.column} {f.description}" for f in fields.values()})
    texts.update({s.id: f"{s.id} {s.description}" for s in ontology.schemas.values()})
    texts.update({t.id: f"{t.id} {t.path}" for t in ontology.tables.values()})
    score = {node: similarity(query, text) for node, text in texts.items()}
    seeds = top([(value, node) for node, value in score.items()], limit)

    neighbours = graph(ontology)
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
    schema_of = {f.id: s.id for s in ontology.schemas.values() for f in s.fields}
    counts = table_counts(ontology)
    tables = [ontology.tables[n] for n in order if n in ontology.tables]
    result = {
        "matched": seeds,
        "concepts": [ontology.concepts[n].model_dump(mode="json") for n in order if n in ontology.concepts],
        "fields": [{**fields[n].model_dump(mode="json"), "schema_id": schema_of[n]} for n in order if n in fields],
        "schemas": [schema_summary(ontology, n, counts) for n in order if n in ontology.schemas],
        "tables": [t.model_dump(mode="json") for t in tables],
        "files": [n for n in order if n not in texts],
        "edges": [{"source": s, "kind": k, "target": t} for s, k, t in walked],
        "paths": [paths[n] for n in order if distance[n] > 0],
    }
    anchors = set(distance) | {t.path for t in tables}
    live = ontology.live_experience()
    anchored = {e.id for e in live if anchors & set(e.anchors)}
    anchored |= set(top([(similarity(query, e.text), e.id) for e in live if e.id not in anchored], limit))
    result["experience"] = [e.labelled() for e in live if e.id in anchored]
    return result
