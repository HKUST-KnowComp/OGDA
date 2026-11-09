"""Distill Experience from one visit's outcomes, and consolidate overlapping entries after a pass."""

from __future__ import annotations

import json

from pydantic import ValidationError

from .agent import Budget
from .claims import anchors
from .llm import InvalidResponse
from .ontology import Claim, ExperienceEntry, Harness
from .prompts import CONSOLIDATION, EXPERIENCE


def distill(llm, ontology: Harness, claims: list[Claim], summary: str, unresolved: list[str],
            iteration: int, budget: Budget) -> list[ExperienceEntry]:
    decided = {claim.id: claim for claim in claims if claim.status != "pending"}
    if not decided and not summary and not unresolved:
        return []
    budget.check(llm)
    request = {
        "outcomes": [{"claim_id": c.id, "kind": c.kind, "payload": c.payload, "status": c.status,
                      "notes": c.notes, "anchors": anchors(c)} for c in decided.values()],
        "proposer_summary": summary,
        "unresolved": unresolved,
    }
    response = llm.complete("experience", EXPERIENCE, json.dumps(request, ensure_ascii=False))
    entries = []
    for item in response.get("entries", []) if isinstance(response.get("entries"), list) else []:
        if not isinstance(item, dict):
            continue
        claim_ids = [cid for cid in item.get("claim_ids", []) if cid in decided]
        try:
            entry = ExperienceEntry(
                id=f"experience-{iteration:06d}-{len(entries)}", kind=item.get("kind"), text=item.get("text"),
                anchors=item.get("anchors", []), claim_ids=claim_ids,
                provisional=any(decided[cid].status == "undecidable" for cid in claim_ids),
            )
        except ValidationError:
            continue
        if entry.text.strip():
            entries.append(entry)
    return entries


def consolidate(llm, ontology: Harness, pass_number: int, budget: Budget) -> int:
    """Merge live entries of the same kind that share an anchor. Return the number of merged entries."""
    groups: list[list[ExperienceEntry]] = []
    for entry in ontology.live_experience():
        matching = [group for group in groups if group[0].kind == entry.kind
                    and set(entry.anchors) & {a for member in group for a in member.anchors}]
        merged = [entry] + [member for group in matching for member in group]
        groups = [group for group in groups if group not in matching] + [merged]
    count = 0
    for group in groups:
        if len(group) < 2:
            continue
        budget.check(llm)
        try:
            response = llm.complete("experience", CONSOLIDATION,
                                    json.dumps({"entries": [e.model_dump() for e in group]}, ensure_ascii=False))
        except InvalidResponse:
            continue
        if not isinstance(response.get("text"), str) or not response["text"].strip():
            continue
        merged = ExperienceEntry(
            id=f"experience-merge-{pass_number:04d}-{count:04d}", kind=group[0].kind, text=response["text"],
            anchors=sorted({a for e in group for a in e.anchors}),
            claim_ids=sorted({c for e in group for c in e.claim_ids}),
            provisional=any(e.provisional for e in group), stale=any(e.stale for e in group),
        )
        ontology.experience.append(merged)
        for entry in group:
            entry.superseded_by = merged.id
        count += 1
    return count
