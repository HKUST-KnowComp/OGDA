"""The concept layer: visits review every Concept against the whole Ontology. They merge duplicates, turn Attributes
into References, add References and containment between Entity types, set compound Identifiers and subtypes, and
remove what nothing uses. Concept-layer Claims are accepted once code checks pass, without a Verifier.

The work list is every Concept without a review decision. Construction converges when it is empty. A visit gets the
first unreviewed Concept by name and the unreviewed Concepts most similar to it, so duplicates meet in one visit, and
the link candidates code finds across the Lake: Fields whose values identify Entities they are not bound to.
"""

from __future__ import annotations

import json

from smolagents import tool

from .agent import run_agent
from .claims import KINDS, ORDER, answer_problems, apply_claim, is_noop, parse_claim, trial
from .experience import consolidate
from .ontology import Harness
from .prompts import CONCEPT_LAYER
from .retrieval import (concept_detail, dataset_summary, link_candidates, ontology_summary, ontology_tools,
                        representatives, similar_concepts)
from .sandbox.executor import ToolCorrection

NOTE = "Accepted after code checks: the Ontology rules, every Binding and the affected data checks passed"
ANSWER_KEYS = {"claims", "reviewed", "summary", "unresolved"}


def work(harness: Harness, progress: dict) -> list[str]:
    """The Concepts that have no review decision yet, by name."""
    return sorted(c.concept for c in harness.ontology.ontology if c.concept not in progress["reviewed"])


def task(run, harness: Harness, progress: dict, names: list[str]) -> dict:
    batch = [names[0], *similar_concepts(harness, names[0], names[1:], run.config.concepts_per_visit - 1)]
    return {
        "stage": "concepts", "pass": progress["pass"],
        "reason": "Review each Concept against the whole Ontology: duplicates, References, Identifiers and subtypes",
        "review": [concept_detail(harness, name) for name in batch],
        "ontology": ontology_summary(harness),
        "datasets": [dataset_summary(harness, name) for name in representatives(harness)],
        "link_candidates": link_candidates(harness, run.store.lake_root, run.config.check_rows, run.field_values),
        "remaining": len(names) - len(batch),
        "remaining_work": progress["remaining_work"],
    }


def propose(run, harness: Harness, progress: dict, pending: dict) -> None:
    """One concept-layer visit. Each answer the Proposer submits, with submit or final_answer, is checked against the
    Harness as it stands, and its Claims are applied and saved together; a rejected answer leaves earlier answers in
    place. When a run stops during the visit, resume keeps the accepted answers and ends the visit."""
    root = run.store.lake_root
    resumed = pending.get("answers", 0) > 0
    for key, empty in (("answers", 0), ("claim_ids", []), ("reviewed", []), ("summary", ""), ("unresolved", [])):
        pending.setdefault(key, type(empty)())

    def accept(answer) -> None:
        if not isinstance(answer, dict):
            raise ValueError("Submit one JSON object")
        if unknown := sorted(map(str, set(answer) - ANSWER_KEYS)):
            raise ValueError(f"Unknown answer keys {unknown}. An answer is {{'claims': [...], 'reviewed': "
                             "[{'concept', 'reason'}], 'summary': str, 'unresolved': [str]}; each Claim goes in its "
                             "claims list")
        claims, summary, unresolved = answer.get("claims", []), answer.get("summary", ""), answer.get("unresolved", [])
        decisions = answer.get("reviewed", [])
        if (not isinstance(claims, list) or not isinstance(summary, str) or not isinstance(unresolved, list)
                or not isinstance(decisions, list)):
            raise ValueError("Submit {'claims': [...], 'reviewed': [{'concept', 'reason'}], 'summary': str, "
                             "'unresolved': [str]}")
        earlier = [(c.kind, c.payload) for c in (harness.claims[cid] for cid in pending["claim_ids"])]
        parsed, errors = [], []
        for index, item in enumerate(claims):
            try:
                kind, payload = parse_claim(item, harness, root)
                if kind not in KINDS["concepts"]:
                    raise ValueError("The concept layer permits only concept, relationship, merge, binding and remove "
                                     "Claims")
                if (kind, payload) not in earlier:  # A Claim accepted earlier in this visit is not repeated.
                    parsed.append((index, kind, payload))
            except (ValueError, TypeError) as error:
                errors.append(f"claims[{index}]: {error}")
        after = harness
        if not errors:
            try:
                after = trial(harness, [(kind, payload) for _, kind, payload in parsed])
            except ValueError as error:
                errors.append(str(error))
            else:
                errors += answer_problems(harness, after, set(), root, run.config.check_rows)
        known = {c.concept for c in harness.ontology.ontology} | {c.concept for c in after.ontology.ontology}
        decided = []
        for item in decisions:
            if (not isinstance(item, dict) or not isinstance(item.get("concept"), str)
                    or not str(item.get("reason", "")).strip()):
                errors.append("Each reviewed entry needs the name of a Concept and a reason")
            elif item["concept"] not in known:
                errors.append(f"reviewed: no Concept {item['concept']!r}")
            else:
                decided.append({"concept": item["concept"], "reason": str(item["reason"])})
        if errors:
            raise ValueError("Correct these Claims and submit this answer again; earlier answers stay saved:\n"
                             + "\n".join(errors))
        for _, kind, payload in sorted(parsed, key=lambda item: (ORDER[item[1]], item[0])):
            if is_noop(kind, payload, harness):
                run.log("claim_skipped", iteration=pending["iteration"], kind=kind, reason="unchanged")
                continue
            claim = run.claim(harness, kind, payload, pending["iteration"], pending["visit_name"])
            claim.status, claim.notes = "accepted", NOTE
            apply_claim(claim, harness)
            run.log("claim_outcome", claim_id=claim.id, kind=kind, status=claim.status, notes=NOTE,
                    acceptance_route="concepts")
            pending["claim_ids"].append(claim.id)
        seen = {item["concept"] for item in pending["reviewed"]}
        pending["reviewed"] += [item for item in decided if item["concept"] not in seen]
        if summary not in pending["summary"]:
            pending["summary"] = "\n".join(text for text in (pending["summary"], summary) if text)
        pending["unresolved"] = list(dict.fromkeys(pending["unresolved"] + [str(u) for u in unresolved]))
        pending["answers"] += 1
        run.store.save(harness)  # An answer's Claims are saved together, so a crash never splits them.

    @tool
    def submit(answer: dict) -> dict:
        """Save finished work now: its Claims are accepted at once and its review decisions are kept, even if the
        visit ends early. A rejected answer can be corrected and submitted again.

        Args:
            answer: The same object as final_answer takes: claims, reviewed, summary and unresolved.
        """
        try:
            accept(answer)
        except ValueError as error:
            raise ToolCorrection(str(error)) from None
        return {"accepted_claims": len(pending["claim_ids"]), "reviewed": len(pending["reviewed"])}

    visit = None
    if not resumed:
        visit = run_agent(run.llm, role="proposer", task=json.dumps(pending["task"]),
                          instructions=CONCEPT_LAYER,
                          validate=accept, tools=ontology_tools(harness) + [submit],
                          **run.visit_paths("proposer", pending["visit_name"]))
    if visit is None or not visit.completed:
        run.log("proposer_incomplete", iteration=pending["iteration"], steps=visit.steps if visit else None,
                stage="concepts", answers=pending["answers"])
        if not pending["answers"]:
            pending["unresolved"].append("The Proposer did not finish this concept-layer visit")
    pending["phase"] = "experience"
    run.store.save(harness)


def finish(run, harness: Harness, progress: dict, pending: dict) -> bool:
    """Record review decisions and end the Pass. Returns False when the visit accepted no answer, which stops
    construction for an explicit resume; a Pass whose answers change and decide nothing ends construction."""
    new = [item for item in pending["reviewed"] if item["concept"] not in progress["reviewed"]]
    for item in new:
        progress["reviewed"][item["concept"]] = item["reason"]
    accepted = sum(harness.claims[cid].status == "accepted" for cid in pending["claim_ids"])
    merged = consolidate(run.llm, harness, progress["pass"], run.budget)
    result = {"pass": progress["pass"], "stage": "concepts", "new_accepted": accepted, "reviewed": len(new),
              "merged_experience": merged}
    harness.metadata.setdefault("passes", []).append(result)
    run.log("pass_end", **result)
    progress["pass"] += 1
    if accepted == 0 and not new:
        if not pending.get("answers", 1):
            return False
        progress["stage"] = "done"
        harness.metadata["stop_reason"] = "unresolved"
    return True
