"""Shared test helpers: a toy Lake, its Claims, and a scripted model that answers by role."""

import json
import os
import re
from pathlib import Path

from ogda.agent import AgentConfig
from ogda.claims import apply_claim, parse_claim
from ogda.llm import ScriptedClient
from ogda.ontology import Claim, Harness, OntologyStore

HOUSEHOLD_ID = {"kind": "concept", "concept": {"concept": "HouseholdId", "type": "ValueType", "extends": ["String"],
                                               "description": "A household identifier"}}
INCOME_USD = {"kind": "concept", "concept": {"concept": "IncomeUsd", "type": "ValueType", "extends": ["Decimal"],
                                             "description": "An amount of income in US dollars"}}
HOUSEHOLD = {"kind": "concept", "concept": {"concept": "Household", "type": "EntityType", "identify_by": ["household_id"],
                                            "description": "A household in the survey"}}
HOUSEHOLD_ID_REL = {"kind": "relationship", "concept": "Household", "relationship": {
    "name": "household_id", "roles": [{"concept": "HouseholdId"}], "multiplicity": "OneToOne",
    "verbalizes": ["{Household} has id {HouseholdId}"]}}
ANNUAL_INCOME_REL = {"kind": "relationship", "concept": "Household", "relationship": {
    "name": "annual_income", "roles": [{"concept": "IncomeUsd"}], "multiplicity": "ManyToOne",
    "verbalizes": ["{Household} earns {IncomeUsd} a year"], "description": "Annual household income"}}
INCOME_CLAIMS = [HOUSEHOLD_ID, INCOME_USD, HOUSEHOLD, HOUSEHOLD_ID_REL, ANNUAL_INCOME_REL]
INCOME_FIELDS = [
    {"name": "household_id", "column": "household_id", "datatype": "String", "description": "Household identifier, e.g. H1"},
    {"name": "annual_income", "column": "annual_income", "datatype": "Decimal",
     "description": "Annual household income in US dollars"}]
INCOME_BINDING = [{"concept": "Household", "identify": {"household_id": "household_id"},
                   "relationships": {"annual_income": ["annual_income"]}}]
INCOME_DATASET = {"kind": "dataset", "dataset": {
    "name": "income", "path": "income.csv", "header_row": 0, "start_row": 1, "end_row": 3,
    "description": "Annual income of each household", "origin": "declared", "documentation": ["README.md"],
    "fields": INCOME_FIELDS, "rows": "Household", "binding": INCOME_BINDING}}


def make_lake(root: Path) -> Path:
    lake = root / "original"
    lake.mkdir()
    (lake / "income.csv").write_text("household_id,annual_income\nH1,100\nH2,200\n")
    (lake / "README.md").write_text("# income.csv\nhousehold_id: Household identifier\n"
                                    "annual_income: Annual household income in US dollars\n")
    return lake


def accept(harness: Harness, items: list[dict], lake_root: Path, status: str = "verified") -> Harness:
    """Parse and apply Claims as if they were accepted, numbering them after the Harness's Claims."""
    for item in items:
        kind, payload = parse_claim(item, harness, lake_root)
        claim = Claim(id=f"claim-{len(harness.claims) + 1:06d}", kind=kind, payload=payload, status=status)
        harness.claims[claim.id] = claim
        apply_claim(claim, harness)
    return harness


def income_harness(lake_root: Path) -> Harness:
    """The toy Lake's Harness: Household, its two Value types and the declared income Dataset."""
    return accept(Harness.empty("toy"), INCOME_CLAIMS + [INCOME_DATASET], lake_root)


def code(text: str) -> str:
    return f"<code>\n{text}\n</code>"


def call(source: str, thought: str = "I run this step.", text: str = "") -> dict:
    """A scripted agent response with one native `code` tool call."""
    return {"text": text, "tool_calls": [{"name": "code", "arguments": json.dumps({"thought": thought, "code": source})}]}


def answer(value: dict) -> str:
    return code(f"final_answer({value!r})")


class Script:
    """Answer each role from a list of responses; the last response repeats."""

    def __init__(self, proposer, verifier=None, experience=None):
        self.responses = {"proposer": list(proposer), "verifier": list(verifier or [verdict("verified")])}
        self.experience = experience or (lambda prompt: {"entries": []})
        self.prompts = {"proposer": [], "verifier": [], "experience": []}

    def __call__(self, role, system, prompt):
        self.prompts[role].append(prompt)
        if role == "experience":
            return self.experience(json.loads(prompt))
        queue = self.responses[role]
        response = queue.pop(0) if len(queue) > 1 else queue[0]
        return response(prompt) if callable(response) else response


def verdict(status: str, notes: str = "Checked against the File", **corrections):
    """A Verifier response that gives the same verdict to every Claim in its task."""
    def respond(prompt):
        ids = list(dict.fromkeys(re.findall(r"claim-\d{6}", prompt)))
        return answer({"verdicts": [{"claim_id": cid, "status": status, "notes": notes, "evidence_refs": ["income.csv"],
                                     **corrections} for cid in ids]})
    return respond


def construction(tmp_path: Path, script: Script, *, model: str = "scripted",
                 role_models: dict[str, str] | None = None, max_steps: int = 3, **settings):
    from ogda.construction import Construction, ConstructionConfig  # Imported here: helpers load without it.

    config = ConstructionConfig(agent=AgentConfig(max_steps=max_steps), **settings)
    return Construction(OntologyStore(tmp_path / "out" / "ontology"), ScriptedClient(script, model, role_models), config,
                        scratch_root=tmp_path / "scratch", log_root=tmp_path / "out" / "logs")


def process_of(item: dict) -> tuple[dict, int]:
    """The item and the id of the process that ran it, for the benchmark worker tests."""
    return item, os.getpid()


def die_on_b(item: dict) -> bool:
    """Ends its process at once on Lake "b", as an out-of-memory kill would."""
    if item["lake_id"] == "b":
        os._exit(1)
    return True
