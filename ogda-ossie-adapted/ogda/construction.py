"""Build the Ontology of one Lake as Ossie documents: induce each File's Datasets and Bindings, then review every
Concept in the concept layer.

The scheduler runs one Iteration at a time. An induction Iteration is a Proposer visit on a batch of Files, code
checks and Verifier visits, Experience, File dispositions, and Datasets for the other Files of each induced layout
group. A concept-layer Iteration is a Proposer visit that reviews a batch of Concepts against the whole Ontology, then
Experience. Progress is saved after each phase, so `run` resumes where an interrupted run stopped.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, fields
from pathlib import Path

from . import concept_layer, induction
from .agent import AgentConfig, Budget
from .claims import apply_claim
from .experience import distill
from .lake import copy_lake
from .llm import BudgetExceeded, InvalidResponse
from .ontology import Claim, Harness, OntologyStore

PROGRESS_VERSION = 3


@dataclass(frozen=True)
class ConstructionConfig:
    max_tokens: int = 2_000_000      # Total token cap for all construction model calls.
    files_per_visit: int = 8         # Files in one induction visit; a layout group counts as one File.
    max_datasets_per_visit: int = 20  # dataset Claims in one induction visit; the Verifier judges these.
    concepts_per_visit: int = 30     # Concepts reviewed in one concept-layer visit.
    max_passes: int = 20             # Concept-layer visits.
    check_rows: int = 100_000        # Rows read per Dataset by the data checks.
    agent: AgentConfig = field(default_factory=AgentConfig)

    @classmethod
    def from_dict(cls, data: dict | None) -> "ConstructionConfig":
        data = dict(data or {})
        agent = AgentConfig(**(data.pop("agent", None) or {}))
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown construction settings: {sorted(unknown)}")
        return cls(agent=agent, **data)


class Construction:
    def __init__(self, store: OntologyStore, llm, config: ConstructionConfig, scratch_root: Path, log_root: Path):
        self.store, self.llm, self.config = store, llm, config
        self.scratch_root, self.log_root = Path(scratch_root).resolve(), Path(log_root).resolve()
        self.budget = Budget(config.max_tokens)
        self.field_values: dict = {}  # Each Dataset's Field values, read once for the concept layer's link candidates.

    def start(self, lake_id: str, source: Path, exclude: list[str] = (), stripped_keys: list[str] = ()) -> Harness:
        """Copy the Lake and save an empty Harness."""
        if self.store.exists():
            raise FileExistsError(f"{self.store.root} holds a Harness; resume it or use a new output directory")
        files = copy_lake(source, self.store.lake_root, list(exclude), list(stripped_keys))
        ontology = Harness.empty(lake_id, metadata={"source": str(source), "exclude": list(exclude),
                                                    "stripped_keys": list(stripped_keys), "files": len(files)})
        self.store.save(ontology)
        return ontology

    def run(self) -> Harness:
        """Run or resume construction until it converges, stalls or reaches a limit."""
        ontology = self.store.load()
        progress = self._progress(ontology)
        if progress["stage"] == "done":
            if not induction.new_files(self, progress):
                return ontology
            progress["stage"] = "induction"
        ontology.metadata.pop("stop_reason", None)
        try:
            while True:
                if progress.get("pending_iteration"):
                    if stalled := self._finish_iteration(ontology, progress):
                        ontology.metadata["stop_reason"] = stalled
                        break
                elif progress["stage"] == "induction":
                    files = induction.next_files(self, progress)
                    if files:
                        self._begin(ontology, progress, induction.task(self, ontology, progress, files))
                    else:
                        progress["stage"] = "concepts"
                        self.store.save(ontology)
                elif progress["stage"] == "done":
                    break  # A concept-layer Pass that made no progress stopped construction.
                else:
                    work = concept_layer.work(ontology, progress)
                    if not work:
                        ontology.metadata["stop_reason"] = "unresolved" if self.unresolved(progress) else "converged"
                        progress["stage"] = "done"
                        break
                    if progress["pass"] > self.config.max_passes:
                        ontology.metadata["stop_reason"] = "max_passes"
                        break
                    self._begin(ontology, progress, concept_layer.task(self, ontology, progress, work))
        except BudgetExceeded as error:
            ontology.metadata["stop_reason"] = "budget"
            self.log("budget_exceeded", message=str(error))
        ontology.metadata["llm_usage"] = dict(self.llm.usage)
        self.store.save(ontology)
        return ontology

    @staticmethod
    def unresolved(progress: dict) -> list[str]:
        return progress["remaining_work"] + [f"{path}: {item['reason']}" for path, item in progress["files"].items()
                                            if item["status"] == "unresolved"]

    def _progress(self, ontology: Harness) -> dict:
        progress = ontology.metadata.get("progress")
        if progress is None:
            progress = ontology.metadata["progress"] = {
                "version": PROGRESS_VERSION, "stage": "induction", "seen": [], "groups": {}, "files": {},
                "pass": 1, "reviewed": {}, "remaining_work": []}
            self.store.save(ontology)
        elif progress.get("version") != PROGRESS_VERSION:
            raise ValueError(f"Unsupported scheduler progress version {progress.get('version')}; rebuild the Ontology")
        return progress

    def _begin(self, ontology: Harness, progress: dict, task: dict) -> None:
        iteration = ontology.metadata["iteration"] = ontology.metadata.get("iteration", 0) + 1
        progress["pending_iteration"] = {"iteration": iteration, "visit_name": f"iteration-{iteration:06d}",
                                          "task": task, "phase": "propose"}
        self.log("iteration_start", iteration=iteration, stage=task["stage"], reason=task["reason"],
                 files=[induction.relative(path, self.store.lake_root) for path in task.get("files", [])],
                 concepts=[item["concept"] for item in task.get("review", [])])
        self.store.save(ontology)

    def _finish_iteration(self, ontology: Harness, progress: dict) -> str | None:
        """Finish the pending Iteration. Returns the stop reason when it stalls: verification that recorded no new
        judgment (`verification_incomplete`), or a Proposer visit that accepted no Claim and moved no File out of the
        queue or, in the concept layer, accepted no answer (`proposer_incomplete`). Both wait for an explicit resume."""
        pending = progress["pending_iteration"]
        stage = pending["task"]["stage"]
        if pending["phase"] == "propose":
            (induction.propose if stage == "induction" else concept_layer.propose)(self, ontology, progress, pending)
        claims = [ontology.claims[cid] for cid in pending["claim_ids"]]
        if pending["phase"] == "verify":
            if not induction.decide(self, ontology, pending, claims):
                return "verification_incomplete"
            pending["phase"] = "experience"
            self.store.save(ontology)
        if not pending.get("experience_written"):
            try:
                ontology.experience.extend(distill(self.llm, ontology, claims, pending["summary"], pending["unresolved"],
                                                   pending["iteration"], self.budget))
            except InvalidResponse as error:
                self.log("experience_failed", iteration=pending["iteration"], message=str(error))
            pending["experience_written"] = True
            self.store.save(ontology)
        progressed = (induction.finish if stage == "induction" else concept_layer.finish)(self, ontology, progress, pending)
        progress["remaining_work"] = list(dict.fromkeys(pending["unresolved"] + [c.notes for c in claims
                                                                                if c.status == "undecidable"]))
        self.log("iteration_end", iteration=pending["iteration"], stage=stage,
                 new_accepted=sum(c.status in {"verified", "accepted"} for c in claims),
                 unresolved=progress["remaining_work"])
        del progress["pending_iteration"]
        self.store.save(ontology)
        return None if progressed else "proposer_incomplete"

    def claim(self, ontology: Harness, kind: str, payload: dict, iteration: int, visit_name: str) -> Claim:
        number = ontology.metadata["next_claim"] = ontology.metadata.get("next_claim", 0) + 1
        claim = Claim(id=f"claim-{number:06d}", kind=kind, payload=payload, iteration=iteration,
                      proposer_visit=f"proposer/{visit_name}")
        ontology.claims[claim.id] = claim
        return claim

    def settle(self, ontology: Harness, claim: Claim, status: str, notes: str, refs: list[str] = (),
               *, route: str, save: bool = True) -> None:
        """Record a Claim's outcome and install it when accepted. Callers that settle several Claims at once pass
        save=False and save once afterwards."""
        claim.status, claim.notes, claim.evidence_refs = status, notes, list(refs)
        if status in {"verified", "accepted"}:
            apply_claim(claim, ontology)
        if save:
            self.store.save(ontology)
        self.log("claim_outcome", claim_id=claim.id, kind=claim.kind, status=status, notes=notes, acceptance_route=route)

    def visit_paths(self, role: str, name: str) -> dict:
        return {"lake_root": self.store.lake_root, "scratch_root": self.scratch_root,
                "artifact_root": self.log_root / role / name, "config": self.config.agent, "budget": self.budget}

    def log(self, event: str, **data) -> None:
        self.log_root.mkdir(parents=True, exist_ok=True)
        with (self.log_root / "construction.jsonl").open("a") as stream:
            stream.write(json.dumps({"event": event, "time": time.time(), **data}, default=str) + "\n")
