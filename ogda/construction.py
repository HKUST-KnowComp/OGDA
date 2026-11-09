"""Map Files once with verified Schemas, then enrich them with an unverified graph of Concepts and Links."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, fields
from pathlib import Path

import openpyxl
from smolagents import tool

from .agent import AgentConfig, Budget, run_agent
from .claims import (ORDER, TABULAR, apply_claim, field_id_conflicts, is_noop, lake_path, missing_references, parse_claim,
                     join_check, read_rows, static_check, table_id_conflicts)
from .experience import consolidate, distill
from .lake import batches, copy_lake, documentation_candidate, list_files
from .llm import BudgetExceeded, InvalidResponse
from .ontology import Claim, Ontology, OntologyStore
from .prompts import PROPOSER, VERIFIER
from .retrieval import retrieve, schema_detail, schema_summary, table_counts

STATUSES = {"verified", "refuted", "undecidable"}
GRAPH_KINDS = {"concept", "link", "remove"}
PROGRESS_VERSION = 1
VERIFIER_VISITS_PER_RUN = 3
STATIC_NOTE = "Static check: each Table File exists, its columns and row range match the accepted Schema"
DOCUMENTATION_NOTE = "Documentation: declared Schema accepted from recorded Documentation; supported structural checks passed"
GRAPH_NOTE = "Accepted without verification: concept graph"


@dataclass(frozen=True)
class ConstructionConfig:
    max_passes: int = 3
    max_tokens: int = 2_000_000
    files_per_visit: int = 8
    max_claims_per_visit: int = 20
    agent: AgentConfig = field(default_factory=AgentConfig)

    @classmethod
    def from_dict(cls, data: dict | None) -> "ConstructionConfig":
        data = dict(data or {})
        agent = AgentConfig(**data.pop("agent", {}))
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown construction settings: {sorted(unknown)}")
        return cls(agent=agent, **data)


def ontology_tools(ontology: Ontology) -> list:
    @tool
    def search_ontology(query: str, hops: int = 2) -> dict:
        """Return the Ontology objects that match a text query and their neighbours, with the edges and paths walked.

        Args:
            query: Words to match against Concept, Schema, Field and Table names and descriptions.
            hops: How many edges to walk from the matched objects.
        """
        return retrieve(ontology, query, hops=hops)

    @tool
    def get_schema(schema_id: str) -> dict:
        """Return one Schema with all its Fields, its first Tables and its table_count.

        Args:
            schema_id: The id of an accepted Schema.
        """
        return schema_detail(ontology, schema_id)

    return [search_ontology, get_schema]


def schema_context(ontology: Ontology) -> dict:
    """Accepted evidence for semantic work, without old Claim ids or Proposer reasoning: Schema summaries
    (get_schema opens one), Concepts and Links."""
    counts = table_counts(ontology)
    return {"schemas": {sid: schema_summary(ontology, sid, counts) for sid in ontology.schemas},
            **ontology.model_dump(mode="json", include={"concepts", "links"})}


class Construction:
    def __init__(self, store: OntologyStore, llm, config: ConstructionConfig, scratch_root: Path, log_root: Path):
        self.store, self.llm, self.config = store, llm, config
        self.scratch_root, self.log_root = Path(scratch_root).resolve(), Path(log_root).resolve()
        self.budget = Budget(config.max_tokens)

    def start(self, lake_id: str, source: Path, exclude: list[str] = (), stripped_keys: list[str] = ()) -> Ontology:
        """Copy the Lake and save an empty Ontology."""
        if self.store.exists():
            raise FileExistsError(f"{self.store.path} exists; resume it or use a new output directory")
        files = copy_lake(source, self.store.lake_root, list(exclude), list(stripped_keys))
        ontology = Ontology(lake_id=lake_id, metadata={"source": str(source), "exclude": list(exclude),
                                                       "stripped_keys": list(stripped_keys), "files": len(files)})
        self.store.save(ontology)
        return ontology

    def run(self) -> Ontology:
        """Resume the saved stage, including submitted Claims and partial Verifier judgments."""
        ontology = self.store.load()
        progress = self._progress(ontology)
        if progress["stage"] == "done":
            if not set(list_files(self.store.lake_root)) - set(progress["files"]):
                return ontology
            progress.update({"stage": "file_mapping", "pass": 1, "new": 0, "reviewed": False})
        ontology.metadata.pop("stop_reason", None)
        try:
            while True:
                if progress.get("pending_iteration"):
                    if not self._finish_iteration(ontology, progress):
                        ontology.metadata["stop_reason"] = "verification_incomplete"
                        break
                elif progress["stage"] == "file_mapping":
                    remaining = [p for p in list_files(self.store.lake_root) if p not in progress["files"]]
                    if remaining:
                        files = batches(remaining, self.config.files_per_visit)[0]
                        self._begin_iteration(ontology, progress, self._task(ontology, "file_mapping", files))
                    else:
                        progress["stage"] = "schema_enrichment"
                        self.store.save(ontology)
                elif progress["reviewed"]:
                    merged = consolidate(self.llm, ontology, progress["pass"], self.budget)
                    result = {"pass": progress["pass"], "stage": "schema_enrichment", "new_accepted": progress["new"],
                              "merged_experience": merged}
                    ontology.metadata.setdefault("passes", []).append(result)
                    self._log("pass_end", **result)
                    if progress["new"] == 0:
                        ontology.metadata["stop_reason"] = "unresolved" if self._unresolved(progress) else "converged"
                        progress["stage"] = "done"
                        break
                    progress.update({"pass": progress["pass"] + 1, "new": 0, "reviewed": False})
                    self.store.save(ontology)
                elif not ontology.schemas:
                    ontology.metadata["stop_reason"] = "unresolved" if self._unresolved(progress) else "converged"
                    progress["stage"] = "done"
                    break
                elif progress["pass"] > self.config.max_passes:
                    ontology.metadata["stop_reason"] = "max_passes"
                    break
                else:
                    self._begin_iteration(ontology, progress, self._task(ontology, "schema_enrichment", []))
        except BudgetExceeded as error:
            ontology.metadata["stop_reason"] = "budget"
            self._log("budget_exceeded", message=str(error))
        ontology.metadata["llm_usage"] = dict(self.llm.usage)
        self.store.save(ontology)
        return ontology

    @staticmethod
    def _unresolved(progress: dict) -> list[str]:
        return progress["remaining_work"] + [f"{path}: {item['reason']}" for path, item in progress["files"].items()
                                            if item["status"] == "unresolved"]

    def _progress(self, ontology: Ontology) -> dict:
        """Migrate old Pass-local progress using accepted mappings, never the old visited list."""
        old = ontology.metadata.get("progress", {})
        if old.get("version") == PROGRESS_VERSION:
            return old
        if "version" in old:
            raise ValueError(f"Unsupported scheduler progress version {old['version']}")
        progress = {"version": PROGRESS_VERSION, "stage": "file_mapping", "files": {}, "pass": 1,
                    "new": 0, "reviewed": False, "remaining_work": []}
        if old:
            ontology.metadata["legacy_progress"] = old
        for path in list_files(self.store.lake_root):
            tables = [t for t in ontology.tables.values() if t.path == path and t.schema_id in ontology.schemas]
            if tables and self._full_mapping(path, tables):
                progress["files"][path] = {"status": "mapped", "complete": True,
                                            "table_ids": [t.id for t in tables], "reason": "Migrated accepted full-File mapping"}
        self._record_documentation(ontology, progress)
        ontology.metadata["progress"] = progress
        pending = [c for c in ontology.claims.values() if c.status == "pending"]
        if pending:
            # Older runs saved Claims before verification, but did not save the original task or summary.
            iteration = max(c.iteration for c in pending)
            if any(c.iteration != iteration for c in pending):
                raise ValueError("Legacy artifact has pending Claims from several iterations; rebuild it")
            stage = "file_mapping" if any(c.kind in {"schema", "tables"} for c in pending) else "schema_enrichment"
            files = sorted({t["path"] for c in pending for t in c.payload.get("tables", [])})
            task = self._task(ontology, stage, files)
            columns = self._columns(ontology, [(c.kind, c.payload) for c in pending])
            routes, verdicts = {}, {}
            for claim in pending:
                check = static_check(claim.kind, claim.payload,
                                     columns.get(self._schema_id(claim.kind, claim.payload)), self.store.lake_root)
                routes[claim.id] = self._route(claim.kind, claim.payload, check)
                if check:
                    routes[claim.id] = "reference"
                    verdicts[claim.id] = {"status": "undecidable", "notes": "Invalid legacy Claim: " + "; ".join(check),
                                          "evidence_refs": []}
            progress["pending_iteration"] = {
                "iteration": iteration, "visit_name": f"iteration-{iteration:06d}", "task": task, "phase": "verify",
                "claim_ids": [c.id for c in pending], "routes": routes, "verdicts": verdicts, "verification_attempts": 0,
                "summary": "Recovered submitted Claims from legacy progress", "unresolved": [], "file_dispositions": [],
            }
            self._log("legacy_pending_recovered", claim_ids=[c.id for c in pending])
        if old:
            self._log("scheduler_migrated", mapped_files=sorted(progress["files"]))
        self.store.save(ontology)
        return progress

    def _full_mapping(self, path: str, tables: list) -> bool:
        """Conservatively migrate only supported Tables spanning every nonempty sheet."""
        if Path(path).suffix.lower() not in TABULAR:
            return False
        sheets, default_sheet = self._sheets(path)
        if not sheets:
            sheets = [None]
        for sheet in sheets:
            rows = read_rows(self.store.lake_root / path, sheet)
            if not any((t.sheet == sheet or (t.sheet is None and sheet == default_sheet))
                       and t.header_row == 0 and t.start_row == 1
                       and (t.end_row is None or t.end_row == len(rows)) for t in tables):
                return False
        return True

    def _sheets(self, path: str) -> tuple[list[str], str | None]:
        if Path(path).suffix.lower() not in {".xlsx", ".xlsm"}:
            return [], None
        workbook = openpyxl.load_workbook(self.store.lake_root / path, read_only=True, data_only=True)
        try:
            return ([sheet.title for sheet in workbook if any(any(v is not None for v in row)
                                                              for row in sheet.iter_rows(values_only=True))],
                    workbook.sheetnames[0])
        finally:
            workbook.close()

    def _task(self, ontology: Ontology, stage: str, files: list[str]) -> dict:
        progress = ontology.metadata.get("progress", {})
        task = {"stage": stage, "files": [str(self.store.lake_root / p) for p in files],
                "schema_ids": sorted(ontology.schemas), "context": schema_context(ontology),
                "remaining_work": self._unresolved(progress) if progress else [],
                "reason": "Unprocessed Files need complete Schema/Table mappings" if stage == "file_mapping"
                          else "Generate supported Concepts and Links from accepted Schemas"}
        if stage == "file_mapping":
            task["documentation"] = [str(self.store.lake_root / p) for p, item in progress.get("files", {}).items()
                                     if item["status"] == "documentation"]
            task["documentation_candidates"] = [str(self.store.lake_root / p) for p in list_files(self.store.lake_root)
                                                if documentation_candidate(p) and progress.get("files", {}).get(p, {}).get(
                                                    "status") not in {"mapped", "documentation"}]
        else:
            task["decided_proposals"] = [{"kind": c.kind, "payload": c.payload, "status": c.status, "notes": c.notes}
                                         for c in ontology.claims.values() if c.status in {"refuted", "undecidable"}]
        return task

    def _begin_iteration(self, ontology: Ontology, progress: dict, task: dict) -> None:
        iteration = ontology.metadata["iteration"] = ontology.metadata.get("iteration", 0) + 1
        progress["pending_iteration"] = {"iteration": iteration, "visit_name": f"iteration-{iteration:06d}",
                                          "task": task, "phase": "propose"}
        self._log("iteration_start", iteration=iteration, files=[lake_path(p, self.store.lake_root) for p in task["files"]],
                  stage=task["stage"], schema_ids=task["schema_ids"], reason=task["reason"])
        self.store.save(ontology)

    @staticmethod
    def _schema_id(kind: str, payload: dict) -> str | None:
        return payload["schema"]["id"] if kind == "schema" else payload.get("schema_id")

    @staticmethod
    def _columns(ontology: Ontology, parsed: list[tuple[str, dict]]) -> dict:
        columns = {sid: [f.column for f in schema.fields] for sid, schema in ontology.schemas.items()}
        columns.update({p["schema"]["id"]: [f["column"] for f in p["schema"]["fields"]]
                        for kind, p in parsed if kind == "schema"})
        return columns

    @staticmethod
    def _route(kind: str, payload: dict, check: list | None) -> str:
        if kind in GRAPH_KINDS:
            return "graph"
        if check != [] or (kind == "schema" and payload["schema"]["origin"] == "inferred"):
            return "verifier"
        return "documentation" if kind == "schema" else "structural"

    def _propose(self, ontology: Ontology, pending: dict) -> None:
        submitted = {}
        stage = pending["task"]["stage"]

        def validate(answer: dict) -> None:
            claims, summary, unresolved = answer.get("claims"), answer.get("summary", ""), answer.get("unresolved", [])
            if not isinstance(claims, list) or not isinstance(summary, str) or not isinstance(unresolved, list):
                raise ValueError("Submit {'claims': [...], 'summary': str, 'unresolved': [str]}")
            if len(claims) > self.config.max_claims_per_visit:
                raise ValueError(f"Submit at most {self.config.max_claims_per_visit} Claims")
            parsed, errors = [], []
            for index, item in enumerate(claims):
                try:
                    kind, payload = parse_claim(item, self.store.lake_root)
                    allowed = {"schema", "tables"} if stage == "file_mapping" else {"concept", "link", "remove"}
                    if kind not in allowed:
                        raise ValueError(f"{stage} permits only {sorted(allowed)} Claims")
                    if kind == "remove" and payload["id"] not in ontology.concepts and payload["id"] not in ontology.links:
                        raise ValueError("Schema enrichment can remove only Concepts or Links")
                    parsed.append((kind, payload))
                except (ValueError, TypeError) as error:
                    errors.append(f"claims[{index}]: {error}")
            columns = self._columns(ontology, parsed)
            checks = [static_check(kind, p, columns.get(self._schema_id(kind, p)), self.store.lake_root)
                      for kind, p in parsed]
            errors += [f"claims[{index}]: " + "; ".join(problems) for index, problems in enumerate(checks) if problems]
            errors += field_id_conflicts(parsed, ontology)
            errors += table_id_conflicts(parsed, ontology)
            join_errors, join_notes = (join_check(parsed, ontology, self.store.lake_root) if stage == "schema_enrichment"
                                       else ([], {}))
            errors += join_errors
            dispositions = self._parse_dispositions(answer.get("file_dispositions", [])) if stage == "file_mapping" else []
            if errors:
                raise ValueError("Correct these Claims and submit again:\n" + "\n".join(errors))
            submitted.update(claims=[(kind, p, self._route(kind, p, check), join_notes.get(index))
                                     for index, ((kind, p), check) in enumerate(zip(parsed, checks))],
                             summary=summary, unresolved=[str(u) for u in unresolved], file_dispositions=dispositions)

        visit = run_agent(self.llm, role="proposer", task=json.dumps(pending["task"]),
                          instructions=PROPOSER + f"\nPropose at most {self.config.max_claims_per_visit} Claims in this visit.",
                          validate=validate, tools=ontology_tools(ontology),
                          **self._visit_paths("proposer", pending["visit_name"]))
        if not visit.completed:
            self._log("proposer_incomplete", iteration=pending["iteration"], steps=visit.steps, stage=stage)
            submitted.update(claims=[], summary="", unresolved=["The Proposer did not finish this work item"],
                             file_dispositions=[])
        claims, routes, graph_notes = [], {}, {}
        for kind, payload, route, note in submitted.pop("claims"):
            previous = next((c for c in ontology.claims.values() if c.kind == kind and c.payload == payload
                             and c.status in {"refuted", "undecidable"}), None)
            if is_noop(kind, payload, ontology) or previous:
                self._log("claim_skipped", iteration=pending["iteration"], kind=kind,
                          reason="unchanged" if previous is None else previous.status)
                if previous and previous.status == "undecidable":
                    submitted["unresolved"].append(previous.notes)
                continue
            claim = self._claim(ontology, kind, payload, pending["iteration"], pending["visit_name"])
            claims.append(claim.id)
            routes[claim.id] = route
            if note:
                graph_notes[claim.id] = f"{GRAPH_NOTE}. {note}"
        pending.update(submitted, phase="verify", claim_ids=claims, routes=routes, graph_notes=graph_notes, verdicts={},
                       verification_attempts=0)
        self.store.save(ontology)  # Submitted Claims survive a Verifier crash.

    def _parse_dispositions(self, items) -> list[dict]:
        if not isinstance(items, list):
            raise ValueError("file_dispositions must be a list")
        result = []
        for item in items:
            if not isinstance(item, dict) or item.get("status") not in {"mapped", "documentation", "unresolved"}:
                raise ValueError("Each File disposition needs path and status: mapped, documentation or unresolved")
            if not isinstance(item.get("path"), str) or not item["path"]:
                raise ValueError("A File disposition needs a nonempty path")
            path = lake_path(item.get("path", ""), self.store.lake_root)
            status = item["status"]
            if status == "mapped":
                table_ids = item.get("table_ids")
                if item.get("complete") is not True or not isinstance(table_ids, list) or not table_ids or any(
                        not isinstance(tid, str) for tid in table_ids):
                    raise ValueError(f"{path}: mapped requires complete=True and all identified table_ids; otherwise use unresolved")
                result.append({"path": path, "status": status, "complete": True, "table_ids": list(dict.fromkeys(table_ids))})
            else:
                if not isinstance(item.get("reason"), str) or not item["reason"].strip():
                    raise ValueError(f"{path}: {status} needs a reason")
                result.append({"path": path, "status": status, "reason": item["reason"]})
        if len({item["path"] for item in result}) != len(result):
            raise ValueError("Submit one disposition per File")
        return result

    def _claim(self, ontology: Ontology, kind: str, payload: dict, iteration: int, visit_name: str) -> Claim:
        number = ontology.metadata["next_claim"] = ontology.metadata.get("next_claim", 0) + 1
        claim = Claim(id=f"claim-{number:06d}", kind=kind, payload=payload, iteration=iteration,
                      proposer_visit=f"proposer/{visit_name}")
        ontology.claims[claim.id] = claim
        return claim

    def _finish_iteration(self, ontology: Ontology, progress: dict) -> bool:
        pending = progress["pending_iteration"]
        if pending["phase"] == "propose":
            self._propose(ontology, pending)
        claims = [ontology.claims[cid] for cid in pending["claim_ids"]]
        if pending["phase"] == "verify":
            if not self._decide(ontology, pending, claims):
                return False
            pending["phase"] = "experience"
            self.store.save(ontology)
        try:
            ontology.experience.extend(distill(self.llm, ontology, claims, pending["summary"], pending["unresolved"],
                                               pending["iteration"], self.budget))
        except InvalidResponse as error:
            self._log("experience_failed", iteration=pending["iteration"], message=str(error))
        if pending["task"]["stage"] == "file_mapping":
            self._finish_files(ontology, progress, pending)
        else:
            progress["new"] += sum(c.status in {"verified", "accepted"} for c in claims)
            progress["reviewed"] = True
        progress["remaining_work"] = list(dict.fromkeys(pending["unresolved"] + [c.notes for c in claims
                                                                                if c.status == "undecidable"]))
        self._log("iteration_end", iteration=pending["iteration"], stage=pending["task"]["stage"],
                  new_accepted=sum(c.status in {"verified", "accepted"} for c in claims),
                  unresolved=progress["remaining_work"])
        del progress["pending_iteration"]
        self.store.save(ontology)
        return True

    def _record_documentation(self, ontology: Ontology, progress: dict) -> None:
        for schema in ontology.schemas.values():
            for path in schema.documentation:
                if (progress["files"].get(path, {}).get("status") not in {"mapped", "documentation"}
                        and (self.store.lake_root / path).is_file()):
                    progress["files"][path] = {"status": "documentation", "reason": f"Defines accepted Schema {schema.id}"}
                    self._log("file_disposition", iteration=ontology.metadata.get("iteration", 0), path=path,
                              **progress["files"][path])

    def _finish_files(self, ontology: Ontology, progress: dict, pending: dict) -> None:
        dispositions = {item["path"]: item for item in pending["file_dispositions"]}
        self._record_documentation(ontology, progress)
        paths = {lake_path(path, self.store.lake_root) for path in pending["task"]["files"]} | set(dispositions)
        for path in sorted(paths):
            item = dict(dispositions.get(path, {"status": "unresolved", "reason": "No complete File disposition was submitted"}))
            item.pop("path", None)
            if item["status"] == "mapped":
                tables = [ontology.tables.get(tid) for tid in item["table_ids"]]
                if any(t is None or t.path != path or t.schema_id not in ontology.schemas for t in tables):
                    item = {"status": "unresolved", "reason": "Some identified Tables or their Schemas were not accepted"}
                else:
                    sheets, default_sheet = self._sheets(path)
                    covered = {t.sheet for t in tables}
                    if None in covered and sheets:
                        covered.add(default_sheet)
                    known = {t.id for t in ontology.tables.values() if t.path == path}
                    if not known <= set(item["table_ids"]):
                        item = {"status": "unresolved", "reason": "The disposition omitted known Table regions"}
                    elif sheets and not set(sheets) <= covered:
                        item = {"status": "unresolved", "reason": f"Unmapped worksheets: {sorted(set(sheets) - covered)}"}
            if progress["files"].get(path, {}).get("status") in {"mapped", "documentation"}:
                continue
            progress["files"][path] = item
            self._log("file_disposition", iteration=pending["iteration"], path=path, **item)

    def _apply_verdicts(self, ontology: Ontology, pending: dict, claims: list[Claim]) -> None:
        """Apply available judgments in dependency order, retaining judgments waiting on pending objects."""
        for claim in sorted(claims, key=lambda c: ORDER[c.kind]):
            if claim.status != "pending":
                continue
            missing = missing_references(claim, ontology)
            outstanding = [c for c in claims if c.status == "pending"]
            if missing and missing_references(claim, ontology, outstanding):
                self._settle(ontology, claim, "undecidable", f"It depends on objects that were not accepted: {missing}",
                             route="reference")
            elif not missing and claim.id in pending["verdicts"]:
                verdict = pending["verdicts"][claim.id]
                self._settle(ontology, claim, verdict["status"], verdict["notes"], verdict["evidence_refs"],
                             route=pending["routes"][claim.id])
            elif claim.id in pending["verdicts"] and pending["verdicts"][claim.id]["status"] not in {"verified", "accepted"}:
                verdict = pending["verdicts"][claim.id]
                self._settle(ontology, claim, verdict["status"], verdict["notes"], verdict["evidence_refs"], route="verifier")

    def _decide(self, ontology: Ontology, pending: dict, claims: list[Claim]) -> bool:
        for claim in claims:
            if claim.kind in GRAPH_KINDS and pending["routes"][claim.id] == "verifier":
                pending["routes"][claim.id] = "graph"  # Saved before the graph was accepted without a Verifier.
            route = pending["routes"][claim.id]
            if route not in {"verifier", "graph", "documentation", "structural", "reference"}:
                raise ValueError(f"Unsupported saved acceptance route {route!r}; rebuild this Ontology")
            if route == "graph" and claim.id not in pending["verdicts"]:
                pending["verdicts"][claim.id] = {"status": "accepted", "evidence_refs": [],
                                                "notes": pending.get("graph_notes", {}).get(claim.id, GRAPH_NOTE)}
            elif route != "verifier" and claim.id not in pending["verdicts"]:
                pending["verdicts"][claim.id] = {"status": "verified", "notes": DOCUMENTATION_NOTE if route == "documentation"
                                                else STATIC_NOTE, "evidence_refs": []}
        self._apply_verdicts(ontology, pending, claims)
        for _ in range(VERIFIER_VISITS_PER_RUN):
            outstanding = [c for c in claims if c.status == "pending" and c.id not in pending["verdicts"]]
            if not outstanding:
                break
            pending["verification_attempts"] += 1
            self.store.save(ontology)
            attempt = pending["verification_attempts"]
            name = pending["visit_name"] if attempt == 1 else f"{pending['visit_name']}-retry-{attempt:03d}"
            completed = len(pending["verdicts"])
            self._verify(ontology, pending, claims, outstanding, name)
            self._apply_verdicts(ontology, pending, claims)
            if len(pending["verdicts"]) == completed:
                break  # A stalled visit needs an explicit resume, not another identical automatic visit.
        return all(c.status != "pending" for c in claims)

    def _settle(self, ontology: Ontology, claim: Claim, status: str, notes: str, refs: list[str] = (),
                *, route: str = "verifier") -> None:
        claim.status, claim.notes, claim.evidence_refs = status, notes, list(refs)
        if status in {"verified", "accepted"}:
            apply_claim(claim, ontology)
        self.store.save(ontology)
        self._log("claim_outcome", claim_id=claim.id, status=status, notes=notes, acceptance_route=route)

    def _verify(self, ontology: Ontology, pending: dict, batch: list[Claim], claims: list[Claim], visit_name: str) -> None:
        expected = {claim.id for claim in claims}

        def submit(verdicts) -> None:
            if not isinstance(verdicts, list) or any(not isinstance(v, dict) for v in verdicts):
                raise ValueError("Submit a list of verdicts with claim_id, status, notes and evidence_refs")
            ids = [v.get("claim_id") for v in verdicts]
            if (any(not isinstance(cid, str) for cid in ids) or len(set(map(str, ids))) != len(ids)
                    or any(cid not in expected for cid in ids)):
                raise ValueError(f"Submit at most one verdict per outstanding Claim: {sorted(expected)}")
            for v in verdicts:
                if v.get("status") not in STATUSES or not isinstance(v.get("notes"), str) or not v["notes"].strip():
                    raise ValueError(f"{v['claim_id']}: needs a valid status and nonempty notes")
                refs = v.get("evidence_refs", [])
                if not isinstance(refs, list) or any(not isinstance(ref, str) for ref in refs):
                    raise ValueError(f"{v['claim_id']}: evidence_refs must be a list of strings")
                value = {"status": v["status"], "notes": v["notes"], "evidence_refs": refs}
                if v["claim_id"] in pending["verdicts"] and pending["verdicts"][v["claim_id"]] != value:
                    raise ValueError(f"{v['claim_id']}: a submitted verdict cannot be changed")
            for v in verdicts:
                pending["verdicts"][v["claim_id"]] = {"status": v["status"], "notes": v["notes"],
                                                      "evidence_refs": list(v.get("evidence_refs", []))}
                ontology.claims[v["claim_id"]].verifier_visit = f"verifier/{visit_name}"
            self.store.save(ontology)
            self._apply_verdicts(ontology, pending, batch)

        @tool
        def submit_verdicts(verdicts: list[dict]) -> dict:
            """Persist completed judgments now, before continuing to other Claims.

            Args:
                verdicts: Completed verdicts, each with claim_id, status, notes and optional evidence_refs.
            """
            submit(verdicts)
            return {"outstanding_claim_ids": sorted(expected - set(pending["verdicts"]))}

        def validate(answer: dict) -> None:
            submit(answer.get("verdicts"))

        task = {"stage": "file_mapping",
                "claims": [{"id": c.id, "kind": c.kind, "payload": c.payload} for c in claims],
                "context": schema_context(ontology)}
        visit = run_agent(self.llm, role="verifier", instructions=VERIFIER, validate=validate,
                          tools=ontology_tools(ontology) + [submit_verdicts], task=json.dumps(task),
                          **self._visit_paths("verifier", visit_name))
        if not visit.completed:
            self._log("verifier_incomplete", visit=visit_name, steps=visit.steps,
                      outstanding_claim_ids=sorted(expected - set(pending["verdicts"])))

    def _visit_paths(self, role: str, name: str) -> dict:
        return {"lake_root": self.store.lake_root, "scratch_root": self.scratch_root,
                "artifact_root": self.log_root / role / name, "config": self.config.agent, "budget": self.budget}

    def _log(self, event: str, **data) -> None:
        self.log_root.mkdir(parents=True, exist_ok=True)
        with (self.log_root / "construction.jsonl").open("a") as stream:
            stream.write(json.dumps({"event": event, "time": time.time(), **data}, default=str) + "\n")
