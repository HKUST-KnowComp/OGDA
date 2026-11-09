"""Schema induction: a Proposer visit gives a batch of Files Datasets and Bindings to the Lake's one Ontology, code
checks each answer, a Verifier judges inferred Datasets, and code gives the other Files of each layout group the
representative's Datasets.

concept and relationship Claims take effect as soon as their answer passes the checks; dataset Claims are accepted
from their Documentation, from an accepted Dataset they copy (`same_as`), from layout fan-out, or by the Verifier.
For a layout group whose regions move or change length from File to File, the Proposer writes a locator: Python
source for find_region(rows) -> (header_row, start_row, end_row). Code runs it in an isolated worker, without Lake
access, on the representative (it must reproduce that region) and then on each member to place its region.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path

from smolagents import tool

from .agent import run_agent
from .binding import (Binding, binding_problems, data_problems, field_source, named_concepts, quote_literal,
                      structure_problems)
from .claims import KINDS, ORDER, answer_problems, apply_claim, is_noop, lake_path, parse_claim, path_values, trial
from .lake import (TABULAR, RowReader, batches, documentation_candidate, layout_groups, list_files, read_rows, sheets,
                   split_documentation)
from .ontology import Claim, Harness
from .ossie import BUILTINS, Dataset, Expression, OgdaExtension, ogda_extension, with_ogda_extension
from .prompts import PROPOSER, VERIFIER
from .retrieval import concept_detail, dataset_summary, ontology_summary, ontology_tools, representatives
from .sandbox.executor import IsolatedExecutor, ToolCorrection

STATUSES = {"verified", "refuted", "undecidable"}
CORRECTIONS = {"descriptions": "description", "datatypes": "datatype"}  # What a verified verdict may correct, by Field id.
DISPOSITIONS = {"mapped", "documentation", "unresolved"}
FINAL = {"mapped", "documentation"}
ANSWER_KEYS = {"claims", "file_dispositions", "summary", "unresolved", "separate", "locators"}
VERIFIER_VISITS_PER_RUN = 3
LISTED_MEMBERS = 20
ONTOLOGY_NOTE = "Accepted after code checks: the Ontology rules, Bindings and data checks of its answer passed"
STRUCTURAL_NOTE = "Same Fields and Binding as an accepted Dataset: its File, region, header and data checked in code"
DOCUMENTATION_NOTE = "Documentation: declared Dataset accepted from its Documentation; structure and data checked in code"
LAYOUT_NOTE = "Same layout as {rep}: each File's header, region and data checked in code"


def relative(path: str, root: Path) -> str:
    return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()


def new_files(run, progress: dict) -> list[str]:
    seen = set(progress["seen"])
    return [path for path in list_files(run.store.lake_root) if path not in seen]


def next_files(run, progress: dict) -> list[str]:
    """The next batch of Files. New Files are grouped by layout first; the other Files of a group wait for their
    representative, which stands for the group in a batch."""
    if fresh := new_files(run, progress):
        progress["groups"].update(layout_groups(run.store.lake_root, fresh))
        progress["seen"] += fresh
    waiting = {m for rep, members in progress["groups"].items() if rep not in progress["files"] for m in members}
    queue = [path for path in progress["seen"] if path not in progress["files"] and path not in waiting]
    return batches(queue, run.config.files_per_visit)[0] if queue else []


def task(run, harness: Harness, progress: dict, files: list[str]) -> dict:
    root, known = run.store.lake_root, progress["files"]
    groups = {}
    for path in files:
        if members := progress["groups"].get(path):
            groups[str(root / path)] = {"files": len(members) + 1, "members": members[:LISTED_MEMBERS],
                                        "more": max(0, len(members) - LISTED_MEMBERS)}
    return {
        "stage": "induction", "reason": "Unmapped Files need Datasets and Bindings",
        "files": [str(root / path) for path in files],
        "layout_groups": groups,
        "documentation": [str(root / p) for p, item in known.items() if item["status"] == "documentation"],
        "documentation_candidates": [str(root / p) for p in progress["seen"]
                                     if documentation_candidate(p) and p not in known],
        "ontology": ontology_summary(harness),
        "datasets": [dataset_summary(harness, name) for name in representatives(harness)],
        "retry": {str(root / path): progress.get("retried", {})[path] for path in files
                  if path in progress.get("retried", {})},
        "decided_proposals": [{"kind": c.kind, "payload": c.payload, "status": c.status, "notes": c.notes}
                              for c in harness.claims.values()
                              if c.kind == "dataset" and c.status in {"refuted", "undecidable"}],
        "remaining_work": run.unresolved(progress),
    }


def region(value) -> tuple:
    """A locator's result: (header_row, start_row, end_row) with header_row and end_row int or None."""
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"find_region must return (header_row, start_row, end_row), not {value!r}")
    header, start, end = value
    if any(v is not None and (isinstance(v, bool) or not isinstance(v, int)) for v in (header, end)) or (
            isinstance(start, bool) or not isinstance(start, int)):
        raise ValueError(f"find_region must return integers (header_row and end_row may be None), not {value!r}")
    return header, start, end


def locate(run, source: str, jobs: list[tuple[str, str | None]], name: str) -> list[tuple | str]:
    """Run a locator's find_region on the rows of each (path, sheet) in one isolated worker without Lake access.

    Returns a region or an error message per job; a failing locator never stops construction.
    """
    executor = IsolatedExecutor(run.store.lake_root, run.scratch_root / "locators", run.log_root / "locators" / name,
                                timeout=run.config.agent.cell_timeout_seconds, memory_mb=run.config.agent.memory_mb,
                                max_output_chars=2000, lake="none")
    results = []
    try:
        for path, sheet in jobs:
            try:
                executor.send_variables({"ROWS": read_rows(run.store.lake_root / path, sheet)})
                results.append(region(executor(f"{source}\nfind_region(ROWS)").output))
            except Exception as error:  # The File is released to the queue with this reason.
                results.append(f"{type(error).__name__}: {error}"[:500])
    finally:
        executor.cleanup()
    return results


def check_locators(run, locators, datasets: dict[str, Dataset], reps: set[str], visit_name: str) -> list[str]:
    """Errors for locators that name no Dataset of a layout group representative in this visit, or do not reproduce
    its region."""
    if not isinstance(locators, dict) or any(not isinstance(k, str) or not isinstance(v, str) or not v.strip()
                                             for k, v in locators.items()):
        raise ValueError("locators maps representative Dataset names to Python source that defines find_region(rows)")
    errors = []
    for name, source in locators.items():
        dataset = datasets.get(name)
        extension = ogda_extension(dataset) if dataset is not None else None
        if extension is None or extension.path not in reps:
            errors.append(f"locators: {name!r} is not a Dataset of a layout group representative in this visit")
            continue
        expected = (extension.header_row, extension.start_row, extension.end_row)
        found = locate(run, source, [(extension.path, extension.sheet)], f"{visit_name}-check-{name}")[0]
        if found != expected:
            errors.append(f"locators[{name!r}]: on {extension.path} find_region returned {found!r}; it must "
                          f"reproduce the Dataset's (header_row, start_row, end_row) = {expected!r}")
    return errors


def route(payload: dict) -> str:
    """A copy of an accepted Dataset is checked in code when code can read its File and verified otherwise, since its
    Documentation was written for another File; a declared Dataset is accepted from its Documentation."""
    extension = ogda_extension(Dataset.model_validate(payload["dataset"]))
    if payload.get("same_as"):
        return "structural" if Path(extension.path).suffix.lower() in TABULAR else "verifier"
    return "documentation" if extension.origin == "declared" else "verifier"


def parse_dispositions(items, root: Path) -> list[dict]:
    if not isinstance(items, list):
        raise ValueError("file_dispositions must be a list")
    result = []
    for item in items:
        if not isinstance(item, dict) or item.get("status") not in DISPOSITIONS:
            raise ValueError("Each File disposition needs path and status: mapped, documentation or unresolved")
        if not isinstance(item.get("path"), str) or not item["path"]:
            raise ValueError("A File disposition needs a nonempty path")
        path, status = lake_path(item["path"], root), item["status"]
        if status == "mapped":
            names = item.get("datasets")
            if item.get("complete") is not True or not isinstance(names, list) or not names or any(
                    not isinstance(name, str) for name in names):
                raise ValueError(f"{path}: mapped requires complete=True and the datasets of all its regions; leave a "
                                 "File you have not finished out of the answer")
            result.append({"path": path, "status": status, "complete": True, "datasets": list(dict.fromkeys(names))})
        else:
            if not isinstance(item.get("reason"), str) or not item["reason"].strip():
                raise ValueError(f"{path}: {status} needs a reason")
            result.append({"path": path, "status": status, "reason": item["reason"]})
    if len({item["path"] for item in result}) != len(result):
        raise ValueError("Submit one disposition per File")
    return result


def coverage_problem(root: Path, path: str, names: list[str], harness: Harness) -> str | None:
    """Why the Datasets `names` do not cover the File at `path`, or None.

    They must be Datasets of that File in `harness`, and include every Dataset of the File there. Each nonempty
    worksheet of a workbook must hold one of them or be cited as Documentation by a Dataset.
    """
    datasets = harness.datasets()
    if missing := [name for name in names if name not in datasets]:
        return f"Datasets {missing} are not accepted or proposed Datasets"
    if elsewhere := [name for name in names if ogda_extension(datasets[name]).path != path]:
        return f"Datasets {elsewhere} are Datasets of other Files"
    if omitted := sorted({n for n, d in datasets.items() if ogda_extension(d).path == path} - set(names)):
        return f"the disposition omits the File's Datasets {omitted}"
    try:
        worksheets, default_sheet = sheets(root / path)
    except Exception as error:  # A corrupt workbook cannot be mapped; the reason says why.
        return f"cannot read its worksheets: {type(error).__name__}: {error}"
    covered = {ogda_extension(datasets[name]).sheet or default_sheet for name in names}
    covered |= {sheet for d in datasets.values() for ref in ogda_extension(d).documentation
                for file, sheet in [split_documentation(ref)] if file == path}
    if uncovered := [name for name in worksheets if name not in covered]:
        return f"worksheets {uncovered} are in no Dataset and no Dataset cites them as Documentation"
    return None


def pending_datasets(harness: Harness, pending: dict) -> list[tuple[str, dict]]:
    """The dataset Claims of earlier answers in this visit that are not decided yet."""
    claims = (harness.claims[cid] for cid in pending["claim_ids"])
    return [(c.kind, c.payload) for c in claims if c.kind == "dataset" and c.status == "pending"]


def propose(run, harness: Harness, progress: dict, pending: dict) -> None:
    """One induction visit. Each answer the Proposer submits, with submit or final_answer, is checked on its own and
    saved at once; a rejected answer leaves earlier answers in place.

    A File gets its disposition from an answer only, and once it has one its Datasets and disposition are fixed for
    the visit. Files that no answer covered stay in the queue. When a run stops during the visit, resume keeps the
    saved answers and ends the visit instead of repeating it.
    """
    root = run.store.lake_root
    batch = {relative(path, root) for path in pending["task"]["files"]}
    members = {m for rep in batch for m in progress["groups"].get(rep, [])}
    resumed = pending.get("answers", 0) > 0
    for key, empty in (("answers", 0), ("claim_ids", []), ("routes", {}), ("file_dispositions", []), ("separate", []),
                       ("locators", {}), ("summary", ""), ("unresolved", [])):
        pending.setdefault(key, copy.deepcopy(empty))

    def submitted_files() -> set[str]:
        return {item["path"] for item in pending["file_dispositions"]}

    def accept(answer) -> None:
        if not isinstance(answer, dict):
            raise ValueError("Submit one JSON object")
        if unknown := sorted(map(str, set(answer) - ANSWER_KEYS)):
            raise ValueError(f"Unknown answer keys {unknown}. An answer is {{'claims': [...], 'file_dispositions': "
                             "[...], 'summary': str, 'unresolved': [str], 'separate': [...], 'locators': {...}}; "
                             "each Claim goes in its claims list")
        claims, summary, unresolved = answer.get("claims", []), answer.get("summary", ""), answer.get("unresolved", [])
        if not isinstance(claims, list) or not isinstance(summary, str) or not isinstance(unresolved, list):
            raise ValueError("Submit {'claims': [...], 'summary': str, 'unresolved': [str], 'file_dispositions': [...]}")
        earlier = [(c.kind, c.payload) for c in (harness.claims[cid] for cid in pending["claim_ids"])]
        waiting = pending_datasets(harness, pending)
        saved = {item["path"]: item for item in pending["file_dispositions"]}
        parsed, errors = {}, []
        for index, item in enumerate(claims):
            try:
                kind, payload = parse_claim(item, harness, root)
                if kind not in KINDS["induction"]:
                    raise ValueError("Schema induction permits only concept, relationship and dataset Claims")
                if (kind, payload) in earlier:
                    continue  # Already submitted in this visit: repeating it changes nothing.
                if kind == "dataset" and (path := ogda_extension(Dataset.model_validate(payload["dataset"])).path) in saved:
                    raise ValueError(f"{path} already has a disposition in this visit; its Datasets are fixed")
                parsed[index] = (kind, payload)
            except (ValueError, TypeError) as error:
                errors.append(f"claims[{index}]: {error}")
        submitted = sum(harness.claims[cid].kind == "dataset" for cid in pending["claim_ids"])
        if submitted + sum(kind == "dataset" for kind, _ in parsed.values()) > run.config.max_datasets_per_visit:
            errors.append(f"Submit at most {run.config.max_datasets_per_visit} Datasets in this visit; {submitted} are "
                          "already submitted (concept and relationship Claims do not count)")
        new = {index: Dataset.model_validate(p["dataset"]) for index, (k, p) in parsed.items() if k == "dataset"}
        if not errors:
            owner = {d.name: ogda_extension(d).path for d in harness.semantic_model.datasets}
            owner.update({p["dataset"]["name"]: p["dataset"]["source"].split("#")[0] for _, p in waiting})
            for index, dataset in sorted(new.items()):
                if owner.setdefault(dataset.name, ogda_extension(dataset).path) != ogda_extension(dataset).path:
                    errors.append(f"claims[{index}]: Dataset name {dataset.name} already maps another File; give each "
                                  "File's Dataset its own name")
            for index, dataset in sorted(new.items()):
                if (target := ogda_extension(dataset).relation_to) and target not in owner:
                    errors.append(f"claims[{index}]: relation_to {target!r} names no Dataset")
        after = harness
        if not errors:
            try:
                after = trial(harness, waiting + list(parsed.values()))
            except ValueError as error:
                errors.append(str(error))
            else:
                errors += answer_problems(harness, after, {d.name for d in new.values()}, root, run.config.check_rows)
        dispositions = [item for item in parse_dispositions(answer.get("file_dispositions", []), root)
                        if saved.get(item["path"]) != item]  # A repeated disposition changes nothing.
        if changed := sorted({item["path"] for item in dispositions} & set(saved)):
            errors.append(f"file_dispositions: {changed} already have a different disposition in this visit")
        if not errors:  # Check each mapping against the Datasets this visit has proposed so far.
            errors += [f"file_dispositions: {item['path']}: {problem}" for item in dispositions if item["status"] == "mapped"
                       and (problem := coverage_problem(root, item["path"], item["datasets"], after))]
        separate = answer.get("separate", [])
        if not isinstance(separate, list) or any(not isinstance(p, str) for p in separate):
            raise ValueError("separate must be a list of paths")
        separate = [lake_path(p, root) for p in separate]
        if outside := [p for p in separate if p not in members]:
            errors.append(f"separate lists Files that are not members of a layout group in this visit: {outside}")
        locators = answer.get("locators", {})
        if not isinstance(locators, dict):
            raise ValueError("locators maps representative Dataset names to Python source that defines find_region(rows)")
        locators = {name: source for name, source in locators.items() if pending["locators"].get(name) != source}
        if not errors and locators:
            if changed := sorted(set(locators) & set(pending["locators"])):
                errors.append(f"locators: {changed} already have a different locator in this visit")
            errors += check_locators(run, locators, {d.name: d for d in new.values()}, batch & set(progress["groups"]),
                                     pending["visit_name"])
        if errors:
            raise ValueError("Correct these Claims and submit this answer again; earlier answers stay saved:\n"
                             + "\n".join(errors))
        for index in sorted(parsed, key=lambda i: (ORDER[parsed[i][0]], i)):
            kind, payload = parsed[index]
            previous = next((c for c in harness.claims.values() if c.kind == kind and c.payload == payload
                             and c.status in {"refuted", "undecidable"}), None)
            if is_noop(kind, payload, harness) or previous:
                run.log("claim_skipped", iteration=pending["iteration"], kind=kind,
                        reason="unchanged" if previous is None else previous.status)
                if previous and previous.status == "undecidable":
                    pending["unresolved"].append(previous.notes)
                continue
            claim = run.claim(harness, kind, payload, pending["iteration"], pending["visit_name"])
            pending["claim_ids"].append(claim.id)
            if kind == "dataset":
                pending["routes"][claim.id] = route(payload)
            else:
                run.settle(harness, claim, "accepted", ONTOLOGY_NOTE, route="ontology", save=False)
        for name, source in locators.items():
            run.log("locator", iteration=pending["iteration"], dataset=name, source=source)
        pending["file_dispositions"] += dispositions
        pending["separate"] = list(dict.fromkeys(pending["separate"] + separate))
        pending["locators"].update(locators)
        if summary not in pending["summary"]:
            pending["summary"] = "\n".join(text for text in (pending["summary"], summary) if text)
        pending["unresolved"] = list(dict.fromkeys(pending["unresolved"] + [str(u) for u in unresolved]))
        pending["answers"] += 1
        run.store.save(harness)  # Submitted Claims and dispositions survive a crash later in the visit.

    @tool
    def submit(answer: dict) -> dict:
        """Save finished work now, before continuing with the other Files. A rejected answer can be corrected and
        submitted again; accepted answers stay saved even if the visit ends early.

        Args:
            answer: The same object as final_answer takes, for the Files finished so far: claims, file_dispositions,
                summary, unresolved, separate and locators.
        """
        try:
            accept(answer)
        except ValueError as error:
            raise ToolCorrection(str(error)) from None
        return {"submitted_files": sorted(submitted_files()), "remaining_files": sorted(batch - submitted_files())}

    visit = None
    if not resumed:
        visit = run_agent(run.llm, role="proposer", task=json.dumps(pending["task"]),
                          instructions=PROPOSER + f"\nPropose at most {run.config.max_datasets_per_visit} Datasets in "
                                                  "this visit; concept and relationship Claims do not count.",
                          validate=accept, tools=ontology_tools(harness) + [submit],
                          **run.visit_paths("proposer", pending["visit_name"]))
    if visit is None or not visit.completed:
        remaining = sorted(batch - submitted_files())
        run.log("proposer_incomplete", iteration=pending["iteration"], steps=visit.steps if visit else None,
                stage="induction", answers=pending["answers"], remaining_files=remaining)
        if remaining:
            pending["unresolved"].append(f"The visit ended before these Files were submitted: {remaining}")
    pending.update(phase="verify", verdicts={}, verification_attempts=0)
    run.store.save(harness)


def missing_references(claim: Claim, harness: Harness, outstanding: list[Claim] = ()) -> list[str]:
    """The Datasets a dataset Claim relates to that neither the Harness nor the `outstanding` Claims provide."""
    target = ogda_extension(Dataset.model_validate(claim.payload["dataset"])).relation_to
    provided = {c.payload["dataset"]["name"] for c in outstanding if c.id != claim.id}
    return [target] if target and harness.dataset(target) is None and target not in provided else []


def apply_verdicts(run, harness: Harness, pending: dict, claims: list[Claim]) -> None:
    """Apply available judgments, retaining judgments that wait on a Dataset another pending Claim provides, and save
    once."""
    settled = False
    for claim in claims:
        if claim.status != "pending":
            continue
        verdict = pending["verdicts"].get(claim.id)
        outstanding = [c for c in claims if c.status == "pending"]
        if missing := missing_references(claim, harness, outstanding):
            run.settle(harness, claim, "undecidable", f"It relates to Datasets that were not accepted: {missing}",
                       route="reference", save=False)
            settled = True
        elif verdict and (not missing_references(claim, harness) or verdict["status"] not in {"verified", "accepted"}):
            correct_fields(run, claim, verdict)
            run.settle(harness, claim, verdict["status"], verdict["notes"], verdict["evidence_refs"],
                       route=pending["routes"][claim.id], save=False)
            settled = True
    if settled:
        run.store.save(harness)


def corrected(dataset: dict, verdict: dict) -> list[tuple[str, str, str, str]]:
    """(Field id, attribute, old, new) for each correction a verdict makes to a Dataset payload."""
    changes = []
    for key, attribute in CORRECTIONS.items():
        for field in dataset["fields"]:
            new = verdict.get(key, {}).get(f"{dataset['name']}.{field['name']}")
            if new is not None and field.get(attribute) != new:
                changes.append((f"{dataset['name']}.{field['name']}", attribute, field.get(attribute), new))
    return changes


def correct_fields(run, claim: Claim, verdict: dict) -> None:
    """Write the Verifier's corrected Field descriptions and datatypes into a dataset Claim before it is applied."""
    fields = {f"{claim.payload['dataset']['name']}.{f['name']}": f for f in claim.payload["dataset"]["fields"]}
    for field_id, attribute, old, new in corrected(claim.payload["dataset"], verdict):
        run.log(f"{attribute}_corrected", claim_id=claim.id, field_id=field_id, old=old, new=new)
        fields[field_id][attribute] = new


def decide(run, harness: Harness, pending: dict, claims: list[Claim]) -> bool:
    for claim in claims:
        claim_route = pending["routes"].get(claim.id)
        if claim_route in {"documentation", "structural"} and claim.id not in pending["verdicts"]:
            note = DOCUMENTATION_NOTE if claim_route == "documentation" else STRUCTURAL_NOTE
            pending["verdicts"][claim.id] = {"status": "verified", "evidence_refs": [], "notes": note}
    apply_verdicts(run, harness, pending, claims)
    for _ in range(VERIFIER_VISITS_PER_RUN):
        outstanding = [c for c in claims if c.status == "pending" and c.id not in pending["verdicts"]]
        if not outstanding:
            break
        pending["verification_attempts"] += 1
        run.store.save(harness)
        attempt = pending["verification_attempts"]
        name = pending["visit_name"] if attempt == 1 else f"{pending['visit_name']}-retry-{attempt:03d}"
        completed = len(pending["verdicts"])
        verify(run, harness, pending, claims, outstanding, name)
        apply_verdicts(run, harness, pending, claims)
        if len(pending["verdicts"]) == completed:
            break  # A stalled visit needs an explicit resume, not another identical automatic visit.
    return all(c.status != "pending" for c in claims)


def verify(run, harness: Harness, pending: dict, batch: list[Claim], claims: list[Claim], visit_name: str) -> None:
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
            dataset = harness.claims[v["claim_id"]].payload["dataset"]
            own = {f"{dataset['name']}.{f['name']}" for f in dataset["fields"]}
            corrections = {key: v.get(key, {}) for key in CORRECTIONS}
            for key, values in corrections.items():
                if not isinstance(values, dict) or any(not isinstance(text, str) or not text.strip()
                                                       for text in values.values()):
                    raise ValueError(f"{v['claim_id']}: {key} maps Field ids to corrected {key}")
                if values and v["status"] != "verified":
                    raise ValueError(f"{v['claim_id']}: corrected {key} go only with a verified verdict")
                if unknown := sorted(set(values) - own):
                    raise ValueError(f"{v['claim_id']}: {unknown} are not Fields of this Dataset")
            if wrong := sorted(set(corrections["datatypes"].values()) - set(BUILTINS)):
                raise ValueError(f"{v['claim_id']}: a datatype is one of {list(BUILTINS)}, not {wrong}")
            if corrections["datatypes"]:
                fixed = copy.deepcopy(dataset)
                for field_id, attribute, _, new in corrected(fixed, corrections):
                    next(f for f in fixed["fields"] if f"{fixed['name']}.{f['name']}" == field_id)[attribute] = new
                binding = harness.claims[v["claim_id"]].payload["binding"]
                if problems := binding_problems(Binding.model_validate(binding), Dataset.model_validate(fixed),
                                                harness.ontology):
                    raise ValueError(f"{v['claim_id']}: these datatypes break the Binding: {'; '.join(problems)}")
            value = {"status": v["status"], "notes": v["notes"], "evidence_refs": refs, **corrections}
            if v["claim_id"] in pending["verdicts"] and pending["verdicts"][v["claim_id"]] != value:
                raise ValueError(f"{v['claim_id']}: a submitted verdict cannot be changed")
        for v in verdicts:
            pending["verdicts"][v["claim_id"]] = {"status": v["status"], "notes": v["notes"],
                                                  "evidence_refs": list(v.get("evidence_refs", [])),
                                                  **{key: dict(v.get(key, {})) for key in CORRECTIONS}}
            harness.claims[v["claim_id"]].verifier_visit = f"verifier/{visit_name}"
        run.store.save(harness)
        apply_verdicts(run, harness, pending, batch)

    @tool
    def submit_verdicts(verdicts: list[dict]) -> dict:
        """Persist completed judgments now, before continuing to other Claims.

        Args:
            verdicts: Completed verdicts, each with claim_id, status, notes, optional evidence_refs and, for verified
                verdicts only, optional descriptions (Field id -> corrected description) and datatypes (Field id ->
                corrected datatype).
        """
        try:
            submit(verdicts)
        except ValueError as error:
            raise ToolCorrection(str(error)) from None
        return {"outstanding_claim_ids": sorted(expected - set(pending["verdicts"]))}

    def validate(answer: dict) -> None:
        if not isinstance(answer, dict):
            raise ValueError("final_answer needs one JSON object")
        submit(answer.get("verdicts"))

    named = set()
    for claim in claims:
        if claim.payload.get("binding"):
            named |= named_concepts(Binding.model_validate(claim.payload["binding"]), harness.ontology)
    task = {"stage": "induction",
            "claims": [{"id": c.id, "dataset": c.payload["dataset"], "binding": c.payload.get("binding")} for c in claims],
            "concepts": [concept_detail(harness, name) for name in sorted(named) if harness.concept(name)],
            "datasets": [dataset_summary(harness, name) for name in representatives(harness)
                         if (b := harness.binding_of(name)) and named_concepts(b, harness.ontology) & named]}
    visit = run_agent(run.llm, role="verifier", instructions=VERIFIER, validate=validate,
                      tools=ontology_tools(harness) + [submit_verdicts], task=json.dumps(task),
                      **run.visit_paths("verifier", visit_name))
    if not visit.completed:
        run.log("verifier_incomplete", visit=visit_name, steps=visit.steps,
                outstanding_claim_ids=sorted(expected - set(pending["verdicts"])))


def record_documentation(run, harness: Harness, progress: dict) -> None:
    for dataset in harness.semantic_model.datasets:
        for ref in ogda_extension(dataset).documentation:
            path, sheet = split_documentation(ref)
            if (sheet is None and progress["files"].get(path, {}).get("status") not in FINAL
                    and (run.store.lake_root / path).is_file()):  # A cited worksheet leaves its workbook a data File.
                progress["files"][path] = {"status": "documentation", "reason": f"Defines accepted Dataset {dataset.name}"}
                run.log("file_disposition", path=path, **progress["files"][path])


def check_disposition(run, harness: Harness, path: str, item: dict) -> dict:
    """The disposition to record: a mapped File must still be covered by accepted Datasets."""
    root = run.store.lake_root
    if not (root / path).is_file():
        return {"status": "unresolved", "reason": "The File no longer exists in the Lake copy"}
    item = {key: value for key, value in item.items() if key != "path"}
    if item["status"] != "mapped":
        return item
    if problem := coverage_problem(root, path, item["datasets"], harness):
        return {"status": "unresolved", "reason": f"Not mapped after verification: {problem}"}
    return item


def refutations(harness: Harness, pending: dict) -> dict[str, list[str]]:
    """The Verifier's notes on this Iteration's refuted Datasets, by the Lake path of their File."""
    notes: dict[str, list[str]] = {}
    for claim in (harness.claims[cid] for cid in pending["claim_ids"]):
        if claim.status == "refuted" and claim.kind == "dataset":
            path = ogda_extension(Dataset.model_validate(claim.payload["dataset"])).path
            notes.setdefault(path, []).append(f"dataset Claim refuted: {claim.notes}")
    return notes


def finish(run, harness: Harness, progress: dict, pending: dict) -> bool:
    """Record File dispositions, then give the other Files of each layout group in the batch the representative's
    Datasets. Returns whether the visit made progress: it accepted a Claim, or a File of the batch left the queue or
    came back for its retry. A File too large for one visit is finished over several, each building on the Claims the
    earlier ones accepted.

    A File without a submitted disposition stays in the queue, and the other Files of its layout group wait for it.
    A File left unmapped because the Verifier refuted one of its Datasets returns to the queue once, with the
    Verifier's notes; the other Files of its layout group wait for it. A second failure is final.
    """
    root = run.store.lake_root
    record_documentation(run, harness, progress)
    dispositions = {item["path"]: item for item in pending["file_dispositions"]}
    batch = {relative(path, root) for path in pending["task"]["files"]}
    refuted, retried = refutations(harness, pending), progress.setdefault("retried", {})
    returned = []
    for path in sorted(batch | set(dispositions)):
        if progress["files"].get(path, {}).get("status") in FINAL:
            continue
        if path in dispositions:
            item = check_disposition(run, harness, path, dispositions[path])
        elif not (root / path).is_file():
            item = {"status": "unresolved", "reason": "The File no longer exists in the Lake copy"}
        else:
            continue  # No answer covered it: it stays in the queue.
        if item["status"] == "unresolved" and path in refuted and path not in retried:
            retried[path] = refuted[path]
            progress["files"].pop(path, None)
            returned.append(path)
            run.log("file_retry", iteration=pending["iteration"], path=path, notes=refuted[path])
            continue
        progress["files"][path] = item
        run.log("file_disposition", iteration=pending["iteration"], path=path, **progress["files"][path])
    for path in sorted(batch | set(dispositions)):
        if path in progress["groups"] and path in progress["files"]:
            fan_out(run, harness, progress, pending, path)
    accepted = any(harness.claims[cid].status in {"verified", "accepted"} for cid in pending["claim_ids"])
    return accepted or bool(returned) or any(path in progress["files"] for path in batch)


def digest(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def member_name(base: str, path: str, harness: Harness, taken: dict[str, str]) -> str:
    """A Dataset name for a group member: the representative's name with the member's file name, unique in the Lake."""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", Path(path).stem).strip("_").lower() or "file"
    candidate, number = f"{base}__{slug}", 2
    while taken.get(candidate, path) != path or (
            (existing := harness.dataset(candidate)) is not None and ogda_extension(existing).path != path):
        candidate, number = f"{base}__{slug}_{number}", number + 1
    taken[candidate] = path
    return candidate


def row_counts(path: Path, datasets: list[Dataset], reader: RowReader | None = None) -> dict:
    reader = reader or RowReader()
    return {sheet: len(reader(path, sheet)) for sheet in sorted({ogda_extension(d).sheet for d in datasets}, key=str)}


def member_dataset(dataset: Dataset, name: str, path: str, **region) -> Dataset:
    """The representative's Dataset for one member File: its path Fields filled from the member's path."""
    extension = OgdaExtension.model_validate({**ogda_extension(dataset).model_dump(), "path": path,
                                              "layout_of": dataset.name, **region})
    values = path_values(extension.path_pattern, path)
    fields = [f.model_copy(update={"expression": Expression(sql=quote_literal(values[f.name]))})
              if field_source(f)[0] == "literal" else f for f in dataset.fields]
    source = f"{path}#{extension.sheet}" if extension.sheet else path
    return with_ogda_extension(dataset.model_copy(update={"name": name, "source": source, "fields": fields}), extension)


def fan_out(run, harness: Harness, progress: dict, pending: dict, rep: str) -> None:
    """Give the other Files of a layout group the representative's Datasets, checked in code, and save once.

    A member's Dataset shares the representative's Fields and Binding, with its path Fields filled from its own path.
    A member that already has Datasets, from the Proposer, is mapped with them when they cover it. A Dataset with a
    locator gets each member's region from it. Otherwise a member gets the representative's region, and when that
    region ends before the end of the File, only a member with the same number of rows gets it. A member whose
    locator fails, or whose structure, path or data checks fail, joins the queue as a single File, as do members the
    Proposer listed in `separate`. A member with the same bytes as an earlier File becomes a copy of its Dataset.
    """
    root = run.store.lake_root
    separate = set(pending.get("separate", []))
    members = [m for m in progress["groups"][rep] if m not in progress["files"] and m not in separate]
    item = progress["files"].get(rep, {})
    if item.get("status") != "mapped":
        reason = f"Same layout as {rep}, which was not mapped: {item.get('reason', 'no disposition')}"
        for member in members:
            progress["files"][member] = {"status": "unresolved", "reason": reason}
        run.log("layout_group_unmapped", representative=rep, files=len(members), reason=reason)
        progress["groups"].pop(rep)
        run.store.save(harness)
        return
    datasets = [harness.dataset(name) for name in item["datasets"]]
    locators = {name: source for name, source in pending.get("locators", {}).items() if name in item["datasets"]}
    fixed = [d for d in datasets if ogda_extension(d).end_row is not None and d.name not in locators]
    lengths = row_counts(root / rep, fixed) if fixed else None
    own: dict[str, list[str]] = {}
    for dataset in harness.semantic_model.datasets:
        own.setdefault(ogda_extension(dataset).path, []).append(dataset.name)
    placed = [m for m in members if m not in own]
    located = {name: dict(zip(placed, locate(run, source, [(m, ogda_extension(harness.dataset(name)).sheet)
                                                           for m in placed], f"{pending['visit_name']}-{name}")))
               for name, source in locators.items()}
    copies = {digest(root / rep): {ogda_extension(d).sheet: d.name for d in datasets}}
    accepted, mapped, released, taken, proposed = [], {}, {}, {}, {}
    for member in members:
        if member in own:
            done = check_disposition(run, harness, member, {"status": "mapped", "complete": True,
                                                            "datasets": sorted(own[member])})
            if done["status"] == "mapped":
                proposed[member] = done["datasets"]
            else:
                released[member] = [done["reason"]]
            continue
        reader = RowReader()  # The member's File is read once for its row counts and checks.
        if lengths is not None:
            try:
                same_length = row_counts(root / member, fixed, reader) == lengths
            except Exception as error:  # An unreadable member gets its own visit, where the Proposer sees why.
                same_length, error_text = False, f"cannot read {member}: {type(error).__name__}: {error}"
            else:
                error_text = f"Its row count differs from {rep}, whose Datasets end before the end of the File"
            if not same_length:
                released[member] = [error_text]
                continue
        built, problems, same = [], [], copies.get(digest(root / member), {})
        for dataset in datasets:
            extension, region = ogda_extension(dataset), {"relation": "part", "relation_to": None}
            if original := same.get(extension.sheet):
                region.update(relation="copy", relation_to=original)
            if dataset.name in located:
                found = located[dataset.name][member]
                if isinstance(found, str):
                    problems.append(f"The locator of {dataset.name} failed on {member}: {found}")
                    continue
                region.update(header_row=found[0], start_row=found[1], end_row=found[2])
            try:
                candidate = member_dataset(dataset, member_name(dataset.name, member, harness, taken), member, **region)
            except ValueError as error:
                problems.append(str(error))
                continue
            if not (found := structure_problems(candidate, root, run.config.check_rows, reader)):
                found = data_problems(candidate, harness.binding_of(dataset.name), harness.ontology, root,
                                      run.config.check_rows, reader)
            problems += found
            built.append(candidate)
        if problems:
            released[member] = problems
            continue
        copies.setdefault(digest(root / member), {ogda_extension(d).sheet: d.name for d in built})
        mapped[member] = [d.name for d in built]
        accepted += built
    for dataset in accepted:
        payload = {"dataset": dataset.model_dump(mode="json"), "binding": None}
        if is_noop("dataset", payload, harness):
            continue
        claim = run.claim(harness, "dataset", payload, pending["iteration"], pending["visit_name"])
        claim.status, claim.notes, claim.evidence_refs = "verified", LAYOUT_NOTE.format(rep=rep), [rep]
        apply_claim(claim, harness)
        run.log("claim_outcome", claim_id=claim.id, kind=claim.kind, status=claim.status, notes=claim.notes,
                acceptance_route="layout")
    for member, names in mapped.items():
        progress["files"][member] = {"status": "mapped", "complete": True, "datasets": names,
                                     "reason": f"Same layout as {rep}"}
    for member, names in proposed.items():
        progress["files"][member] = {"status": "mapped", "complete": True, "datasets": names,
                                     "reason": "Mapped by its own Datasets"}
    run.log("layout_group_mapped", representative=rep, mapped=len(mapped), proposed=len(proposed), released=released,
            separate=sorted(separate & set(progress["groups"][rep])), located=sorted(located))
    progress["groups"].pop(rep)
    run.store.save(harness)
