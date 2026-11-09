"""Command line: build an Ontology for a Lake, search it, and answer a Task with it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .construction import Construction, ConstructionConfig
from .data_agent import DataAgentConfig, answer
from .llm import OpenAICompatibleClient
from .ontology import OntologyStore
from .retrieval import retrieve


def build(args) -> None:
    document = yaml.safe_load(Path(args.config).read_text()) or {}
    settings = dict(document.get("construction") or {})
    for name in ("max_passes", "max_tokens", "files_per_visit", "max_claims_per_visit"):
        if getattr(args, name) is not None:
            settings[name] = getattr(args, name)
    output = Path(args.output)
    llm = OpenAICompatibleClient(args.config, record_path=output / "logs" / "llm.jsonl")
    store = OntologyStore(output / "ontology")
    construction = Construction(store, llm, ConstructionConfig.from_dict(settings),
                                scratch_root=Path(args.scratch), log_root=output / "logs")
    if not (args.resume and store.exists()):
        construction.start(args.lake_id or Path(args.lake).name, Path(args.lake), args.exclude, args.strip_key)
    ontology = construction.run()
    summary = {
        "lake_id": ontology.lake_id, "stop_reason": ontology.metadata.get("stop_reason"),
        **{name: len(getattr(ontology, name)) for name in ("schemas", "tables", "concepts", "links", "claims")},
        "claim_outcomes": {s: sum(c.status == s for c in ontology.claims.values())
                           for s in ("verified", "accepted", "refuted", "undecidable", "pending")},
        "experience": len(ontology.live_experience()),
        "llm_usage": llm.usage,
    }
    (output / "construction-summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def search(args) -> None:
    print(json.dumps(retrieve(OntologyStore(args.ontology).load(), args.query, limit=args.limit, hops=args.hops), indent=2))


def ask(args) -> None:
    document = yaml.safe_load(Path(args.config).read_text()) or {}
    config = DataAgentConfig.from_dict(document.get("data_agent"))
    task = json.loads(Path(args.task).read_text())
    output = Path(args.output)
    llm = OpenAICompatibleClient(args.config, record_path=output / "logs" / "llm.jsonl")
    try:
        result = answer(OntologyStore(args.ontology), llm, task, config, Path(args.scratch), output)
    finally:
        llm.close()
    summary = {key: result[key] for key in ("task", "status", "stop_reason", "answer", "explanation", "steps", "usage",
                                            "wall_seconds")}
    summary["grounded"] = [node["id"] for node in result["graph"]["nodes"] if node["kind"] in {"schema", "table", "file"}]
    summary["outside_ontology"] = result["graph"]["outside_ontology"]
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ogda")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("build", help="Build or resume the Ontology of one Lake")
    run.add_argument("--config", required=True, help="YAML with an llm block and an optional construction block")
    run.add_argument("--lake", required=True, help="Original Lake directory; it is copied, never changed")
    run.add_argument("--lake-id")
    run.add_argument("--exclude", nargs="*", default=[], help="Glob patterns of Files to leave out of the copy")
    run.add_argument("--strip-key", nargs="*", default=[], help="Keys to remove from JSON and YAML Files")
    run.add_argument("--output", required=True, help="Run directory, for example results/<benchmark>/<run-id>")
    run.add_argument("--scratch", required=True, help="Scratch directory, for example exp_env/<benchmark>/<run-id>")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--max-passes", type=int, help="Maximum Schema-enrichment Passes after File mapping")
    run.add_argument("--max-tokens", type=int)
    run.add_argument("--files-per-visit", type=int)
    run.add_argument("--max-claims-per-visit", type=int)

    find = commands.add_parser("search", help="Retrieve Ontology objects and Experience for a query")
    find.add_argument("--ontology", required=True, help="The ontology directory of a build")
    find.add_argument("--limit", type=int, default=10)
    find.add_argument("--hops", type=int, default=2)
    find.add_argument("query")

    task = commands.add_parser("ask", help="Answer one Task with a finished Harness")
    task.add_argument("--config", required=True, help="YAML with an llm block and an optional data_agent block")
    task.add_argument("--ontology", required=True, help="The ontology directory of a build; it is never changed")
    task.add_argument("--task", required=True, help='JSON file: {"id", "question", optional "answer_format"}')
    task.add_argument("--output", required=True, help="Task directory, for example results/<benchmark>/<run-id>/tasks/<task-id>")
    task.add_argument("--scratch", required=True, help="Task scratch, for example exp_env/<benchmark>/<run-id>/tasks/<task-id>")

    args = parser.parse_args(argv)
    {"build": build, "search": search, "ask": ask}[args.command](args)


if __name__ == "__main__":
    main()
