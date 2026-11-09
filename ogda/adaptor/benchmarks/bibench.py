"""BI-Bench: one Lake of CSV tables per case, a table or value as the answer, and the benchmark's own comparison."""

from __future__ import annotations

import math
from pathlib import Path
import shutil

from ..artifacts import file_digest, read_json, safe_id
from . import GENERATION_FAILURES, clean_lake, identifiers, load_metadata

# BI-Bench's rule (tools/utils/prompt.py, PROMPT_PY): the result is a DataFrame, a string or a number, never a
# Series. Its scorer (compare_dataframes in tools/run_large_models.py) drops the index, so labels must be columns.
ANSWER_FORMAT = ("The answer is the query's result: a number, a string or a table. Give a number or a string directly. "
                 "For a table, a Step saves it as a CSV file in the Task's Workspace without the index, with every label "
                 "(such as the group of each row) in a column, and the answer is {'prediction_file': '<Workspace-relative path>.csv'}.")


def inventory(root: Path, lake_root: Path) -> tuple[list[dict], list[dict]]:
    from method.adaptor.bibench.adapter import validate_assets

    lakes, tasks = [], []
    official = root / "bi-bench"
    for case_id, question in sorted(read_json(official / "queries.json").items()):
        safe_id(case_id)
        validate_assets(root, case_id)
        path = lake_root / case_id
        expected = {p.name: file_digest(p) for p in (official / case_id).glob("*.csv")
                    if p.is_file() and not p.name.startswith("_")}
        # The prepared BI documentation contains only names, types and descriptions.
        docs = path / "documentation" / "schema.json"
        document = load_metadata(docs)
        validate_documentation(document)
        expected["documentation/schema.json"] = file_digest(docs)
        hashes = clean_lake(path, expected)
        lakes.append({"lake_id": case_id, "source": str(path.resolve()), "source_files_sha256": hashes})
        tasks.append({"task_id": case_id, "lake_id": case_id, "case_id": case_id, "question": question})
    return lakes, tasks


def validate_documentation(value) -> None:
    """Reject answer-bearing metadata rather than silently sanitizing an existing Lake."""
    allowed = {"datasets", "tables", "name", "description", "columns", "dtype", "type", "table", "table_name", "column_name", "data_type",
               "relationships", "from_table", "from_column", "to_table", "to_column"}
    if isinstance(value, dict):
        if set(value) - allowed:
            raise ValueError(f"Unexpected BI documentation keys: {sorted(set(value) - allowed)}")
        for item in value.values():
            validate_documentation(item)
    elif isinstance(value, list):
        for item in value:
            validate_documentation(item)


def check_reference(task: dict, root: Path) -> None:
    from method.adaptor.bibench.adapter import case_inputs

    question, _, _ = case_inputs(root / "bi-bench", task["case_id"])
    if question != task["question"]:
        raise ValueError(f"BI question changed: {task['task_id']}")


def answer_format(task: dict) -> str:
    return ANSWER_FORMAT


def validate_prediction(answer, workspace: Path) -> str | int | float | Path:
    """Return a number or string answer, or the CSV table in the Task Workspace that the answer names."""
    import pandas as pd

    if isinstance(answer, str) and answer.strip():
        return answer
    if isinstance(answer, (int, float)) and not isinstance(answer, bool):
        if not math.isfinite(answer):
            raise ValueError("A BI number answer must be finite")
        return answer
    if not isinstance(answer, dict) or set(answer) != {"prediction_file"} or not isinstance(answer["prediction_file"], str):
        raise ValueError("A BI answer is a number, a string or {'prediction_file': '<Workspace-relative path>.csv'}")
    relative = Path(answer["prediction_file"])
    path = (workspace / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(workspace.resolve()) or (workspace / relative).is_symlink() or not path.is_file():
        raise ValueError(f"{relative} is not a file in the Task Workspace {workspace}")
    try:
        frame = pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as error:
        raise ValueError(f"{relative} is not a readable CSV table: {error}") from None
    if len(frame.columns) and frame.columns[0] == "Unnamed: 0" and frame.iloc[:, 0].tolist() == list(range(len(frame))):
        raise ValueError(f"{relative} has the row numbers of the index as its first column: save the table with index=False")
    return path


def export_prediction(answer, workspace: Path, artifact: Path) -> Path:
    """Save the answer where the scorer reads it; a number or string becomes a one-cell table, as in BI-Bench."""
    import pandas as pd

    value = validate_prediction(answer, workspace)
    target = artifact / "prediction.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, Path):
        shutil.copyfile(value, target)
    else:
        pd.DataFrame({"result": [value]}).to_csv(target, index=False)
    return target


def check(value, task: dict, workspace: Path) -> None:
    validate_prediction(value, workspace)


def prediction(value, task: dict, workspace: Path, artifact: Path, explanation: str) -> dict:
    path = export_prediction(value, workspace, artifact)
    return {"prediction_file": str(path), "prediction_sha256": file_digest(path), "workflow": explanation}


def accepted(prediction: dict, task: dict) -> bool:
    path = Path(prediction.get("prediction_file", ""))
    return path.is_file() and file_digest(path) == prediction.get("prediction_sha256")


def evaluate(job: dict, prediction: dict) -> dict:
    import pandas as pd
    from method.adaptor.bibench.adapter import benchmark_modules, case_inputs, evaluate as compare

    task, root = job["task"], Path(job["benchmark_root"])
    result = dict(identifiers(task), generation_status=prediction["status"], metric="native compare_dataframes")
    if prediction["status"] in GENERATION_FAILURES:
        return dict(result, status="generation_failed", score=0.0)
    query, _, gold_files = case_inputs(root / "bi-bench", task["case_id"])
    if query != task["question"]:
        raise ValueError("BI prediction differs from the official question")
    path = Path(prediction["prediction_file"])
    if file_digest(path) != prediction["prediction_sha256"]:
        raise ValueError("BI prediction CSV changed after generation")
    with benchmark_modules(root):
        correct = compare(root, gold_files, pd.read_csv(path))
    return dict(result, status="scored", score=float(correct), correctness=correct)
