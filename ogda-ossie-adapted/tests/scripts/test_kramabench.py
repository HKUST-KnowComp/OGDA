"""Tests for the KramaBench benchmark module (scripts/benchmarks/kramabench.py)."""

from __future__ import annotations

import pytest

from benchmarks import kramabench


def test_lakes_are_domain_inputs(fake_kramabench):
    assert kramabench.lakes(fake_kramabench) == [
        {
            "lake_id": "environment",
            "source": fake_kramabench / "data/environment/input",
            "exclude": [],
            "strip_keys": [],
        }
    ]


def test_tasks_keep_the_question_verbatim(fake_kramabench):
    task = kramabench.tasks(fake_kramabench)[0]
    assert (
        task["question"] == "How many rows?"
        and task["answer_type"] == "numeric_exact"
        and task["lake_id"] == "environment"
    )


def test_check_names_the_json_kind_not_the_answer_type():
    with pytest.raises(ValueError, match="^The answer must be a JSON number$"):
        kramabench.check("3", {"answer_type": "numeric_exact"})
    kramabench.check([1, 2], {"answer_type": "list_exact"})
    with pytest.raises(ValueError, match="^The answer must be a JSON number, not a boolean$"):
        kramabench.check(True, {"answer_type": "numeric_exact"})


def test_summarize_counts_a_generation_failed_task_as_zero(monkeypatch, fake_kramabench, tmp_path):
    from benchmarks import Judge

    jobs = []

    def run_scorer(script, job, judge, cwd, log=None):
        jobs.append(job)
        return {"aggregates": []}

    monkeypatch.setattr(kramabench, "run_scorer", run_scorer)
    t1, t2 = kramabench.tasks(fake_kramabench)
    judge = Judge(model="m", api_key="k", base_url="http://judge.test/v1")
    failed = kramabench.evaluate({"task_id": "t2", "status": "incomplete", "answer": None, "pipeline_code": ""}, t2,
                                 judge, fake_kramabench, fake_kramabench, tmp_path)
    scored = {"task_id": "t1", "workload": "environment", "metric": "success", "status": "scored", "score": 1.0,
              "native_evaluation": {"task_id": "t1", "success": True, "llm_code_eval": [True]}}
    kramabench.summarize([scored, failed], fake_kramabench, judge)
    [job] = jobs
    assert {row["task_id"] for row in job["rows"]} == {"t1", "t2"}
    assert sorted((row["workload"], row["metric"], row["value"]) for row in job["rows"] if row["task_id"] == "t2") \
        == [("environment", "f1", 0), ("environment", "llm_code_eval", '"[]"'),
            ("overall", "f1", 0), ("overall", "llm_code_eval", '"[]"')]


def test_prediction_keeps_answer_and_pipeline():
    result = {"status": "answered", "answer": 3}
    assert kramabench.prediction(result, {"answer_type": "numeric_exact"}, "x = 1") == {
        "answer": 3,
        "pipeline_code": "x = 1",
    }
