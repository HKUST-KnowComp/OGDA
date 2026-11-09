"""scripts/benchmark.py: run ids, preflight and the construct and infer stages."""

from __future__ import annotations

import json
import os
import pickle
from datetime import datetime, timezone

import pytest

from benchmark import default_run_id, main, model_slug


def test_model_slug_is_filesystem_safe():
    assert model_slug("openrouter/anthropic/claude-x:free") == "claude-x-free"
    assert default_run_id("baseline", "openai/gpt-5", datetime(2026, 10, 29, 15, 30, tzinfo=timezone.utc)) \
        == "baseline_gpt-5_20261029T153000Z"


@pytest.mark.parametrize("problem, message", [
    ("no_lake_dir", "prepare_lakes.py"), ("unknown_lake", "Unknown Lake"), ("unknown_task", "Unknown Task"),
    ("existing_run", "--resume"), ("no_sandbox", "sandbox"), ("bad_config", "model")])
def test_preflight_reports_each_problem(problem, message, env, capsys):
    assert main(env.argv(problem)) == 2
    err = capsys.readouterr().err
    assert err.startswith("preflight: ") and message in err
    assert not (env.results / "construction").exists() or problem == "existing_run"


def test_preflight_rejects_a_changed_lake(env, capsys):
    (env.lake_dir / "environment" / "a.csv").write_text("changed\n")
    assert main(env.argv("construct")) == 2
    assert "prepare_lakes.py" in capsys.readouterr().err


def test_default_construct_run_id_names_the_proposer_model(env):
    import benchmark

    plan = benchmark.preflight(benchmark.parse_args(env.argv("construct")))
    assert plan.run_id.startswith("ontology_scripted-model_") and plan.run_id.endswith("Z")
    assert plan.out == env.results / "construction" / plan.run_id
    assert plan.env == env.envdir / plan.run_id
    assert [lake["lake_id"] for lake in plan.lakes] == ["environment"]


@pytest.mark.requires_sandbox
def test_construct_writes_ontology_logs_and_summary(env):
    assert main(env.argv("construct", run_id="r1")) == 0
    lake = env.results / "construction/r1/environment"
    assert (lake / "ontology").is_dir() and (lake / "logs").is_dir()
    summary = json.loads((lake / "summary.json").read_text())
    assert summary["lake_id"] == "environment" and summary["datasets"] == 1 and summary["entity_types"] == 1
    assert summary["stop_reason"] == "converged"
    run = json.loads((env.results / "construction/r1/run.json").read_text())
    assert run["lakes"] == ["environment"] and "api_key" not in json.dumps(run) and "test-key" not in json.dumps(run)
    assert run["stage"] == "construct" and run["benchmark"] == "Kramabench" and run["run_id"] == "r1"
    assert run["models"] == {role: "openai/scripted-model" for role in ("proposer", "verifier", "experience")}
    assert run["construction"] == {"max_passes": 1, "agent": {"max_steps": 3}}
    assert (env.envdir / "r1/construction/environment").is_dir()


@pytest.mark.requires_sandbox
def test_construct_resume_skips_finished_lakes(env):
    assert main(env.argv("construct", run_id="r1")) == 0
    summary = (env.results / "construction/r1/environment/summary.json").read_text()
    clients = len(env.clients)
    assert main(env.argv("construct", run_id="r1", extra=("--resume",))) == 0
    assert len(env.clients) == clients  # No Lake was rebuilt, so no model was called.
    assert (env.results / "construction/r1/environment/summary.json").read_text() == summary


@pytest.mark.requires_sandbox
@pytest.mark.parametrize("stalled", ["verification_incomplete", "proposer_incomplete"])
def test_construct_resume_continues_a_stalled_lake(env, stalled):
    assert main(env.argv("construct", run_id="r1")) == 0
    summary_path = env.results / "construction/r1/environment/summary.json"
    summary_path.write_text(json.dumps({**json.loads(summary_path.read_text()), "stop_reason": stalled}))
    clients = len(env.clients)
    assert main(env.argv("construct", run_id="r1", extra=("--resume",))) == 0
    assert len(env.clients) == clients + 1  # The Lake's construction ran again.
    assert json.loads(summary_path.read_text())["stop_reason"] == "converged"


def test_a_failing_lake_is_recorded_and_the_stage_fails(env, monkeypatch):
    import benchmark

    def broken(config, record_path):
        raise RuntimeError("endpoint down")

    monkeypatch.setattr(benchmark, "make_client", broken)
    assert main(env.argv("construct", run_id="r1")) == 1
    summary = json.loads((env.results / "construction/r1/environment/summary.json").read_text())
    assert summary == {"lake_id": "environment", "status": "error", "error": "RuntimeError: endpoint down"}


@pytest.mark.requires_sandbox
def test_baseline_infer_writes_traces_and_predictions(env):
    assert main(env.argv("infer", method="baseline", run_id="b1")) == 0
    run = env.results / "inference/b1"
    assert (run / "tasks/t1/result.json").exists() and (run / "tasks/t1/logs/baseline/steps.json").exists()
    rows = [json.loads(line) for line in (run / "predictions.jsonl").read_text().splitlines()]
    assert [r["task_id"] for r in rows] == ["t1", "t2"] and rows[0]["pipeline_code"]
    assert [r["answer"] for r in rows] == [2, [1, 2]] and {r["status"] for r in rows} == {"answered"}
    info = json.loads((run / "run.json").read_text())
    assert info["method"] == "baseline" and info["tasks"] == ["t1", "t2"] and info["model"] == "openai/scripted-model"
    assert "test-key" not in json.dumps(info)
    assert (env.envdir / "b1/inference/tasks/t1").is_dir()


def test_ogda_infer_requires_a_finished_ontology_run(env, capsys):
    assert main(env.argv("infer", method="ogda", run_id="d1")) == 2 and "--ontology-run" in capsys.readouterr().err


@pytest.mark.requires_sandbox
def test_ogda_infer_grounds_the_ontology(env):
    env.build_ontology("r1")
    assert main(env.argv("infer", method="ogda", run_id="d1", extra=("--ontology-run", "r1"))) == 0
    result = json.loads((env.results / "inference/d1/tasks/t1/result.json").read_text())
    assert result["status"] == "answered" and result["answer"] == 2 and "searches" not in result
    assert result["graph"]["answer"] == {"typed": False, "from": ["a"]}
    info = json.loads((env.results / "inference/d1/run.json").read_text())
    assert info["ontology_run"] == "r1" and info["method"] == "ogda"


@pytest.mark.requires_sandbox
def test_one_failing_task_does_not_stop_the_run(env, monkeypatch):
    import benchmark
    from ogda.llm import LLMError

    def client(config, record_path):
        if "t1" in record_path.parts:
            raise LLMError("endpoint down")
        return env.make_client(config, record_path)

    monkeypatch.setattr(benchmark, "make_client", client)
    assert main(env.argv("infer", method="baseline", run_id="b2")) == 1
    assert json.loads((env.results / "inference/b2/tasks/t1/result.json").read_text())["status"] == "error"
    assert json.loads((env.results / "inference/b2/tasks/t2/result.json").read_text())["status"] == "answered"
    rows = [json.loads(line) for line in (env.results / "inference/b2/predictions.jsonl").read_text().splitlines()]
    assert [(r["task_id"], r["status"], r["answer"]) for r in rows] == [("t1", "error", None), ("t2", "answered", [1, 2])]


@pytest.mark.requires_sandbox
def test_infer_resume_skips_answered_tasks(env):
    assert main(env.argv("infer", method="baseline", run_id="b1")) == 0
    run = env.results / "inference/b1"
    t1 = (run / "tasks/t1/result.json").read_text()
    (run / "tasks/t2/result.json").write_text(json.dumps({"task": {}, "status": "error", "error": "LLMError: down"}))
    clients = len(env.clients)
    assert main(env.argv("infer", method="baseline", run_id="b1", extra=("--resume",))) == 0
    assert len(env.clients) == clients + 1  # Only t2, whose run failed, ran again.
    assert json.loads((run / "tasks/t2/result.json").read_text())["status"] == "answered"
    assert (run / "tasks/t1/result.json").read_text() == t1
    assert [json.loads(line)["task_id"] for line in (run / "predictions.jsonl").read_text().splitlines()] == ["t1", "t2"]


@pytest.mark.requires_sandbox
def test_infer_resume_of_some_tasks_keeps_every_prediction(env):
    assert main(env.argv("infer", method="baseline", run_id="b1")) == 0
    run = env.results / "inference/b1"
    assert main(env.argv("infer", method="baseline", run_id="b1", extra=("--resume", "--tasks", "t2"))) == 0
    assert [json.loads(line)["task_id"] for line in (run / "predictions.jsonl").read_text().splitlines()] == ["t1", "t2"]
    assert json.loads((run / "run.json").read_text())["tasks"] == ["t1", "t2"]


# The evaluate stage: official scorers run in subprocesses, stubbed here through run_scorer.

def judge():
    from benchmarks import Judge

    return Judge(model="gpt-5-mini", api_key="k", base_url="http://judge.test/v1")


def fake_kramabench_scorer(calls: list):
    def run(script, job, judge, cwd, log=None):
        calls.append(job)
        if job["mode"] == "aggregate":
            return {"aggregates": [{"workload": "overall.json", "metric": "llm_code_eval", "value_mean": 1.0}]}
        return {"native": [{"task_id": job["task_id"], "success": 1.0, "f1": 1.0, "llm_code_eval": [True]}]}
    return run


def test_kramabench_evaluate_uses_primary_metric(monkeypatch, fake_kramabench, tmp_path):
    from benchmarks import kramabench

    monkeypatch.setattr(kramabench, "run_scorer", lambda script, job, judge, cwd, log=None: {
        "native": [{"task_id": "t1", "success": 1.0, "llm_code_eval": [True]}]})
    score = kramabench.evaluate({"task_id": "t1", "status": "answered", "answer": 3, "pipeline_code": "x"},
                                kramabench.tasks(fake_kramabench)[0], judge(), fake_kramabench, fake_kramabench,
                                tmp_path)
    assert score["status"] == "scored" and score["score"] == 1.0 and score["metric"] == "success"
    assert score["metrics"] == {"success": 1.0, "llm_code_eval": [True]}


def test_incomplete_prediction_is_generation_failed_without_judge(monkeypatch, fake_discoverybench, tmp_path):
    from benchmarks import discoverybench

    monkeypatch.setattr(discoverybench, "run_scorer", lambda *a, **k: pytest.fail("judge called"))
    score = discoverybench.evaluate(
        {"task_id": "toy__metadata_0__q0", "status": "incomplete", "hypothesis": "", "workflow": ""},
        discoverybench.tasks(fake_discoverybench)[0], judge(), fake_discoverybench, fake_discoverybench, tmp_path)
    assert score == {"task_id": "toy__metadata_0__q0", "status": "generation_failed", "score": 0.0}


def test_evaluate_writes_a_model_named_folder(env, monkeypatch):
    from benchmarks import kramabench

    monkeypatch.setattr(kramabench, "run_scorer", fake_kramabench_scorer([]))
    env.infer_run("b1")
    assert main(env.argv("evaluate", run_id="b1")) == 0
    [folder] = (env.results / "inference/b1/evaluation").iterdir()
    assert folder.name.startswith("gpt-5-mini_")
    assert {p.name for p in folder.iterdir()} >= {"run.json", "scores.jsonl", "summary.json"}
    run = json.loads((folder / "run.json").read_text())
    assert run["judge_model"] == "gpt-5-mini" and run["infer_run"] == "b1" and "test-key" not in json.dumps(run)
    summary = json.loads((folder / "summary.json").read_text())
    assert summary["judge_model"] == "gpt-5-mini" and summary["score"] == 1.0 and summary["scored"] == 2


def test_use_judge_replaces_the_hardcoded_model(monkeypatch):
    from scorers.kramabench_score import use_judge

    monkeypatch.setenv("OPENAI_API_KEY", "k")

    class Fake:
        def __init__(self, model):
            self.model = model

    use_judge(Fake, "my-judge")
    assert Fake(model="gpt-5-mini").model == "my-judge"


def test_evaluate_stops_when_the_judge_rejects_json_mode(env, monkeypatch, capsys):
    import benchmark
    from benchmark import PreflightError

    def reject(judge):
        raise PreflightError("JSON-mode call")

    monkeypatch.setattr(benchmark, "check_judge", reject)
    env.infer_run("b1")
    assert main(env.argv("evaluate", run_id="b1")) == 2 and "JSON-mode" in capsys.readouterr().err


def test_evaluate_without_predictions_says_to_resume(env, capsys):
    env.infer_run("interrupted", predictions=False)
    assert main(env.argv("evaluate", run_id="interrupted")) == 2
    assert "rerun infer with --resume" in capsys.readouterr().err


def test_evaluate_resume_reuses_the_newest_folder_and_skips_scored(env, monkeypatch):
    from benchmarks import kramabench

    calls = []
    monkeypatch.setattr(kramabench, "run_scorer", fake_kramabench_scorer(calls))
    run = env.infer_run("b1")
    (run / "evaluation/gpt-5-mini_20000101T000000Z").mkdir(parents=True)
    assert main(env.argv("evaluate", run_id="b1")) == 0
    newest = max((run / "evaluation").iterdir())
    scores = newest / "scores.jsonl"
    scores.write_text(next(line for line in scores.read_text().splitlines(True) if '"t1"' in line))
    calls.clear()
    assert main(env.argv("evaluate", run_id="b1", extra=("--resume",))) == 0
    assert len(list((run / "evaluation").iterdir())) == 2
    assert [job.get("task_id") for job in calls if job["mode"] == "score"] == ["t2"]
    assert sorted(json.loads(line)["task_id"] for line in scores.read_text().splitlines()) == ["t1", "t2"]


def test_evaluate_resume_rescores_evaluator_failed_rows(env, monkeypatch):
    from benchmarks import kramabench

    calls = []
    monkeypatch.setattr(kramabench, "run_scorer", fake_kramabench_scorer(calls))
    run = env.infer_run("b1")
    folder = run / "evaluation/gpt-5-mini_20000101T000000Z"
    folder.mkdir(parents=True)
    rows = [{"task_id": "t1", "workload": "environment", "status": "scored", "score": 1.0, "native_evaluation": {}},
            {"task_id": "t2", "status": "evaluator_failed", "score": 0.0, "error": "RuntimeError: judge down"}]
    (folder / "scores.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert main(env.argv("evaluate", run_id="b1", extra=("--resume",))) == 0
    assert [job.get("task_id") for job in calls if job["mode"] == "score"] == ["t2"]
    last = [json.loads(line) for line in (folder / "scores.jsonl").read_text().splitlines()][-1]
    assert last["task_id"] == "t2" and last["status"] == "scored"
    summary = json.loads((folder / "summary.json").read_text())
    assert (summary["scored"], summary["evaluator_failed"]) == (2, 0)


def test_one_worker_runs_the_items_in_this_process():
    import benchmark
    from helpers import process_of

    items = [{"lake_id": "a"}, {"lake_id": "b"}]
    assert benchmark.run_each(process_of, items, 1) == [(item, os.getpid()) for item in items]


def test_each_item_runs_in_its_own_process():
    import benchmark
    from helpers import process_of

    items = [{"lake_id": name} for name in "abc"]
    results = benchmark.run_each(process_of, items, 2)
    assert [item for item, _ in results] == items
    processes = [process for _, process in results]
    assert len(set(processes)) == 3 and os.getpid() not in processes


def test_an_item_whose_process_dies_counts_as_failed(capsys):
    import benchmark
    from helpers import die_on_b

    results = benchmark.run_each(die_on_b, [{"lake_id": name} for name in "abc"], 2)
    assert len(results) == 3 and results[1] is False
    assert "b: its worker process stopped" in capsys.readouterr().err


@pytest.mark.requires_sandbox
def test_construct_and_infer_send_their_lakes_and_tasks_to_the_workers(env, monkeypatch):
    import benchmark

    real, calls = benchmark.run_each, []

    def spy(work, items, workers):
        pickle.dumps(work)  # A worker process receives the work by pickling it.
        calls.append(([item.get("task_id") or item["lake_id"] for item in items], workers))
        return real(work, items, 1)

    monkeypatch.setattr(benchmark, "run_each", spy)
    assert main(env.argv("construct", run_id="r1", extra=("--workers", "3"))) == 0
    assert main(env.argv("infer", method="ogda", run_id="d1", extra=("--ontology-run", "r1", "--workers", "3"))) == 0
    assert calls == [(["environment"], 3), (["t1", "t2"], 3)]
