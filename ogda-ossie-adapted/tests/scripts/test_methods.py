"""Tests for scripts/methods.py: the Data agent method, the baseline method and pipeline code extraction."""

from __future__ import annotations

import json

import pytest
from helpers import make_lake
from methods import pipeline_code, run_baseline
from ogda.agent import AgentConfig
from ogda.data_agent import DataAgentConfig
from ogda.llm import ScriptedClient

TASK = {"id": "t1", "question": "How many households?", "answer_format": "One integer."}


def code(text: str) -> str:
    return f"<code>\n{text}\n</code>"


@pytest.fixture
def clean_lake(tmp_path):
    return make_lake(tmp_path)


def test_pipeline_code_joins_code_actions(tmp_path):
    (tmp_path / "logs/baseline").mkdir(parents=True)
    (tmp_path / "logs/baseline/steps.json").write_text(
        json.dumps([{"code_action": "a = 1"}, {"other": 1}, {"code_action": "b = 2"}]))
    assert pipeline_code(tmp_path / "logs") == "a = 1\n\nb = 2"


def test_pipeline_code_with_no_steps_files_is_empty(tmp_path):
    (tmp_path / "logs").mkdir()
    assert pipeline_code(tmp_path / "logs") == ""


@pytest.mark.requires_sandbox
def test_baseline_answers_with_no_tools(tmp_path, clean_lake):
    llm = ScriptedClient(lambda role, s, p: code(
        "import pandas as pd\nn = len(pd.read_csv(LAKE_ROOT / 'income.csv'))\nfinal_answer(n)"))
    result = run_baseline(TASK, clean_lake, llm, DataAgentConfig(), tmp_path / "env", tmp_path / "out",
                          check=lambda v: None)
    assert result["status"] == "answered" and result["answer"] == 2
    assert "ground(" not in (tmp_path / "out/logs/baseline/steps.json").read_text()
    assert json.loads((tmp_path / "out/result.json").read_text())["answer"] == 2
    assert result["stop_reason"] == "answered" and result["task"] == TASK
    assert isinstance(result["usage"], dict) and isinstance(result["wall_seconds"], float)


@pytest.mark.requires_sandbox
def test_baseline_rejects_wrong_shape_then_accepts(tmp_path, clean_lake):
    responses = [code("final_answer('two')"), code("final_answer(2)")]

    def respond(role, system, prompt):
        return responses.pop(0) if len(responses) > 1 else responses[0]

    def integer(value):
        if not isinstance(value, int):
            raise ValueError("answer must be a JSON integer")

    llm = ScriptedClient(respond)
    result = run_baseline(TASK, clean_lake, llm, DataAgentConfig(), tmp_path / "env", tmp_path / "out", check=integer)
    assert result["status"] == "answered" and result["answer"] == 2


@pytest.mark.requires_sandbox
def test_baseline_saves_a_json_serialized_string_answer_unwrapped(tmp_path, clean_lake):
    llm = ScriptedClient(lambda role, s, p: code("import json\nfinal_answer(json.dumps('two'))"))
    result = run_baseline(TASK, clean_lake, llm, DataAgentConfig(), tmp_path / "env", tmp_path / "out",
                          check=lambda value: None)
    assert result["answer"] == "two" and result["steps"] == 1
    assert json.loads((tmp_path / "out/result.json").read_text())["answer"] == "two"


@pytest.mark.requires_sandbox
def test_baseline_without_answer_is_incomplete(tmp_path, clean_lake):
    llm = ScriptedClient(lambda role, s, p: code("print('still working')"))
    config = DataAgentConfig(agent=AgentConfig(max_steps=2))
    result = run_baseline(TASK, clean_lake, llm, config, tmp_path / "env", tmp_path / "out", check=lambda v: None)
    assert result["status"] == "incomplete" and result["answer"] is None
    assert result["stop_reason"] == "incomplete" and result["steps"] == 2
