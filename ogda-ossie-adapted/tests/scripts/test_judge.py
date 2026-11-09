"""Tests for the shared Judge and run_scorer helpers in scripts/benchmarks."""

from __future__ import annotations

import pytest
import yaml

from benchmarks import Judge, judge_from_config, run_scorer


def test_judge_uses_the_evaluator_role(tmp_path):
    config = tmp_path / "c.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "llm": {
                    "model": "openai/main",
                    "base_url": "https://a/v1",
                    "api_key": "k1",
                    "roles": {
                        "evaluator": {
                            "model": "openai/deepseek/deepseek-chat",
                            "base_url": "https://judge/v1",
                            "api_key": "k2",
                        }
                    },
                }
            }
        )
    )
    judge = judge_from_config(config)
    assert (judge.model, judge.base_url, judge.api_key) == (
        "deepseek/deepseek-chat",
        "https://judge/v1",
        "k2",
    )
    assert "k2" not in repr(judge)


def test_judge_needs_the_openai_prefix(tmp_path):
    config = tmp_path / "c.yaml"
    config.write_text(
        yaml.safe_dump(
            {"llm": {"model": "anthropic/claude", "base_url": "https://a/v1", "api_key": "k"}}
        )
    )
    with pytest.raises(ValueError, match="openai/"):
        judge_from_config(config)


def test_run_scorer_sends_the_judge_to_the_subprocess(tmp_path):
    script = tmp_path / "s.py"
    script.write_text(
        "import json, os, sys\n"
        "job = json.load(sys.stdin)\n"
        "print('noise')\n"
        "print(json.dumps({'job': job, 'model': os.environ['JUDGE_MODEL'], "
        "'url': os.environ['OPENAI_BASE_URL']}))\n"
        "print('judging', file=sys.stderr)"
    )
    log = tmp_path / "logs" / "t1.log"
    out = run_scorer(script, {"x": 1}, Judge("m", "k", "https://judge/v1"), tmp_path, log=log)
    assert out == {"job": {"x": 1}, "model": "m", "url": "https://judge/v1"}
    assert log.read_text() == "judging\n"
