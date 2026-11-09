"""Tests for the DiscoveryBench module (scripts/benchmarks/discoverybench.py)."""

from __future__ import annotations

import pytest

from benchmarks import discoverybench


def test_lakes_strip_answer_keys(fake_discoverybench):
    assert discoverybench.lakes(fake_discoverybench) == [
        {
            "lake_id": "toy",
            "source": fake_discoverybench / "discoverybench/real/test/toy",
            "exclude": [],
            "strip_keys": ["queries", "hypotheses", "intermediate"],
        }
    ]


def test_tasks_flatten_nested_queries(fake_discoverybench):
    assert discoverybench.tasks(fake_discoverybench) == [
        {
            "task_id": "toy__metadata_0__q0",
            "lake_id": "toy",
            "question": "Is y rising?",
            "dataset": "toy",
            "metadata_id": "0",
            "query_id": "0",
        }
    ]


def test_check_needs_hypothesis_and_workflow():
    with pytest.raises(ValueError, match="hypothesis"):
        discoverybench.check({"workflow": "w"}, {})
    discoverybench.check({"hypothesis": "h", "workflow": "w"}, {})


def test_gold_reads_answer_key_and_metadata(fake_discoverybench):
    task = discoverybench.tasks(fake_discoverybench)[0]
    gold = discoverybench.gold(task, fake_discoverybench)
    assert gold["gold_hypo"] == "y rises" and gold["dataset_meta"]["queries"]
