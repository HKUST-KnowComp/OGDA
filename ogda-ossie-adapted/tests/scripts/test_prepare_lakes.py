"""Tests for scripts/prepare_lakes.py: one clean Lake folder per Lake, plus manifest."""

from __future__ import annotations

import json

import pytest

from prepare_lakes import prepare, verify


def test_prepare_writes_one_clean_folder_per_lake(fake_discoverybench, tmp_path):
    manifest = prepare("discoverybench", fake_discoverybench, tmp_path / "lake")
    meta = json.loads((tmp_path / "lake/toy/metadata_0.json").read_text())
    assert set(meta) >= {"datasets", "workflow"} and not {"queries", "hypotheses", "intermediate"} & set(meta)
    assert sorted(p.name for p in (tmp_path / "lake").iterdir()) == ["manifest.json", "toy"]
    assert set(manifest["lakes"]["toy"]["clean_sha256"]) == {"data.csv", "metadata_0.json"}


def test_prepare_keeps_existing_folder_without_force(fake_discoverybench, tmp_path):
    prepare("discoverybench", fake_discoverybench, tmp_path / "lake")
    with pytest.raises(FileExistsError):
        prepare("discoverybench", fake_discoverybench, tmp_path / "lake")
    prepare("discoverybench", fake_discoverybench, tmp_path / "lake", force=True)


def test_verify_rejects_a_changed_lake(fake_discoverybench, tmp_path):
    prepare("discoverybench", fake_discoverybench, tmp_path / "lake")
    verify(tmp_path / "lake", ["toy"])
    (tmp_path / "lake/toy/data.csv").write_text("changed")
    with pytest.raises(ValueError, match="toy.*prepare_lakes.py"):
        verify(tmp_path / "lake", ["toy"])
    with pytest.raises(ValueError, match="missing"):
        verify(tmp_path / "lake", ["absent"])
