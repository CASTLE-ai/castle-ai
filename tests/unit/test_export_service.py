"""Unit tests for castle.service.export_service.build_run_manifest."""

import json
import os

from castle.service.export_service import build_run_manifest


def _make_project(tmp_path):
    pp = tmp_path / "proj"
    (pp / "cluster" / "sessions" / "session_001").mkdir(parents=True)
    (pp / "config.json").write_text(json.dumps({
        "source": ["b.mp4", "a.mp4"],
        "latent": {"a/x": "a.mp4", "a/y": "a.mp4"},
    }))
    (pp / "cluster" / "sessions" / "session_001" / "manifest.json").write_text(
        json.dumps({"session_id": "session_001", "n_clusters": 5})
    )
    return str(pp)


def test_build_run_manifest_core_fields(tmp_path):
    pp = _make_project(tmp_path)
    m = build_run_manifest(
        pp, project_name="proj", components=["latent", "cluster"],
        generated_at="20260619_120000",
    )
    assert m["manifest_schema_version"] == 1
    assert m["project_name"] == "proj"
    assert m["generated_at"] == "20260619_120000"
    # components are normalised (sorted)
    assert m["components"] == ["cluster", "latent"]
    # environment provenance present + JSON-serialisable
    assert "environment" in m and "packages" in m["environment"]
    json.dumps(m)


def test_build_run_manifest_reads_project_and_session(tmp_path):
    pp = _make_project(tmp_path)
    m = build_run_manifest(pp, project_name="proj", session_id="session_001")
    assert m["project"]["videos"] == ["a.mp4", "b.mp4"]  # sorted
    assert m["project"]["latent_count"] == 2
    assert m["session"]["session_id"] == "session_001"
    assert m["session"]["n_clusters"] == 5


def test_build_run_manifest_missing_inputs_are_best_effort(tmp_path):
    # No config.json, no session → still returns a valid manifest, no raise.
    m = build_run_manifest(str(tmp_path / "nope"), project_name="ghost")
    assert m["project_name"] == "ghost"
    assert "project" not in m and "session" not in m
    assert "environment" in m
