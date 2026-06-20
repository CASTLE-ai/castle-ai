"""Unit tests for castle.service.annotator_loader."""

import json
import os
import shutil

import numpy as np
import pytest

from castle.service.annotator_loader import (
    AnnotatorData,
    get_annotator_frame,
    load_annotator_data,
)


# ---------------------------
# Fixtures
# ---------------------------

def _make_project(tmp_path, n_bins=20, bin_size=5):
    """Create a minimal on-disk project structure for testing."""
    project_name = "test_project"
    project_path = tmp_path / project_name
    cluster_path = project_path / "cluster"
    source_path = project_path / "sources"
    cluster_path.mkdir(parents=True)
    source_path.mkdir(parents=True)

    # config.json
    config = {
        "project_name": project_name,
        "source": ["vid_a.mp4"],
        "latent": {
            "vid_a_ROI_1_model.npz": "vid_a.mp4",
        },
    }
    (project_path / "config.json").write_text(json.dumps(config))

    # id.csv
    id_csv_content = "Id,Name,Color\n0,init,grey\n1,cluster_a,#FF0000\n2,cluster_b,#00FF00\n"
    (cluster_path / "id.csv").write_text(id_csv_content)

    # cluster_.npz
    cls = np.array([0] * 5 + [1] * 8 + [2] * 7, dtype=np.int16)
    emb = np.random.rand(n_bins, 2).astype(np.float64)
    np.savez(str(cluster_path / "cluster_.npz"), cls=cls, emb=emb)

    # Session manifest
    sessions_path = cluster_path / "sessions"
    sessions_path.mkdir()
    session_dir = sessions_path / "session_001"
    session_dir.mkdir()
    manifest = {
        "session_id": "session_001",
        "name": "Test Session",
        "created_at": "2026-01-01T00:00:00",
        "updated_at": "2026-01-01T00:00:00",
        "model": "test_model",
        "roi_id": 1,
        "bin_size": bin_size,
        "n_clusters": 3,
        "total_frames": n_bins * bin_size,
        "status": "in_progress",
        "description": "",
    }
    (session_dir / "manifest.json").write_text(json.dumps(manifest))
    # Copy cluster files into session
    shutil.copyfile(str(cluster_path / "id.csv"), str(session_dir / "id.csv"))
    shutil.copyfile(str(cluster_path / "cluster_.npz"), str(session_dir / "cluster_.npz"))

    # Mark session as active
    (sessions_path / "_active.txt").write_text("session_001")

    return str(tmp_path), project_name


# ---------------------------
# Tests: load_annotator_data
# ---------------------------

def test_load_annotator_data_basic(tmp_path):
    """load_annotator_data returns AnnotatorData with correct fields."""
    storage_path, project_name = _make_project(tmp_path, n_bins=20, bin_size=5)

    data = load_annotator_data(storage_path, project_name)

    assert isinstance(data, AnnotatorData)
    assert data.cluster.shape == (20,)
    assert data.embedding.shape == (20, 2)
    assert isinstance(data.cluster_meta, dict)


def test_load_annotator_data_cluster_meta(tmp_path):
    """cluster_meta correctly maps IDs to name/color dicts."""
    storage_path, project_name = _make_project(tmp_path)

    data = load_annotator_data(storage_path, project_name)

    assert 0 in data.cluster_meta
    assert data.cluster_meta[0]["name"] == "init"
    assert data.cluster_meta[0]["color"] == "grey"
    assert 1 in data.cluster_meta
    assert data.cluster_meta[1]["name"] == "cluster_a"
    assert data.cluster_meta[2]["name"] == "cluster_b"


def test_load_annotator_data_bin_size_from_session(tmp_path):
    """bin_size is read from the active session manifest."""
    storage_path, project_name = _make_project(tmp_path, bin_size=7)

    data = load_annotator_data(storage_path, project_name)

    assert data.bin_size == 7


def test_load_annotator_data_missing_id_csv(tmp_path):
    """FileNotFoundError raised when id.csv is absent."""
    storage_path, project_name = _make_project(tmp_path)
    id_csv = os.path.join(storage_path, project_name, "cluster", "id.csv")
    os.remove(id_csv)

    with pytest.raises(FileNotFoundError, match="id.csv"):
        load_annotator_data(storage_path, project_name)


def test_load_annotator_data_missing_npz(tmp_path):
    """FileNotFoundError raised when cluster_.npz is absent."""
    storage_path, project_name = _make_project(tmp_path)
    npz_path = os.path.join(storage_path, project_name, "cluster", "cluster_.npz")
    os.remove(npz_path)

    with pytest.raises(FileNotFoundError, match=r"cluster_\*\.npz"):
        load_annotator_data(storage_path, project_name)


def test_load_annotator_data_cluster_array_dtype(tmp_path):
    """cluster array is cast to int32 regardless of on-disk dtype."""
    storage_path, project_name = _make_project(tmp_path)

    data = load_annotator_data(storage_path, project_name)

    assert data.cluster.dtype == np.int32


def test_load_annotator_data_project_paths(tmp_path):
    """project_path and source_path are set correctly."""
    storage_path, project_name = _make_project(tmp_path)

    data = load_annotator_data(storage_path, project_name)

    expected_project = os.path.join(storage_path, project_name)
    expected_source = os.path.join(expected_project, "sources")
    assert data.project_path == expected_project
    assert data.source_path == expected_source


def test_load_annotator_data_with_session_id(tmp_path):
    """Providing session_id activates that session before loading."""
    storage_path, project_name = _make_project(tmp_path, bin_size=3)

    data = load_annotator_data(storage_path, project_name, session_id="session_001")

    assert data.bin_size == 3
    assert isinstance(data.cluster_meta, dict)


# ---------------------------
# Tests: get_annotator_frame
# ---------------------------

def _make_annotator_data(n_bins=10, bin_size=4, videos_meta=None):
    """Build a minimal AnnotatorData without requiring real video files."""
    if videos_meta is None:
        videos_meta = [(n_bins, "fake_video.mp4")]
    cls = np.zeros(n_bins, dtype=np.int32)
    emb = np.zeros((n_bins, 2), dtype=np.float64)
    return AnnotatorData(
        cluster=cls,
        cluster_meta={0: {"name": "init", "color": "grey"}},
        embedding=emb,
        bin_size=bin_size,
        project_path="/fake/project",
        source_path="/fake/project/sources",
        videos_meta=videos_meta,
        fps=30.0,
    )


def test_get_annotator_frame_empty_videos_meta():
    """Returns None when videos_meta is empty."""
    data = _make_annotator_data()
    data.videos_meta = []

    result = get_annotator_frame(data, 0)

    assert result is None


def test_get_annotator_frame_out_of_range():
    """Returns None when bin_idx exceeds total bins."""
    data = _make_annotator_data(n_bins=5)

    result = get_annotator_frame(data, 100)

    assert result is None


def test_get_annotator_frame_returns_none_on_missing_video(tmp_path):
    """Returns None gracefully when the video file does not exist."""
    data = _make_annotator_data(n_bins=5, bin_size=2)
    data.source_path = str(tmp_path)
    data.videos_meta = [(5, "nonexistent_video.mp4")]

    result = get_annotator_frame(data, 2)

    assert result is None


def test_get_annotator_frame_bin_routing():
    """Correct video is selected for multi-video projects.

    We monkey-patch _get_cached_reader to capture which video was requested.
    """
    import castle.service.annotator_loader as module

    captured = {}

    def fake_reader(annotator_data, video_path):
        captured["video_path"] = video_path

        class _FakeReader:
            def get_frame(self, idx):
                return np.zeros((32, 32, 3), dtype=np.uint8)

        return _FakeReader()

    original = module._get_cached_reader
    module._get_cached_reader = fake_reader

    try:
        videos_meta = [(5, "vid_a.mp4"), (5, "vid_b.mp4")]
        data = _make_annotator_data(n_bins=10, bin_size=1, videos_meta=videos_meta)
        data.source_path = "/fake/sources"

        # bin_idx=7 → second video (7 - 5 = 2 in vid_b)
        result = get_annotator_frame(data, 7)

        assert result is not None
        assert "vid_b.mp4" in captured["video_path"]
    finally:
        module._get_cached_reader = original


def test_get_annotator_frame_frame_idx_calculation():
    """frame_idx = local_bin * bin_size + bin_size // 2."""
    import castle.service.annotator_loader as module

    captured = {}

    def fake_reader(annotator_data, video_path):
        class _FakeReader:
            def get_frame(self, idx):
                captured["frame_idx"] = idx
                return np.zeros((32, 32, 3), dtype=np.uint8)

        return _FakeReader()

    original = module._get_cached_reader
    module._get_cached_reader = fake_reader

    try:
        bin_size = 6
        data = _make_annotator_data(n_bins=10, bin_size=bin_size)
        data.source_path = "/fake/sources"

        # Global bin 3 → local bin 3, frame_idx = 3*6 + 3 = 21
        get_annotator_frame(data, 3)

        assert captured["frame_idx"] == 3 * bin_size + bin_size // 2
    finally:
        module._get_cached_reader = original
