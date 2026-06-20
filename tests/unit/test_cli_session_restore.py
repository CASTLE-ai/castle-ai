"""CLI clustering runs must be UI-restorable (project decision 2026-06-20).

A ``castle cluster run`` should leave behind exactly what the Behavior
Microscope UI leaves behind: a SessionManager session+manifest (#4) and a
``node_{parent}_meta.json`` sidecar whose parent name matches the clustered
node and whose payload shape matches the UI's (#5). These tests exercise the
new ClusteringSession helpers in isolation (no GPU/latents needed).
"""

import json
import os

from castle.service.clustering_service import ClusteringSession
from castle.service.session_manager import SessionManager


class _Aggregator:
    """Minimal stand-in for LatentAggregator (only ``latents`` is read)."""
    def __init__(self, n):
        self.latents = list(range(n))


def _make_session(tmp_path, *, n_frames=120, prepare_id=None, k_prime=None):
    """Build a ClusteringSession without running the heavy __init__."""
    sess = object.__new__(ClusteringSession)
    sess.storage_path = str(tmp_path)
    sess.project_name = "proj"
    sess.roi = 1
    sess.bin_size = 1
    sess.model = "dinov3_vitb16"
    sess._prepare_id = prepare_id
    sess._k_prime = k_prime
    sess.aggregator = _Aggregator(n_frames)
    sess._current_cluster_name = "init"
    sess._last_umap_config = [{"n_neighbors": 100, "min_dist": 0.0}]
    sess._last_umap_seeds = [12345]
    sess._last_eps = 1.5
    # latents.cluster_meta drives the n_clusters count on snapshot
    sess.latents = type("L", (), {"cluster_meta": {
        0: {"name": "init"}, 1: {"name": "init_a0"}, 2: {"name": "init_a1"},
    }})()
    os.makedirs(os.path.join(tmp_path, "proj", "cluster"), exist_ok=True)
    return sess


def test_start_new_session_writes_manifest_with_provenance(tmp_path):
    sess = _make_session(tmp_path, n_frames=90, prepare_id="prep_x", k_prime=8)
    sid = sess.start_new_session(variance_pct=95.0)

    mgr = SessionManager(str(tmp_path), "proj")
    assert mgr.get_active_session_id() == sid
    info = mgr.get_session(sid)
    assert info is not None
    assert info.model == "dinov3_vitb16"
    assert info.total_frames == 90
    assert info.prepare_id == "prep_x"
    assert info.k_prime == 8
    assert info.variance_pct == 95.0


def test_node_meta_sidecar_matches_ui_shape(tmp_path):
    sess = _make_session(tmp_path)
    cluster_path = os.path.join(tmp_path, "proj", "cluster")
    sess._write_node_meta(cluster_path, os.path.join(cluster_path, "cluster_init_a0_.npz"))

    meta_path = os.path.join(cluster_path, "node_init_meta.json")
    assert os.path.exists(meta_path), "sidecar named after the clustered node"
    meta = json.loads(open(meta_path).read())
    # Exact key set the UI submit path writes — restore reads these.
    assert set(meta) == {
        "parent_cluster_name", "umap_config", "eps",
        "min_samples", "preset", "umap_seed", "embedding_npz",
    }
    assert meta["parent_cluster_name"] == "init"
    assert meta["eps"] == 1.5
    assert meta["umap_seed"] == 12345
    assert meta["embedding_npz"] == "cluster_init_a0_.npz"
    assert json.loads(meta["umap_config"]) == [{"n_neighbors": 100, "min_dist": 0.0}]


def test_snapshot_copies_cluster_artifacts_into_session(tmp_path):
    sess = _make_session(tmp_path)
    sid = sess.start_new_session()
    cluster_path = os.path.join(tmp_path, "proj", "cluster")
    # Lay down the artifacts a real submit would produce.
    for fn in ("id.csv", "cluster_init_a0_.npz", "time_series_v1.csv",
               "node_init_meta.json"):
        with open(os.path.join(cluster_path, fn), "w") as f:
            f.write("x")

    used = sess._snapshot_to_session()
    assert used == sid

    sess_dir = SessionManager(str(tmp_path), "proj").get_session_dir(sid)
    for fn in ("id.csv", "cluster_init_a0_.npz", "time_series_v1.csv",
               "node_init_meta.json"):
        assert os.path.exists(os.path.join(sess_dir, fn)), f"{fn} not snapshotted"
    # n_clusters counts non-'init' named clusters (2 here).
    assert SessionManager(str(tmp_path), "proj").get_session(sid).n_clusters == 2


def test_snapshot_auto_creates_session_when_none_active(tmp_path):
    # submit() called via the service API without start_new_session() must still
    # produce a restorable session.
    sess = _make_session(tmp_path)
    mgr = SessionManager(str(tmp_path), "proj")
    assert mgr.get_active_session_id() is None
    used = sess._snapshot_to_session()
    assert used is not None
    assert mgr.get_active_session_id() == used
