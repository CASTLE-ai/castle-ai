"""Unit tests for castle.service.cluster_npz — the cluster_*.npz filename grammar.

This grammar encodes child cluster names into the embedding filename, so it is a
fragile data channel; these tests pin the parse/round-trip behaviour after the
module was extracted out of clustering_service.
"""

import json
import os

from castle.service import cluster_npz


def test_parent_match_and_child_extraction_roundtrip():
    # Two immediate children of parent "init" (depth 1 → 2 segments per child).
    fn = "cluster_init_a0_init_a1_.npz"
    assert cluster_npz._parent_from_cluster_filename(fn, "init") is True
    assert cluster_npz._extract_child_names_from_filename(fn, "init") == [
        "init_a0", "init_a1",
    ]


def test_parent_mismatch_and_model_file_rejected():
    assert cluster_npz._parent_from_cluster_filename("cluster_init_a0_.npz", "other") is False
    # model/data artefacts are not parent embeddings
    assert cluster_npz._parent_from_cluster_filename("cluster_model.npz", "init") is False
    # malformed → empty child list, no raise
    assert cluster_npz._extract_child_names_from_filename("not_a_cluster.npz", "init") == []


def test_find_latest_cluster_npz_skips_non_embedding(tmp_path):
    cdir = tmp_path
    # newest is a non-embedding artefact that MUST be skipped
    (cdir / "cluster_init_a0_.npz").write_bytes(b"x")
    (cdir / "cluster_model.npz").write_bytes(b"y")
    (cdir / "cluster_data.npz").write_bytes(b"z")
    latest = cluster_npz.find_latest_cluster_npz(str(cdir))
    assert latest is not None
    assert os.path.basename(latest) == "cluster_init_a0_.npz"


def test_load_node_meta_missing_and_present(tmp_path):
    assert cluster_npz.load_node_meta(str(tmp_path), "init") is None  # no sidecar
    (tmp_path / "node_init_meta.json").write_text(json.dumps({"eps": 0.5}))
    meta = cluster_npz.load_node_meta(str(tmp_path), "init")
    assert meta == {"eps": 0.5}


def test_reexport_identity_from_clustering_service():
    """Backward-compat: the old import path still resolves to the same objects."""
    from castle.service import clustering_service as cs
    assert cs.find_latest_cluster_npz is cluster_npz.find_latest_cluster_npz
    assert cs.load_node_meta is cluster_npz.load_node_meta
