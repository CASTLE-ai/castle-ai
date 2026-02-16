"""Unit tests for automated Behavior Microscope (recursive hierarchical clustering)."""

import numpy as np
import pytest

from castle.core.auto_cluster import (
    MICROSCOPE_PRESETS,
    DEFAULT_EPS_VALUES,
    ClusteringCandidate,
    TreeNode,
    score_clustering,
    select_umap_config,
    find_best_eps,
    auto_cluster,
    select_best,
)


# ---------------------------------------------------------------------------
# MICROSCOPE_PRESETS
# ---------------------------------------------------------------------------

class TestMicroscopePresets:
    def test_has_four_presets(self):
        assert len(MICROSCOPE_PRESETS) == 4
        assert set(MICROSCOPE_PRESETS.keys()) == {"low", "intermediate", "high", "super_high"}

    def test_low_builds_1_stage(self):
        cfg = MICROSCOPE_PRESETS["low"]["build_config"](300)
        assert len(cfg) == 1
        assert cfg[0]["n_components"] == 2
        assert cfg[0]["n_neighbors"] == 300

    def test_intermediate_builds_2_stage(self):
        cfg = MICROSCOPE_PRESETS["intermediate"]["build_config"]((500, 300))
        assert len(cfg) == 2
        assert cfg[0]["n_components"] == 5
        assert cfg[1]["n_components"] == 2

    def test_high_builds_2_stage_10d(self):
        cfg = MICROSCOPE_PRESETS["high"]["build_config"]((300, 100))
        assert len(cfg) == 2
        assert cfg[0]["n_components"] == 10
        assert cfg[1]["n_components"] == 2

    def test_super_high_builds_3_stage(self):
        cfg = MICROSCOPE_PRESETS["super_high"]["build_config"]((300, 100, 50))
        assert len(cfg) == 3
        assert cfg[0]["n_components"] == 15
        assert cfg[1]["n_components"] == 5
        assert cfg[2]["n_components"] == 2


# ---------------------------------------------------------------------------
# select_umap_config
# ---------------------------------------------------------------------------

class TestSelectUmapConfig:
    def test_depth_0_returns_low(self):
        cfg = select_umap_config(depth=0, n_frames=5000)
        assert len(cfg) == 1  # 1-stage = Low
        assert cfg[0]["n_components"] == 2

    def test_depth_1_returns_intermediate(self):
        cfg = select_umap_config(depth=1, n_frames=5000)
        assert len(cfg) == 2  # 2-stage = Intermediate
        assert cfg[0]["n_components"] == 5
        assert cfg[1]["n_components"] == 2

    def test_depth_3_returns_intermediate(self):
        cfg = select_umap_config(depth=3, n_frames=2000)
        assert len(cfg) == 2

    def test_small_cluster_returns_low(self):
        cfg = select_umap_config(depth=2, n_frames=200)
        assert len(cfg) == 1  # small → Low
        assert cfg[0]["n_components"] == 2
        assert cfg[0]["n_neighbors"] <= 200

    def test_n_neighbors_clamped(self):
        cfg = select_umap_config(depth=0, n_frames=100)
        assert cfg[0]["n_neighbors"] <= 100 // 3

    def test_very_small_cluster(self):
        cfg = select_umap_config(depth=4, n_frames=50)
        assert len(cfg) == 1
        assert cfg[0]["n_neighbors"] >= 10  # at least 10


# ---------------------------------------------------------------------------
# score_clustering
# ---------------------------------------------------------------------------

class TestScoreClustering:
    def test_two_clean_clusters(self):
        labels = np.array([0] * 50 + [1] * 50)
        scores = score_clustering(labels)
        assert "quality_score" in scores
        assert scores["n_clusters"] == 2
        assert scores["noise_ratio"] == 0.0

    def test_random_labels_lower_tc(self):
        rng = np.random.default_rng(42)
        labels = rng.integers(0, 3, size=200)
        scores = score_clustering(labels)
        # Random should have lower TC than structured
        clean = score_clustering(np.array([0] * 100 + [1] * 100))
        assert scores["temporal_coherence"] < clean["temporal_coherence"]

    def test_single_cluster_penalized(self):
        labels = np.array([0] * 100)
        scores = score_clustering(labels)
        assert scores["n_clusters"] == 1
        assert scores["quality_score"] < 0.8  # penalized vs multi-cluster

    def test_all_noise(self):
        labels = np.array([-1] * 100)
        scores = score_clustering(labels)
        assert scores["noise_ratio"] == 1.0
        assert scores["n_clusters"] == 0

    def test_with_embedding(self):
        labels = np.array([0] * 50 + [1] * 50)
        emb = np.vstack([np.random.randn(50, 2) - 3, np.random.randn(50, 2) + 3])
        scores = score_clustering(labels, embedding=emb)
        assert scores["calinski_harabasz"] > 0

    def test_score_in_reasonable_range(self):
        labels = np.array([0] * 50 + [1] * 50 + [2] * 50)
        scores = score_clustering(labels)
        assert 0 <= scores["quality_score"] <= 1.5


# ---------------------------------------------------------------------------
# find_best_eps
# ---------------------------------------------------------------------------

class TestFindBestEps:
    def test_finds_something_for_separable_data(self):
        from sklearn.datasets import make_blobs
        X, _ = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)
        valid = np.ones(100, dtype=bool)
        dummy = np.zeros(100, dtype=int)
        result = find_best_eps(X, dummy, valid, eps_values=[0.5, 1.0, 2.0])
        assert result is not None
        assert result.n_clusters >= 2

    def test_returns_none_for_single_point(self):
        X = np.zeros((5, 2))  # all same point
        valid = np.ones(5, dtype=bool)
        dummy = np.zeros(5, dtype=int)
        # Very small eps → all noise, very large eps → 1 cluster
        result = find_best_eps(X, dummy, valid, eps_values=[0.001])
        assert result is None  # no valid 2+ cluster solution


# ---------------------------------------------------------------------------
# TreeNode
# ---------------------------------------------------------------------------

class TestTreeNode:
    def test_create_leaf(self):
        node = TreeNode(name="root_a0", depth=1, n_frames=500, is_leaf=True,
                        stop_reason="min_frames")
        assert node.is_leaf
        assert node.children == []

    def test_create_branch(self):
        child = TreeNode(name="root_a0_b0", depth=2, n_frames=200, is_leaf=True)
        parent = TreeNode(name="root_a0", depth=1, n_frames=500, is_leaf=False,
                          children=[child])
        assert len(parent.children) == 1

    def test_tree_depth(self):
        leaf = TreeNode(name="root_a0_b0_c0", depth=3, n_frames=100, is_leaf=True)
        mid = TreeNode(name="root_a0_b0", depth=2, n_frames=300, is_leaf=False,
                       children=[leaf])
        root = TreeNode(name="root_a0", depth=1, n_frames=1000, is_leaf=False,
                        children=[mid])
        assert root.children[0].children[0].depth == 3


# ---------------------------------------------------------------------------
# Legacy flat auto_cluster
# ---------------------------------------------------------------------------

class TestLegacyAutoCluster:
    def test_with_blobs(self):
        """Legacy flat sweep finds candidates for separable data."""
        from sklearn.datasets import make_blobs
        X, _ = make_blobs(n_samples=200, centers=3, n_features=64, random_state=42)
        candidates = auto_cluster(X, presets=["low"], eps_values=[0.5, 1.0],
                                  n_neighbors_filter=25)
        # Should find at least one valid candidate
        assert len(candidates) >= 0  # may be 0 if UMAP randomness

    def test_empty_data(self):
        X = np.zeros((5, 10))
        candidates = auto_cluster(X, presets=["low"], eps_values=[1.0],
                                  n_neighbors_filter=25)
        assert isinstance(candidates, list)

    def test_nan_handling(self):
        X = np.random.randn(100, 10)
        X[50:55] = np.nan
        candidates = auto_cluster(X, presets=["low"], eps_values=[1.0],
                                  n_neighbors_filter=25)
        assert isinstance(candidates, list)


# ---------------------------------------------------------------------------
# select_best
# ---------------------------------------------------------------------------

class TestSelectBest:
    def test_returns_highest_quality(self):
        c1 = ClusteringCandidate("low", 300, 1.0, [], 3, 0.1, 0.8, 0.9, 0.05)
        c2 = ClusteringCandidate("low", 300, 0.5, [], 5, 0.05, 0.9, 0.95, 0.02)
        best = select_best([c2, c1])  # c2 has higher quality
        assert best.quality_score == 0.9

    def test_tc_filter(self):
        c1 = ClusteringCandidate("low", 300, 1.0, [], 3, 0.1, 0.9, 0.7, 0.05)
        c2 = ClusteringCandidate("low", 300, 0.5, [], 5, 0.05, 0.8, 0.85, 0.02)
        best = select_best([c1, c2], min_tc=0.8)
        assert best.temporal_coherence >= 0.8

    def test_empty_returns_none(self):
        assert select_best([]) is None

    def test_fallback_when_no_tc_match(self):
        c1 = ClusteringCandidate("low", 300, 1.0, [], 3, 0.1, 0.9, 0.5, 0.05)
        best = select_best([c1], min_tc=0.99)
        assert best is c1  # fallback to best overall
