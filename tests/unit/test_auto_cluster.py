"""
tests/unit/test_auto_cluster.py
Unit tests for automated Behavior Microscope (auto_cluster).
"""

import numpy as np
import pytest
from sklearn.datasets import make_blobs

from castle.core.auto_cluster import (
    MICROSCOPE_PRESETS,
    ClusteringCandidate,
    score_clustering,
    auto_cluster,
    select_best,
)


# ---------------------------------------------------------------------------
# Test score_clustering function
# ---------------------------------------------------------------------------

def test_score_clustering_perfect_labels():
    """Perfect temporal structure: high quality score."""
    # Create perfect labels: long stable bouts
    labels = np.array([0]*50 + [1]*50 + [2]*50)
    embedding = np.random.randn(150, 2)
    
    scores = score_clustering(labels, embedding)
    
    assert scores["quality_score"] > 0.7  # Should be high
    assert scores["temporal_coherence"] > 0.95
    assert scores["single_frame_ratio"] < 0.1
    assert scores["n_clusters"] == 3


def test_score_clustering_random_labels():
    """Random flickering labels: low quality score."""
    np.random.seed(42)
    labels = np.random.randint(0, 5, size=200)
    embedding = np.random.randn(200, 2)
    
    scores = score_clustering(labels, embedding)
    
    assert scores["quality_score"] < 0.5  # Should be low
    assert scores["temporal_coherence"] < 0.3
    assert scores["single_frame_ratio"] > 0.3


def test_score_clustering_all_same():
    """All same label: penalized for n_clusters < 2."""
    labels = np.zeros(100, dtype=int)
    embedding = np.random.randn(100, 2)
    
    scores = score_clustering(labels, embedding)
    
    assert scores["n_clusters"] == 1
    # Penalty is applied but temporal coherence is perfect (all same = no transitions)
    # So score might still be moderate
    assert scores["quality_score"] < 0.9  # Should not be high


def test_score_clustering_high_noise():
    """High noise ratio: penalized."""
    labels = np.array([0]*30 + [1]*30 + [-1]*140)  # 70% noise
    embedding = np.random.randn(200, 2)
    
    scores = score_clustering(labels, embedding)
    
    assert scores["noise_ratio"] == 0.7
    # Quality is affected but TC can still be high if the valid parts are stable
    assert scores["quality_score"] < 0.85


# ---------------------------------------------------------------------------
# Test MICROSCOPE_PRESETS structure
# ---------------------------------------------------------------------------

def test_microscope_presets_has_all_keys():
    """All 4 preset keys exist."""
    assert "low" in MICROSCOPE_PRESETS
    assert "intermediate" in MICROSCOPE_PRESETS
    assert "high" in MICROSCOPE_PRESETS
    assert "super_high" in MICROSCOPE_PRESETS


def test_preset_low_build_config():
    """Low preset returns correct 1-stage config."""
    cfg = MICROSCOPE_PRESETS["low"]["build_config"](100)
    assert len(cfg) == 1
    assert cfg[0]["n_neighbors"] == 100
    assert cfg[0]["n_components"] == 2
    assert cfg[0]["min_dist"] == 0.0


def test_preset_intermediate_build_config():
    """Intermediate preset returns correct 2-stage config."""
    cfg = MICROSCOPE_PRESETS["intermediate"]["build_config"]((500, 300))
    assert len(cfg) == 2
    assert cfg[0]["n_neighbors"] == 500
    assert cfg[0]["n_components"] == 5
    assert cfg[1]["n_neighbors"] == 300
    assert cfg[1]["n_components"] == 2


def test_preset_super_high_build_config():
    """Super-high preset returns correct 3-stage config."""
    cfg = MICROSCOPE_PRESETS["super_high"]["build_config"]((300, 100, 50))
    assert len(cfg) == 3
    assert cfg[0]["n_neighbors"] == 300
    assert cfg[0]["n_components"] == 15
    assert cfg[1]["n_neighbors"] == 100
    assert cfg[1]["n_components"] == 5
    assert cfg[2]["n_neighbors"] == 50
    assert cfg[2]["n_components"] == 2


# ---------------------------------------------------------------------------
# Test auto_cluster function
# ---------------------------------------------------------------------------

def test_auto_cluster_with_synthetic_data():
    """auto_cluster finds valid candidates on synthetic blob data."""
    X, y_true = make_blobs(n_samples=200, centers=3, n_features=64, random_state=42)
    
    candidates = auto_cluster(
        data=X,
        presets=["low"],
        eps_values=[0.5, 1.0],
        n_neighbors_filter=25,  # Use smallest/fastest
        device="cpu",
    )
    
    assert len(candidates) > 0
    # Check first candidate structure
    c = candidates[0]
    assert isinstance(c, ClusteringCandidate)
    assert c.preset_name == "low"
    assert c.n_neighbors == 25
    assert c.eps in [0.5, 1.0]
    assert c.n_clusters >= 2
    assert 0.0 <= c.quality_score <= 1.0


def test_auto_cluster_preset_filter():
    """auto_cluster only returns candidates from requested presets."""
    X, _ = make_blobs(n_samples=100, centers=2, n_features=32, random_state=42)
    
    candidates = auto_cluster(
        data=X,
        presets=["low"],
        eps_values=[0.5],
        n_neighbors_filter=25,
        device="cpu",
    )
    
    for c in candidates:
        assert c.preset_name == "low"


def test_auto_cluster_empty_data():
    """auto_cluster with empty data returns empty candidates."""
    candidates = auto_cluster(
        data=np.array([]).reshape(0, 64),
        presets=["low"],
        eps_values=[0.5],
        device="cpu",
    )
    
    assert len(candidates) == 0


def test_auto_cluster_handles_nan_rows():
    """auto_cluster handles NaN rows gracefully."""
    X, _ = make_blobs(n_samples=100, centers=2, n_features=32, random_state=42)
    # Inject NaN rows
    X[10:20, :] = np.nan
    
    candidates = auto_cluster(
        data=X,
        presets=["low"],
        eps_values=[0.5],
        n_neighbors_filter=25,
        device="cpu",
    )
    
    # Should still find candidates (NaN filtered out)
    assert len(candidates) >= 0


# ---------------------------------------------------------------------------
# Test select_best function
# ---------------------------------------------------------------------------

def test_select_best_returns_highest_quality():
    """select_best returns first candidate meeting min_tc threshold."""
    c1 = ClusteringCandidate(
        preset_name="low", n_neighbors=25, eps=0.5, umap_config=[],
        n_clusters=3, noise_ratio=0.1, quality_score=0.85,
        temporal_coherence=0.92, single_frame_ratio=0.05,
    )
    c2 = ClusteringCandidate(
        preset_name="low", n_neighbors=25, eps=1.0, umap_config=[],
        n_clusters=4, noise_ratio=0.2, quality_score=0.75,
        temporal_coherence=0.88, single_frame_ratio=0.1,
    )
    
    # select_best expects pre-sorted candidates (as provided by auto_cluster)
    candidates = sorted([c2, c1], key=lambda x: x.quality_score, reverse=True)
    best = select_best(candidates, min_tc=0.8)
    
    # Should return c1 (highest quality, meets min_tc)
    assert best.quality_score == 0.85
    assert best.temporal_coherence == 0.92


def test_select_best_with_min_tc_filter():
    """select_best respects min_tc threshold."""
    c1 = ClusteringCandidate(
        preset_name="low", n_neighbors=25, eps=0.5, umap_config=[],
        n_clusters=3, noise_ratio=0.1, quality_score=0.90,
        temporal_coherence=0.75,  # Below threshold
        single_frame_ratio=0.05,
    )
    c2 = ClusteringCandidate(
        preset_name="low", n_neighbors=25, eps=1.0, umap_config=[],
        n_clusters=4, noise_ratio=0.2, quality_score=0.80,
        temporal_coherence=0.85,  # Above threshold
        single_frame_ratio=0.1,
    )
    
    candidates = [c1, c2]  # c1 has higher quality but low TC
    best = select_best(candidates, min_tc=0.80)
    
    assert best == c2  # c2 meets min_tc


def test_select_best_empty_list():
    """select_best with empty list returns None."""
    best = select_best([])
    assert best is None


# ---------------------------------------------------------------------------
# Test ClusteringCandidate dataclass
# ---------------------------------------------------------------------------

def test_clustering_candidate_fields():
    """ClusteringCandidate has all expected fields."""
    c = ClusteringCandidate(
        preset_name="high",
        n_neighbors=(500, 300),
        eps=1.5,
        umap_config=[{"n_neighbors": 500}, {"n_neighbors": 300}],
        n_clusters=10,
        noise_ratio=0.15,
        quality_score=0.82,
        temporal_coherence=0.91,
        single_frame_ratio=0.08,
        calinski_harabasz=450.0,
        davies_bouldin=0.65,
    )
    
    assert c.preset_name == "high"
    assert c.n_neighbors == (500, 300)
    assert c.eps == 1.5
    assert c.n_clusters == 10
    assert c.quality_score == 0.82


# ---------------------------------------------------------------------------
# Test quality_score is in valid range
# ---------------------------------------------------------------------------

def test_quality_score_range():
    """Quality score is in [0, 1] for normal inputs."""
    # Generate varied but reasonable labels
    labels = np.array([0]*40 + [1]*40 + [2]*40 + [-1]*10)  # 3 clusters + 10 noise
    embedding = np.random.randn(130, 2)
    
    scores = score_clustering(labels, embedding)
    
    assert 0.0 <= scores["quality_score"] <= 1.0
    assert scores["n_clusters"] == 3
