"""Tests for ``suggest_clustering_params`` (PERF-06 / P3-D)."""

from __future__ import annotations

import pytest

from castle.service.clustering_service import (
    ClusteringParamSuggestion,
    suggest_clustering_params,
)


def test_basic_suggestion_for_typical_dataset() -> None:
    s = suggest_clustering_params(10_000)
    assert isinstance(s, ClusteringParamSuggestion)
    # 10k / 200 = 50 → min_cluster_size = 50
    assert s.min_cluster_size == 50
    # 10k / 500 = 20 → min_samples = 20
    assert s.min_samples == 20
    assert s.min_samples < s.min_cluster_size, "min_samples must be < min_cluster_size for HDBSCAN"
    assert s.eps_range  # non-empty
    assert 1.0 in s.eps_range


def test_tiny_dataset_clamps_to_lower_bounds() -> None:
    """Small datasets shouldn't get sub-10 min_cluster_size."""
    s = suggest_clustering_params(100)
    assert s.min_cluster_size >= 10
    assert s.min_samples >= 5
    assert s.min_samples < s.min_cluster_size


def test_large_dataset_scales_up() -> None:
    s = suggest_clustering_params(1_000_000)
    assert s.min_cluster_size == 5_000
    assert s.min_samples == 2_000


def test_zero_or_negative_raises() -> None:
    with pytest.raises(ValueError, match="positive"):
        suggest_clustering_params(0)
    with pytest.raises(ValueError, match="positive"):
        suggest_clustering_params(-5)


def test_eps_range_is_monotone_increasing() -> None:
    s = suggest_clustering_params(5000)
    assert s.eps_range == sorted(s.eps_range)
