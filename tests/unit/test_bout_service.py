"""Unit tests for castle.service.bout_service."""

import numpy as np
import pytest

from castle.service.bout_service import (
    find_bouts,
    _select_representative_bouts,
    _compute_aligned_range,
)


def test_find_bouts_simple():
    cluster = np.array([0, 0, 0, 1, 1, 0, 0, 0, 0, 1])
    bouts = find_bouts(cluster, cluster_id=0)
    # Should find 2 bouts: [0:3] and [5:9]
    assert len(bouts) == 2
    # Sorted by length (longest first): [5:9] (4 bins) then [0:3] (3 bins)
    assert bouts[0] == (5, 9)
    assert bouts[1] == (0, 3)


def test_find_bouts_empty():
    cluster = np.array([1, 1, 1])
    bouts = find_bouts(cluster, cluster_id=0)
    assert len(bouts) == 0


def test_find_bouts_all_same():
    cluster = np.array([2, 2, 2, 2, 2])
    bouts = find_bouts(cluster, cluster_id=2)
    assert len(bouts) == 1
    assert bouts[0] == (0, 5)


def test_find_bouts_single_element():
    cluster = np.array([0, 1, 0, 1, 0])
    bouts = find_bouts(cluster, cluster_id=0)
    assert len(bouts) == 3
    # Each bout is length 1, sorted longest-first (all equal)
    for start, end in bouts:
        assert end - start == 1


def test_find_bouts_at_boundaries():
    """Bouts at the start and end of the array."""
    cluster = np.array([3, 3, 1, 1, 3, 3, 3])
    bouts = find_bouts(cluster, cluster_id=3)
    assert len(bouts) == 2
    # [4:7] (3 bins) is longest, then [0:2] (2 bins)
    assert bouts[0] == (4, 7)
    assert bouts[1] == (0, 2)


# -----------------------------------------------------------------------
# Tests for _select_representative_bouts
# -----------------------------------------------------------------------


def _make_embedding_and_cluster(bouts_with_offsets, n_bins=50):
    """Helper: build synthetic embedding and cluster arrays.

    Each bout is a run of cluster-0 bins.  Embeddings for bout bins are
    placed at (offset, 0) so distance to centroid is deterministic.

    Args:
        bouts_with_offsets: list of (start, end, emb_x_offset) triples.
        n_bins: total number of bins.
    """
    cluster = np.ones(n_bins, dtype=np.int32)  # everything else is cluster 1
    embedding = np.zeros((n_bins, 2), dtype=np.float64)

    for start, end, emb_x in bouts_with_offsets:
        cluster[start:end] = 0
        embedding[start:end, 0] = emb_x

    return embedding, cluster


def test_select_representative_bouts_returns_n():
    """Should return exactly n bouts (when enough exist)."""
    bouts_def = [
        (0, 5, 1.0),
        (10, 15, 2.0),
        (20, 25, 3.0),
        (30, 35, 4.0),
    ]
    bouts = [(s, e) for s, e, _ in bouts_def]
    embedding, cluster = _make_embedding_and_cluster(bouts_def)

    selected = _select_representative_bouts(bouts, 0, embedding, cluster, n=3)
    assert len(selected) == 3


def test_select_representative_bouts_fewer_than_n():
    """When fewer bouts exist than n, return all of them."""
    bouts_def = [(0, 5, 0.5), (10, 15, 1.5)]
    bouts = [(s, e) for s, e, _ in bouts_def]
    embedding, cluster = _make_embedding_and_cluster(bouts_def)

    selected = _select_representative_bouts(bouts, 0, embedding, cluster, n=9)
    assert len(selected) == 2


def test_select_representative_bouts_ranking():
    """The bout closest to centroid should come first.

    Centroid of cluster 0 = mean of all cluster-0 embeddings.
    With two bouts at x=0.5 and x=10.0, the one at x=0.5 is closer
    to the centroid (which will be dominated by the many bins at 0.5
    if they're longer).

    Instead, use equal-length bouts and put centroid clearly between them.
    Bout A at x=0, Bout B at x=4.  Centroid ≈ 2.  Bout at x=4 is closer to 2
    only if |4-2| < |0-2|.  Actually |4-2|=2 == |0-2|=2, so make asymmetric.
    Bout A at x=0 (5 bins), Bout B at x=3 (5 bins), Bout C at x=9 (5 bins).
    Centroid ≈ (0*5 + 3*5 + 9*5) / 15 = 4.  Closest to 4 → B (dist=1), then A (4), C (5).
    """
    n_bins = 60
    cluster = np.ones(n_bins, dtype=np.int32)
    embedding = np.zeros((n_bins, 2), dtype=np.float64)

    # Bout A: bins 0-5, x=0
    cluster[0:5] = 0
    embedding[0:5, 0] = 0.0

    # Bout B: bins 10-15, x=3
    cluster[10:15] = 0
    embedding[10:15, 0] = 3.0

    # Bout C: bins 20-25, x=9
    cluster[20:25] = 0
    embedding[20:25, 0] = 9.0

    bouts = [(0, 5), (10, 15), (20, 25)]
    selected = _select_representative_bouts(bouts, 0, embedding, cluster, n=3)

    # Centroid = (0*5 + 3*5 + 9*5) / 15 = 12/3 = 4
    # dist² A = (0-4)²=16, B = (3-4)²=1, C = (9-4)²=25 → order B, A, C
    assert selected[0] == (10, 15), "Bout B (x=3) should be most representative"
    assert selected[1] == (0, 5), "Bout A (x=0) should be second"
    assert selected[2] == (20, 25), "Bout C (x=9) should be last"


def test_select_representative_bouts_empty():
    """Empty bout list → empty result."""
    embedding = np.zeros((20, 2))
    cluster = np.zeros(20, dtype=np.int32)
    result = _select_representative_bouts([], 0, embedding, cluster, n=5)
    assert result == []


# -----------------------------------------------------------------------
# Tests for _compute_aligned_range
# -----------------------------------------------------------------------


def test_compute_aligned_range_uniform():
    """All bouts same length → half_len = bout_half + pad_bins."""
    # Bouts of length 10 each; pad=5; centre at 15, 35, 55
    bouts = [(10, 20), (30, 40), (50, 60)]
    pad_bins = 5
    total_bins = 100

    half_len = _compute_aligned_range(bouts, pad_bins, total_bins)
    # Centre of (10,20) = 15; padded_start = max(0,10-5)=5; padded_end = min(100,20+5)=25
    # left = 15-5=10, right = 25-15=10 → min=10
    # Same for others.  min over all bouts = 10
    assert half_len == 10


def test_compute_aligned_range_shortest_limits():
    """The shortest padded bout constrains the result."""
    # Short bout (length 2) padded 3 each side → padded length 8, half=4
    # Long bout (length 20) padded 3 → half = 10
    # Result should be min(4, 10) = 4
    bouts = [(40, 42), (10, 30)]  # short first
    pad_bins = 3
    total_bins = 100

    half_len = _compute_aligned_range(bouts, pad_bins, total_bins)
    # (40,42): centre=41, padded_start=37, padded_end=45; left=4, right=4 → 4
    # (10,30): centre=20, padded_start=7, padded_end=33; left=13, right=13 → 13
    assert half_len == 4


def test_compute_aligned_range_boundary_clamp():
    """A bout near the start/end of the array should be clamped."""
    # Bout at bins 0-4, centre=2, pad=10
    # padded_start = max(0, 0-10)=0; padded_end = min(20, 4+10)=14
    # left = 2-0=2, right = 14-2=12 → min=2
    bouts = [(0, 4)]
    pad_bins = 10
    total_bins = 20

    half_len = _compute_aligned_range(bouts, pad_bins, total_bins)
    assert half_len == 2


def test_compute_aligned_range_empty():
    """Empty bouts list → half_len == 0."""
    assert _compute_aligned_range([], 5, 100) == 0
