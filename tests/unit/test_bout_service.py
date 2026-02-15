"""Unit tests for castle.service.bout_service."""

import numpy as np
from castle.service.bout_service import find_bouts


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
