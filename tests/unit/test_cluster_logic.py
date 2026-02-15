"""Unit tests for pure-logic helpers in castle.core.cluster."""

import numpy as np
from castle.core.cluster import (
    auto_generate_cluster_name,
    find_nearest_embedding,
    frame_to_timestamp,
)


# ---- auto_generate_cluster_name ----

def test_auto_generate_cluster_name_root():
    assert auto_generate_cluster_name("root", 0) == "root_a0"
    assert auto_generate_cluster_name("root", 5) == "root_a5"


def test_auto_generate_cluster_name_nested():
    assert auto_generate_cluster_name("root_a0", 1) == "root_a0_b1"
    assert auto_generate_cluster_name("root_a0_b1", 2) == "root_a0_b1_c2"


def test_auto_generate_cluster_name_none():
    result = auto_generate_cluster_name(None, 0)
    assert result == "root_a0"


def test_auto_generate_cluster_name_deep():
    name = auto_generate_cluster_name("root_a0_b1_c2", 3)
    assert name == "root_a0_b1_c2_d3"


# ---- find_nearest_embedding ----

def test_find_nearest_embedding():
    data = np.array([[0, 0], [1, 1], [2, 2], [10, 10]], dtype=float)
    idx, dist = find_nearest_embedding(data, 0.9, 0.9)
    assert idx == 1  # Closest to (1,1)
    assert dist < 0.2


def test_find_nearest_embedding_exact():
    data = np.array([[0, 0], [5, 5], [10, 10]], dtype=float)
    idx, dist = find_nearest_embedding(data, 5.0, 5.0)
    assert idx == 1
    assert dist < 1e-10


def test_find_nearest_embedding_with_tree():
    from scipy.spatial import KDTree
    data = np.array([[0, 0], [3, 4], [6, 8]], dtype=float)
    tree = KDTree(data)
    idx, dist = find_nearest_embedding(data, 3.1, 4.1, tree=tree)
    assert idx == 1


# ---- frame_to_timestamp ----

def test_frame_to_timestamp_zero():
    ts = frame_to_timestamp(0, 30.0)
    assert ts == "00:00:00,000"


def test_frame_to_timestamp_one_second():
    ts = frame_to_timestamp(30, 30.0)
    assert ts == "00:00:01,000"


def test_frame_to_timestamp_complex():
    # 3661 seconds = 1h 1m 1s at 1 fps
    ts = frame_to_timestamp(3661, 1.0)
    assert ts.startswith("01:01:01")
