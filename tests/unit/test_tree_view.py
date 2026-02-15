"""Unit tests for the cluster tree building logic.

cluster_tree.py is a pure module with no Gradio dependency, so it can be
imported directly without mocking.
"""

import numpy as np

from castle.ui.cluster_tree import build_cluster_tree_markdown


def test_build_cluster_tree_markdown():
    cluster_meta = {
        0: {'name': 'init', 'color': 'grey'},
        1: {'name': 'root_a0', 'color': '#FF0000'},
        2: {'name': 'root_a1', 'color': '#00FF00'},
        3: {'name': 'root_a0_b0', 'color': '#0000FF'},
    }
    cluster_array = np.array([0] * 10 + [1] * 20 + [2] * 30 + [3] * 15)

    md = build_cluster_tree_markdown(cluster_meta, cluster_array)
    assert 'root_a0' in md
    assert 'root_a1' in md
    assert 'root_a0_b0' in md
    # init should be skipped
    assert 'init' not in md
    # Should have header
    assert 'Cluster Tree' in md


def test_build_cluster_tree_markdown_empty():
    cluster_meta = {
        0: {'name': 'init', 'color': 'grey'},
    }
    cluster_array = np.array([0] * 10)

    md = build_cluster_tree_markdown(cluster_meta, cluster_array)
    assert 'No clusters yet' in md


def test_build_cluster_tree_markdown_counts():
    cluster_meta = {
        0: {'name': 'init', 'color': 'grey'},
        1: {'name': 'root_a0', 'color': '#FF0000'},
    }
    cluster_array = np.array([0] * 10 + [1] * 42)

    md = build_cluster_tree_markdown(cluster_meta, cluster_array)
    assert '42 bins' in md
