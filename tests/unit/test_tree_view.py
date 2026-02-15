"""Unit tests for the cluster tree building logic.

cluster_page_ui.py imports gradio at module level, so we mock it
to avoid the heavy import and keep this as a fast unit test.
"""

import sys
from unittest.mock import MagicMock

import numpy as np


def test_build_cluster_tree_markdown():
    # Pre-mock gradio before importing the module
    _original_gradio = sys.modules.get('gradio')
    sys.modules['gradio'] = MagicMock()
    try:
        from castle.ui.cluster_page_ui import build_cluster_tree_markdown

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
    finally:
        if _original_gradio is not None:
            sys.modules['gradio'] = _original_gradio
        else:
            del sys.modules['gradio']


def test_build_cluster_tree_markdown_empty():
    _original_gradio = sys.modules.get('gradio')
    sys.modules['gradio'] = MagicMock()
    try:
        from castle.ui.cluster_page_ui import build_cluster_tree_markdown

        cluster_meta = {
            0: {'name': 'init', 'color': 'grey'},
        }
        cluster_array = np.array([0] * 10)

        md = build_cluster_tree_markdown(cluster_meta, cluster_array)
        assert 'No clusters yet' in md
    finally:
        if _original_gradio is not None:
            sys.modules['gradio'] = _original_gradio
        else:
            del sys.modules['gradio']


def test_build_cluster_tree_markdown_counts():
    _original_gradio = sys.modules.get('gradio')
    sys.modules['gradio'] = MagicMock()
    try:
        from castle.ui.cluster_page_ui import build_cluster_tree_markdown

        cluster_meta = {
            0: {'name': 'init', 'color': 'grey'},
            1: {'name': 'root_a0', 'color': '#FF0000'},
        }
        cluster_array = np.array([0] * 10 + [1] * 42)

        md = build_cluster_tree_markdown(cluster_meta, cluster_array)
        assert '42 bins' in md
    finally:
        if _original_gradio is not None:
            sys.modules['gradio'] = _original_gradio
        else:
            del sys.modules['gradio']
