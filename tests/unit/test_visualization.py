"""Unit tests for castle.visualization.embedding_plots.

Uses matplotlib Agg backend to avoid display requirements.
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from castle.visualization.embedding_plots import (
    plot_embedding,
    plot_named_embedding,
    plot_syllables,
    plot_syllables_bar,
)


def test_plot_embedding_no_cluster():
    """plot_embedding with no cluster should not crash."""
    embedding = np.random.randn(100, 2)
    plt.figure()
    plot_embedding(embedding)
    plt.close('all')


def test_plot_embedding_with_cluster():
    embedding = np.random.randn(100, 2)
    cluster = np.array([0] * 50 + [1] * 50)

    def palette(c):
        return ['#FF0000', '#00FF00'][c % 2]

    plt.figure()
    plot_embedding(embedding, cluster=cluster, palette_fn=palette)
    plt.close('all')


def test_plot_embedding_with_unclustered():
    embedding = np.random.randn(100, 2)
    cluster = np.array([0] * 40 + [1] * 40 + [-1] * 20)

    def palette(c):
        return '#FF0000' if c >= 0 else '#CCCCCC'

    plt.figure()
    plot_embedding(embedding, cluster=cluster, palette_fn=palette)
    plt.close('all')


def test_plot_named_embedding():
    embedding = np.random.randn(50, 2)
    cluster = np.array([0] * 25 + [1] * 25)
    export = {
        0: {'name': 'running', 'color': '#FF0000'},
        1: {'name': 'walking', 'color': '#00FF00'},
    }

    plt.figure()
    plot_named_embedding(embedding, cluster, export, palette_fn=lambda c: '#CCCCCC')
    plt.close('all')


def test_plot_syllables():
    cluster = np.array([0, 0, 0, 1, 1, 1, 0, 0])
    key_frames = [0, 3, 6, 8]
    cluster_meta = {
        0: {'name': 'idle', 'color': '#AAAAAA'},
        1: {'name': 'active', 'color': '#FF0000'},
    }

    plt.figure()
    plot_syllables(cluster, key_frames, cluster_meta)
    plt.close('all')


def test_plot_syllables_bar():
    syllables = np.array([0, 0, 1, 1, 2, 2])
    key_frames = [0, 2, 4, 6]
    meta = [
        {'name': 'A', 'color': '#FF0000'},
        {'name': 'B', 'color': '#00FF00'},
        {'name': 'C', 'color': '#0000FF'},
    ]

    plt.figure()
    plot_syllables_bar(syllables, key_frames, meta)
    plt.close('all')


def test_plot_syllables_with_palette_fn():
    cluster = np.array([0, 0, 1, 1])
    key_frames = [0, 2, 4]
    cluster_meta = {
        0: {'name': 'a', 'color': '#AA0000'},
        1: {'name': 'b', 'color': '#00AA00'},
    }

    plt.figure()
    plot_syllables(cluster, key_frames, cluster_meta,
                   palette_fn=lambda c: cluster_meta.get(c, {}).get('color', 'grey'))
    plt.close('all')
