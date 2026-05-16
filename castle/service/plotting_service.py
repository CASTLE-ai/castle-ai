"""Plotting service — Gradio-free wrappers for embedding scatter plots.

The Gradio handlers used to instantiate :class:`EmbeddingScatterPlot`
inline. That coupled algorithm code to the UI module and made the
"build figure" step impossible to call from a notebook or PyQt without
dragging the Gradio import path along for the ride.

This service exposes two thin helpers that return the plot object plus
its rendered image, so the handler reduces to "call helper, slot two
items into Gradio outputs".

It deliberately does **not** depend on ``gradio`` or ``PyQt6`` — only on
matplotlib (already a CASTLE dep).
"""

from __future__ import annotations

import os
from typing import Any, Tuple

import numpy as np

__all__ = [
    "build_scatter_plot",
    "build_named_scatter_plot",
    "plot_syllables_per_video",
]


def build_scatter_plot(local_latents: Any) -> Tuple[Any, np.ndarray]:
    """Build an :class:`EmbeddingScatterPlot` and render the cluster-coloured
    figure.

    Args:
        local_latents: A :class:`castle.utils.latent_explorer.LocalLatent`
            with ``embedding`` and ``cluster`` attributes populated.

    Returns:
        ``(scatter_plot, image)`` — the plot object (so callers can keep
        it in state for click-to-frame interactions) plus the rendered
        ``np.ndarray`` ready to hand to Gradio's :class:`~gradio.Image`
        component.
    """
    from castle.ui.embedding_scatter import EmbeddingScatterPlot

    plot = EmbeddingScatterPlot(local_latents)
    return plot, plot.plot()


def build_named_scatter_plot(local_latents: Any) -> Tuple[Any, np.ndarray]:
    """Like :func:`build_scatter_plot` but renders the *named* embedding
    (post-submit view with cluster names overlaid).

    Args:
        local_latents: LocalLatent whose ``export`` dict carries the
            user-assigned cluster names.

    Returns:
        ``(scatter_plot, image)``.
    """
    from castle.ui.embedding_scatter import EmbeddingScatterPlot

    plot = EmbeddingScatterPlot(local_latents)
    return plot, plot.plot_named_embedding()


def plot_syllables_per_video(latents: Any, aggregator: Any):
    """Build a per-video syllable timeline figure (matplotlib).

    Renders one horizontal coloured bar per video, x-axis in seconds.
    Cluster colours come from ``latents.cluster_meta[cid]['color']`` and
    bar widths derive from the ``aggregator.bin_size`` × ``aggregator.fps``
    conversion. Used by the Behavior Microscope post-submit view.

    Args:
        latents: :class:`Latent` with ``cluster`` and ``cluster_meta``.
        aggregator: :class:`LatentAggregator` providing ``videos_meta``,
            ``fps``, ``bin_size``.

    Returns:
        A matplotlib ``Figure``. Caller is responsible for closing /
        rendering it.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    cluster = latents.cluster
    cluster_meta = latents.cluster_meta
    videos_meta = aggregator.videos_meta
    fps = aggregator.fps
    bin_size = aggregator.bin_size

    n_videos = len(videos_meta)
    fig, axes = plt.subplots(n_videos, 1, figsize=(14, 0.8 * n_videos), squeeze=False)
    axes = axes.flatten()

    def palette(c):
        if c in cluster_meta:
            return cluster_meta[c]['color']
        return 'grey'

    cum = 0
    for video_idx, (vn, video_name) in enumerate(videos_meta):
        ax = axes[video_idx]
        video_cluster = cluster[cum:cum + vn]
        n = len(video_cluster)
        key_frames = (
            [0]
            + [i + 1 for i in range(n - 1) if video_cluster[i] != video_cluster[i + 1]]
            + [n]
        )
        widths = [(key_frames[j + 1] - key_frames[j]) * bin_size / fps for j in range(len(key_frames) - 1)]
        colors = [palette(video_cluster[key_frames[j]]) for j in range(len(key_frames) - 1)]
        lefts = [key_frames[j] * bin_size / fps for j in range(len(key_frames) - 1)]
        total_seconds = n * bin_size / fps

        ax.bar(lefts, height=[1] * len(widths), width=widths, color=colors,
               align='edge', edgecolor='none')
        ax.set_xlim(0, total_seconds)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        video_basename = os.path.basename(video_name).split('.')[0]
        ax.set_title(video_basename, fontsize=9, loc='left')
        cum += vn

    unique_clusters = sorted(set(cluster))
    if -1 in unique_clusters:
        unique_clusters.remove(-1)

    legend_handles = [
        Patch(color=palette(cat), label=cluster_meta[cat]['name'])
        for cat in unique_clusters if cat in cluster_meta
    ]
    plt.tight_layout()
    if legend_handles:
        axes[-1].legend(
            handles=legend_handles, loc='upper center',
            bbox_to_anchor=(0.5, -0.3),
            ncol=min(len(legend_handles), 6), fontsize=8,
        )
        fig.subplots_adjust(bottom=0.2)

    return fig
