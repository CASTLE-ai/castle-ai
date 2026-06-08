"""
castle/visualization/embedding_plots.py
Standalone visualization functions for embeddings and syllables.

Extracted from castle.utils.latent_explorer (Latent, LocalLatent) and
castle.utils.explorer (Latent, FocusLatent) to separate data from visualization (B-01).

All functions take plain data (numpy arrays, dicts) as arguments,
not class instances, so they can be used from any frontend.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from typing import List, Dict, Optional, Callable


def plot_embedding(embedding: np.ndarray,
                   cluster: Optional[np.ndarray] = None,
                   palette_fn: Optional[Callable[[int], str]] = None,
                   dims: Optional[List[int]] = None,
                   legend: bool = True) -> None:
    """
    Plot 2D embedding scatter with optional cluster coloring.
    
    Replaces LocalLatent.plot_embedding() from latent_explorer.py.
    
    Args:
        embedding: (N, D) array of embedding coordinates.
        cluster: (N,) integer array of cluster assignments. -1 = unclustered.
        palette_fn: Callable(cluster_id) -> color string. If None, uses grey.
        dims: Which two dimensions of embedding to plot.
        legend: Whether to show legend.
    """
    if dims is None:
        dims = [0, 1]
    assert len(dims) == 2, 'dims must have exactly 2 elements'

    # Density-aware markers. With min_dist=0 UMAP stacks similar points onto the
    # same spot, so at ~10^5-10^6 points the default large opaque dots hide the
    # data — it looks like a few hundred blobs instead of a dense cloud. Shrink
    # the markers and add transparency so overlapping points read as density.
    n = len(embedding)
    _s = 36 if n < 2_000 else 8 if n < 50_000 else 2
    _alpha = 1.0 if n < 2_000 else 0.4 if n < 50_000 else 0.15

    if cluster is not None and palette_fn is not None:
        for cid in range(0, cluster.max() + 1):
            mask = cluster == cid
            if mask.any():
                plt.scatter(
                    x=embedding[mask, dims[0]],
                    y=embedding[mask, dims[1]],
                    c=palette_fn(cid),
                    label=f'{cid}',
                    s=_s, alpha=_alpha, linewidths=0,
                )
        if -1 in cluster:
            mask = cluster == -1
            plt.scatter(
                x=embedding[mask, dims[0]],
                y=embedding[mask, dims[1]],
                c='grey',
                label='-1',
                s=_s, alpha=_alpha, linewidths=0,
            )
        if legend:
            plt.legend()
    else:
        plt.scatter(
            x=embedding[:, dims[0]],
            y=embedding[:, dims[1]],
            c='grey',
            s=_s, alpha=_alpha, linewidths=0,
        )


def plot_named_embedding(embedding: np.ndarray,
                         cluster: np.ndarray,
                         export: Dict[int, Dict],
                         palette_fn: Callable[[int], str],
                         dims: Optional[List[int]] = None,
                         legend: bool = True) -> None:
    """
    Plot 2D embedding scatter colored by named cluster labels.
    
    Replaces LocalLatent.plot_name_embedding() from latent_explorer.py.
    
    Args:
        embedding: (N, D) array of embedding coordinates.
        cluster: (N,) integer array of cluster assignments.
        export: Dict mapping cluster_id -> {'name': str, 'color': str}.
        palette_fn: Fallback callable(cluster_id) -> color string for unnamed clusters.
        dims: Which two dimensions to plot.
        legend: Whether to show legend.
    """
    if dims is None:
        dims = [0, 1]
    assert len(dims) == 2, 'dims must have exactly 2 elements'

    if cluster is not None:
        for cid in range(0, cluster.max() + 1):
            mask = cluster == cid
            if not mask.any():
                continue
            if cid in export:
                c = export[cid]['color'] or palette_fn(cid)
                label = export[cid]['name']
            else:
                c = palette_fn(-1)
                label = cid
            plt.scatter(
                x=embedding[mask, dims[0]],
                y=embedding[mask, dims[1]],
                c=c,
                label=label
            )
        if legend:
            plt.legend()
    else:
        plt.scatter(
            x=embedding[:, dims[0]],
            y=embedding[:, dims[1]],
            c='grey'
        )


def plot_syllables(cluster: np.ndarray,
                   key_frames: List[int],
                   cluster_meta: Dict[int, Dict],
                   palette_fn: Optional[Callable[[int], str]] = None) -> None:
    """
    Plot behavioral syllables as a horizontal bar timeline.
    
    Replaces Latent.plot_syllables() from latent_explorer.py.
    
    Args:
        cluster: (N,) integer array of cluster IDs per bin.
        key_frames: List of frame indices where cluster identity changes.
        cluster_meta: Dict mapping cluster_id -> {'name': str, 'color': str}.
        palette_fn: Optional callable(cluster_id) -> color string.
                    If None, uses cluster_meta colors with grey fallback.
    """
    if palette_fn is None:
        def palette_fn(c):
            if c in cluster_meta:
                return cluster_meta[c]['color']
            return 'grey'

    widths = [key_frames[j + 1] - key_frames[j] for j in range(len(key_frames) - 1)]
    colors = [palette_fn(cluster[key_frames[j]]) for j in range(len(key_frames) - 1)]
    lefts = key_frames[:-1]

    plt.bar(lefts, height=[1] * len(widths), width=widths, color=colors,
            align='edge', edgecolor='none')
    plt.xlim(0, key_frames[-1])
    plt.ylim(0, 1)
    plt.yticks([])

    unique_categories = sorted(set(cluster[key_frames[j]] for j in range(len(key_frames) - 1)))
    if -1 in unique_categories:
        unique_categories.remove(-1)

    legend_handles = [
        Patch(color=palette_fn(cat), label=cluster_meta[cat]['name'])
        for cat in unique_categories
        if cat in cluster_meta
    ]
    if legend_handles:
        plt.legend(handles=legend_handles, title="Categories")


def plot_syllables_bar(syllables: np.ndarray,
                       key_frames: List[int],
                       meta: List[Dict],
                       palette_fn: Optional[Callable[[int], str]] = None,
                       legend: bool = True) -> None:
    """
    Plot behavioral syllables as a horizontal bar timeline.
    
    Replaces Latent.plot() from explorer.py.
    Uses list-based meta (explorer.py style) instead of dict-based cluster_meta.
    
    Args:
        syllables: (N,) integer array of syllable IDs per bin.
        key_frames: List of frame indices where syllable identity changes.
        meta: List of dicts with 'name' and 'color' keys, indexed by syllable ID.
        palette_fn: Optional callable(syllable_id) -> color string.
        legend: Whether to show legend.
    """
    if palette_fn is None:
        def palette_fn(c):
            if 0 <= c < len(meta):
                return meta[c]['color']
            return 'grey'

    widths = [key_frames[j + 1] - key_frames[j] for j in range(len(key_frames) - 1)]
    colors = [palette_fn(syllables[key_frames[j]]) for j in range(len(key_frames) - 1)]
    lefts = key_frames[:-1]

    plt.bar(lefts, height=[1] * len(widths), width=widths, color=colors,
            align='edge', edgecolor='none')
    plt.xlim(0, key_frames[-1])
    plt.ylim(0, 1)
    plt.yticks([])

    if legend:
        unique_categories = sorted(set(syllables[key_frames[j]] for j in range(len(key_frames) - 1)))
        if -1 in unique_categories:
            unique_categories.remove(-1)

        legend_handles = [
            Patch(color=palette_fn(cat), label=meta[cat]['name'])
            for cat in unique_categories
            if 0 <= cat < len(meta)
        ]
        if legend_handles:
            plt.legend(handles=legend_handles, title="Categories")


def plot_focus_embedding(embedding: np.ndarray,
                         focus: np.ndarray,
                         cluster: Optional[np.ndarray] = None,
                         palette_fn: Optional[Callable[[int], str]] = None,
                         dims: Optional[List[int]] = None,
                         legend: bool = True) -> None:
    """
    Plot 2D embedding scatter for FocusLatent (explorer.py style).
    
    Replaces FocusLatent.plot() from explorer.py.
    Similar to plot_embedding but handles focus mask (NaN-containing embeddings).
    
    Args:
        embedding: (N, D) array, may contain NaN for non-focus points.
        focus: (N,) boolean array indicating valid (focused) points.
        cluster: (N,) integer array of cluster assignments. -1 = unclustered.
        palette_fn: Callable(cluster_id) -> color string.
        dims: Which two dimensions to plot.
        legend: Whether to show legend.
    """
    if dims is None:
        dims = [0, 1]
    assert len(dims) == 2, 'dims must have exactly 2 elements'

    if cluster is not None and palette_fn is not None:
        for cid in range(0, cluster.max() + 1):
            mask = cluster == cid
            if mask.any():
                plt.scatter(
                    x=embedding[mask, dims[0]],
                    y=embedding[mask, dims[1]],
                    c=palette_fn(cid),
                    label=f'{cid}'
                )
        if -1 in cluster:
            mask = cluster == -1
            plt.scatter(
                x=embedding[mask, dims[0]],
                y=embedding[mask, dims[1]],
                c='grey',
                label='-1'
            )
        if legend:
            plt.legend()
    else:
        plt.scatter(
            x=embedding[focus, dims[0]],
            y=embedding[focus, dims[1]],
            c='grey'
        )
