"""Ethogram visualization: raster plots, transition heatmaps, bout distributions.

All functions return a :class:`matplotlib.figure.Figure` that the caller
can ``savefig()`` or display.
"""

from __future__ import annotations

from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import matplotlib
    import matplotlib.figure
    from castle.core.ethogram import BoutStatistics, Ethogram, TransitionMatrix


def _get_cluster_colors(names: Dict[int, str], n: int) -> List[str]:
    """Generate a list of distinct colours for *n* clusters."""
    try:
        from castle.core.config import PALETTE_HEX
        palette = (PALETTE_HEX * ((n // len(PALETTE_HEX)) + 1))[:n]
    except Exception:
        import matplotlib.cm as cm
        cmap = cm.get_cmap("tab20", n)
        palette = [
            "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))
            for r, g, b, _ in [cmap(i) for i in range(n)]
        ]
    return palette


def plot_ethogram_raster(
    ethogram: "Ethogram",
    figsize: tuple = (16, 4),
) -> "matplotlib.figure.Figure":
    """Plot ethogram raster: time on x-axis, behaviours as coloured blocks.

    Each time point is coloured by its cluster assignment, producing a
    horizontal bar showing behavioural state over time.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    labels = ethogram.cluster_labels
    unique_ids = sorted(ethogram.cluster_names.keys())
    n = len(unique_ids)
    id_to_idx = {cid: i for i, cid in enumerate(unique_ids)}

    colors = _get_cluster_colors(ethogram.cluster_names, n)
    cmap = ListedColormap(colors)

    mapped = np.array([id_to_idx.get(int(l), 0) for l in labels])

    fig, ax = plt.subplots(figsize=figsize)
    time_s = np.arange(len(labels)) / ethogram.fps

    ax.imshow(
        mapped[np.newaxis, :],
        aspect="auto",
        cmap=cmap,
        vmin=0,
        vmax=n - 1,
        extent=[time_s[0], time_s[-1], 0, 1],
        interpolation="nearest",
    )

    ax.set_xlabel("Time (s)")
    ax.set_yticks([])
    ax.set_title("Ethogram Raster")

    # Legend
    import matplotlib.patches as mpatches
    patches = [
        mpatches.Patch(color=colors[i], label=ethogram.cluster_names.get(cid, f"c{cid}"))
        for i, cid in enumerate(unique_ids)
    ]
    ax.legend(handles=patches, loc="upper right", fontsize="small", ncol=min(n, 6))

    fig.tight_layout()
    return fig


def plot_transition_heatmap(
    transition_matrix: "TransitionMatrix",
    figsize: tuple = (8, 8),
) -> "matplotlib.figure.Figure":
    """Plot transition probability matrix as an annotated heatmap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mat = transition_matrix.matrix
    names = transition_matrix.cluster_names
    K = len(names)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=1 if mat.max() <= 1 else None)

    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("To")
    ax.set_ylabel("From")
    ax.set_title(
        f"Transition Probabilities  (H={transition_matrix.entropy:.2f} bits, "
        f"n={transition_matrix.n_transitions})"
    )

    # Annotate cells
    for i in range(K):
        for j in range(K):
            val = mat[i, j]
            if val > 0.005:
                color = "white" if val > 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color=color)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def plot_bout_duration_distribution(
    bout_stats: "Dict[int, BoutStatistics]",
    bouts: Optional[List] = None,
    figsize: tuple = (12, 6),
) -> "matplotlib.figure.Figure":
    """Plot bout duration distributions per cluster as a box plot.

    If *bouts* (list of :class:`BoutInfo`) is provided, a full box plot is
    drawn.  Otherwise, a simplified bar chart with error bars is used.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sorted_ids = sorted(bout_stats.keys())
    names = [bout_stats[cid].cluster_name for cid in sorted_ids]

    if bouts is not None:
        from collections import defaultdict
        grouped: Dict[int, list] = defaultdict(list)
        for b in bouts:
            grouped[b.cluster_id].append(b.duration_seconds)
        data = [grouped.get(cid, []) for cid in sorted_ids]

        fig, ax = plt.subplots(figsize=figsize)
        bp = ax.boxplot(data, labels=names, patch_artist=True)
        colors = _get_cluster_colors({}, len(sorted_ids))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    else:
        means = [bout_stats[cid].mean_duration_s for cid in sorted_ids]
        stds = [bout_stats[cid].std_duration_s for cid in sorted_ids]
        fig, ax = plt.subplots(figsize=figsize)
        colors = _get_cluster_colors({}, len(sorted_ids))
        ax.bar(names, means, yerr=stds, color=colors, alpha=0.7, capsize=4)

    ax.set_ylabel("Duration (s)")
    ax.set_title("Bout Duration Distribution")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


def plot_bout_frequency_bar(
    bout_stats: "Dict[int, BoutStatistics]",
    figsize: tuple = (10, 5),
) -> "matplotlib.figure.Figure":
    """Plot behaviour frequency bar chart (fraction of total time)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sorted_ids = sorted(bout_stats.keys())
    names = [bout_stats[cid].cluster_name for cid in sorted_ids]
    freqs = [bout_stats[cid].frequency for cid in sorted_ids]
    colors = _get_cluster_colors({}, len(sorted_ids))

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(names, freqs, color=colors, alpha=0.8)

    # Value labels
    for bar, freq in zip(bars, freqs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{freq:.1%}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_ylabel("Frequency (fraction of time)")
    ax.set_title("Behaviour Frequency")
    ax.set_ylim(0, min(1.0, max(freqs) * 1.3) if freqs else 1.0)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig
