"""Group comparison visualization.

Provides radar charts, transition heatmap diffs, volcano plots, and
forest plots for comparing behavioral patterns between groups.

All functions return a :class:`matplotlib.figure.Figure`.
"""

from __future__ import annotations

from typing import List, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import matplotlib
    import matplotlib.figure


def plot_fingerprint_radar(
    fingerprints_a: List,
    fingerprints_b: List,
    group_names: Tuple[str, str],
    figsize: tuple = (10, 10),
) -> "matplotlib.figure.Figure":
    """Radar chart comparing mean behavioral profiles between groups.

    Plots the mean frequency per cluster for each group on a polar plot
    with shaded SEM bands.

    Args:
        fingerprints_a: List of BehavioralFingerprint for group A.
        fingerprints_b: List of BehavioralFingerprint for group B.
        group_names: (group_a_name, group_b_name).
        figsize: Figure size.

    Returns:
        matplotlib Figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cluster_names = fingerprints_a[0].cluster_names
    K = len(cluster_names)

    # Compute mean and SEM of frequencies for each group
    freqs_a = np.array([fp.frequencies for fp in fingerprints_a])
    freqs_b = np.array([fp.frequencies for fp in fingerprints_b])

    mean_a = freqs_a.mean(axis=0)
    mean_b = freqs_b.mean(axis=0)
    sem_a = freqs_a.std(axis=0) / max(np.sqrt(len(fingerprints_a)), 1)
    sem_b = freqs_b.std(axis=0) / max(np.sqrt(len(fingerprints_b)), 1)

    # Angles
    angles = np.linspace(0, 2 * np.pi, K, endpoint=False).tolist()
    # Close the polygon
    angles += angles[:1]
    mean_a_c = np.concatenate([mean_a, mean_a[:1]])
    mean_b_c = np.concatenate([mean_b, mean_b[:1]])
    sem_a_c = np.concatenate([sem_a, sem_a[:1]])
    sem_b_c = np.concatenate([sem_b, sem_b[:1]])

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    ax.plot(angles, mean_a_c, "o-", linewidth=2, label=group_names[0], color="#1f77b4")
    ax.fill_between(angles, mean_a_c - sem_a_c, mean_a_c + sem_a_c, alpha=0.15, color="#1f77b4")
    ax.plot(angles, mean_b_c, "s-", linewidth=2, label=group_names[1], color="#ff7f0e")
    ax.fill_between(angles, mean_b_c - sem_b_c, mean_b_c + sem_b_c, alpha=0.15, color="#ff7f0e")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cluster_names, fontsize=9)
    ax.set_title("Behavioral Fingerprint Radar", fontsize=14, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    fig.tight_layout()
    return fig


def plot_transition_heatmap_diff(
    tm_a: np.ndarray,
    tm_b: np.ndarray,
    cluster_names: List[str],
    group_names: Tuple[str, str],
    figsize: tuple = (16, 5),
) -> "matplotlib.figure.Figure":
    """Side-by-side transition heatmaps + difference map.

    Three panels: Group A | Group B | (B - A) difference.

    Args:
        tm_a: K×K transition matrix for group A.
        tm_b: K×K transition matrix for group B.
        cluster_names: List of cluster names.
        group_names: (group_a_name, group_b_name).
        figsize: Figure size.

    Returns:
        matplotlib Figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    K = len(cluster_names)
    diff = tm_b - tm_a
    max_val = max(np.max(np.abs(tm_a)), np.max(np.abs(tm_b)), 1e-6)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Group A
    im0 = axes[0].imshow(tm_a, cmap="YlOrRd", vmin=0, vmax=max_val)
    axes[0].set_title(group_names[0])
    axes[0].set_xticks(range(K))
    axes[0].set_yticks(range(K))
    axes[0].set_xticklabels(cluster_names, rotation=45, ha="right", fontsize=7)
    axes[0].set_yticklabels(cluster_names, fontsize=7)
    axes[0].set_xlabel("To")
    axes[0].set_ylabel("From")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Group B
    im1 = axes[1].imshow(tm_b, cmap="YlOrRd", vmin=0, vmax=max_val)
    axes[1].set_title(group_names[1])
    axes[1].set_xticks(range(K))
    axes[1].set_yticks(range(K))
    axes[1].set_xticklabels(cluster_names, rotation=45, ha="right", fontsize=7)
    axes[1].set_yticklabels(cluster_names, fontsize=7)
    axes[1].set_xlabel("To")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Difference
    vabs = max(np.max(np.abs(diff)), 1e-6)
    im2 = axes[2].imshow(diff, cmap="RdBu_r", vmin=-vabs, vmax=vabs)
    axes[2].set_title(f"{group_names[1]} − {group_names[0]}")
    axes[2].set_xticks(range(K))
    axes[2].set_yticks(range(K))
    axes[2].set_xticklabels(cluster_names, rotation=45, ha="right", fontsize=7)
    axes[2].set_yticklabels(cluster_names, fontsize=7)
    axes[2].set_xlabel("To")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle("Transition Matrix Comparison", fontsize=13)
    fig.tight_layout()
    return fig


def plot_volcano(
    feature_names: List[str],
    effect_sizes: np.ndarray,
    pvalues_adj: np.ndarray,
    alpha: float = 0.05,
    figsize: tuple = (10, 8),
) -> "matplotlib.figure.Figure":
    """Volcano plot: effect size vs -log10(p-value).

    Significant features with |g| > 0.5 are labelled.

    Args:
        feature_names: Feature names.
        effect_sizes: Hedges' g values.
        pvalues_adj: BH-FDR adjusted p-values.
        alpha: Significance threshold.
        figsize: Figure size.

    Returns:
        matplotlib Figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    neg_log_p = -np.log10(np.clip(pvalues_adj, 1e-300, 1.0))
    significant = pvalues_adj < alpha

    fig, ax = plt.subplots(figsize=figsize)

    # Non-significant
    ns_mask = ~significant
    ax.scatter(
        effect_sizes[ns_mask],
        neg_log_p[ns_mask],
        c="grey",
        alpha=0.5,
        s=30,
        label="n.s.",
    )
    # Significant
    ax.scatter(
        effect_sizes[significant],
        neg_log_p[significant],
        c="red",
        alpha=0.8,
        s=50,
        label=f"p_adj < {alpha}",
    )

    # Threshold line
    ax.axhline(-np.log10(alpha), color="grey", linestyle="--", linewidth=0.8)
    ax.axvline(-0.5, color="grey", linestyle=":", linewidth=0.8)
    ax.axvline(0.5, color="grey", linestyle=":", linewidth=0.8)

    # Label significant features with large effect
    for i in range(len(feature_names)):
        if significant[i] and abs(effect_sizes[i]) > 0.5:
            ax.annotate(
                feature_names[i],
                (effect_sizes[i], neg_log_p[i]),
                fontsize=7,
                alpha=0.8,
                xytext=(5, 5),
                textcoords="offset points",
            )

    ax.set_xlabel("Effect Size (Hedges' g)")
    ax.set_ylabel("-log₁₀(adjusted p-value)")
    ax.set_title("Volcano Plot")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def plot_forest(
    feature_names: List[str],
    effect_sizes: np.ndarray,
    ci_lower: np.ndarray,
    ci_upper: np.ndarray,
    figsize: tuple = (8, 12),
    max_features: int = 30,
) -> "matplotlib.figure.Figure":
    """Forest plot: effect sizes with confidence intervals.

    Features are sorted by absolute effect size (largest at top).

    Args:
        feature_names: Feature names.
        effect_sizes: Hedges' g values.
        ci_lower: 95% CI lower bounds.
        ci_upper: 95% CI upper bounds.
        figsize: Figure size.
        max_features: Maximum number of features to show.

    Returns:
        matplotlib Figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Sort by absolute effect size (descending)
    order = np.argsort(np.abs(effect_sizes))[::-1][:max_features]

    names = [feature_names[i] for i in order]
    es = effect_sizes[order]
    lo = ci_lower[order]
    hi = ci_upper[order]

    n = len(names)
    y_pos = np.arange(n)

    fig, ax = plt.subplots(figsize=figsize)
    ax.axvline(0, color="grey", linestyle="--", linewidth=1)

    # Color by sign of effect
    colors = ["#d62728" if e > 0 else "#1f77b4" for e in es]

    for i in range(n):
        ax.plot([lo[i], hi[i]], [y_pos[i], y_pos[i]], color=colors[i], linewidth=1.5)
        ax.plot(es[i], y_pos[i], "o", color=colors[i], markersize=6)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Effect Size (Hedges' g)")
    ax.set_title("Forest Plot — Effect Sizes with 95% CI")
    fig.tight_layout()
    return fig
