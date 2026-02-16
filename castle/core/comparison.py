"""Group-level behavioral comparison.

Implements BehaviorFlow Analysis (BFA), behavioral fingerprinting,
and permutation-based statistical tests for comparing experimental groups.

Reference: von Ziegler et al., Nature Methods 2024 (BehaviorFlow)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class BehavioralFingerprint:
    """Per-animal behavioral summary vector.

    Combines bout statistics and transition probabilities into
    a single feature vector for group comparison.
    """

    animal_id: str
    group: str
    # Per-cluster statistics (K values each)
    frequencies: np.ndarray  # fraction of time in each behavior
    bout_counts: np.ndarray  # number of bouts per behavior
    mean_bout_durations: np.ndarray  # mean bout duration per behavior
    median_bout_durations: np.ndarray
    cv_bout_durations: np.ndarray  # coefficient of variation
    inter_bout_intervals: np.ndarray  # mean IBI per behavior
    # Transition matrix
    transition_matrix: np.ndarray  # K×K probability matrix
    # Metadata
    cluster_names: List[str]
    n_frames: int
    fps: float

    def to_feature_vector(self, include_transitions: bool = True) -> np.ndarray:
        """Convert to flat feature vector for multivariate analysis.

        Features order: frequencies (K) + bout_counts (K) +
        mean_bout_durations (K) + median_bout_durations (K) +
        cv_bout_durations (K) + inter_bout_intervals (K) +
        [transition_matrix (K*K)].

        NaN values are replaced with 0.0.
        """
        features = [
            self.frequencies,
            self.bout_counts,
            self.mean_bout_durations,
            self.median_bout_durations,
            self.cv_bout_durations,
            self.inter_bout_intervals,
        ]
        vec = np.concatenate(features).astype(np.float64)
        if include_transitions:
            vec = np.concatenate([vec, self.transition_matrix.flatten()])
        # Replace NaN with 0
        vec = np.nan_to_num(vec, nan=0.0)
        return vec

    def feature_names(self, include_transitions: bool = True) -> List[str]:
        """Get human-readable feature names."""
        names: List[str] = []
        prefixes = [
            "freq",
            "bout_count",
            "mean_dur",
            "median_dur",
            "cv_dur",
            "ibi",
        ]
        for prefix in prefixes:
            for cname in self.cluster_names:
                names.append(f"{prefix}_{cname}")
        if include_transitions:
            for from_name in self.cluster_names:
                for to_name in self.cluster_names:
                    names.append(f"trans_{from_name}_to_{to_name}")
        return names


@dataclass
class ComparisonResult:
    """Statistical comparison between two groups."""

    group_a_name: str
    group_b_name: str
    n_a: int
    n_b: int

    # Omnibus tests
    bfa_distance: float  # BFA Manhattan distance
    bfa_pvalue: float  # permutation p-value
    energy_distance: Optional[float] = None
    energy_pvalue: Optional[float] = None

    # Per-feature tests
    feature_names: List[str] = field(default_factory=list)
    feature_pvalues: Optional[np.ndarray] = None  # raw p-values
    feature_pvalues_adj: Optional[np.ndarray] = None  # BH-FDR adjusted
    feature_effect_sizes: Optional[np.ndarray] = None  # Hedges' g
    feature_ci_lower: Optional[np.ndarray] = None  # 95% CI lower
    feature_ci_upper: Optional[np.ndarray] = None  # 95% CI upper
    feature_means_a: Optional[np.ndarray] = None
    feature_means_b: Optional[np.ndarray] = None

    # Significant features
    significant_features: List[str] = field(default_factory=list)

    # Summary
    summary: str = ""


def compute_fingerprint(
    animal_id: str,
    group: str,
    cluster_labels: np.ndarray,
    fps: float = 30.0,
    cluster_names: Optional[Dict[int, str]] = None,
) -> BehavioralFingerprint:
    """Compute behavioral fingerprint for one animal.

    Uses ethogram engine to extract bout stats and transition matrix,
    then packages into a BehavioralFingerprint.

    Args:
        animal_id: Identifier for this animal/video.
        group: Group label (e.g. "control", "treatment").
        cluster_labels: 1-D integer array of per-frame cluster assignments.
        fps: Frames per second.
        cluster_names: Optional mapping of cluster_id → human name.

    Returns:
        BehavioralFingerprint with all statistics computed.
    """
    from castle.core.ethogram import compute_ethogram

    labels = np.asarray(cluster_labels)
    ethogram = compute_ethogram(labels, fps=fps, cluster_names=cluster_names)

    # Use sorted cluster_ids for consistent ordering
    sorted_ids = sorted(ethogram.cluster_names.keys())
    K = len(sorted_ids)
    name_list = [ethogram.cluster_names[cid] for cid in sorted_ids]

    frequencies = np.zeros(K, dtype=np.float64)
    bout_counts_arr = np.zeros(K, dtype=np.float64)
    mean_durs = np.zeros(K, dtype=np.float64)
    median_durs = np.zeros(K, dtype=np.float64)
    cv_durs = np.zeros(K, dtype=np.float64)
    ibis = np.zeros(K, dtype=np.float64)

    for i, cid in enumerate(sorted_ids):
        if cid in ethogram.bout_stats:
            bs = ethogram.bout_stats[cid]
            frequencies[i] = bs.frequency
            bout_counts_arr[i] = bs.n_bouts
            mean_durs[i] = bs.mean_duration_s
            median_durs[i] = bs.median_duration_s
            cv_durs[i] = bs.cv_duration
            ibis[i] = bs.mean_inter_bout_interval_s

    return BehavioralFingerprint(
        animal_id=animal_id,
        group=group,
        frequencies=frequencies,
        bout_counts=bout_counts_arr,
        mean_bout_durations=mean_durs,
        median_bout_durations=median_durs,
        cv_bout_durations=cv_durs,
        inter_bout_intervals=ibis,
        transition_matrix=ethogram.transition_matrix.matrix,
        cluster_names=name_list,
        n_frames=ethogram.n_frames,
        fps=fps,
    )


def bfa_test(
    group_a: List[BehavioralFingerprint],
    group_b: List[BehavioralFingerprint],
    n_permutations: int = 10000,
    random_state: int = 42,
) -> Tuple[float, float]:
    """Behavioral Flow Analysis (BFA) omnibus test.

    Computes Manhattan distance between group-mean transition matrices,
    tests significance via permutation.

    Reference: BehaviorFlow (von Ziegler et al., Nature Methods 2024)

    Args:
        group_a: Fingerprints for group A.
        group_b: Fingerprints for group B.
        n_permutations: Number of permutations.
        random_state: Random seed for reproducibility.

    Returns:
        (distance, p_value)
    """
    trans_a = [fp.transition_matrix for fp in group_a]
    trans_b = [fp.transition_matrix for fp in group_b]
    all_trans = trans_a + trans_b
    n_a = len(trans_a)
    n_total = len(all_trans)

    # Stack for efficient computation
    all_stack = np.array(all_trans)  # (n_total, K, K)

    # Observed statistic
    mean_a = np.mean(all_stack[:n_a], axis=0)
    mean_b = np.mean(all_stack[n_a:], axis=0)
    observed = float(np.sum(np.abs(mean_a - mean_b)))

    # Permutation null distribution
    rng = np.random.default_rng(random_state)
    null_dist = np.empty(n_permutations)
    for i in range(n_permutations):
        perm = rng.permutation(n_total)
        m_a = np.mean(all_stack[perm[:n_a]], axis=0)
        m_b = np.mean(all_stack[perm[n_a:]], axis=0)
        null_dist[i] = float(np.sum(np.abs(m_a - m_b)))

    # p-value: fraction of permuted distances >= observed (with floor)
    p_value = float(np.mean(null_dist >= observed))
    p_value = max(p_value, 1.0 / (n_permutations + 1))

    return observed, p_value


def energy_distance_test(
    group_a: List[BehavioralFingerprint],
    group_b: List[BehavioralFingerprint],
    n_permutations: int = 10000,
    random_state: int = 42,
) -> Tuple[float, float]:
    """Energy distance multivariate two-sample test.

    Tests whether behavioral fingerprint distributions differ between groups.
    Energy distance = 2*mean(||a-b||) - mean(||a-a'||) - mean(||b-b'||)

    Args:
        group_a: Fingerprints for group A.
        group_b: Fingerprints for group B.
        n_permutations: Number of permutations.
        random_state: Random seed for reproducibility.

    Returns:
        (energy_distance, p_value)
    """
    from scipy.spatial.distance import cdist

    X = np.array([fp.to_feature_vector() for fp in group_a])
    Y = np.array([fp.to_feature_vector() for fp in group_b])
    n_a = len(X)
    Z = np.vstack([X, Y])

    def _energy_stat(idx_a: np.ndarray, idx_b: np.ndarray) -> float:
        A = Z[idx_a]
        B = Z[idx_b]
        cross = cdist(A, B).mean()
        within_a = cdist(A, A).mean() if len(A) > 1 else 0.0
        within_b = cdist(B, B).mean() if len(B) > 1 else 0.0
        return 2.0 * cross - within_a - within_b

    a_idx = np.arange(n_a)
    b_idx = np.arange(n_a, len(Z))
    observed = _energy_stat(a_idx, b_idx)

    rng = np.random.default_rng(random_state)
    null_dist = np.empty(n_permutations)
    for i in range(n_permutations):
        perm = rng.permutation(len(Z))
        null_dist[i] = _energy_stat(perm[:n_a], perm[n_a:])

    p_value = float(np.mean(null_dist >= observed))
    p_value = max(p_value, 1.0 / (n_permutations + 1))

    return float(observed), p_value


def hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    """Hedges' g effect size (bias-corrected Cohen's d).

    Args:
        a: Values for group A.
        b: Values for group B.

    Returns:
        Hedges' g value.
    """
    n_a = len(a)
    n_b = len(b)
    if n_a + n_b < 3:
        return 0.0
    mean_diff = float(np.mean(a) - np.mean(b))
    var_a = float(np.var(a, ddof=1)) if n_a > 1 else 0.0
    var_b = float(np.var(b, ddof=1)) if n_b > 1 else 0.0
    pooled_std = np.sqrt(
        ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    )
    if pooled_std < 1e-12:
        return 0.0
    d = mean_diff / pooled_std
    # Hedges' correction factor
    correction = 1.0 - 3.0 / (4.0 * (n_a + n_b - 2) - 1)
    return float(d * correction)


def hedges_g_ci(
    g: float, n_a: int, n_b: int, alpha: float = 0.05
) -> Tuple[float, float]:
    """Approximate confidence interval for Hedges' g.

    Uses the non-central t distribution approximation.

    Args:
        g: Hedges' g value.
        n_a: Size of group A.
        n_b: Size of group B.
        alpha: Significance level (default 0.05 for 95% CI).

    Returns:
        (ci_lower, ci_upper)
    """
    from scipy import stats as sp_stats

    se = np.sqrt((n_a + n_b) / (n_a * n_b) + g**2 / (2.0 * (n_a + n_b)))
    z = sp_stats.norm.ppf(1.0 - alpha / 2.0)
    return float(g - z * se), float(g + z * se)


def benjamini_hochberg(pvalues: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg FDR correction.

    Args:
        pvalues: Array of raw p-values.
        alpha: Significance threshold (not used in adjustment, kept for API).

    Returns:
        Array of adjusted p-values (same shape as input).
    """
    pvals = np.asarray(pvalues, dtype=np.float64)
    n = len(pvals)
    if n == 0:
        return np.array([], dtype=np.float64)

    # Sort p-values
    sorted_idx = np.argsort(pvals)
    sorted_pvals = pvals[sorted_idx]

    # BH adjustment: p_adj[i] = p[i] * n / rank
    ranks = np.arange(1, n + 1, dtype=np.float64)
    adjusted = sorted_pvals * n / ranks

    # Enforce monotonicity (step-up): go from largest to smallest
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]

    # Clip to [0, 1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    # Map back to original order
    result = np.empty(n, dtype=np.float64)
    result[sorted_idx] = adjusted
    return result


def permutation_test_per_feature(
    group_a: List[BehavioralFingerprint],
    group_b: List[BehavioralFingerprint],
    n_permutations: int = 10000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> dict:
    """Per-feature permutation tests with BH-FDR correction.

    Tests each behavioral feature independently, then corrects for
    multiple comparisons.

    Args:
        group_a: Fingerprints for group A.
        group_b: Fingerprints for group B.
        n_permutations: Number of permutations.
        alpha: Significance threshold for BH-FDR.
        random_state: Random seed.

    Returns:
        dict with keys:
            feature_names, pvalues, pvalues_adj, effect_sizes,
            ci_lower, ci_upper, means_a, means_b
    """
    vecs_a = np.array([fp.to_feature_vector() for fp in group_a])
    vecs_b = np.array([fp.to_feature_vector() for fp in group_b])
    n_a = vecs_a.shape[0]
    n_b = vecs_b.shape[0]
    n_features = vecs_a.shape[1]

    feature_names = group_a[0].feature_names()

    all_vecs = np.vstack([vecs_a, vecs_b])  # (n_total, n_features)
    n_total = n_a + n_b

    # Observed difference in means
    means_a = vecs_a.mean(axis=0)
    means_b = vecs_b.mean(axis=0)
    observed_diff = np.abs(means_a - means_b)

    # Permutation test: vectorised over features
    rng = np.random.default_rng(random_state)
    count_ge = np.zeros(n_features, dtype=np.float64)
    for _ in range(n_permutations):
        perm = rng.permutation(n_total)
        perm_a = all_vecs[perm[:n_a]]
        perm_b = all_vecs[perm[n_a:]]
        perm_diff = np.abs(perm_a.mean(axis=0) - perm_b.mean(axis=0))
        count_ge += (perm_diff >= observed_diff).astype(np.float64)

    pvalues = count_ge / n_permutations
    # Floor p-values
    pvalues = np.maximum(pvalues, 1.0 / (n_permutations + 1))

    # BH-FDR correction
    pvalues_adj = benjamini_hochberg(pvalues, alpha=alpha)

    # Effect sizes + CIs
    effect_sizes = np.zeros(n_features, dtype=np.float64)
    ci_lower = np.zeros(n_features, dtype=np.float64)
    ci_upper = np.zeros(n_features, dtype=np.float64)
    for j in range(n_features):
        g = hedges_g(vecs_a[:, j], vecs_b[:, j])
        effect_sizes[j] = g
        lo, hi = hedges_g_ci(g, n_a, n_b, alpha=alpha)
        ci_lower[j] = lo
        ci_upper[j] = hi

    return {
        "feature_names": feature_names,
        "pvalues": pvalues,
        "pvalues_adj": pvalues_adj,
        "effect_sizes": effect_sizes,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "means_a": means_a,
        "means_b": means_b,
    }


def compare_groups(
    group_a: List[BehavioralFingerprint],
    group_b: List[BehavioralFingerprint],
    n_permutations: int = 10000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> ComparisonResult:
    """Complete group comparison pipeline.

    1. BFA omnibus test
    2. Energy distance test
    3. Per-feature permutation tests + BH-FDR
    4. Hedges' g effect sizes + 95% CIs
    5. Summary generation

    Args:
        group_a: Fingerprints for group A.
        group_b: Fingerprints for group B.
        n_permutations: Number of permutations.
        alpha: Significance threshold.
        random_state: Random seed.

    Returns:
        ComparisonResult with all statistics.
    """
    if len(group_a) == 0 or len(group_b) == 0:
        raise ValueError("Both groups must have at least one animal.")

    group_a_name = group_a[0].group
    group_b_name = group_b[0].group

    # 1. BFA omnibus test
    bfa_dist, bfa_p = bfa_test(
        group_a, group_b, n_permutations=n_permutations, random_state=random_state
    )

    # 2. Energy distance test
    energy_dist, energy_p = energy_distance_test(
        group_a, group_b, n_permutations=n_permutations, random_state=random_state
    )

    # 3. Per-feature permutation tests
    per_feat = permutation_test_per_feature(
        group_a,
        group_b,
        n_permutations=n_permutations,
        alpha=alpha,
        random_state=random_state,
    )

    # 4. Identify significant features
    sig_mask = per_feat["pvalues_adj"] < alpha
    sig_features = [
        per_feat["feature_names"][i]
        for i in range(len(per_feat["feature_names"]))
        if sig_mask[i]
    ]

    # 5. Summary
    bfa_sig = "***" if bfa_p < 0.001 else "**" if bfa_p < 0.01 else "*" if bfa_p < 0.05 else "n.s."
    energy_sig = "***" if energy_p < 0.001 else "**" if energy_p < 0.01 else "*" if energy_p < 0.05 else "n.s."

    summary_lines = [
        f"=== Group Comparison: {group_a_name} vs {group_b_name} ===",
        f"Sample sizes: n_a={len(group_a)}, n_b={len(group_b)}",
        "",
        "--- Omnibus Tests ---",
        f"  BFA: distance={bfa_dist:.4f}, p={bfa_p:.4f} {bfa_sig}",
        f"  Energy: distance={energy_dist:.4f}, p={energy_p:.4f} {energy_sig}",
        "",
        f"--- Significant Features ({len(sig_features)}/{len(per_feat['feature_names'])}) ---",
    ]

    # Show top significant features sorted by absolute effect size
    if sig_features:
        sig_indices = [i for i in range(len(per_feat["feature_names"])) if sig_mask[i]]
        sig_sorted = sorted(sig_indices, key=lambda i: abs(per_feat["effect_sizes"][i]), reverse=True)
        for idx in sig_sorted[:15]:
            name = per_feat["feature_names"][idx]
            g = per_feat["effect_sizes"][idx]
            lo = per_feat["ci_lower"][idx]
            hi = per_feat["ci_upper"][idx]
            padj = per_feat["pvalues_adj"][idx]
            summary_lines.append(
                f"  {name}: g={g:+.3f} [{lo:+.3f}, {hi:+.3f}], p_adj={padj:.4f}"
            )
    else:
        summary_lines.append("  (none)")

    return ComparisonResult(
        group_a_name=group_a_name,
        group_b_name=group_b_name,
        n_a=len(group_a),
        n_b=len(group_b),
        bfa_distance=bfa_dist,
        bfa_pvalue=bfa_p,
        energy_distance=energy_dist,
        energy_pvalue=energy_p,
        feature_names=per_feat["feature_names"],
        feature_pvalues=per_feat["pvalues"],
        feature_pvalues_adj=per_feat["pvalues_adj"],
        feature_effect_sizes=per_feat["effect_sizes"],
        feature_ci_lower=per_feat["ci_lower"],
        feature_ci_upper=per_feat["ci_upper"],
        feature_means_a=per_feat["means_a"],
        feature_means_b=per_feat["means_b"],
        significant_features=sig_features,
        summary="\n".join(summary_lines),
    )
