"""Group-level behavioral comparison.

Implements BehaviorFlow Analysis (BFA), behavioral fingerprinting,
and permutation-based statistical tests for comparing experimental groups.

Reference: von Ziegler et al., Nature Methods 2024 (BehaviorFlow)
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Fingerprint output schema version (docs/behavior_data_contract.md C-6). Bumped
# to 2.0 when fingerprints became dimensioned by the GLOBAL cluster-id set
# (so every animal's vector has identical length), frequency switched to the
# valid-frames-only definition, and absent-behavior durations became NaN.
FINGERPRINT_SCHEMA_VERSION = "2.0"


def _perm_p_floor(n_total: int, n_a: int, n_permutations: int) -> float:
    """Smallest reportable permutation p-value.

    A permutation test over groups of size ``n_a`` / ``n_b`` has at most
    ``C(n_total, n_a)`` distinct labellings, so the true minimum achievable
    p-value is ``1 / C(n_total, n_a)`` — for small samples this is far larger
    than ``1 / (n_permutations + 1)`` and using the latter as the floor reports
    impossibly-significant p-values (e.g. n=1 per group can never be
    significant). The floor is the coarser (larger) of the two resolutions.
    """
    try:
        comb = math.comb(n_total, n_a)
    except ValueError:
        comb = 1
    comb = max(comb, 1)
    return max(1.0 / (n_permutations + 1), 1.0 / comb)


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
    # Global cluster-id order this vector is dimensioned by (contract C-6).
    cluster_id_order: List[int] = field(default_factory=list)
    schema_version: str = FINGERPRINT_SCHEMA_VERSION

    def to_feature_vector(
        self, include_transitions: bool = True, fill_nan: bool = True
    ) -> np.ndarray:
        """Convert to flat feature vector for multivariate analysis.

        Features order: frequencies (K) + bout_counts (K) +
        mean_bout_durations (K) + median_bout_durations (K) +
        cv_bout_durations (K) + inter_bout_intervals (K) +
        [transition_matrix (K*K)].

        Args:
            include_transitions: append the flattened K×K transition matrix.
            fill_nan: when True (default), NaN entries (absent-behavior
                durations) are replaced with 0.0 — required for the multivariate
                omnibus distances (BFA / energy). Set False for per-feature
                tests that must distinguish "absent" (NaN) from "0 seconds".
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
        if fill_nan:
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
    all_cluster_ids: Optional[List[int]] = None,
) -> BehavioralFingerprint:
    """Compute behavioral fingerprint for one animal.

    Uses the ethogram engine to extract bout stats and the transition matrix,
    then packages them into a fixed-length :class:`BehavioralFingerprint`.

    Args:
        animal_id: Identifier for this animal/video.
        group: Group label (e.g. "control", "treatment").
        cluster_labels: 1-D integer array of per-frame cluster assignments.
        fps: Frames per second.
        cluster_names: Optional mapping of cluster_id → human name.
        all_cluster_ids: The GLOBAL set of cluster ids shared across every
            animal being compared. **Pass this** so all fingerprints have the
            same dimension and the i-th feature means the same behavior for
            every animal (contract C-6). A behavior absent in this animal is
            padded — structural fields (frequency, bout_count) with 0, and
            undefined duration / IBI fields with NaN. When ``None`` the animal's
            own present clusters are used (back-compat; do not use for
            cross-animal comparison).

    Returns:
        BehavioralFingerprint with all statistics computed.
    """
    from castle.core.ethogram import compute_ethogram

    labels = np.asarray(cluster_labels)
    ethogram = compute_ethogram(labels, fps=fps, cluster_names=cluster_names)

    # Dimension by the GLOBAL cluster set when supplied, so every animal's
    # vector is the same length and aligned feature-for-feature.
    if all_cluster_ids is not None:
        sorted_ids = sorted(int(c) for c in all_cluster_ids)
    else:
        sorted_ids = sorted(ethogram.cluster_names.keys())
    K = len(sorted_ids)
    name_list = [
        ethogram.cluster_names.get(cid, (cluster_names or {}).get(cid, f"cluster_{cid}"))
        for cid in sorted_ids
    ]

    # Structural fields default to 0 (a real "did not occur"); duration/IBI
    # default to NaN ("undefined" for an absent behavior, not 0 seconds).
    frequencies = np.zeros(K, dtype=np.float64)
    bout_counts_arr = np.zeros(K, dtype=np.float64)
    mean_durs = np.full(K, np.nan, dtype=np.float64)
    median_durs = np.full(K, np.nan, dtype=np.float64)
    cv_durs = np.full(K, np.nan, dtype=np.float64)
    ibis = np.full(K, np.nan, dtype=np.float64)

    for i, cid in enumerate(sorted_ids):
        if cid in ethogram.bout_stats:
            bs = ethogram.bout_stats[cid]
            frequencies[i] = bs.frequency_valid_only
            bout_counts_arr[i] = bs.n_bouts
            mean_durs[i] = bs.mean_duration_s
            median_durs[i] = bs.median_duration_s
            cv_durs[i] = bs.cv_duration
            ibis[i] = bs.mean_inter_bout_interval_s

    # Map the ethogram's (present-cluster) transition matrix into the global
    # K×K, zero-padding rows/cols for behaviors absent in this animal.
    global_idx = {cid: i for i, cid in enumerate(sorted_ids)}
    tm = ethogram.transition_matrix
    trans = np.zeros((K, K), dtype=np.float64)
    for ai, a_cid in enumerate(tm.cluster_ids):
        for aj, b_cid in enumerate(tm.cluster_ids):
            if a_cid in global_idx and b_cid in global_idx:
                trans[global_idx[a_cid], global_idx[b_cid]] = tm.matrix[ai, aj]

    return BehavioralFingerprint(
        animal_id=animal_id,
        group=group,
        frequencies=frequencies,
        bout_counts=bout_counts_arr,
        mean_bout_durations=mean_durs,
        median_bout_durations=median_durs,
        cv_bout_durations=cv_durs,
        inter_bout_intervals=ibis,
        transition_matrix=trans,
        cluster_names=name_list,
        n_frames=ethogram.n_frames,
        fps=fps,
        cluster_id_order=list(sorted_ids),
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
    p_value = max(p_value, _perm_p_floor(n_total, n_a, n_permutations))

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
    # PERF-04: route through the GPU-aware helper so large fingerprint
    # sets benefit from torch.cdist on CUDA; CPU path keeps scipy.
    from castle.utils.distance import pairwise_distance

    X = np.array([fp.to_feature_vector() for fp in group_a])
    Y = np.array([fp.to_feature_vector() for fp in group_b])
    n_a = len(X)
    Z = np.vstack([X, Y])
    n_total = len(Z)

    # PERF-04: compute the full N×N pairwise-distance matrix ONCE (a single
    # GPU/scipy call), then every permutation's energy statistic is plain numpy
    # indexing into it — avoids ~3*n_permutations tiny pairwise_distance calls
    # (each a host↔device round-trip on CUDA). Numerically identical: each
    # sub-block mean equals pairwise_distance(subset, subset).mean(), diagonal
    # zeros included exactly as before.
    D = np.asarray(pairwise_distance(Z, Z))

    def _energy_stat(idx_a: np.ndarray, idx_b: np.ndarray) -> float:
        cross = D[np.ix_(idx_a, idx_b)].mean()
        within_a = D[np.ix_(idx_a, idx_a)].mean() if len(idx_a) > 1 else 0.0
        within_b = D[np.ix_(idx_b, idx_b)].mean() if len(idx_b) > 1 else 0.0
        return 2.0 * cross - within_a - within_b

    a_idx = np.arange(n_a)
    b_idx = np.arange(n_a, n_total)
    observed = _energy_stat(a_idx, b_idx)

    rng = np.random.default_rng(random_state)
    null_dist = np.empty(n_permutations)
    for i in range(n_permutations):
        perm = rng.permutation(n_total)
        null_dist[i] = _energy_stat(perm[:n_a], perm[n_a:])

    p_value = float(np.mean(null_dist >= observed))
    p_value = max(p_value, _perm_p_floor(n_total, n_a, n_permutations))

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
    # Keep NaN (do not 0-fill): an absent-behavior duration is undefined, not
    # 0 seconds, so only animals that exhibited the behavior contribute to that
    # feature's mean / effect size (contract C-6 missing_duration_policy=NaN).
    vecs_a = np.array([fp.to_feature_vector(fill_nan=False) for fp in group_a])
    vecs_b = np.array([fp.to_feature_vector(fill_nan=False) for fp in group_b])
    n_a = vecs_a.shape[0]
    n_b = vecs_b.shape[0]
    n_features = vecs_a.shape[1]

    feature_names = group_a[0].feature_names()

    all_vecs = np.vstack([vecs_a, vecs_b])  # (n_total, n_features)
    n_total = n_a + n_b

    # NaN-aware means + permutation. A permutation can land every animal that
    # exhibits a behavior in one group, leaving the other group's slice all-NaN
    # → np.nanmean emits "Mean of empty slice"; the result (NaN) is correct and
    # handled by the `untestable` mask, so silence the cosmetic warning.
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        means_a = np.nanmean(vecs_a, axis=0)
        means_b = np.nanmean(vecs_b, axis=0)
        observed_diff = np.abs(means_a - means_b)
        # Features with no valid observation in a group are not testable.
        untestable = ~np.isfinite(observed_diff)

        rng = np.random.default_rng(random_state)
        count_ge = np.zeros(n_features, dtype=np.float64)
        for _ in range(n_permutations):
            perm = rng.permutation(n_total)
            perm_a = np.nanmean(all_vecs[perm[:n_a]], axis=0)
            perm_b = np.nanmean(all_vecs[perm[n_a:]], axis=0)
            perm_diff = np.abs(perm_a - perm_b)
            # NaN >= x is False — harmless for untestable features (set to 1 below).
            count_ge += (perm_diff >= observed_diff).astype(np.float64)

    pvalues = count_ge / n_permutations
    # Floor at the combinatorially-achievable minimum (small-n honesty).
    pvalues = np.maximum(pvalues, _perm_p_floor(n_total, n_a, n_permutations))
    pvalues[untestable] = 1.0

    # BH-FDR correction
    pvalues_adj = benjamini_hochberg(pvalues, alpha=alpha)

    # Effect sizes + CIs (per feature, over animals with a valid value).
    effect_sizes = np.zeros(n_features, dtype=np.float64)
    ci_lower = np.zeros(n_features, dtype=np.float64)
    ci_upper = np.zeros(n_features, dtype=np.float64)
    for j in range(n_features):
        aj = vecs_a[:, j]
        bj = vecs_b[:, j]
        aj = aj[np.isfinite(aj)]
        bj = bj[np.isfinite(bj)]
        if len(aj) == 0 or len(bj) == 0:
            continue  # leave effect 0 / CI (0, 0) — not estimable
        g = hedges_g(aj, bj)
        effect_sizes[j] = g
        lo, hi = hedges_g_ci(g, len(aj), len(bj), alpha=alpha)
        ci_lower[j] = lo
        ci_upper[j] = hi

    return {
        "feature_names": feature_names,
        "pvalues": pvalues,
        "pvalues_adj": pvalues_adj,
        "effect_sizes": effect_sizes,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "means_a": np.nan_to_num(means_a, nan=0.0),
        "means_b": np.nan_to_num(means_b, nan=0.0),
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

    # Fail loud with a clear message if fingerprints are not dimension-aligned
    # (build them with a shared all_cluster_ids — contract C-6).
    lengths = {len(fp.to_feature_vector()) for fp in (*group_a, *group_b)}
    if len(lengths) > 1:
        from castle.core.types import CastleDataError
        raise CastleDataError(
            "Fingerprints have mismatched feature-vector lengths "
            f"({sorted(lengths)}); build every animal's fingerprint with the "
            "same all_cluster_ids so features align (contract C-6)."
        )

    group_a_name = group_a[0].group
    group_b_name = group_b[0].group

    # Small samples cannot reach significance in a permutation test; the
    # p-value floor reflects this, but flag it explicitly too.
    small_sample = len(group_a) < 2 or len(group_b) < 2

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
    ]
    if small_sample:
        summary_lines.append(
            "  ⚠️  WARNING: a group has < 2 animals — a permutation test cannot "
            "reach significance (p is floored at the combinatorial minimum) and "
            "effect-size CIs are unreliable. Interpret with caution."
        )
    summary_lines += [
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


# ---------------------------------------------------------------------------
# Paired / within-subject tests
# ---------------------------------------------------------------------------


def paired_hedges_g(diffs: np.ndarray) -> float:
    """Hedges' g for paired differences (one-sample effect size).

    Computes g = mean(diffs) / SD(diffs) with Hedges' bias correction.

    Args:
        diffs: Array of paired differences.

    Returns:
        Hedges' g value.
    """
    n = len(diffs)
    if n < 2:
        return 0.0
    mean_d = float(np.mean(diffs))
    sd_d = float(np.std(diffs, ddof=1))
    if sd_d < 1e-12:
        return 0.0
    d = mean_d / sd_d
    # Hedges' correction factor
    correction = 1.0 - 3.0 / (4.0 * (n - 1) - 1)
    return float(d * correction)


def paired_permutation_test(
    fingerprints_before: List[BehavioralFingerprint],
    fingerprints_after: List[BehavioralFingerprint],
    n_permutations: int = 10000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> dict:
    """Paired permutation test for within-subject designs.

    Each animal is measured twice (e.g., pre-drug vs post-drug).
    Tests whether the paired differences are significantly different from zero.

    For each permutation: randomly flip the sign of each animal's difference
    (This is the paired permutation test — respects within-subject structure).

    Args:
        fingerprints_before: Pre-treatment fingerprints (one per animal).
        fingerprints_after: Post-treatment fingerprints (same animals, same order).
        n_permutations: Number of permutations.
        alpha: Significance threshold for BH-FDR.
        random_state: Random seed for reproducibility.

    Returns:
        dict with:
            - paired_bfa_distance: BFA distance on paired differences
            - paired_bfa_pvalue: permutation p-value
            - per_feature_pvalues: per-feature paired permutation tests
            - per_feature_pvalues_adj: BH-FDR corrected
            - per_feature_effect_sizes: paired Hedges' g
            - feature_names: feature name list
            - mean_diffs: mean of paired differences per feature
    """
    n = len(fingerprints_before)
    if n != len(fingerprints_after):
        raise ValueError(
            f"Paired test requires equal number of before ({n}) and after "
            f"({len(fingerprints_after)}) fingerprints."
        )
    if n == 0:
        raise ValueError("Need at least one pair of fingerprints.")

    rng = np.random.default_rng(random_state)

    # --- Transition-matrix-based BFA paired test ---
    trans_before = np.array([fp.transition_matrix for fp in fingerprints_before])
    trans_after = np.array([fp.transition_matrix for fp in fingerprints_after])
    trans_diffs = trans_after - trans_before  # (n, K, K)

    observed_bfa = float(np.sum(np.abs(np.mean(trans_diffs, axis=0))))

    null_bfa = np.empty(n_permutations)
    for i in range(n_permutations):
        signs = rng.choice([-1, 1], size=n)
        flipped = trans_diffs * signs[:, None, None]
        null_bfa[i] = float(np.sum(np.abs(np.mean(flipped, axis=0))))
    bfa_pvalue = float(np.mean(null_bfa >= observed_bfa))
    bfa_pvalue = max(bfa_pvalue, 1.0 / (n_permutations + 1))

    # --- Per-feature paired permutation test (NaN-aware, contract C-6) ---
    # Build vectors WITHOUT 0-filling so an absent behaviour's duration/IBI stays
    # NaN ("undefined", not 0 seconds). 0-filling would treat "behaviour absent at
    # one timepoint" as a 0-second observation and bias the paired mean-difference,
    # the sign-flip null, and paired Hedges' g — matching the unpaired path which
    # already keeps NaN. A pair is dropped per-feature when either side is NaN.
    vecs_before = np.array([fp.to_feature_vector(fill_nan=False) for fp in fingerprints_before])
    vecs_after = np.array([fp.to_feature_vector(fill_nan=False) for fp in fingerprints_after])
    diffs = vecs_after - vecs_before  # (n, n_features); NaN where either side absent
    n_features = diffs.shape[1]
    feature_names = fingerprints_before[0].feature_names()

    finite = np.isfinite(diffs)            # valid-pair mask per (animal, feature)
    n_valid = finite.sum(axis=0)           # valid pairs per feature
    testable = n_valid >= 2                # need >= 2 valid pairs to test
    denom = np.maximum(n_valid, 1)         # avoid /0 for untestable features

    # Mean over valid pairs only (NaN contributes 0 to the sum, excluded from count).
    mean_diffs = np.where(finite, diffs, 0.0).sum(axis=0) / denom
    observed_mean_abs = np.abs(mean_diffs)

    count_ge = np.zeros(n_features, dtype=np.float64)
    for _ in range(n_permutations):
        signs = rng.choice([-1, 1], size=n)
        flipped = np.where(finite, diffs * signs[:, None], 0.0)
        perm_mean_abs = np.abs(flipped.sum(axis=0) / denom)
        count_ge += (perm_mean_abs >= observed_mean_abs).astype(np.float64)

    pvalues = count_ge / n_permutations
    pvalues = np.maximum(pvalues, 1.0 / (n_permutations + 1))
    pvalues = np.where(testable, pvalues, 1.0)  # untestable features → p = 1.0

    # BH-FDR correction
    pvalues_adj = benjamini_hochberg(pvalues, alpha=alpha)

    # Paired Hedges' g per feature, over that feature's valid pairs only.
    effect_sizes = np.zeros(n_features, dtype=np.float64)
    for j in range(n_features):
        col = diffs[:, j][finite[:, j]]
        effect_sizes[j] = paired_hedges_g(col) if col.size >= 2 else 0.0

    mean_diffs = np.where(testable, mean_diffs, 0.0)

    return {
        "paired_bfa_distance": observed_bfa,
        "paired_bfa_pvalue": bfa_pvalue,
        "per_feature_pvalues": pvalues,
        "per_feature_pvalues_adj": pvalues_adj,
        "per_feature_effect_sizes": effect_sizes,
        "feature_names": feature_names,
        "mean_diffs": mean_diffs,
    }


def compare_paired(
    fingerprints_before: List[BehavioralFingerprint],
    fingerprints_after: List[BehavioralFingerprint],
    n_permutations: int = 10000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> ComparisonResult:
    """Complete paired comparison pipeline.

    1. Paired BFA omnibus test (sign-flip on transition differences)
    2. Per-feature paired permutation tests + BH-FDR
    3. Paired Hedges' g effect sizes
    4. Summary generation

    Args:
        fingerprints_before: Pre-treatment fingerprints (one per animal).
        fingerprints_after: Post-treatment fingerprints (same animals, same order).
        n_permutations: Number of permutations.
        alpha: Significance threshold.
        random_state: Random seed.

    Returns:
        ComparisonResult with all paired statistics.
    """
    n = len(fingerprints_before)
    if n != len(fingerprints_after):
        raise ValueError(
            f"Paired test requires equal number of before ({n}) and after "
            f"({len(fingerprints_after)}) fingerprints."
        )
    if n == 0:
        raise ValueError("Need at least one pair of fingerprints.")

    # Fail loud with a clear message if fingerprints are not dimension-aligned
    # (build them with a shared all_cluster_ids — contract C-6). Without this the
    # downstream np.array(...) over ragged vectors/matrices raises an opaque numpy
    # "inhomogeneous shape" ValueError.
    lengths = {len(fp.to_feature_vector()) for fp in (*fingerprints_before, *fingerprints_after)}
    if len(lengths) > 1:
        from castle.core.types import CastleDataError
        raise CastleDataError(
            "Fingerprints have mismatched feature-vector lengths "
            f"({sorted(lengths)}); build every animal's fingerprint with the "
            "same all_cluster_ids so features align (contract C-6)."
        )

    group_before_name = fingerprints_before[0].group
    group_after_name = fingerprints_after[0].group

    # Run paired permutation test
    paired_result = paired_permutation_test(
        fingerprints_before,
        fingerprints_after,
        n_permutations=n_permutations,
        alpha=alpha,
        random_state=random_state,
    )

    # Identify significant features
    sig_mask = paired_result["per_feature_pvalues_adj"] < alpha
    feature_names = paired_result["feature_names"]
    sig_features = [
        feature_names[i] for i in range(len(feature_names)) if sig_mask[i]
    ]

    # Summary
    bfa_p = paired_result["paired_bfa_pvalue"]
    bfa_sig = "***" if bfa_p < 0.001 else "**" if bfa_p < 0.01 else "*" if bfa_p < 0.05 else "n.s."

    summary_lines = [
        f"=== Paired Comparison: {group_before_name} vs {group_after_name} ===",
        f"Paired samples: n={n}",
        "",
        "--- Omnibus Tests ---",
        f"  Paired BFA: distance={paired_result['paired_bfa_distance']:.4f}, p={bfa_p:.4f} {bfa_sig}",
        "",
        f"--- Significant Features ({len(sig_features)}/{len(feature_names)}) ---",
    ]

    if sig_features:
        sig_indices = [i for i in range(len(feature_names)) if sig_mask[i]]
        sig_sorted = sorted(
            sig_indices,
            key=lambda i: abs(paired_result["per_feature_effect_sizes"][i]),
            reverse=True,
        )
        for idx in sig_sorted[:15]:
            name = feature_names[idx]
            g = paired_result["per_feature_effect_sizes"][idx]
            padj = paired_result["per_feature_pvalues_adj"][idx]
            summary_lines.append(f"  {name}: g={g:+.3f}, p_adj={padj:.4f}")
    else:
        summary_lines.append("  (none)")

    return ComparisonResult(
        group_a_name=group_before_name,
        group_b_name=group_after_name,
        n_a=n,
        n_b=n,
        bfa_distance=paired_result["paired_bfa_distance"],
        bfa_pvalue=bfa_p,
        energy_distance=None,
        energy_pvalue=None,
        feature_names=feature_names,
        feature_pvalues=paired_result["per_feature_pvalues"],
        feature_pvalues_adj=paired_result["per_feature_pvalues_adj"],
        feature_effect_sizes=paired_result["per_feature_effect_sizes"],
        feature_ci_lower=None,
        feature_ci_upper=None,
        feature_means_a=None,
        feature_means_b=None,
        significant_features=sig_features,
        summary="\n".join(summary_lines),
    )
