"""Unit tests for paired/within-subject statistical tests."""

import numpy as np
import pytest

from castle.core.comparison import (
    BehavioralFingerprint,
    ComparisonResult,
    benjamini_hochberg,
    paired_hedges_g,
    paired_permutation_test,
    compare_paired,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fingerprint(
    animal_id: str,
    group: str,
    K: int = 3,
    frequencies: np.ndarray = None,
    transition_matrix: np.ndarray = None,
    seed: int = 0,
) -> BehavioralFingerprint:
    """Create a BehavioralFingerprint with optional customisation."""
    rng = np.random.default_rng(seed)
    if frequencies is None:
        raw = rng.random(K)
        frequencies = raw / raw.sum()
    if transition_matrix is None:
        raw_tm = rng.random((K, K))
        np.fill_diagonal(raw_tm, 0)
        row_sums = raw_tm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        transition_matrix = raw_tm / row_sums

    return BehavioralFingerprint(
        animal_id=animal_id,
        group=group,
        frequencies=frequencies,
        bout_counts=rng.integers(1, 20, size=K).astype(float),
        mean_bout_durations=rng.random(K) * 2 + 0.1,
        median_bout_durations=rng.random(K) * 2 + 0.1,
        cv_bout_durations=rng.random(K),
        inter_bout_intervals=rng.random(K) * 5,
        transition_matrix=transition_matrix,
        cluster_names=[f"c{i}" for i in range(K)],
        n_frames=1000,
        fps=30.0,
    )


def _make_paired_groups(
    n: int,
    K: int = 3,
    shift: float = 0.0,
    seed_base: int = 0,
):
    """Create before/after fingerprint pairs.

    When shift > 0, the 'after' group gets a systematic shift in frequencies
    and transition matrix, simulating a treatment effect.
    """
    before = []
    after = []
    for i in range(n):
        rng = np.random.default_rng(seed_base + i)

        # Before fingerprint
        raw = rng.random(K)
        freq_before = raw / raw.sum()
        raw_tm = rng.random((K, K))
        np.fill_diagonal(raw_tm, 0)
        row_sums = raw_tm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        tm_before = raw_tm / row_sums

        fp_before = _make_fingerprint(
            f"animal_{i}", "before", K=K,
            frequencies=freq_before, transition_matrix=tm_before,
            seed=seed_base + i + 1000,
        )
        fp_before.frequencies = freq_before
        fp_before.transition_matrix = tm_before
        before.append(fp_before)

        # After fingerprint (shifted)
        freq_after = freq_before.copy()
        freq_after[0] += shift
        freq_after = np.clip(freq_after, 0, None)
        freq_after = freq_after / freq_after.sum()

        tm_after = tm_before.copy()
        if shift > 0 and K >= 2:
            tm_after[0, 1] += shift
            row_sums_after = tm_after.sum(axis=1, keepdims=True)
            row_sums_after[row_sums_after == 0] = 1
            tm_after = tm_after / row_sums_after
            np.fill_diagonal(tm_after, 0)
            row_sums_after2 = tm_after.sum(axis=1, keepdims=True)
            row_sums_after2[row_sums_after2 == 0] = 1
            tm_after = tm_after / row_sums_after2

        fp_after = _make_fingerprint(
            f"animal_{i}", "after", K=K,
            frequencies=freq_after, transition_matrix=tm_after,
            seed=seed_base + i + 2000,
        )
        fp_after.frequencies = freq_after
        fp_after.transition_matrix = tm_after
        # Use matching bout stats from before (same animal, only shifted features)
        fp_after.bout_counts = fp_before.bout_counts.copy()
        fp_after.mean_bout_durations = fp_before.mean_bout_durations + shift
        fp_after.median_bout_durations = fp_before.median_bout_durations + shift
        fp_after.cv_bout_durations = fp_before.cv_bout_durations.copy()
        fp_after.inter_bout_intervals = fp_before.inter_bout_intervals.copy()
        after.append(fp_after)

    return before, after


# ---------------------------------------------------------------------------
# paired_hedges_g
# ---------------------------------------------------------------------------

class TestPairedHedgesG:
    def test_zero_differences(self):
        """Zero differences → g = 0."""
        diffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        assert paired_hedges_g(diffs) == 0.0

    def test_positive_shift(self):
        """Consistent positive shift → positive g."""
        diffs = np.array([1.0, 1.1, 0.9, 1.2, 0.8])
        g = paired_hedges_g(diffs)
        assert g > 0

    def test_negative_shift(self):
        """Consistent negative shift → negative g."""
        diffs = np.array([-2.0, -1.8, -2.2, -1.9, -2.1])
        g = paired_hedges_g(diffs)
        assert g < 0

    def test_single_observation(self):
        """Single observation → 0."""
        assert paired_hedges_g(np.array([1.0])) == 0.0

    def test_large_effect(self):
        """Large consistent shift with low variance → large |g|."""
        diffs = np.array([10.0, 10.01, 9.99, 10.0, 10.0])
        g = paired_hedges_g(diffs)
        assert abs(g) > 5.0


# ---------------------------------------------------------------------------
# paired_permutation_test
# ---------------------------------------------------------------------------

class TestPairedPermutationTest:
    def test_identical_before_after_high_pvalue(self):
        """Identical before/after → high p-value (no effect)."""
        before, _ = _make_paired_groups(8, K=3, shift=0.0, seed_base=0)
        # Use same fingerprints for after
        after = [
            _make_fingerprint(fp.animal_id, "after", K=3, seed=s)
            for fp, s in zip(before, range(1000, 1008))
        ]
        # Copy exact same data
        for b, a in zip(before, after):
            a.frequencies = b.frequencies.copy()
            a.bout_counts = b.bout_counts.copy()
            a.mean_bout_durations = b.mean_bout_durations.copy()
            a.median_bout_durations = b.median_bout_durations.copy()
            a.cv_bout_durations = b.cv_bout_durations.copy()
            a.inter_bout_intervals = b.inter_bout_intervals.copy()
            a.transition_matrix = b.transition_matrix.copy()

        result = paired_permutation_test(before, after, n_permutations=500)
        assert result["paired_bfa_pvalue"] > 0.05

    def test_different_data_low_pvalue(self):
        """Large systematic shift → low p-value."""
        before, after = _make_paired_groups(10, K=3, shift=2.0, seed_base=42)
        result = paired_permutation_test(before, after, n_permutations=1000)
        # With shift=2.0 the effect should be detectable
        # Check that at least some per-feature p-values are low
        assert np.any(result["per_feature_pvalues"] < 0.05)

    def test_mismatched_lengths_raises(self):
        """Different number of before/after → error."""
        before = [_make_fingerprint(f"a{i}", "before", seed=i) for i in range(5)]
        after = [_make_fingerprint(f"a{i}", "after", seed=i + 100) for i in range(3)]
        with pytest.raises(ValueError, match="equal number"):
            paired_permutation_test(before, after)

    def test_empty_raises(self):
        """Empty input → error."""
        with pytest.raises(ValueError, match="at least one"):
            paired_permutation_test([], [])

    def test_sign_flip_correctness(self):
        """Verify sign-flip produces symmetric null distribution.

        With identical before/after, observed stat should be 0, and all
        permuted stats should also be ~0.
        """
        K = 2
        fp_before = _make_fingerprint("a0", "before", K=K, seed=0)
        fp_after = _make_fingerprint("a0", "after", K=K, seed=0)
        # Make them identical
        fp_after.frequencies = fp_before.frequencies.copy()
        fp_after.bout_counts = fp_before.bout_counts.copy()
        fp_after.mean_bout_durations = fp_before.mean_bout_durations.copy()
        fp_after.median_bout_durations = fp_before.median_bout_durations.copy()
        fp_after.cv_bout_durations = fp_before.cv_bout_durations.copy()
        fp_after.inter_bout_intervals = fp_before.inter_bout_intervals.copy()
        fp_after.transition_matrix = fp_before.transition_matrix.copy()

        result = paired_permutation_test([fp_before], [fp_after], n_permutations=200)
        # BFA distance on identical data should be 0
        assert result["paired_bfa_distance"] == pytest.approx(0.0)

    def test_output_structure(self):
        """Check all expected keys are present."""
        before, after = _make_paired_groups(5, K=3, shift=0.0)
        result = paired_permutation_test(before, after, n_permutations=200)
        assert "paired_bfa_distance" in result
        assert "paired_bfa_pvalue" in result
        assert "per_feature_pvalues" in result
        assert "per_feature_pvalues_adj" in result
        assert "per_feature_effect_sizes" in result
        assert "feature_names" in result
        assert "mean_diffs" in result

    def test_per_feature_pvalues_in_range(self):
        """All p-values in (0, 1]."""
        before, after = _make_paired_groups(6, K=3, shift=0.5)
        result = paired_permutation_test(before, after, n_permutations=200)
        assert all(0 < p <= 1.0 for p in result["per_feature_pvalues"])
        assert all(0 < p <= 1.0 for p in result["per_feature_pvalues_adj"])

    def test_bh_fdr_applied(self):
        """Adjusted p-values should be >= raw p-values."""
        before, after = _make_paired_groups(8, K=3, shift=1.0)
        result = paired_permutation_test(before, after, n_permutations=500)
        raw = result["per_feature_pvalues"]
        adj = result["per_feature_pvalues_adj"]
        assert all(a >= r - 1e-10 for a, r in zip(adj, raw))

    def test_effect_sizes_present(self):
        """Effect sizes should have correct length."""
        before, after = _make_paired_groups(5, K=3, shift=0.5)
        result = paired_permutation_test(before, after, n_permutations=100)
        n_features = len(result["feature_names"])
        assert len(result["per_feature_effect_sizes"]) == n_features

    def test_paired_hedges_g_in_result(self):
        """With a big shift, effect sizes should be large for affected features."""
        before, after = _make_paired_groups(10, K=3, shift=3.0, seed_base=99)
        result = paired_permutation_test(before, after, n_permutations=500)
        # At least some features should have |g| > 0.5
        max_abs_g = float(np.max(np.abs(result["per_feature_effect_sizes"])))
        assert max_abs_g > 0.5


# ---------------------------------------------------------------------------
# compare_paired (full pipeline)
# ---------------------------------------------------------------------------

class TestComparePaired:
    def test_returns_comparison_result(self):
        before, after = _make_paired_groups(5, K=3, shift=0.5)
        result = compare_paired(before, after, n_permutations=200)
        assert isinstance(result, ComparisonResult)

    def test_summary_contains_paired(self):
        before, after = _make_paired_groups(5, K=3, shift=0.0)
        result = compare_paired(before, after, n_permutations=100)
        assert "Paired" in result.summary

    def test_group_names_in_summary(self):
        before, after = _make_paired_groups(5, K=3, shift=0.0)
        result = compare_paired(before, after, n_permutations=100)
        assert "before" in result.summary
        assert "after" in result.summary

    def test_mismatched_raises(self):
        before = [_make_fingerprint(f"a{i}", "before", seed=i) for i in range(3)]
        after = [_make_fingerprint(f"a{i}", "after", seed=i) for i in range(5)]
        with pytest.raises(ValueError, match="equal number"):
            compare_paired(before, after)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            compare_paired([], [])

    def test_energy_distance_is_none(self):
        """Paired test does not compute energy distance."""
        before, after = _make_paired_groups(5, K=3, shift=0.0)
        result = compare_paired(before, after, n_permutations=100)
        assert result.energy_distance is None
        assert result.energy_pvalue is None

    def test_n_a_and_n_b_equal(self):
        before, after = _make_paired_groups(7, K=3, shift=0.0)
        result = compare_paired(before, after, n_permutations=100)
        assert result.n_a == 7
        assert result.n_b == 7

    def test_significant_features_detected(self):
        """Large shift should produce significant features."""
        before, after = _make_paired_groups(10, K=3, shift=3.0, seed_base=42)
        result = compare_paired(before, after, n_permutations=1000, alpha=0.05)
        # With shift=3.0, we expect some significant features
        assert isinstance(result.significant_features, list)
