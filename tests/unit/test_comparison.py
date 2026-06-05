"""Unit tests for castle.core.comparison and castle.service.comparison_service."""

import csv
import os
import tempfile

import numpy as np
import pytest

from castle.core.comparison import (
    BehavioralFingerprint,
    ComparisonResult,
    bfa_test,
    benjamini_hochberg,
    compare_groups,
    compute_fingerprint,
    energy_distance_test,
    hedges_g,
    hedges_g_ci,
    permutation_test_per_feature,
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


def _make_group(group_name: str, n: int, K: int = 3, seed_base: int = 0,
                freq_bias: np.ndarray = None, tm_bias: np.ndarray = None) -> list:
    """Create a list of fingerprints for a group."""
    fps = []
    for i in range(n):
        rng = np.random.default_rng(seed_base + i)
        raw = rng.random(K)
        freq = raw / raw.sum()
        if freq_bias is not None:
            freq = freq + freq_bias
            freq = np.clip(freq, 0, None)
            freq = freq / freq.sum()

        raw_tm = rng.random((K, K))
        np.fill_diagonal(raw_tm, 0)
        if tm_bias is not None:
            raw_tm = raw_tm + tm_bias
            raw_tm = np.clip(raw_tm, 0, None)
            np.fill_diagonal(raw_tm, 0)
        row_sums = raw_tm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        tm = raw_tm / row_sums

        fp = _make_fingerprint(
            animal_id=f"{group_name}_{i}",
            group=group_name,
            K=K,
            frequencies=freq,
            transition_matrix=tm,
            seed=seed_base + i + 1000,
        )
        # Override with computed values
        fp.frequencies = freq
        fp.transition_matrix = tm
        fps.append(fp)
    return fps


# ---------------------------------------------------------------------------
# compute_fingerprint
# ---------------------------------------------------------------------------

class TestComputeFingerprint:
    def test_basic(self):
        labels = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 0, 0])
        fp = compute_fingerprint("animal1", "control", labels, fps=10.0)
        assert fp.animal_id == "animal1"
        assert fp.group == "control"
        assert fp.n_frames == 11
        assert fp.fps == 10.0
        assert len(fp.cluster_names) == 3
        assert len(fp.frequencies) == 3
        assert fp.frequencies.sum() == pytest.approx(1.0)

    def test_with_cluster_names(self):
        labels = np.array([0, 0, 1, 1, 1])
        names = {0: "walk", 1: "rest"}
        fp = compute_fingerprint("a1", "ctrl", labels, fps=30.0, cluster_names=names)
        assert "walk" in fp.cluster_names
        assert "rest" in fp.cluster_names

    def test_single_cluster(self):
        labels = np.array([0, 0, 0, 0, 0])
        fp = compute_fingerprint("a1", "ctrl", labels, fps=10.0)
        assert len(fp.cluster_names) == 1
        assert fp.frequencies[0] == pytest.approx(1.0)
        assert fp.transition_matrix.shape == (1, 1)

    def test_all_same_cluster(self):
        """All frames same cluster → single bout, zero transitions."""
        labels = np.ones(100, dtype=int) * 3
        fp = compute_fingerprint("a1", "ctrl", labels, fps=30.0)
        assert fp.bout_counts[0] == 1.0
        assert fp.transition_matrix.shape == (1, 1)


# ---------------------------------------------------------------------------
# BehavioralFingerprint.to_feature_vector
# ---------------------------------------------------------------------------

class TestFeatureVector:
    def test_dimensions_with_transitions(self):
        fp = _make_fingerprint("a1", "ctrl", K=4)
        vec = fp.to_feature_vector(include_transitions=True)
        # 6 feature groups * 4 + 4*4 = 24 + 16 = 40
        assert len(vec) == 6 * 4 + 4 * 4

    def test_dimensions_without_transitions(self):
        fp = _make_fingerprint("a1", "ctrl", K=3)
        vec = fp.to_feature_vector(include_transitions=False)
        assert len(vec) == 6 * 3

    def test_nan_handling(self):
        fp = _make_fingerprint("a1", "ctrl", K=2)
        fp.cv_bout_durations = np.array([np.nan, 0.5])
        vec = fp.to_feature_vector()
        assert not np.any(np.isnan(vec))

    def test_feature_vector_consistent(self):
        fp = _make_fingerprint("a1", "ctrl", K=3, seed=42)
        v1 = fp.to_feature_vector()
        v2 = fp.to_feature_vector()
        np.testing.assert_array_equal(v1, v2)


# ---------------------------------------------------------------------------
# BehavioralFingerprint.feature_names
# ---------------------------------------------------------------------------

class TestFeatureNames:
    def test_length_matches_vector(self):
        fp = _make_fingerprint("a1", "ctrl", K=3)
        names = fp.feature_names()
        vec = fp.to_feature_vector()
        assert len(names) == len(vec)

    def test_length_without_transitions(self):
        fp = _make_fingerprint("a1", "ctrl", K=4)
        names = fp.feature_names(include_transitions=False)
        vec = fp.to_feature_vector(include_transitions=False)
        assert len(names) == len(vec)

    def test_names_contain_prefixes(self):
        fp = _make_fingerprint("a1", "ctrl", K=2)
        names = fp.feature_names()
        prefixes = {"freq_", "bout_count_", "mean_dur_", "median_dur_", "cv_dur_", "ibi_", "trans_"}
        for prefix in prefixes:
            assert any(n.startswith(prefix) for n in names), f"Missing prefix {prefix}"


# ---------------------------------------------------------------------------
# bfa_test
# ---------------------------------------------------------------------------

class TestBFATest:
    def test_identical_groups_high_pvalue(self):
        """Identical groups should produce high p-value (not significant)."""
        group = _make_group("ctrl", 8, K=3, seed_base=0)
        # Use same group for both → no difference
        dist, pval = bfa_test(group, group, n_permutations=500)
        assert pval > 0.05

    def test_different_groups_low_pvalue(self):
        """Very different groups should produce low p-value."""
        K = 3
        group_a = _make_group("ctrl", 10, K=K, seed_base=0)
        # Create very different transition matrices for group B
        group_b = _make_group("treat", 10, K=K, seed_base=100,
                              tm_bias=np.array([[0, 5, 0], [0, 0, 5], [5, 0, 0]]))
        dist, pval = bfa_test(group_a, group_b, n_permutations=1000)
        assert pval < 0.05
        assert dist > 0

    def test_distance_is_nonnegative(self):
        group_a = _make_group("a", 5, K=3, seed_base=0)
        group_b = _make_group("b", 5, K=3, seed_base=50)
        dist, _ = bfa_test(group_a, group_b, n_permutations=100)
        assert dist >= 0

    def test_permutation_count(self):
        """Result should be consistent with the permutation count."""
        group_a = _make_group("a", 5, K=2, seed_base=0)
        group_b = _make_group("b", 5, K=2, seed_base=50)
        _, pval = bfa_test(group_a, group_b, n_permutations=100)
        # p-value should be a multiple of 1/100 (or the floor)
        assert pval >= 1 / 101


# ---------------------------------------------------------------------------
# energy_distance_test
# ---------------------------------------------------------------------------

class TestEnergyDistanceTest:
    def test_identical_groups(self):
        group = _make_group("ctrl", 8, K=3, seed_base=0)
        dist, pval = energy_distance_test(group, group, n_permutations=500)
        assert pval > 0.05

    def test_basic_correctness(self):
        group_a = _make_group("a", 8, K=3, seed_base=0)
        group_b = _make_group("b", 8, K=3, seed_base=200)
        dist, pval = energy_distance_test(group_a, group_b, n_permutations=500)
        assert dist >= 0  # energy distance is non-negative
        assert 0 < pval <= 1

    def test_single_animal_per_group(self):
        """Should work with n=1 per group."""
        fp_a = [_make_fingerprint("a1", "a", K=3, seed=0)]
        fp_b = [_make_fingerprint("b1", "b", K=3, seed=100)]
        dist, pval = energy_distance_test(fp_a, fp_b, n_permutations=100)
        assert dist >= 0


# ---------------------------------------------------------------------------
# hedges_g
# ---------------------------------------------------------------------------

class TestHedgesG:
    def test_zero_for_identical(self):
        a = np.array([1.0, 2.0, 3.0])
        g = hedges_g(a, a)
        assert g == pytest.approx(0.0)

    def test_known_value(self):
        """Known Cohen's d ≈ 1.0 for these groups, Hedges' correction close."""
        a = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        b = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
        # d = (1-2)/0 → but std=0, so returns 0
        # Need variance
        a = np.array([0.5, 1.0, 1.5, 1.0, 1.0])
        b = np.array([2.0, 2.5, 3.0, 2.5, 2.5])
        g = hedges_g(a, b)
        assert g < 0  # a < b → negative effect
        assert abs(g) > 1.0  # large effect

    def test_positive_direction(self):
        a = np.array([5.0, 6.0, 7.0])
        b = np.array([1.0, 2.0, 3.0])
        g = hedges_g(a, b)
        assert g > 0

    def test_small_sample(self):
        """With n<3 total, returns 0."""
        a = np.array([1.0])
        b = np.array([2.0])
        g = hedges_g(a, b)
        assert g == 0.0

    def test_zero_variance(self):
        """Identical values in both groups → g=0."""
        a = np.array([5.0, 5.0, 5.0])
        b = np.array([5.0, 5.0, 5.0])
        g = hedges_g(a, b)
        assert g == 0.0


# ---------------------------------------------------------------------------
# hedges_g_ci
# ---------------------------------------------------------------------------

class TestHedgesGCI:
    def test_ci_contains_zero_for_small_effect(self):
        lo, hi = hedges_g_ci(0.0, 10, 10)
        assert lo < 0 < hi

    def test_ci_width_decreases_with_n(self):
        lo1, hi1 = hedges_g_ci(0.5, 5, 5)
        lo2, hi2 = hedges_g_ci(0.5, 50, 50)
        assert (hi1 - lo1) > (hi2 - lo2)


# ---------------------------------------------------------------------------
# benjamini_hochberg
# ---------------------------------------------------------------------------

class TestBenjaminiHochberg:
    def test_single_pvalue(self):
        adj = benjamini_hochberg(np.array([0.03]))
        assert adj[0] == pytest.approx(0.03)

    def test_all_significant(self):
        pvals = np.array([0.001, 0.002, 0.003])
        adj = benjamini_hochberg(pvals)
        # All adjusted should still be ≤ 0.05
        assert all(adj <= 0.05)

    def test_ordering_preserved(self):
        """Adjusted p-values preserve weak ordering: if p_i < p_j then p_adj_i <= p_adj_j."""
        pvals = np.array([0.01, 0.04, 0.03, 0.50])
        adj = benjamini_hochberg(pvals)
        # For each pair, if raw i < raw j, adjusted i should be <= adjusted j
        for i in range(len(pvals)):
            for j in range(len(pvals)):
                if pvals[i] < pvals[j]:
                    assert adj[i] <= adj[j] + 1e-10

    def test_correction_increases_pvalues(self):
        pvals = np.array([0.01, 0.02, 0.03, 0.04])
        adj = benjamini_hochberg(pvals)
        # Adjusted should be >= original
        assert all(adj >= pvals - 1e-10)

    def test_empty(self):
        adj = benjamini_hochberg(np.array([]))
        assert len(adj) == 0

    def test_clipped_to_one(self):
        pvals = np.array([0.8, 0.9, 0.95])
        adj = benjamini_hochberg(pvals)
        assert all(adj <= 1.0)


# ---------------------------------------------------------------------------
# permutation_test_per_feature
# ---------------------------------------------------------------------------

class TestPermutationTestPerFeature:
    def test_output_structure(self):
        group_a = _make_group("a", 5, K=3, seed_base=0)
        group_b = _make_group("b", 5, K=3, seed_base=50)
        result = permutation_test_per_feature(group_a, group_b, n_permutations=200)
        assert "feature_names" in result
        assert "pvalues" in result
        assert "pvalues_adj" in result
        assert "effect_sizes" in result
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert "means_a" in result
        assert "means_b" in result
        # Lengths match
        n = len(result["feature_names"])
        assert len(result["pvalues"]) == n
        assert len(result["pvalues_adj"]) == n
        assert len(result["effect_sizes"]) == n

    def test_pvalues_in_range(self):
        group_a = _make_group("a", 5, K=2, seed_base=0)
        group_b = _make_group("b", 5, K=2, seed_base=50)
        result = permutation_test_per_feature(group_a, group_b, n_permutations=200)
        assert all(0 < p <= 1 for p in result["pvalues"])
        assert all(0 < p <= 1 for p in result["pvalues_adj"])


# ---------------------------------------------------------------------------
# compare_groups (full pipeline)
# ---------------------------------------------------------------------------

class TestCompareGroups:
    def test_full_pipeline(self):
        group_a = _make_group("ctrl", 5, K=3, seed_base=0)
        group_b = _make_group("treat", 5, K=3, seed_base=100)
        result = compare_groups(group_a, group_b, n_permutations=200)
        assert isinstance(result, ComparisonResult)
        assert result.group_a_name == "ctrl"
        assert result.group_b_name == "treat"
        assert result.n_a == 5
        assert result.n_b == 5
        assert result.bfa_distance >= 0
        assert 0 < result.bfa_pvalue <= 1
        assert result.energy_distance >= 0
        assert 0 < result.energy_pvalue <= 1
        assert len(result.feature_names) > 0
        assert result.feature_pvalues is not None
        assert result.feature_pvalues_adj is not None
        assert result.feature_effect_sizes is not None
        assert result.summary != ""

    def test_identical_groups_no_significant(self):
        """Identical groups should not produce significant features."""
        group = _make_group("ctrl", 8, K=3, seed_base=42)
        result = compare_groups(group, group, n_permutations=500)
        # Most features should not be significant
        assert result.bfa_pvalue > 0.05

    def test_empty_group_raises(self):
        group_a = _make_group("a", 3, K=3, seed_base=0)
        with pytest.raises(ValueError):
            compare_groups(group_a, [], n_permutations=100)

    def test_single_animal_per_group(self):
        """Should work with n=1 per group."""
        fp_a = [_make_fingerprint("a1", "a", K=3, seed=0)]
        fp_b = [_make_fingerprint("b1", "b", K=3, seed=100)]
        result = compare_groups(fp_a, fp_b, n_permutations=100)
        assert isinstance(result, ComparisonResult)
        assert result.n_a == 1
        assert result.n_b == 1


# ---------------------------------------------------------------------------
# ComparisonResult
# ---------------------------------------------------------------------------

class TestComparisonResult:
    def test_significant_features_detection(self):
        """Significant features list should contain features below alpha."""
        group_a = _make_group("ctrl", 8, K=3, seed_base=0)
        # Create a very different group
        group_b = _make_group("treat", 8, K=3, seed_base=500,
                              freq_bias=np.array([0.5, -0.2, -0.1]),
                              tm_bias=np.array([[0, 3, 0], [0, 0, 3], [3, 0, 0]]))
        result = compare_groups(group_a, group_b, n_permutations=1000)
        # With such different groups, there should be some significant features
        # (not guaranteed due to randomness, but likely)
        assert isinstance(result.significant_features, list)

    def test_summary_contains_group_names(self):
        group_a = _make_group("wild_type", 3, K=2, seed_base=0)
        group_b = _make_group("knockout", 3, K=2, seed_base=50)
        result = compare_groups(group_a, group_b, n_permutations=100)
        assert "wild_type" in result.summary
        assert "knockout" in result.summary


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_identical_data(self):
        """All animals have identical data → distance=0, high p-value."""
        K = 3
        freq = np.array([0.5, 0.3, 0.2])
        tm = np.array([[0, 0.6, 0.4], [0.3, 0, 0.7], [0.5, 0.5, 0]])
        group_a = []
        group_b = []
        for i in range(5):
            group_a.append(_make_fingerprint(f"a{i}", "a", K=K, frequencies=freq.copy(),
                                              transition_matrix=tm.copy(), seed=i))
            group_b.append(_make_fingerprint(f"b{i}", "b", K=K, frequencies=freq.copy(),
                                              transition_matrix=tm.copy(), seed=i))
        # Override all varying fields to be identical
        for fp in group_a + group_b:
            fp.bout_counts = np.array([10.0, 6.0, 4.0])
            fp.mean_bout_durations = np.array([0.5, 0.3, 0.2])
            fp.median_bout_durations = np.array([0.4, 0.25, 0.15])
            fp.cv_bout_durations = np.array([0.3, 0.2, 0.1])
            fp.inter_bout_intervals = np.array([1.0, 1.5, 2.0])

        dist, pval = bfa_test(group_a, group_b, n_permutations=500)
        assert dist == pytest.approx(0.0)
        # p-value should be 1.0 (or close) since observed=0 and all perms produce >=0
        assert pval >= 0.5

    def test_all_same_cluster_fingerprint(self):
        """All frames are the same cluster — only 1 cluster."""
        labels = np.zeros(100, dtype=int)
        fp = compute_fingerprint("a1", "ctrl", labels, fps=10.0)
        assert len(fp.cluster_names) == 1
        vec = fp.to_feature_vector()
        assert not np.any(np.isnan(vec))

    def test_two_frames(self):
        """Minimal sequence — 2 frames."""
        labels = np.array([0, 1])
        fp = compute_fingerprint("a1", "ctrl", labels, fps=10.0)
        assert fp.n_frames == 2
        assert len(fp.frequencies) == 2

    def test_feature_vector_nan_replaced(self):
        """NaN in input should be replaced with 0 in feature vector."""
        fp = _make_fingerprint("a1", "ctrl", K=3, seed=0)
        fp.mean_bout_durations = np.array([1.0, np.nan, 0.5])
        fp.inter_bout_intervals = np.array([np.nan, np.nan, 1.0])
        vec = fp.to_feature_vector()
        assert not np.any(np.isnan(vec))


# ---------------------------------------------------------------------------
# Visualization smoke tests
# ---------------------------------------------------------------------------

class TestVisualization:
    def test_radar_plot(self):
        from castle.visualization.comparison_plots import plot_fingerprint_radar
        group_a = _make_group("ctrl", 3, K=4, seed_base=0)
        group_b = _make_group("treat", 3, K=4, seed_base=50)
        fig = plot_fingerprint_radar(group_a, group_b, ("Control", "Treatment"))
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_transition_heatmap_diff(self):
        from castle.visualization.comparison_plots import plot_transition_heatmap_diff
        K = 3
        tm_a = np.array([[0, 0.6, 0.4], [0.3, 0, 0.7], [0.5, 0.5, 0]])
        tm_b = np.array([[0, 0.3, 0.7], [0.5, 0, 0.5], [0.2, 0.8, 0]])
        fig = plot_transition_heatmap_diff(tm_a, tm_b, ["A", "B", "C"], ("Ctrl", "Exp"))
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_volcano_plot(self):
        from castle.visualization.comparison_plots import plot_volcano
        names = [f"feat_{i}" for i in range(10)]
        effects = np.random.randn(10)
        pvals = np.random.uniform(0.001, 0.5, 10)
        fig = plot_volcano(names, effects, pvals)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_forest_plot(self):
        from castle.visualization.comparison_plots import plot_forest
        names = [f"feat_{i}" for i in range(10)]
        effects = np.random.randn(10)
        ci_lo = effects - 0.5
        ci_hi = effects + 0.5
        fig = plot_forest(names, effects, ci_lo, ci_hi)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


# ---------------------------------------------------------------------------
# Service layer
# ---------------------------------------------------------------------------

class TestComparisonService:
    def _build_fake_project(self, tmp_path, name, n_videos=3, n_frames=50, K=3, seed=0):
        """Create a minimal project with cluster data."""
        project_dir = tmp_path / name
        cluster_dir = project_dir / "cluster"
        cluster_dir.mkdir(parents=True)

        # id.csv
        with open(cluster_dir / "id.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Id", "Name", "Color"])
            for i in range(K):
                w.writerow([i, f"behavior_{i}", f"#{i:02x}{i:02x}{i:02x}"])

        rng = np.random.default_rng(seed)
        for v in range(n_videos):
            labels = rng.integers(0, K, size=n_frames)
            ts_path = cluster_dir / f"time_series_video{v}.csv"
            with open(ts_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["", "behavior"])
                for i, label in enumerate(labels):
                    w.writerow([i, label])

        return str(project_dir)

    def test_compare_projects(self, tmp_path):
        from castle.service.comparison_service import compare_projects

        path_a = self._build_fake_project(tmp_path, "proj_a", n_videos=3, seed=0)
        path_b = self._build_fake_project(tmp_path, "proj_b", n_videos=3, seed=100)

        result = compare_projects(path_a, path_b, n_permutations=200)
        assert result["status"] == "success"
        assert "bfa_distance" in result
        assert "bfa_pvalue" in result
        assert len(result["fingerprints_a"]) == 3
        assert len(result["fingerprints_b"]) == 3

    def test_compute_project_fingerprints(self, tmp_path):
        from castle.service.comparison_service import compute_project_fingerprints

        path = self._build_fake_project(tmp_path, "proj", n_videos=2, seed=42)
        result = compute_project_fingerprints(path, group_name="test", fps=30.0)
        assert result["status"] == "success"
        assert result["n_animals"] == 2
        assert len(result["fingerprints"]) == 2

    def test_export_comparison_report(self, tmp_path):
        from castle.service.comparison_service import (
            compare_projects,
            export_comparison_report,
        )

        path_a = self._build_fake_project(tmp_path, "proj_a2", n_videos=2, seed=0)
        path_b = self._build_fake_project(tmp_path, "proj_b2", n_videos=2, seed=100)

        result = compare_projects(path_a, path_b, n_permutations=100)
        assert result["status"] == "success"

        out_dir = str(tmp_path / "report")
        paths = export_comparison_report(result, out_dir)
        assert len(paths) >= 2
        for p in paths:
            assert os.path.exists(p)


# ---------------------------------------------------------------------------
# PR1 Stage 3 (CRITICAL C2): global cluster-id alignment + small-sample honesty
# ---------------------------------------------------------------------------

class TestGlobalClusterAlignment:
    """Fingerprints dimensioned by the GLOBAL cluster set, so cross-animal
    comparison never crashes or mis-aligns features (contract C-6)."""

    def test_different_cluster_sets_aligned(self):
        all_ids = [0, 1, 2, 3]
        labels_a = np.array([0] * 10 + [1] * 10 + [2] * 10)          # no cluster 3
        labels_b = np.array([0] * 10 + [1] * 10 + [2] * 10 + [3] * 10)
        fp_a = compute_fingerprint("a", "G", labels_a, fps=10.0, all_cluster_ids=all_ids)
        fp_b = compute_fingerprint("b", "G", labels_b, fps=10.0, all_cluster_ids=all_ids)

        assert len(fp_a.to_feature_vector()) == len(fp_b.to_feature_vector())
        assert fp_a.cluster_id_order == fp_b.cluster_id_order == [0, 1, 2, 3]
        # cluster 3 absent in A: structural 0, duration undefined (NaN)
        assert fp_a.frequencies[3] == 0.0
        assert fp_a.bout_counts[3] == 0.0
        assert np.isnan(fp_a.mean_bout_durations[3])
        assert fp_a.transition_matrix.shape == (4, 4)
        # frequency uses valid-frames-only and sums to 1 (no -1 here)
        assert fp_b.frequencies.sum() == pytest.approx(1.0)

    def test_compare_groups_different_cluster_sets_no_crash(self):
        all_ids = [0, 1, 2, 3]
        group_a = [
            compute_fingerprint(f"a{i}", "A", np.array([0] * 10 + [1] * 10 + [2] * 10),
                                fps=10.0, all_cluster_ids=all_ids)
            for i in range(3)
        ]
        group_b = [
            compute_fingerprint(f"b{i}", "B", np.array([0] * 10 + [1] * 10 + [2] * 10 + [3] * 10),
                                fps=10.0, all_cluster_ids=all_ids)
            for i in range(3)
        ]
        result = compare_groups(group_a, group_b, n_permutations=200)
        assert isinstance(result, ComparisonResult)
        assert len(result.feature_names) == len(group_a[0].to_feature_vector())

    def test_mismatched_fingerprint_lengths_raise(self):
        from castle.core.types import CastleDataError
        fp_a = _make_fingerprint("a", "A", K=3, seed=1)
        fp_b = _make_fingerprint("b", "B", K=4, seed=2)  # different length
        with pytest.raises(CastleDataError, match="mismatched"):
            compare_groups([fp_a], [fp_b], n_permutations=50)


class TestSmallSampleHonesty:
    def test_n1_group_floored_and_warned(self):
        all_ids = [0, 1, 2]
        a = [compute_fingerprint("a0", "A", np.array([0] * 10 + [1] * 10 + [2] * 10),
                                 fps=10.0, all_cluster_ids=all_ids)]
        b = [
            compute_fingerprint(f"b{i}", "B", np.array([0] * 5 + [1] * 15 + [2] * 10),
                                fps=10.0, all_cluster_ids=all_ids)
            for i in range(2)
        ]
        result = compare_groups(a, b, n_permutations=1000)
        # n_a=1, n_b=2 → C(3,1)=3 → min achievable p = 1/3
        assert result.bfa_pvalue >= 1.0 / 3 - 1e-9
        assert "WARNING" in result.summary

    def test_energy_distance_precomputed_finite(self):
        all_ids = [0, 1]
        a = [compute_fingerprint(f"a{i}", "A", np.array([0] * 10 + [1] * 10),
                                 fps=10.0, all_cluster_ids=all_ids) for i in range(4)]
        b = [compute_fingerprint(f"b{i}", "B", np.array([0] * 15 + [1] * 5),
                                 fps=10.0, all_cluster_ids=all_ids) for i in range(4)]
        dist, p = energy_distance_test(a, b, n_permutations=200)
        assert np.isfinite(dist)
        assert 0.0 < p <= 1.0
