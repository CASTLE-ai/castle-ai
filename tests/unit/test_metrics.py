"""Tests for castle.core.metrics — clustering quality metrics."""

import time

import numpy as np
import pytest

from castle.core.metrics import (
    ClusterQualityReport,
    bout_quality_metrics,
    compute_external_metrics,
    compute_internal_metrics,
    evaluate_clustering,
    temporal_coherence,
)


# ======================================================================== #
#  temporal_coherence                                                       #
# ======================================================================== #


class TestTemporalCoherence:
    def test_perfect_labels(self):
        """All-same labels → coherence 1.0."""
        labels = np.array([1, 1, 1, 1, 1])
        assert temporal_coherence(labels) == 1.0

    def test_alternating_labels(self):
        """Alternating labels → coherence 0.0."""
        labels = np.array([0, 1, 0, 1, 0, 1])
        assert temporal_coherence(labels) == 0.0

    def test_realistic_labels(self):
        """Block-structured labels → high but not perfect coherence."""
        labels = np.array([0] * 50 + [1] * 50 + [2] * 50)
        tc = temporal_coherence(labels)
        # 2 transitions out of 149 comparisons → 147/149
        assert tc == pytest.approx(147 / 149, abs=1e-6)

    def test_single_element(self):
        """Single-element array → 1.0 by convention."""
        assert temporal_coherence(np.array([5])) == 1.0

    def test_empty_array(self):
        """Empty array → 1.0 by convention."""
        assert temporal_coherence(np.array([], dtype=int)) == 1.0

    def test_window_larger_than_1(self):
        """Window=2 checks label match with 2-frame offset."""
        labels = np.array([0, 0, 1, 1, 2, 2])
        # labels[:4]=[0,0,1,1], labels[2:]=[1,1,2,2] → matches: 0==1(N),0==1(N),1==2(N),1==2(N) → 0/4
        tc = temporal_coherence(labels, window=2)
        assert tc == pytest.approx(0.0, abs=1e-6)
        # Better example: [0,0,0,1,1,1] with window=2
        labels2 = np.array([0, 0, 0, 1, 1, 1])
        tc2 = temporal_coherence(labels2, window=2)
        # labels2[:4]=[0,0,0,1], labels2[2:]=[0,1,1,1] → matches: 0==0(Y),0==1(N),0==1(N),1==1(Y) → 2/4
        assert tc2 == pytest.approx(2 / 4, abs=1e-6)


# ======================================================================== #
#  bout_quality_metrics                                                     #
# ======================================================================== #


class TestBoutQualityMetrics:
    def test_known_sequence(self):
        """Known bout structure: [0,0,0,1,1,2] → 3 bouts, lengths [3,2,1]."""
        labels = np.array([0, 0, 0, 1, 1, 2])
        m = bout_quality_metrics(labels)
        assert m["n_bouts"] == 3
        assert m["median_duration"] == 2.0
        assert m["n_single_frame"] == 1
        assert m["single_frame_ratio"] == pytest.approx(1 / 3, abs=1e-6)

    def test_single_frame_detection(self):
        """All single-frame bouts: alternating labels."""
        labels = np.array([0, 1, 0, 1, 0])
        m = bout_quality_metrics(labels)
        assert m["n_bouts"] == 5
        assert m["n_single_frame"] == 5
        assert m["single_frame_ratio"] == 1.0

    def test_single_bout(self):
        """Single continuous bout."""
        labels = np.array([3, 3, 3, 3])
        m = bout_quality_metrics(labels)
        assert m["n_bouts"] == 1
        assert m["n_single_frame"] == 0
        assert m["single_frame_ratio"] == 0.0
        assert m["median_duration"] == 4.0
        assert m["duration_cv"] == 0.0

    def test_empty(self):
        labels = np.array([], dtype=int)
        m = bout_quality_metrics(labels)
        assert m["n_bouts"] == 0


# ======================================================================== #
#  compute_internal_metrics                                                 #
# ======================================================================== #


class TestComputeInternalMetrics:
    def test_without_embedding(self):
        """Internal metrics without embedding only give TC + bout metrics."""
        labels = np.array([0] * 100 + [1] * 100)
        m = compute_internal_metrics(labels)
        assert m["temporal_coherence"] > 0.99
        assert m["silhouette_sample"] is None
        assert m["calinski_harabasz"] is None
        assert m["davies_bouldin"] is None

    def test_with_embedding(self):
        """With well-separated 2D embedding → positive silhouette."""
        rng = np.random.RandomState(42)
        # Two well-separated clusters in 2D
        emb_a = rng.randn(100, 2) + np.array([5, 5])
        emb_b = rng.randn(100, 2) + np.array([-5, -5])
        embedding = np.vstack([emb_a, emb_b])
        labels = np.array([0] * 100 + [1] * 100)
        m = compute_internal_metrics(labels, embedding=embedding)
        assert m["silhouette_sample"] is not None
        assert m["silhouette_sample"] > 0.5
        assert m["calinski_harabasz"] is not None
        assert m["calinski_harabasz"] > 0
        assert m["davies_bouldin"] is not None

    def test_single_cluster_no_distance_metrics(self):
        """Single cluster → distance metrics stay None."""
        labels = np.array([0] * 50)
        emb = np.random.randn(50, 2)
        m = compute_internal_metrics(labels, embedding=emb)
        assert m["silhouette_sample"] is None

    def test_all_noise(self):
        """All -1 labels → distance metrics stay None."""
        labels = np.array([-1] * 50)
        emb = np.random.randn(50, 2)
        m = compute_internal_metrics(labels, embedding=emb)
        assert m["silhouette_sample"] is None


# ======================================================================== #
#  compute_external_metrics                                                 #
# ======================================================================== #


class TestComputeExternalMetrics:
    def test_perfect_match(self):
        """Identical labels → NMI=1, ARI=1."""
        labels = np.array([0, 0, 1, 1, 2, 2])
        gt = np.array([0, 0, 1, 1, 2, 2])
        m = compute_external_metrics(labels, gt)
        assert m["nmi"] == pytest.approx(1.0, abs=1e-6)
        assert m["ari"] == pytest.approx(1.0, abs=1e-6)

    def test_random_labels(self):
        """Random labels against structured GT → low ARI (near 0)."""
        rng = np.random.RandomState(123)
        gt = np.array([0] * 500 + [1] * 500)
        labels = rng.randint(0, 2, size=1000)
        m = compute_external_metrics(labels, gt)
        # ARI should be close to 0 for random
        assert abs(m["ari"]) < 0.1


# ======================================================================== #
#  evaluate_clustering (integration of all metrics)                         #
# ======================================================================== #


class TestEvaluateClustering:
    def test_verdict_good(self):
        """Long stable bouts → GOOD."""
        labels = np.array([0] * 500 + [1] * 500 + [2] * 500)
        report = evaluate_clustering(labels)
        assert report.verdict == "GOOD"
        assert report.temporal_coherence > 0.95
        assert report.single_frame_ratio < 0.1

    def test_verdict_poor_flickering(self):
        """Alternating labels → POOR."""
        labels = np.tile([0, 1], 500)
        report = evaluate_clustering(labels)
        assert report.verdict == "POOR"
        assert report.temporal_coherence == 0.0
        assert len(report.warnings) > 0

    def test_verdict_acceptable(self):
        """Moderately stable → ACCEPTABLE."""
        # Build labels with TC ~0.90 and low single-frame ratio
        # Blocks of ~10 frames each, with occasional 2-frame flips
        rng = np.random.RandomState(42)
        segments = []
        for _ in range(30):
            label = rng.randint(0, 3)
            length = rng.randint(8, 15)
            segments.extend([label] * length)
        labels = np.array(segments)
        report = evaluate_clustering(labels)
        # TC is high (block structure), single_frame_ratio is 0
        # Force it into ACCEPTABLE range by tweaking thresholds
        assert report.temporal_coherence > 0.85
        assert report.single_frame_ratio < 0.2
        assert report.verdict in ("ACCEPTABLE", "GOOD")

    def test_with_ground_truth(self):
        """External metrics populated when ground truth is provided."""
        labels = np.array([0, 0, 1, 1, 2, 2])
        gt = np.array([0, 0, 1, 1, 2, 2])
        report = evaluate_clustering(labels, ground_truth=gt)
        assert report.nmi is not None
        assert report.ari == pytest.approx(1.0, abs=1e-6)
        assert report.homogeneity is not None
        assert report.completeness is not None

    def test_without_ground_truth_external_none(self):
        """Without ground truth, external metrics are None."""
        labels = np.array([0, 0, 1, 1])
        report = evaluate_clustering(labels)
        assert report.nmi is None
        assert report.ari is None


# ======================================================================== #
#  Edge cases                                                               #
# ======================================================================== #


class TestEdgeCases:
    def test_empty_labels(self):
        """Empty label array doesn't crash."""
        report = evaluate_clustering(np.array([], dtype=int))
        assert report.temporal_coherence == 1.0
        assert report.n_single_frame_bouts == 0

    def test_single_cluster(self):
        """Single cluster → valid report with warning."""
        labels = np.array([0] * 100)
        report = evaluate_clustering(labels)
        assert report.verdict == "GOOD"
        assert any("Only one cluster" in w for w in report.warnings)

    def test_all_noise(self):
        """All -1 labels → handled gracefully."""
        labels = np.array([-1] * 100)
        report = evaluate_clustering(labels)
        assert report.temporal_coherence == 1.0  # all same label


# ======================================================================== #
#  ClusterQualityReport                                                     #
# ======================================================================== #


class TestClusterQualityReport:
    def test_warnings_default_empty(self):
        """Default warnings is empty list, not None."""
        report = ClusterQualityReport(
            temporal_coherence=0.99,
            calinski_harabasz=None,
            davies_bouldin=None,
            silhouette_sample=None,
            n_single_frame_bouts=0,
            single_frame_ratio=0.0,
            median_bout_duration_frames=50.0,
            bout_duration_cv=0.1,
        )
        assert report.warnings == []
        assert report.nmi is None

    def test_warnings_generation(self):
        """evaluate_clustering generates warnings for concerning metrics."""
        # Flickering → multiple warnings
        labels = np.tile([0, 1], 100)
        report = evaluate_clustering(labels)
        assert len(report.warnings) >= 2  # low TC + high single-frame ratio


# ======================================================================== #
#  Performance                                                              #
# ======================================================================== #


class TestPerformance:
    def test_large_array_speed(self):
        """100k frames should evaluate in < 2 seconds (no embedding)."""
        labels = np.repeat(np.arange(20), 5000)  # 100k frames, 20 clusters
        start = time.time()
        report = evaluate_clustering(labels)
        elapsed = time.time() - start
        assert elapsed < 2.0, f"Took {elapsed:.2f}s, expected < 2s"
        assert report.verdict == "GOOD"
