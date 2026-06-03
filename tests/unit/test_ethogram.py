"""Unit tests for castle.core.ethogram and castle.service.ethogram_service."""

import os
import csv
import tempfile

import numpy as np
import pytest

from castle.core.ethogram import (
    BoutInfo,
    BoutStatistics,
    TransitionMatrix,
    Ethogram,
    extract_bouts,
    compute_bout_statistics,
    compute_transition_matrix,
    compute_temporal_coherence,
    compute_ethogram,
)


# ---------------------------------------------------------------------------
# extract_bouts
# ---------------------------------------------------------------------------

class TestExtractBouts:
    def test_simple_sequence(self):
        labels = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])
        bouts = extract_bouts(labels, fps=10.0)
        assert len(bouts) == 3
        # Should be in temporal order
        assert bouts[0].cluster_id == 0
        assert bouts[0].start_frame == 0
        assert bouts[0].end_frame == 3
        assert bouts[0].duration_frames == 3
        assert bouts[0].duration_seconds == pytest.approx(0.3)

        assert bouts[1].cluster_id == 1
        assert bouts[1].start_frame == 3
        assert bouts[1].end_frame == 5

        assert bouts[2].cluster_id == 2
        assert bouts[2].start_frame == 5
        assert bouts[2].end_frame == 9
        assert bouts[2].duration_seconds == pytest.approx(0.4)

    def test_empty_labels(self):
        bouts = extract_bouts(np.array([], dtype=int), fps=30.0)
        assert bouts == []

    def test_single_element(self):
        bouts = extract_bouts(np.array([5]), fps=10.0)
        assert len(bouts) == 1
        assert bouts[0].cluster_id == 5
        assert bouts[0].duration_frames == 1
        assert bouts[0].duration_seconds == pytest.approx(0.1)

    def test_alternating(self):
        labels = np.array([0, 1, 0, 1, 0])
        bouts = extract_bouts(labels, fps=1.0)
        assert len(bouts) == 5
        for b in bouts:
            assert b.duration_frames == 1

    def test_all_same(self):
        labels = np.array([3, 3, 3, 3, 3])
        bouts = extract_bouts(labels, fps=5.0)
        assert len(bouts) == 1
        assert bouts[0].cluster_id == 3
        assert bouts[0].duration_frames == 5
        assert bouts[0].duration_seconds == pytest.approx(1.0)

    def test_duration_calculation(self):
        """30 frames at 30 fps = 1 second."""
        labels = np.zeros(30, dtype=int)
        bouts = extract_bouts(labels, fps=30.0)
        assert len(bouts) == 1
        assert bouts[0].duration_seconds == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# compute_bout_statistics
# ---------------------------------------------------------------------------

class TestBoutStatistics:
    def test_basic_stats(self):
        labels = np.array([0, 0, 0, 1, 1, 0, 0, 0, 0, 1])
        bouts = extract_bouts(labels, fps=10.0)
        stats = compute_bout_statistics(bouts, labels, fps=10.0)

        assert 0 in stats
        assert 1 in stats

        s0 = stats[0]
        assert s0.n_bouts == 2
        assert s0.total_frames == 7  # 3 + 4
        assert s0.frequency == pytest.approx(0.7)

    def test_inter_bout_interval(self):
        # cluster 0: [0..3), [5..9) → gap is 5-3=2 frames → 0.2s at 10fps
        labels = np.array([0, 0, 0, 1, 1, 0, 0, 0, 0, 1])
        bouts = extract_bouts(labels, fps=10.0)
        stats = compute_bout_statistics(bouts, labels, fps=10.0)
        s0 = stats[0]
        assert s0.mean_inter_bout_interval_s == pytest.approx(0.2)

    def test_single_bout_ibi_zero(self):
        labels = np.array([0, 0, 0])
        bouts = extract_bouts(labels, fps=10.0)
        stats = compute_bout_statistics(bouts, labels, fps=10.0)
        assert stats[0].mean_inter_bout_interval_s == 0.0

    def test_cv_calculation(self):
        """CV = std / mean."""
        labels = np.array([0, 0, 1, 0, 0, 0, 0])
        bouts = extract_bouts(labels, fps=10.0)
        stats = compute_bout_statistics(bouts, labels, fps=10.0)
        s0 = stats[0]
        durations = np.array([0.2, 0.4])  # 2 frames, 4 frames at 10fps
        expected_cv = float(np.std(durations, ddof=0) / np.mean(durations))
        assert s0.cv_duration == pytest.approx(expected_cv, abs=1e-6)

    def test_cluster_names_used(self):
        labels = np.array([0, 1, 1])
        bouts = extract_bouts(labels, fps=10.0)
        names = {0: "walking", 1: "resting"}
        stats = compute_bout_statistics(bouts, labels, fps=10.0, cluster_names=names)
        assert stats[0].cluster_name == "walking"
        assert stats[1].cluster_name == "resting"


# ---------------------------------------------------------------------------
# compute_transition_matrix
# ---------------------------------------------------------------------------

class TestTransitionMatrix:
    def test_probabilities_sum_to_one(self):
        labels = np.array([0, 1, 2, 0, 1, 2, 0, 1, 0, 2])
        tm = compute_transition_matrix(labels)
        # Each row with transitions should sum to ~1
        for i in range(tm.matrix.shape[0]):
            row_sum = tm.matrix[i].sum()
            if row_sum > 0:
                assert row_sum == pytest.approx(1.0, abs=1e-10)

    def test_self_transitions_excluded(self):
        labels = np.array([0, 0, 0, 1, 1, 1, 0])
        tm = compute_transition_matrix(labels)
        # Diagonal should be 0 (self-transitions excluded)
        for i in range(tm.matrix.shape[0]):
            assert tm.counts[i, i] == 0

    def test_known_transitions(self):
        # 0→1, 1→0, 0→1, 1→2  => transitions: 0→1 (2), 1→0 (1), 1→2 (1)
        labels = np.array([0, 1, 0, 1, 2])
        tm = compute_transition_matrix(labels)
        id_to_idx = {cid: i for i, cid in enumerate(tm.cluster_ids)}

        assert tm.counts[id_to_idx[0], id_to_idx[1]] == 2
        assert tm.counts[id_to_idx[1], id_to_idx[0]] == 1
        assert tm.counts[id_to_idx[1], id_to_idx[2]] == 1
        assert tm.n_transitions == 4

        # Row 0: only goes to 1 → P(0→1) = 1.0
        assert tm.matrix[id_to_idx[0], id_to_idx[1]] == pytest.approx(1.0)
        # Row 1: goes to 0 (1x) and 2 (1x) → P = 0.5 each
        assert tm.matrix[id_to_idx[1], id_to_idx[0]] == pytest.approx(0.5)
        assert tm.matrix[id_to_idx[1], id_to_idx[2]] == pytest.approx(0.5)

    def test_single_cluster(self):
        labels = np.array([0, 0, 0, 0])
        tm = compute_transition_matrix(labels)
        assert tm.n_transitions == 0
        assert tm.entropy == 0.0
        assert tm.matrix.shape == (1, 1)

    def test_entropy_two_equal(self):
        """Two equally probable transitions from each state → known entropy."""
        # 0→1, 1→0, 0→1, 1→0 ...
        labels = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        tm = compute_transition_matrix(labels)
        # Each state transitions to exactly one other state with P=1.0
        # So entropy = -sum(1.0 * log2(1.0)) per entry = 0 (each row has one non-zero)
        # Actually: row 0: [0, 1.0], row 1: [1.0, 0] → flat_p = [1.0, 1.0]
        # H = -2*(1.0*log2(1.0)) = 0
        assert tm.entropy == pytest.approx(0.0)

    def test_cluster_names_propagated(self):
        labels = np.array([0, 1, 0])
        names = {0: "walk", 1: "rest"}
        tm = compute_transition_matrix(labels, cluster_names=names)
        assert "walk" in tm.cluster_names
        assert "rest" in tm.cluster_names


# ---------------------------------------------------------------------------
# compute_temporal_coherence
# ---------------------------------------------------------------------------

class TestTemporalCoherence:
    def test_all_same(self):
        labels = np.array([1, 1, 1, 1, 1])
        assert compute_temporal_coherence(labels) == pytest.approx(1.0)

    def test_alternating(self):
        labels = np.array([0, 1, 0, 1, 0, 1])
        assert compute_temporal_coherence(labels) == pytest.approx(0.0)

    def test_half_and_half(self):
        labels = np.array([0, 0, 1, 1])
        # matches: (0==0, 0==1, 1==1) = (T, F, T) = 2/3
        assert compute_temporal_coherence(labels) == pytest.approx(2.0 / 3.0)

    def test_empty(self):
        assert compute_temporal_coherence(np.array([], dtype=int)) == 1.0

    def test_single_element(self):
        assert compute_temporal_coherence(np.array([0])) == 1.0

    def test_window_2(self):
        labels = np.array([0, 0, 0, 1, 1])
        # window=2: compare [0,0,0] with [0,1,1] → matches: (0==0, 0==1, 0==1) = 1/3
        assert compute_temporal_coherence(labels, window=2) == pytest.approx(1.0 / 3.0)


# ---------------------------------------------------------------------------
# compute_ethogram (full pipeline)
# ---------------------------------------------------------------------------

class TestComputeEthogram:
    def test_full_pipeline(self):
        labels = np.array([0, 0, 1, 1, 1, 2, 0, 0])
        names = {0: "walk", 1: "groom", 2: "rear"}
        eth = compute_ethogram(labels, fps=10.0, cluster_names=names)

        assert isinstance(eth, Ethogram)
        assert eth.n_frames == 8
        assert eth.n_clusters == 3
        assert eth.fps == 10.0
        assert len(eth.bouts) == 4  # walk(2), groom(3), rear(1), walk(2)
        assert 0 in eth.bout_stats
        assert 1 in eth.bout_stats
        assert 2 in eth.bout_stats
        assert isinstance(eth.transition_matrix, TransitionMatrix)
        assert 0.0 <= eth.temporal_coherence <= 1.0

    def test_single_cluster_pipeline(self):
        labels = np.array([0, 0, 0])
        eth = compute_ethogram(labels, fps=1.0)
        assert eth.n_clusters == 1
        assert len(eth.bouts) == 1
        assert eth.temporal_coherence == 1.0
        assert eth.transition_matrix.n_transitions == 0

    def test_no_cluster_names(self):
        labels = np.array([0, 1, 2])
        eth = compute_ethogram(labels, fps=1.0)
        assert "cluster_0" in eth.cluster_names.values()

    def test_frequencies_sum_to_one(self):
        labels = np.array([0, 0, 1, 1, 2, 2, 2, 2])
        eth = compute_ethogram(labels, fps=1.0)
        total_freq = sum(bs.frequency for bs in eth.bout_stats.values())
        assert total_freq == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# CSV export (integration test)
# ---------------------------------------------------------------------------

class TestExportCSV:
    def test_export_creates_files(self, tmp_path):
        """Create a minimal fake project and export CSV."""
        # Build fake project structure
        project_dir = tmp_path / "test_project"
        cluster_dir = project_dir / "cluster"
        cluster_dir.mkdir(parents=True)

        # id.csv
        with open(cluster_dir / "id.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Id", "Name", "Color"])
            w.writerow([0, "walk", "#ff0000"])
            w.writerow([1, "rest", "#00ff00"])

        # time_series_video1.csv
        with open(cluster_dir / "time_series_video1.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["", "behavior"])
            for i, label in enumerate([0, 0, 0, 1, 1, 0, 1, 1, 1, 0]):
                w.writerow([i, label])

        output_dir = tmp_path / "export"
        from castle.service.ethogram_service import export_ethogram_csv

        result = export_ethogram_csv(str(project_dir), str(output_dir))
        assert os.path.isdir(result)

        # Per-video export: combined long-format stats/bouts + per-video matrices.
        expected_files = ["bout_stats.csv", "bouts.csv",
                         "transition_matrix_video1.csv", "transition_counts_video1.csv"]
        for fname in expected_files:
            assert (output_dir / fname).exists(), f"Missing {fname}"

        # bout_stats.csv is long-format with a leading `video` column.
        with open(output_dir / "bout_stats.csv") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert "video" in rows[0]
            assert all(r["video"] == "video1" for r in rows)
            assert len(rows) == 2  # walk and rest (single video)
            names_found = {r["cluster_name"] for r in rows}
            assert "walk" in names_found
            assert "rest" in names_found

    def test_export_transition_matrix_csv_content(self, tmp_path):
        """Transition matrix CSV should have correct structure."""
        project_dir = tmp_path / "test_project2"
        cluster_dir = project_dir / "cluster"
        cluster_dir.mkdir(parents=True)

        with open(cluster_dir / "id.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Id", "Name", "Color"])
            w.writerow([0, "A", "red"])
            w.writerow([1, "B", "blue"])

        with open(cluster_dir / "time_series_v.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["", "behavior"])
            for i, label in enumerate([0, 1, 0, 1]):
                w.writerow([i, label])

        output_dir = tmp_path / "export2"
        from castle.service.ethogram_service import export_ethogram_csv
        export_ethogram_csv(str(project_dir), str(output_dir))

        with open(output_dir / "transition_matrix_v.csv") as f:
            reader = csv.reader(f)
            header = next(reader)
            assert header[1:] == ["A", "B"]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_negative_cluster_ids(self):
        """Cluster ID -1 (noise/unclustered) is an unlabeled gap, not a state:
        excluded from bouts/stats/n_clusters and reported separately."""
        labels = np.array([-1, -1, 0, 0, -1])
        eth = compute_ethogram(labels, fps=1.0)
        assert -1 not in eth.bout_stats
        assert eth.n_clusters == 1           # only cluster 0 is real
        assert eth.n_unlabeled == 3
        assert eth.unlabeled_fraction == 0.6

    def test_large_cluster_ids(self):
        labels = np.array([100, 100, 200, 200, 300])
        eth = compute_ethogram(labels, fps=1.0)
        assert eth.n_clusters == 3

    def test_two_frames(self):
        labels = np.array([0, 1])
        eth = compute_ethogram(labels, fps=1.0)
        assert len(eth.bouts) == 2
        assert eth.transition_matrix.n_transitions == 1
        assert eth.temporal_coherence == 0.0
