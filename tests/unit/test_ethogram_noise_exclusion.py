"""Ethogram: -1 (noise / unclustered / dropped) is an unlabeled gap, not a state.

Covers the A1 audit fix: DBSCAN noise and NaN→-1 frames must be excluded from
bouts, the transition matrix, temporal coherence and n_clusters, and reported
separately via n_unlabeled / unlabeled_fraction — while still SEGMENTING the
real bouts on either side (never merging across a gap).
"""
import numpy as np

from castle.core.ethogram import (
    extract_bouts,
    compute_transition_matrix,
    compute_temporal_coherence,
    compute_ethogram,
)


class TestBoutsExcludeNoise:
    def test_minus1_run_is_not_a_bout_but_segments(self):
        # 0,0,-1,0,0 → two separate cluster-0 bouts, NO -1 bout, NOT one merged bout.
        labels = np.array([0, 0, -1, 0, 0])
        bouts = extract_bouts(labels, fps=10.0)
        assert all(b.cluster_id != -1 for b in bouts)
        zero_bouts = [b for b in bouts if b.cluster_id == 0]
        assert len(zero_bouts) == 2  # segmented by the gap, not merged
        assert {(b.start_frame, b.end_frame) for b in zero_bouts} == {(0, 2), (3, 5)}

    def test_all_noise_has_no_bouts(self):
        bouts = extract_bouts(np.array([-1, -1, -1]), fps=10.0)
        assert bouts == []


class TestTransitionExcludesNoise:
    def test_minus1_not_in_axes_and_not_counted(self):
        # 0,0,1,1,-1,2 → only 0→1 is a real transition; 1→-1 and -1→2 are gaps.
        labels = np.array([0, 0, 1, 1, -1, 2])
        tm = compute_transition_matrix(labels)
        assert -1 not in tm.cluster_ids
        assert tm.cluster_ids == [0, 1, 2]
        assert tm.n_transitions == 1  # only 0→1
        i0, i1 = tm.cluster_ids.index(0), tm.cluster_ids.index(1)
        assert tm.counts[i0, i1] == 1

    def test_transition_through_gap_not_merged(self):
        # 0 → gap → 1 must NOT be counted as a 0→1 transition.
        labels = np.array([0, -1, 1])
        tm = compute_transition_matrix(labels)
        assert tm.n_transitions == 0
        assert tm.cluster_ids == [0, 1]

    def test_all_noise_degenerate(self):
        tm = compute_transition_matrix(np.array([-1, -1, -1]))
        assert tm.cluster_ids == []
        assert tm.n_transitions == 0


class TestTemporalCoherenceExcludesNoise:
    def test_pairs_touching_minus1_excluded(self):
        # 0,0,-1,1,1: valid pairs (0,0) and (1,1) both match → 1.0
        # (without exclusion this would be 2/4 = 0.5).
        labels = np.array([0, 0, -1, 1, 1])
        assert compute_temporal_coherence(labels) == 1.0

    def test_all_noise_returns_one(self):
        assert compute_temporal_coherence(np.array([-1, -1, -1])) == 1.0


class TestComputeEthogramNoiseReporting:
    def test_n_clusters_excludes_minus1_and_reports_fraction(self):
        labels = np.array([0, 0, 1, 1, -1, -1, -1, 2])  # 3 real clusters, 3 noise of 8
        eth = compute_ethogram(labels, fps=10.0)
        assert eth.n_clusters == 3
        assert -1 not in eth.cluster_names
        assert eth.n_unlabeled == 3
        assert eth.unlabeled_fraction == 3 / 8
        # No bout / bout_stats entry for -1.
        assert all(b.cluster_id != -1 for b in eth.bouts)
        assert -1 not in eth.bout_stats
        assert -1 not in eth.transition_matrix.cluster_ids

    def test_no_noise_fraction_is_zero(self):
        eth = compute_ethogram(np.array([0, 0, 1, 1]), fps=10.0)
        assert eth.n_unlabeled == 0
        assert eth.unlabeled_fraction == 0.0

    def test_all_noise_does_not_crash(self):
        eth = compute_ethogram(np.array([-1, -1, -1]), fps=10.0)
        assert eth.n_clusters == 0
        assert eth.unlabeled_fraction == 1.0
        assert eth.bouts == []
        assert eth.bout_stats == {}
        assert eth.transition_matrix.n_transitions == 0
