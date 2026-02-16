"""Tests for castle.core.temporal_smooth."""

import numpy as np
import pytest

from castle.core.temporal_smooth import (
    median_smooth,
    min_bout_filter,
    smooth_labels,
)


# ------------------------------------------------------------------ #
# median_smooth
# ------------------------------------------------------------------ #

class TestMedianSmooth:
    def test_single_flicker_removed(self):
        """[1,1,2,1,1] → [1,1,1,1,1] with default window=5."""
        labels = np.array([1, 1, 2, 1, 1])
        result = median_smooth(labels, window=5)
        np.testing.assert_array_equal(result, [1, 1, 1, 1, 1])

    def test_window_3(self):
        """Window=3 removes single-frame flicker."""
        labels = np.array([0, 0, 1, 0, 0, 0])
        result = median_smooth(labels, window=3)
        np.testing.assert_array_equal(result, [0, 0, 0, 0, 0, 0])

    def test_window_5_longer_sequence(self):
        labels = np.array([1, 1, 1, 2, 1, 1, 1, 1])
        result = median_smooth(labels, window=5)
        np.testing.assert_array_equal(result, [1, 1, 1, 1, 1, 1, 1, 1])

    def test_window_7(self):
        """Larger window smooths wider flickers."""
        labels = np.array([0, 0, 0, 1, 1, 0, 0, 0, 0])
        result = median_smooth(labels, window=7)
        np.testing.assert_array_equal(result, [0, 0, 0, 0, 0, 0, 0, 0, 0])

    def test_window_1_identity(self):
        """Window=1 should return identical labels."""
        labels = np.array([3, 1, 2, 1, 3])
        result = median_smooth(labels, window=1)
        np.testing.assert_array_equal(result, labels)

    def test_even_window_raises(self):
        with pytest.raises(ValueError, match="odd"):
            median_smooth(np.array([1, 2, 3]), window=4)

    def test_preserves_long_bouts(self):
        """Long uniform bouts are untouched."""
        labels = np.array([0] * 20 + [1] * 20 + [2] * 20)
        result = median_smooth(labels, window=5)
        np.testing.assert_array_equal(result, labels)


# ------------------------------------------------------------------ #
# min_bout_filter
# ------------------------------------------------------------------ #

class TestMinBoutFilter:
    def test_removes_single_frame_bout(self):
        labels = np.array([1, 1, 2, 1, 1])
        result = min_bout_filter(labels, min_frames=2)
        np.testing.assert_array_equal(result, [1, 1, 1, 1, 1])

    def test_keeps_long_bouts(self):
        labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        result = min_bout_filter(labels, min_frames=3)
        np.testing.assert_array_equal(result, labels)

    def test_min_frames_1_identity(self):
        labels = np.array([0, 1, 0, 1, 0])
        result = min_bout_filter(labels, min_frames=1)
        np.testing.assert_array_equal(result, labels)

    def test_different_min_frames(self):
        """With min_frames=4, bouts of length 3 are removed."""
        labels = np.array([0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0])
        result = min_bout_filter(labels, min_frames=4)
        np.testing.assert_array_equal(result, [0] * 11)

    def test_short_bout_between_different_labels(self):
        """Short bout between two different labels → assign to longer neighbour."""
        labels = np.array([0, 0, 0, 0, 2, 1, 1])
        result = min_bout_filter(labels, min_frames=2)
        # bout of 2 at label=0..0(4) is fine, bout of 2(1) short,
        # prev_len=4 > next_len=2 → replaced with 0
        np.testing.assert_array_equal(result, [0, 0, 0, 0, 0, 1, 1])

    def test_cascading_removal(self):
        """Removing one bout may create a new short bout — should converge."""
        labels = np.array([0, 0, 1, 2, 0, 0])
        result = min_bout_filter(labels, min_frames=2)
        np.testing.assert_array_equal(result, [0, 0, 0, 0, 0, 0])


# ------------------------------------------------------------------ #
# smooth_labels
# ------------------------------------------------------------------ #

class TestSmoothLabels:
    def test_method_median_only(self):
        labels = np.array([1, 1, 2, 1, 1])
        result = smooth_labels(labels, method="median", window=3)
        np.testing.assert_array_equal(result, [1, 1, 1, 1, 1])

    def test_method_min_bout_only(self):
        labels = np.array([0, 0, 0, 1, 0, 0, 0])
        result = smooth_labels(labels, method="min_bout", min_bout_frames=2)
        np.testing.assert_array_equal(result, [0, 0, 0, 0, 0, 0, 0])

    def test_method_both(self):
        labels = np.array([1, 1, 2, 1, 1, 3, 1, 1, 1])
        result = smooth_labels(labels, method="both", window=3, min_bout_frames=2)
        np.testing.assert_array_equal(result, [1, 1, 1, 1, 1, 1, 1, 1, 1])

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="method"):
            smooth_labels(np.array([1, 2]), method="invalid")


# ------------------------------------------------------------------ #
# Edge cases
# ------------------------------------------------------------------ #

class TestEdgeCases:
    def test_empty_array(self):
        empty = np.array([], dtype=np.int32)
        np.testing.assert_array_equal(median_smooth(empty), [])
        np.testing.assert_array_equal(min_bout_filter(empty), [])
        np.testing.assert_array_equal(smooth_labels(empty), [])

    def test_single_element(self):
        single = np.array([5])
        np.testing.assert_array_equal(median_smooth(single), [5])
        np.testing.assert_array_equal(min_bout_filter(single), [5])
        np.testing.assert_array_equal(smooth_labels(single), [5])

    def test_all_same_label(self):
        same = np.array([3, 3, 3, 3, 3])
        np.testing.assert_array_equal(smooth_labels(same), same)

    def test_fewer_transitions_after_smoothing(self):
        """Smoothing should never increase the number of transitions."""
        labels = np.array([0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0])
        smoothed = smooth_labels(labels, method="both", window=3, min_bout_frames=2)
        orig_transitions = np.sum(np.diff(labels) != 0)
        smooth_transitions = np.sum(np.diff(smoothed) != 0)
        assert smooth_transitions <= orig_transitions

    def test_temporal_coherence_improves(self):
        """Temporal coherence (fraction of same-as-previous) should improve."""
        labels = np.array([0, 1, 0, 1, 0, 0, 2, 0, 0, 0, 1, 0, 0])
        smoothed = smooth_labels(labels, method="both", window=3, min_bout_frames=2)
        orig_tc = np.mean(labels[1:] == labels[:-1])
        smooth_tc = np.mean(smoothed[1:] == smoothed[:-1])
        assert smooth_tc >= orig_tc
