"""Tests for castle.analysis.group_ethogram plotting (PR1 Stage 1.3).

Locks two fixes:
- -1 (unlabeled gap) frames must not crash the raster (previously KeyError on
  a colour-map key that excludes -1).
- BrokenBarHCollection (removed in matplotlib >= 3.10) replaced by
  ax.broken_barh, which is stable across versions.
"""

import os

import numpy as np
import pytest


def _ethogram_dict(labels_by_subject, fps=10.0, social_events=None):
    subject_ids = sorted(labels_by_subject)
    n = max(len(v) for v in labels_by_subject.values())
    return {
        "n_subjects": len(subject_ids),
        "subject_ids": subject_ids,
        "per_subject": {
            sid: {"cluster_names": {0: "rest", 1: "move"}, "labels": np.asarray(lbl)}
            for sid, lbl in labels_by_subject.items()
        },
        "time_axis": np.arange(n) / fps,
        "social_events": social_events or [],
        "fps": fps,
    }


class TestPlotGroupEthogramNoise:
    def test_minus_one_does_not_crash(self, tmp_path):
        """A -1 gap frame leaves a blank span instead of raising KeyError."""
        from castle.analysis.group_ethogram import plot_group_ethogram

        d = _ethogram_dict({
            0: [0, 0, -1, 1, 1, 0],   # -1 in the middle
            1: [1, 1, 1, -1, 0, 0],
        }, social_events=[{"start_frame": 1, "end_frame": 2}])
        out = tmp_path / "group_etho.png"
        result = plot_group_ethogram(d, str(out))
        assert os.path.exists(result)
        assert os.path.getsize(result) > 0

    def test_all_noise_subject_does_not_crash(self, tmp_path):
        """A subject whose labels are entirely -1 renders an empty (blank) row."""
        from castle.analysis.group_ethogram import plot_group_ethogram

        d = _ethogram_dict({0: [-1, -1, -1, -1], 1: [0, 0, 1, 1]})
        out = tmp_path / "group_etho2.png"
        result = plot_group_ethogram(d, str(out))
        assert os.path.exists(result)
        assert os.path.getsize(result) > 0
