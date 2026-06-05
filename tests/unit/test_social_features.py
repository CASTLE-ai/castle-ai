"""Tests for castle.analysis.social_features + SubjectTrack.valid_frames (PR2 Stage 5).

Tracking-loss frames are interpolated in SubjectTrack.positions; social metrics
must not treat those fabricated positions as real observations (theme D).
"""

import numpy as np
import pytest

from castle.core.multi_subject import SubjectTrack
from castle.analysis.social_features import (
    compute_pairwise_distance,
    compute_relative_orientation,
    compute_approach_score,
    detect_social_events,
)


def _track(sid, positions, angles=None, valid=None):
    n = len(positions)
    return SubjectTrack(
        subject_id=sid,
        body_roi_id=sid * 2,
        head_roi_id=sid * 2 + 1,
        positions=np.asarray(positions, dtype=float),
        angles=np.zeros(n) if angles is None else np.asarray(angles, dtype=float),
        valid_frames=valid,
    )


def test_subject_track_valid_frames_defaults_all_true():
    """Backward compatible: a track built without valid_frames is all-valid."""
    t = _track(0, [[0, 0]] * 5)
    assert t.valid_frames.dtype == bool
    assert t.valid_frames.all()
    assert len(t.valid_frames) == 5


def test_pairwise_distance_nan_at_invalid_frames():
    N = 10
    p0 = np.zeros((N, 2))
    p1 = np.full((N, 2), 10.0)
    valid1 = np.ones(N, dtype=bool)
    valid1[3:6] = False  # subject 1 interpolated on frames 3,4,5
    dist = compute_pairwise_distance([_track(0, p0), _track(1, p1, valid=valid1)])
    assert np.isnan(dist[3:6, 0, 1]).all()
    assert np.isnan(dist[4, 1, 0])      # symmetric
    assert np.isfinite(dist[0, 0, 1])   # valid frame unaffected


def test_social_event_only_in_valid_window():
    """Close throughout, but subject 1 is interpolated from frame 10 → events
    cover only the real window."""
    N = 30
    p0 = np.zeros((N, 2))
    p1 = np.full((N, 2), 5.0)           # dist ~7 px < threshold
    valid1 = np.ones(N, dtype=bool)
    valid1[10:] = False
    events = detect_social_events(
        [_track(0, p0), _track(1, p1, valid=valid1)],
        distance_threshold=50.0, duration_threshold=5,
    )
    assert events, "expected a proximity event in the valid window"
    assert all(e["end_frame"] < 10 for e in events)


def test_no_fake_event_when_close_only_during_interpolation():
    """If the pair is 'close' ONLY during interpolated frames, no event fires."""
    N = 30
    p0 = np.zeros((N, 2))
    p1 = np.full((N, 2), 100.0)         # normally far (dist ~141 > 50)
    p1[10:20] = 5.0                     # interpolated fill-in happens to be close
    valid1 = np.ones(N, dtype=bool)
    valid1[10:20] = False
    events = detect_social_events(
        [_track(0, p0), _track(1, p1, valid=valid1)],
        distance_threshold=50.0, duration_threshold=3,
    )
    assert events == []


def test_relative_orientation_coincident_is_nan():
    p0 = np.array([[0, 0], [0, 0], [1, 1], [2, 2]], dtype=float)
    p1 = np.array([[0, 0], [5, 0], [1, 1], [9, 9]], dtype=float)  # frames 0,2 coincide
    orient = compute_relative_orientation([_track(0, p0), _track(1, p1)])
    assert np.isnan(orient[0, 0, 1])
    assert np.isnan(orient[2, 0, 1])
    assert np.isfinite(orient[1, 0, 1])


def test_approach_score_fps_scaling():
    N = 20
    p0 = np.zeros((N, 2))
    p1 = np.stack([np.linspace(100.0, 10.0, N), np.zeros(N)], axis=1)  # approaching
    tracks = [_track(0, p0), _track(1, p1)]
    s_frame = compute_approach_score(tracks, window=4)
    s_sec = compute_approach_score(tracks, window=4, fps=30.0)
    m = np.isfinite(s_frame) & np.isfinite(s_sec)
    assert np.allclose(s_sec[m], s_frame[m] * 30.0)
    assert np.nanmax(s_frame) > 0   # approaching → positive score
