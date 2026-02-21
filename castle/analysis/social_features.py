"""Social behaviour feature extraction for multi-subject tracking data.

Provides pairwise geometric measures (distance, relative orientation) and
higher-level interaction scoring (approach/avoidance score, social event
detection) derived from a list of :class:`~castle.core.multi_subject.SubjectTrack`
objects.

All functions operate on the *positions* and *angles* fields of each track and
assume all tracks are synchronised (same number of frames).

Usage example::

    from castle.analysis.social_features import (
        compute_pairwise_distance,
        compute_relative_orientation,
        compute_approach_score,
        detect_social_events,
    )

    dist = compute_pairwise_distance(tracks)   # (N, S, S)
    orient = compute_relative_orientation(tracks)  # (N, S, S)
    approach = compute_approach_score(tracks)  # (N, S, S)
    events = detect_social_events(tracks)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from castle.core.multi_subject import SubjectTrack

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_tracks(tracks: list[SubjectTrack]) -> tuple[int, int]:
    """Return (n_frames, n_subjects) after basic sanity checks.

    Raises
    ------
    ValueError
        If *tracks* is empty or subjects have different frame counts.
    """
    if not tracks:
        raise ValueError("tracks must be a non-empty list.")
    n_frames = tracks[0].n_frames
    for t in tracks[1:]:
        if t.n_frames != n_frames:
            raise ValueError(
                f"Track {t.subject_id} has {t.n_frames} frames; "
                f"expected {n_frames} (from subject {tracks[0].subject_id})."
            )
    return n_frames, len(tracks)


# ---------------------------------------------------------------------------
# Pairwise geometry
# ---------------------------------------------------------------------------


def compute_pairwise_distance(
    tracks: list[SubjectTrack],
) -> np.ndarray:
    """Compute Euclidean pairwise distance between all subjects per frame.

    Parameters
    ----------
    tracks : list[SubjectTrack]
        Synchronised subject tracks (each must have a ``positions`` array).

    Returns
    -------
    np.ndarray, shape (N, n_subjects, n_subjects)
        Symmetric distance matrix per frame.  ``dist[t, i, j]`` is the pixel
        distance between subject *i* and subject *j* at frame *t*.  Diagonal
        entries are 0.
    """
    n_frames, n_subj = _validate_tracks(tracks)

    # (n_subjects, N, 2)
    pos_stack = np.stack([t.positions for t in tracks], axis=0)

    dist = np.zeros((n_frames, n_subj, n_subj), dtype=np.float64)
    for i in range(n_subj):
        for j in range(i + 1, n_subj):
            d = np.linalg.norm(pos_stack[i] - pos_stack[j], axis=1)  # (N,)
            dist[:, i, j] = d
            dist[:, j, i] = d

    logger.debug(
        "compute_pairwise_distance: %d subjects, %d frames, "
        "mean inter-subject distance=%.1f px",
        n_subj,
        n_frames,
        float(dist[dist > 0].mean()) if (dist > 0).any() else 0.0,
    )
    return dist


def compute_relative_orientation(
    tracks: list[SubjectTrack],
) -> np.ndarray:
    """Compute relative heading angle from subject *i* toward subject *j*.

    The relative orientation ``orient[t, i, j]`` is defined as the angle (in
    degrees, range (−180, 180]) between subject *i*'s heading direction and the
    vector pointing from *i* to *j* at frame *t*.

    A value near 0° means *i* is facing *j*; ±180° means *i* faces directly
    away.  Diagonal entries are 0.

    Parameters
    ----------
    tracks : list[SubjectTrack]
        Synchronised subject tracks.

    Returns
    -------
    np.ndarray, shape (N, n_subjects, n_subjects)
        Relative orientation matrix (degrees).
    """
    n_frames, n_subj = _validate_tracks(tracks)

    pos_stack = np.stack([t.positions for t in tracks], axis=0)  # (S, N, 2)
    ang_stack = np.stack([t.angles for t in tracks], axis=0)  # (S, N)

    orient = np.zeros((n_frames, n_subj, n_subj), dtype=np.float64)
    for i in range(n_subj):
        for j in range(n_subj):
            if i == j:
                continue
            # Vector from i to j
            vec_ij = pos_stack[j] - pos_stack[i]  # (N, 2)
            # Angle of that vector
            angle_to_j = np.rad2deg(
                np.arctan2(vec_ij[:, 1], vec_ij[:, 0])
            )  # (N,)
            # Relative angle: angle_to_j minus heading of i
            rel = angle_to_j - ang_stack[i]  # (N,)
            # Wrap to (-180, 180]
            rel = (rel + 180.0) % 360.0 - 180.0
            orient[:, i, j] = rel

    return orient


# ---------------------------------------------------------------------------
# Approach / avoidance
# ---------------------------------------------------------------------------


def compute_approach_score(
    tracks: list[SubjectTrack],
    window: int = 30,
) -> np.ndarray:
    """Compute per-pair approach/avoidance score over a sliding window.

    The score for pair (i, j) at frame *t* is defined as the **negative
    distance derivative** (in pixels/frame) averaged over a centred window of
    *window* frames:

    .. math::

        A_{ij}(t) = -\\frac{1}{w} \\sum_{k=-w/2}^{w/2} \\Delta d_{ij}(t+k)

    where :math:`\\Delta d_{ij}` is the frame-to-frame change in distance.
    Positive values mean the pair is *approaching*; negative means *receding*.

    Edge frames use half-windows (no zero-padding — scores near edges have
    fewer contributing frames but remain unbiased).

    Parameters
    ----------
    tracks : list[SubjectTrack]
        Synchronised subject tracks.
    window : int
        Number of frames for the sliding average (default 30).

    Returns
    -------
    np.ndarray, shape (N, n_subjects, n_subjects)
        Approach score matrix.  ``score[t, i, j] == score[t, j, i]``.
        Diagonal entries are 0.
    """
    dist = compute_pairwise_distance(tracks)  # (N, S, S)
    n_frames, n_subj, _ = dist.shape

    # First derivative of distance: (N-1, S, S)
    delta_d = np.diff(dist, axis=0)

    # Score = –mean(Δd) over a sliding window; shape (N, S, S)
    half = window // 2
    scores = np.zeros_like(dist)

    for t in range(n_frames):
        t_start = max(0, t - half)
        t_end = min(n_frames - 1, t + half)
        # delta_d[k] represents change from frame k to k+1
        d_start = t_start
        d_end = t_end  # delta_d has shape N-1
        if d_end > d_start and d_end <= n_frames - 1:
            window_slice = delta_d[d_start:d_end]
        elif d_end == d_start:
            window_slice = delta_d[d_start : d_start + 1]
        else:
            window_slice = delta_d[d_start:]
        if len(window_slice) > 0:
            scores[t] = -np.mean(window_slice, axis=0)

    logger.debug(
        "compute_approach_score: window=%d, score range [%.2f, %.2f]",
        window,
        float(scores.min()),
        float(scores.max()),
    )
    return scores


# ---------------------------------------------------------------------------
# Social event detection
# ---------------------------------------------------------------------------


def detect_social_events(
    tracks: list[SubjectTrack],
    distance_threshold: float = 50.0,
    duration_threshold: int = 15,
) -> list[dict]:
    """Detect social interaction events from pairwise proximity.

    An *interaction event* occurs when two subjects remain within
    *distance_threshold* pixels for at least *duration_threshold* consecutive
    frames.

    Parameters
    ----------
    tracks : list[SubjectTrack]
        Synchronised subject tracks.
    distance_threshold : float
        Maximum inter-subject distance (pixels) to count as proximate.
        Default 50 px.
    duration_threshold : int
        Minimum consecutive-frame duration (frames) for a valid event.
        Default 15 frames.

    Returns
    -------
    list[dict]
        Each element is a dictionary::

            {
                "type": "proximity",
                "subjects": (subject_id_i, subject_id_j),
                "start_frame": int,
                "end_frame": int,       # inclusive
                "duration": int,        # frames
            }

        Events are sorted by ``start_frame``.
    """
    n_frames, n_subj = _validate_tracks(tracks)
    dist = compute_pairwise_distance(tracks)  # (N, S, S)
    events: list[dict] = []

    for i in range(n_subj):
        for j in range(i + 1, n_subj):
            close = dist[:, i, j] < distance_threshold  # (N,) boolean

            # Find contiguous runs of True
            run_start: int | None = None
            for t in range(n_frames):
                if close[t] and run_start is None:
                    run_start = t
                elif (not close[t]) and run_start is not None:
                    duration = t - run_start
                    if duration >= duration_threshold:
                        events.append(
                            {
                                "type": "proximity",
                                "subjects": (
                                    tracks[i].subject_id,
                                    tracks[j].subject_id,
                                ),
                                "start_frame": run_start,
                                "end_frame": t - 1,
                                "duration": duration,
                            }
                        )
                    run_start = None

            # Close any open run at end of video
            if run_start is not None:
                duration = n_frames - run_start
                if duration >= duration_threshold:
                    events.append(
                        {
                            "type": "proximity",
                            "subjects": (
                                tracks[i].subject_id,
                                tracks[j].subject_id,
                            ),
                            "start_frame": run_start,
                            "end_frame": n_frames - 1,
                            "duration": duration,
                        }
                    )

    events.sort(key=lambda e: e["start_frame"])
    logger.info(
        "detect_social_events: %d events detected "
        "(dist_threshold=%.1f px, dur_threshold=%d frames)",
        len(events),
        distance_threshold,
        duration_threshold,
    )
    return events
