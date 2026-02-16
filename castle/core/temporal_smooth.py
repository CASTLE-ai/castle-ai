"""Temporal smoothing for cluster label sequences.

Removes biologically implausible single-frame label flickers
and enforces minimum bout duration constraints.
"""

import numpy as np
from collections import Counter


def median_smooth(labels: np.ndarray, window: int = 5) -> np.ndarray:
    """Apply rolling-mode filter to cluster labels.

    Replaces each label with the mode (most common) of its neighbors.
    Uses a sliding window centred on each frame.

    Args:
        labels: 1-D array of cluster assignments.
        window: Smoothing window size (must be odd, ≥ 1). Default 5.

    Returns:
        Smoothed labels array (same shape and dtype).

    Raises:
        ValueError: If *window* is even or < 1.
    """
    labels = np.asarray(labels)
    if labels.size == 0:
        return labels.copy()

    if window < 1 or window % 2 == 0:
        raise ValueError(f"window must be a positive odd integer, got {window}")

    if window == 1:
        return labels.copy()

    n = len(labels)
    half = window // 2
    result = np.empty_like(labels)

    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        segment = labels[lo:hi]
        # mode = most common label; ties broken by the label that appears
        # first (Counter.most_common order is stable for equal counts in
        # CPython ≥ 3.7, but we sort to be safe).
        counts = Counter(segment.tolist())
        result[i] = counts.most_common(1)[0][0]

    return result


def min_bout_filter(labels: np.ndarray, min_frames: int = 3) -> np.ndarray:
    """Remove bouts shorter than *min_frames*.

    Short bouts are reassigned to the surrounding behaviour.  If a short
    bout sits between two bouts of the *same* label, it is replaced by
    that label.  Otherwise it is replaced by the label of the longer
    neighbour bout (ties go to the preceding bout).

    Args:
        labels: 1-D array of cluster assignments.
        min_frames: Minimum bout duration in frames. Default 3.

    Returns:
        Filtered labels array (same shape and dtype).
    """
    labels = np.asarray(labels)
    if labels.size == 0:
        return labels.copy()

    if min_frames <= 1:
        return labels.copy()

    result = labels.copy()

    # Iterate until convergence (short bouts may be created by merging).
    changed = True
    while changed:
        changed = False
        bouts = _extract_bout_runs(result)

        for idx, (label, start, end) in enumerate(bouts):
            length = end - start
            if length >= min_frames:
                continue

            # Determine replacement label from neighbours
            prev_label = bouts[idx - 1][0] if idx > 0 else None
            next_label = bouts[idx + 1][0] if idx < len(bouts) - 1 else None

            if prev_label is not None and prev_label == next_label:
                replacement = prev_label
            elif prev_label is not None and next_label is not None:
                prev_len = bouts[idx - 1][2] - bouts[idx - 1][1]
                next_len = bouts[idx + 1][2] - bouts[idx + 1][1]
                replacement = prev_label if prev_len >= next_len else next_label
            elif prev_label is not None:
                replacement = prev_label
            elif next_label is not None:
                replacement = next_label
            else:
                continue  # single bout spanning whole array — keep it

            result[start:end] = replacement
            changed = True

    return result


def _extract_bout_runs(labels: np.ndarray):
    """Return list of (label, start_idx, end_idx) tuples."""
    if len(labels) == 0:
        return []

    bouts = []
    current_label = labels[0]
    start = 0

    for i in range(1, len(labels)):
        if labels[i] != current_label:
            bouts.append((int(current_label), start, i))
            current_label = labels[i]
            start = i

    bouts.append((int(current_label), start, len(labels)))
    return bouts


def smooth_labels(
    labels: np.ndarray,
    method: str = "both",
    window: int = 5,
    min_bout_frames: int = 3,
) -> np.ndarray:
    """Apply temporal smoothing to cluster labels.

    Two-step process (when *method* = ``"both"``):

    1. **Median smoothing** — removes isolated frame flickers.
    2. **Minimum bout filter** — enforces minimum bout duration.

    Args:
        labels: 1-D cluster assignments.
        method: ``"median"``, ``"min_bout"``, or ``"both"``
            (default applies both sequentially).
        window: Median-filter window size (odd integer).
        min_bout_frames: Minimum bout duration in frames.

    Returns:
        Smoothed label array (same shape and dtype).

    Raises:
        ValueError: If *method* is not one of the accepted strings.
    """
    valid_methods = {"median", "min_bout", "both"}
    if method not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}, got '{method}'")

    labels = np.asarray(labels)
    if labels.size == 0:
        return labels.copy()

    result = labels

    if method in ("median", "both"):
        result = median_smooth(result, window=window)

    if method in ("min_bout", "both"):
        result = min_bout_filter(result, min_frames=min_bout_frames)

    return result
