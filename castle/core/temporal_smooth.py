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

    The algorithm runs in a single linear pass over the bout list, avoiding
    the O(n²) re-scan of the previous while-loop implementation.

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

    # Build a mutable list of bouts: each entry is [label, start, end].
    # We work on this list in a single left-to-right pass, merging short bouts
    # into their neighbours without restarting the scan.
    bouts = [[lbl, s, e] for lbl, s, e in _extract_bout_runs(labels)]

    i = 0
    while i < len(bouts):
        lbl, start, end = bouts[i]
        length = end - start

        if length >= min_frames:
            i += 1
            continue

        # Determine replacement label from immediate neighbours.
        prev_lbl = bouts[i - 1][0] if i > 0 else None
        next_lbl = bouts[i + 1][0] if i < len(bouts) - 1 else None

        if prev_lbl is None and next_lbl is None:
            # Single bout spanning the whole array — nothing to merge into.
            i += 1
            continue

        if prev_lbl is not None and prev_lbl == next_lbl:
            replacement = prev_lbl
        elif prev_lbl is not None and next_lbl is not None:
            prev_len = bouts[i - 1][2] - bouts[i - 1][1]
            next_len = bouts[i + 1][2] - bouts[i + 1][1]
            replacement = prev_lbl if prev_len >= next_len else next_lbl
        elif prev_lbl is not None:
            replacement = prev_lbl
        else:
            replacement = next_lbl

        bouts[i][0] = replacement

        # Merge the now-relabelled bout with adjacent bouts that share the
        # same label, so the combined bout is considered in subsequent steps.
        # Merge with next neighbour first (to keep index arithmetic simple).
        if i < len(bouts) - 1 and bouts[i + 1][0] == replacement:
            bouts[i][2] = bouts[i + 1][2]
            del bouts[i + 1]

        if i > 0 and bouts[i - 1][0] == replacement:
            bouts[i - 1][2] = bouts[i][2]
            del bouts[i]
            i -= 1  # re-examine the merged bout in case it's still short

        # Do NOT advance i — re-check the current (possibly merged) bout.

    # Reconstruct the result array from the (possibly modified) bout list.
    result = labels.copy()
    for lbl, start, end in bouts:
        result[start:end] = lbl

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
