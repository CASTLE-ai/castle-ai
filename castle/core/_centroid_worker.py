"""Process-pool worker for parallel centroid extraction (KIT pre-process).

Kept deliberately tiny with **lazy** heavy imports (``cv2``/``h5py``/``numpy``/
``H5IO`` are imported inside the worker, not at module top) so that a
``forkserver``/``spawn`` start-method imports a light template per child rather
than the whole ``castle`` package graph (scipy, torch via transitive imports).

The worker reads a disjoint, contiguous frame range from its OWN read-only HDF5
handle and returns RAW per-frame body+head centroids (NaN where missing). Global
NaN-interpolation is done by the orchestrator AFTER all chunks return — a chunk
must never interpolate locally, because a NaN at a chunk edge needs neighbours
from the adjacent chunk.
"""

from __future__ import annotations


class PreprocessCancelled(Exception):
    """Raised when a pre-process run is cancelled mid-flight (centroid pool,
    encode loop or mask-transform loop). A normal ``Exception`` so it lands in
    the service's ``except BaseException`` partial-output cleanup."""


def _largest_component_centroid(cv2, np, binary):
    """Centroid [x, y] of the largest connected component of a uint8 binary
    mask, or ``None`` if there is no foreground. Mirrors the legacy serial
    logic in ``stabilized_camera.extract_centroids_from_masks``."""
    if binary.sum() == 0:
        return None
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8, ltype=cv2.CV_32S
    )
    if num_labels <= 1:
        return None
    areas = [stats[j, cv2.CC_STAT_AREA] for j in range(1, num_labels)]
    best = int(np.argmax(areas)) + 1  # +1: label 0 is background
    return centroids[best, 0], centroids[best, 1]


def centroid_chunk_worker(args):
    """Extract body+head centroids for frames ``[start, stop)``.

    Parameters
    ----------
    args : tuple
        ``(mask_h5_path, body_roi_id, head_roi_id, start, stop)``. No
        synchronization primitive is passed in — an Event/Condition is
        unpicklable to a pool worker (submit pickles its args), which is why the
        old per-frame cancel check was removed; the orchestrator cancels pending
        chunks via ``shutdown(cancel_futures=True)`` instead.

    Returns
    -------
    (start, body_chunk, head_chunk)
        ``body_chunk`` / ``head_chunk`` are ``float64`` arrays of shape
        ``(stop - start, 2)`` with NaN for frames lacking a valid centroid.
    """
    import cv2  # noqa: F401 — lazy, keeps forkserver template light
    import numpy as np

    from castle.utils.h5_io import H5IO

    (mask_h5_path, body_roi_id, head_roi_id, start, stop) = args

    body_roi_id = int(body_roi_id)
    head_roi_id = int(head_roi_id)
    n = stop - start
    body = np.full((n, 2), np.nan, dtype=np.float64)
    head = np.full((n, 2), np.nan, dtype=np.float64)

    with H5IO(mask_h5_path, read_only=True) as h5:
        for j in range(n):
            i = start + j
            if h5.has_mask(i):
                try:
                    mask = h5.read_mask(i)
                except Exception:
                    mask = None
                if mask is not None:
                    b = _largest_component_centroid(
                        cv2, np, (mask == body_roi_id).astype(np.uint8))
                    if b is not None:
                        body[j, 0], body[j, 1] = b
                    hd = _largest_component_centroid(
                        cv2, np, (mask == head_roi_id).astype(np.uint8))
                    if hd is not None:
                        head[j, 0], head[j, 1] = hd

    return start, body, head
