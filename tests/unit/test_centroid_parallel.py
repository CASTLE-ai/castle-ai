"""Tests for the fused + parallel centroid extractor (Phase A).

`extract_body_head_centroids` reads each mask ONCE and computes body + head
centroids across a process pool. These assert it is numerically identical to the
legacy two-call `extract_centroids_from_masks` path (serial AND pool), that the
global NaN-interpolation spans chunk boundaries, and that an absent ROI raises.
"""

import h5py
import numpy as np
import numpy.testing as npt
import pytest

from castle.core.stabilized_camera import (
    extract_body_head_centroids,
    extract_centroids_from_masks,
)


BODY, HEAD = 1, 2


def _write_masks(path, masks):
    with h5py.File(path, "w") as f:
        for i in range(len(masks)):
            f.create_dataset(str(i), data=masks[i], dtype="uint8",
                             compression="gzip", compression_opts=3)


def _build(tmp_path, n=40, missing=()):
    """n frames; body square + head square at distinct, frame-varying spots.
    Frames in `missing` are left blank (→ NaN → interpolated)."""
    H, W = 160, 160
    masks = np.zeros((n, H, W), dtype=np.uint8)
    for i in range(n):
        if i in missing:
            continue
        bx, by = 40 + (i % 5), 50 + (i % 3)
        hx, hy = 100 + (i % 4), 110 + (i % 6)
        masks[i, by - 4:by + 5, bx - 4:bx + 5] = BODY
        masks[i, hy - 4:hy + 5, hx - 4:hx + 5] = HEAD
    p = str(tmp_path / "masks.h5")
    _write_masks(p, masks)
    return p


def _legacy(p, n):
    return (extract_centroids_from_masks(p, roi_id=BODY, n_frames=n),
            extract_centroids_from_masks(p, roi_id=HEAD, n_frames=n))


def test_serial_matches_legacy(tmp_path):
    p = _build(tmp_path, n=40)
    lb, lh = _legacy(p, 40)
    # max_workers=1 → serial fused path
    b, h = extract_body_head_centroids(p, BODY, HEAD, 40, max_workers=1)
    npt.assert_allclose(b, lb, atol=1e-9)
    npt.assert_allclose(h, lh, atol=1e-9)


def test_pool_matches_legacy(tmp_path):
    p = _build(tmp_path, n=40)
    lb, lh = _legacy(p, 40)
    # max_workers=2 forces the ProcessPool even for a short clip
    b, h = extract_body_head_centroids(p, BODY, HEAD, 40, max_workers=2)
    npt.assert_allclose(b, lb, atol=1e-9)
    npt.assert_allclose(h, lh, atol=1e-9)


def test_pool_interpolates_across_chunk_boundary(tmp_path):
    # Hole frames placed so at least one lands on a chunk edge; the pool path
    # must still interpolate globally (workers return NaN, orchestrator interps).
    missing = (0, 9, 10, 11, 39)
    p = _build(tmp_path, n=40, missing=missing)
    lb, lh = _legacy(p, 40)
    b, h = extract_body_head_centroids(p, BODY, HEAD, 40, max_workers=3)
    npt.assert_allclose(b, lb, atol=1e-9)
    npt.assert_allclose(h, lh, atol=1e-9)
    assert not np.isnan(b).any() and not np.isnan(h).any()


def test_absent_roi_raises(tmp_path):
    # head ROI present, body ROI never appears → body interp has no anchor.
    H, W, n = 100, 100, 20
    masks = np.zeros((n, H, W), dtype=np.uint8)
    masks[:, 40:51, 40:51] = HEAD  # only head, no body
    p = str(tmp_path / "nobody.h5")
    _write_masks(p, masks)
    with pytest.raises(ValueError, match="body"):
        extract_body_head_centroids(p, BODY, HEAD, n, max_workers=2)


def test_progress_callback_reaches_one(tmp_path):
    p = _build(tmp_path, n=30)
    seen = []
    extract_body_head_centroids(p, BODY, HEAD, 30, max_workers=2,
                                progress_callback=lambda f, d="": seen.append(f))
    assert seen and abs(seen[-1] - 1.0) < 1e-9
    assert all(0.0 <= f <= 1.0 for f in seen)
