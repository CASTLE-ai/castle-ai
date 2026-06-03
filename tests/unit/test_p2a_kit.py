"""Tests for P2a KIT pre-process speedups (2026-06).

* _threaded_iter: order-preserving, exception-propagating, deadlock-safe on
  early consumer exit (the producer/consumer overlap used by the encode loops).
* _encode_stabilized_video: the threaded decode→warp→encode pipeline runs
  end-to-end on real PyAV and produces a valid, correctly-sized video in order.
* extract_orientations_from_masks(body_pos=...): identical result to recomputing
  the body centroids (the redundant-H5-sweep removal is output-preserving).
"""

import numpy as np
import pytest


# --------------------------------------------------------------------------- #
# _threaded_iter
# --------------------------------------------------------------------------- #

def test_threaded_iter_preserves_order():
    from castle.service.preprocessing_service import _threaded_iter

    def producer():
        for i in range(200):
            yield (i, i * i)

    assert list(_threaded_iter(producer, maxsize=8)) == [(i, i * i) for i in range(200)]


def test_threaded_iter_propagates_exception():
    from castle.service.preprocessing_service import _threaded_iter

    def producer():
        yield 1
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        list(_threaded_iter(producer))


def test_threaded_iter_early_exit_no_deadlock():
    from castle.service.preprocessing_service import _threaded_iter

    def producer():
        for i in range(100000):   # far more than maxsize → producer blocks on put
            yield i

    it = _threaded_iter(producer, maxsize=4)
    first = next(it)
    it.close()   # consumer stops early — must not hang (stop + drain + join)
    assert first == 0


# --------------------------------------------------------------------------- #
# _encode_stabilized_video (threaded pipeline, real PyAV)
# --------------------------------------------------------------------------- #

class _FakeCam:
    """Minimal stand-in for StabilizedCamera: resize each frame to size×size."""

    def __init__(self, size):
        self.size = size

    def generate_frame(self, img_bgr, i):
        import cv2
        return cv2.resize(img_bgr, (self.size, self.size))


def test_encode_stabilized_video_threaded_smoke(synthetic_video, tmp_path):
    import av
    from castle.service.preprocessing_service import _encode_stabilized_video

    out_path = tmp_path / "kit_out.mp4"
    size = 64
    _encode_stabilized_video(
        video_path=str(synthetic_video),
        cam=_FakeCam(size),
        out_path=str(out_path),
        fps=30.0,
        n_frames=10,
        output_size=size,
    )

    assert out_path.exists()
    with av.open(str(out_path)) as c:
        frames = list(c.decode(c.streams.video[0]))
    assert len(frames) == 10                       # all frames encoded, in order
    assert (frames[0].width, frames[0].height) == (size, size)


# --------------------------------------------------------------------------- #
# extract_orientations_from_masks(body_pos=...) equivalence
# --------------------------------------------------------------------------- #

def _make_mask_h5(path, n_frames=20):
    from castle.utils.h5_io import H5IO

    with H5IO(str(path)) as h5:
        for i in range(n_frames):
            m = np.zeros((50, 50), dtype=np.uint8)
            m[10:20, 10:20] = 1          # body ROI
            m[30:40, 30:40] = 2          # head ROI
            h5.write_mask(i, m)


def test_extract_orientations_body_pos_equivalence(tmp_path):
    from castle.core.stabilized_camera import (
        extract_centroids_from_masks,
        extract_orientations_from_masks,
    )

    h5_path = tmp_path / "mask_list.h5"
    n = 20
    _make_mask_h5(h5_path, n)

    without = extract_orientations_from_masks(str(h5_path), 1, 2, n)
    body = extract_centroids_from_masks(str(h5_path), 1, n)
    with_bp = extract_orientations_from_masks(str(h5_path), 1, 2, n, body_pos=body)

    np.testing.assert_array_equal(without, with_bp)
