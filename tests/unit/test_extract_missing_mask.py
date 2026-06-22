"""Unit tests for tolerating frames with no tracked mask during extraction.

When the tracker loses the object on a frame, that frame's key is absent from
``mask_list.h5`` and ``H5IO.read_mask`` raises ``ValueError("Without mask at
frame N")``. Previously that escaped the per-frame skip logic and crashed the
whole DataLoader worker. The fix routes a missing mask through the same skip
path as an absent ROI: a blank frame + all-zero mask (which the model turns into
a NaN-placeholder latent row, kept in place to preserve frame alignment).

These are pure unit tests — the VideoReader / H5IO are replaced with light fakes,
so no real video, HDF5 file, model, or GPU is needed.
"""

import numpy as np
import pytest

from castle.core.data import VideoDataset, Preprocess
from castle.core.types import ROINotFoundError
from castle.core import extractor as extractor_mod


class _FakeReader:
    def __init__(self, h=64, w=64):
        self.h, self.w = h, w

    def __getitem__(self, idx):
        return np.zeros((self.h, self.w, 3), dtype=np.uint8)

    def __len__(self):
        return 30


class _FakeTracker:
    """``read_mask`` raises ValueError for 'missing' frames, like real H5IO."""

    def __init__(self, missing):
        self.missing = set(missing)

    def read_mask(self, idx):
        if idx in self.missing:
            raise ValueError(f"Without mask at frame {idx}")
        m = np.zeros((64, 64), dtype=np.uint8)
        m[10:40, 10:40] = 1
        return m


def _make_dataset(missing, on_frame_error="skip", center=False):
    pre = Preprocess(
        center_roi_switch=center, center_roi_id=1,
        center_roi_crop_width=48, center_roi_crop_height=48,
    )
    ds = VideoDataset("video.mp4", 30, "mask.h5", pre, select_roi=1,
                      on_frame_error=on_frame_error)
    # Pre-seed the lazily-opened handles so __getitem__ uses the fakes.
    ds.reader = _FakeReader()
    ds.tracker = _FakeTracker(missing)
    return ds


def test_present_frame_returns_real_pair():
    ds = _make_dataset(missing=set())
    frame, mask = ds[0]
    assert mask.shape == frame.shape[:2]
    assert mask.max() == 1


def test_missing_mask_is_skipped_not_crashed():
    ds = _make_dataset(missing={5})
    frame, mask = ds[5]          # must NOT raise
    assert mask.ndim == 2 and mask.shape == frame.shape[:2]
    assert mask.max() == 0       # all-zero → model emits a NaN gap downstream


def test_missing_mask_skip_pair_matches_crop_dims_when_centered():
    # With centering on, real frames are crop-sized; the skip pair must match so
    # it collates into the same batch.
    ds = _make_dataset(missing={5}, center=True)
    frame, mask = ds[5]
    assert frame.shape[:2] == (48, 48)
    assert mask.shape == (48, 48) and mask.max() == 0


def test_missing_mask_raises_in_strict_mode():
    ds = _make_dataset(missing={5}, on_frame_error="raise")
    with pytest.raises(ROINotFoundError):
        _ = ds[5]


# --- extractor accounting/guard helpers ------------------------------------

def test_count_nan_rows_counts_all_nan_rows():
    arr = np.ones((10, 4), dtype=np.float32)
    arr[3] = np.nan
    arr[7] = np.nan
    assert extractor_mod._count_nan_rows(arr) == 2
    assert extractor_mod._count_nan_rows(np.empty((0, 4), np.float32)) == 0


def test_guard_near_total_skip_allows_partial_gaps():
    # 30% gaps is fine — extraction should complete.
    extractor_mod._guard_near_total_skip("vid", n_total=100, n_skipped=30)


def test_guard_near_total_skip_aborts_when_essentially_empty():
    from castle.core.types import ExtractionError
    with pytest.raises(ExtractionError):
        extractor_mod._guard_near_total_skip("vid", n_total=100, n_skipped=100)
    with pytest.raises(ExtractionError):
        extractor_mod._guard_near_total_skip("vid", n_total=0, n_skipped=0)
