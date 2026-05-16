"""Error-handling regression tests (P0-C).

Covers:

- :class:`CastleError` hierarchy shape (3 sub-bases + 7 leaves) and the
  ``ExtractionResult`` dataclass.
- :class:`Preprocess` now raises ``ROINotFoundError`` instead of silently
  returning a blank frame when the centre ROI is absent.
- :class:`VideoDataset` honours ``on_frame_error`` — "raise" propagates, "skip"
  swaps in a blank.
- ``extract_roi_latent_from_video`` raises ``MaskNotFoundError`` when its
  mask file is missing (no silent ``return ""``).
- Internal abort threshold ``max(1, int(rate * total))`` floors at 1.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest


# ---- CastleError hierarchy --------------------------------------------------

def test_castle_error_hierarchy_structure() -> None:
    from castle.core.types import (
        CastleAlgorithmError,
        CastleDataError,
        CastleError,
        CastleIOError,
        ExtractionError,
        InsufficientDataError,
        LatentCorruptError,
        MaskNotFoundError,
        NoClustersFound,
        PreprocessingError,
        ROINotFoundError,
        VideoReadError,
    )

    # Three sub-bases all derive from CastleError
    for base in (CastleIOError, CastleDataError, CastleAlgorithmError):
        assert issubclass(base, CastleError)
        assert issubclass(base, Exception)

    # Seven leaves slot into the expected sub-base
    assert issubclass(VideoReadError, CastleIOError)
    assert issubclass(MaskNotFoundError, CastleIOError)
    assert issubclass(LatentCorruptError, CastleIOError)
    assert issubclass(ROINotFoundError, CastleDataError)
    assert issubclass(InsufficientDataError, CastleDataError)
    assert issubclass(PreprocessingError, CastleDataError)
    assert issubclass(ExtractionError, CastleAlgorithmError)
    assert issubclass(NoClustersFound, CastleAlgorithmError)

    # Catch-the-root pattern works
    try:
        raise VideoReadError("boom")
    except CastleError as e:
        assert "boom" in str(e)


def test_extraction_result_dataclass() -> None:
    from castle.core.types import ExtractionResult

    res = ExtractionResult(
        latent_path=Path("/tmp/foo.npz"),
        n_frames=10,
        n_batches_failed=0,
        feature_dim=768,
    )
    assert res.latent_path == Path("/tmp/foo.npz")
    assert res.n_frames == 10
    assert res.feature_dim == 768
    # frozen dataclass — assignment must fail
    with pytest.raises(Exception):  # FrozenInstanceError subclasses AttributeError
        res.n_frames = 99  # type: ignore[misc]


# ---- Preprocess.transform raises -------------------------------------------

def _make_mask_with_rois(*roi_ids: int, shape=(64, 64)) -> np.ndarray:
    """Build a (H, W) uint8 mask with one band per requested ROI id."""
    mask = np.zeros(shape, dtype=np.uint8)
    h_band = shape[0] // max(1, len(roi_ids))
    for i, rid in enumerate(roi_ids):
        y0 = i * h_band
        y1 = (i + 1) * h_band if i < len(roi_ids) - 1 else shape[0]
        mask[y0:y1, :] = rid
    return mask


def test_preprocess_raises_roi_not_found() -> None:
    from castle.core.data import Preprocess
    from castle.core.types import ROINotFoundError

    pp = Preprocess(center_roi_switch=True, center_roi_id=7,
                    center_roi_crop_width=32, center_roi_crop_height=32)
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    mask = _make_mask_with_rois(1, 2)  # no id 7 present

    with pytest.raises(ROINotFoundError) as ei:
        pp.transform(frame, mask)
    assert "7" in str(ei.value)


def test_preprocess_passthrough_when_no_center_roi_switch() -> None:
    """With center_roi_switch=False the function should not raise on missing ROI."""
    from castle.core.data import Preprocess

    pp = Preprocess(center_roi_switch=False)
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=np.uint8)
    f, m = pp.transform(frame, mask)
    assert f.shape == (64, 64, 3)
    assert m.shape == (64, 64)


# ---- VideoDataset on_frame_error -------------------------------------------


class _StubReader:
    def __init__(self, frames):
        self._frames = frames

    def __getitem__(self, idx):
        return self._frames[idx]


class _StubTracker:
    def __init__(self, masks):
        self._masks = masks

    def read_mask(self, idx):
        return self._masks[idx]


def _build_dataset(masks, frames, on_frame_error, center_roi_id=5):
    """Create a VideoDataset whose I/O is mocked out at the worker layer."""
    from castle.core.data import Preprocess, VideoDataset

    ds = VideoDataset.__new__(VideoDataset)
    ds.video_path = "<stub>"
    ds.video_len = len(frames)
    ds.mask_path = "<stub>"
    ds.preprocess = Preprocess(
        center_roi_switch=True, center_roi_id=center_roi_id,
        center_roi_crop_width=32, center_roi_crop_height=32,
    )
    ds.select_roi = center_roi_id
    ds.rotate_deg = None
    ds.interpolated_points = None
    ds.on_frame_error = on_frame_error
    ds.reader = _StubReader(frames)
    ds.tracker = _StubTracker(masks)
    return ds


def test_video_dataset_skip_returns_blank() -> None:
    """on_frame_error='skip' returns the blank-page sentinel when ROI missing."""
    from castle.utils.video_align import blank_page

    frames = [np.zeros((64, 64, 3), dtype=np.uint8)]
    masks = [_make_mask_with_rois(1, 2)]  # no id 5
    ds = _build_dataset(masks, frames, on_frame_error="skip")

    pf, pm = ds[0]
    expected = blank_page(32, 32)
    assert pf.shape == expected.shape
    np.testing.assert_array_equal(pf, expected)


def test_video_dataset_raise_propagates() -> None:
    """on_frame_error='raise' propagates ROINotFoundError."""
    from castle.core.types import ROINotFoundError

    frames = [np.zeros((64, 64, 3), dtype=np.uint8)]
    masks = [_make_mask_with_rois(1, 2)]
    ds = _build_dataset(masks, frames, on_frame_error="raise")

    with pytest.raises(ROINotFoundError):
        ds[0]


# ---- extract_roi_latent_from_video failure paths ---------------------------

def test_extract_raises_mask_not_found(tmp_path) -> None:
    """A missing mask file raises MaskNotFoundError (no silent return '')."""
    from castle.core.data import Preprocess
    from castle.core.extractor import extract_roi_latent_from_video
    from castle.core.types import MaskNotFoundError

    project_path = tmp_path / "demo"
    (project_path / "sources").mkdir(parents=True)
    (project_path / "config.json").write_text(
        '{"source": ["v.mp4"], "latent": {}}'
    )

    with pytest.raises(MaskNotFoundError) as ei:
        extract_roi_latent_from_video(
            storage_path=str(tmp_path),
            project_name="demo",
            video_name="v.mp4",
            roi_id=1,
            model_name="dinov3_vitb16",
            batch_size=4,
            preprocess_config=Preprocess(),
            skip_existing=False,
        )
    assert "castle track" in str(ei.value).lower()


# ---- Abort threshold floor (BUG-04) ----------------------------------------

@pytest.mark.parametrize("total,rate,expected_floor", [
    (1, 0.05, 1),       # 1 batch → still 1 (floor)
    (10, 0.05, 1),      # 10 * 0.05 = 0.5 → floor to 1
    (20, 0.05, 1),      # 20 * 0.05 = 1
    (40, 0.05, 2),      # 40 * 0.05 = 2
    (100, 0.05, 5),     # 100 * 0.05 = 5
])
def test_abort_threshold_floor(total, rate, expected_floor) -> None:
    """Mirror the floor formula used inside the extractor."""
    assert max(1, int(rate * total)) == expected_floor


# ---- BUG-05 dim mismatch helper --------------------------------------------

def test_dim_mismatch_detection_logic() -> None:
    """Mirror the dim-mismatch detection used inside the extractor."""
    arrs = [np.zeros((4, 768)), np.zeros((4, 768)), np.zeros((4, 384))]
    expected = arrs[0].shape[1]
    bad = [(i, tuple(a.shape)) for i, a in enumerate(arrs) if a.shape[1] != expected]
    assert bad == [(2, (4, 384))]
