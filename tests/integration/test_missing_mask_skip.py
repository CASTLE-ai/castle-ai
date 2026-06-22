"""End-to-end: frames with no tracked mask are skipped, not fatal.

Builds a mask file that is missing a few frame keys (as happens when the tracker
loses the object) and runs the real extraction path with a fake, mask-aware
observer. Asserts the run completes, the missing frames land as NaN gap rows in
the right positions, the skip count is recorded in metadata, and an
all-missing mask file aborts loudly instead of writing a useless latent.
"""

import h5py
import numpy as np
import pytest

TEST_VIDEO_H = 480
TEST_VIDEO_W = 640
TEST_VIDEO_LEN = 30
FEATURE_DIM = 8


class _MaskAwareObserver:
    """Constant features, but NaN for an all-zero mask — mimics the real model's
    empty-mask guard so a skipped (blank) frame becomes a NaN placeholder row."""

    def extract_tensor_batch(self, frames, masks, roi_id,
                             pooling="weighted_average", scales=None, layers=None):
        masks = np.asarray(masks)
        n = int(frames.shape[0])
        out = np.ones((n, FEATURE_DIM), dtype=np.float32)
        for r in range(n):
            if masks[r].sum() == 0:
                out[r] = np.nan
        return out


def _write_mask_file(path, *, missing):
    """A mask_list.h5 with a (H, W) ROI-1 mask for every frame except `missing`."""
    with h5py.File(path, "w") as f:
        for i in range(TEST_VIDEO_LEN):
            if i in missing:
                continue
            mask = np.zeros((TEST_VIDEO_H, TEST_VIDEO_W), dtype=np.uint8)
            mask[100:300, 100:300] = 1
            f.create_dataset(str(i), data=mask, compression="gzip")


def test_missing_masks_become_nan_gaps(dummy_project, tmp_path, monkeypatch):
    from castle.core import extractor as extractor_mod
    from castle.core.data import Preprocess
    from castle.utils.latent_metadata import load_latent_metadata
    from castle.service.extraction_service import latent_gap_summary

    storage_path, project_name, video_name = dummy_project

    monkeypatch.setattr(extractor_mod, "get_num_workers", lambda *a, **k: 0)
    monkeypatch.setattr(extractor_mod, "_get_observer", lambda model_name: _MaskAwareObserver())

    missing = {10, 11, 12}
    mask_path = tmp_path / "mask_missing.h5"
    _write_mask_file(mask_path, missing=missing)

    out_path = extractor_mod.extract_roi_latent_from_video(
        storage_path, project_name, video_name,
        roi_id=1, model_name="fakemodel", batch_size=10,
        preprocess_config=Preprocess(), skip_existing=False,
        mask_path_override=str(mask_path),
        on_frame_error="skip",
    )

    arr = np.load(out_path)["latent"]
    assert arr.shape == (TEST_VIDEO_LEN, FEATURE_DIM)
    # Missing frames are NaN gaps in their exact positions; everything else finite.
    gap_idx = sorted(missing)
    assert np.isnan(arr[gap_idx]).all()
    finite_idx = [i for i in range(TEST_VIDEO_LEN) if i not in missing]
    assert np.isfinite(arr[finite_idx]).all()

    meta = load_latent_metadata(out_path)
    assert meta["tags"]["n_skipped_frames"] == len(missing)
    assert meta["tags"]["n_total_frames"] == TEST_VIDEO_LEN

    summary = latent_gap_summary(out_path)
    assert summary == {"n_skipped": 3, "n_total": 30, "frac": pytest.approx(0.1)}


def test_all_missing_masks_aborts_loudly(dummy_project, tmp_path, monkeypatch):
    from castle.core import extractor as extractor_mod
    from castle.core.data import Preprocess
    from castle.core.types import ExtractionError

    storage_path, project_name, video_name = dummy_project

    monkeypatch.setattr(extractor_mod, "get_num_workers", lambda *a, **k: 0)
    monkeypatch.setattr(extractor_mod, "_get_observer", lambda model_name: _MaskAwareObserver())

    mask_path = tmp_path / "mask_all_missing.h5"
    _write_mask_file(mask_path, missing=set(range(TEST_VIDEO_LEN)))

    with pytest.raises(ExtractionError, match="no usable"):
        extractor_mod.extract_roi_latent_from_video(
            storage_path, project_name, video_name,
            roi_id=1, model_name="fakemodel", batch_size=10,
            preprocess_config=Preprocess(), skip_existing=False,
            mask_path_override=str(mask_path),
            on_frame_error="skip",
        )
