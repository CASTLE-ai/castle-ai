"""Per-scale SPP latents: filename/metadata scale parsing, column slicing, and
the Behavior-Microscope scale-combination (per-scale files AND legacy combined
files), which lets users mix and match 1×1 / 2×2 / 4×4 blocks.

C (base feature dim) is small here so the math is easy to read; real DINOv3 is
768. A multiscale latent's columns are the concatenated per-scale blocks in
ascending-scale order: scale s occupies s²·C columns.
"""

import os
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest

from castle.core.cluster import (
    LatentAggregator,
    _scale_block,
    _spp_scales_of,
)
from castle.core.types import CastleDataError
from castle.utils.latent_metadata import save_latent_with_metadata

C = 5  # tiny base feature dim


def _block(n, scale, fill):
    return np.full((n, scale * scale * C), float(fill), dtype=np.float32)


def _multiscale(n, scales):
    """Concatenated [s1|s2|…] blocks; block for scale s is filled with value s."""
    return np.hstack([_block(n, s, s) for s in sorted(scales)])


# --------------------------------------------------------------------------- #
# _spp_scales_of
# --------------------------------------------------------------------------- #
def test_spp_scales_from_filename():
    assert _spp_scales_of("v_ROI_1_dinov3_vitb16_spp1.npz") == [1]
    assert _spp_scales_of("v_ROI_1_dinov3_vitb16_spp1x2x4.npz") == [1, 2, 4]
    assert _spp_scales_of("v_ROI_1_dinov3_vitb16_spp2x4.npz") == [2, 4]
    assert _spp_scales_of("v_ROI_1_dinov3_vitb16.npz") == []          # weighted_average


def test_spp_scales_metadata_hint_wins():
    # Metadata hint overrides the filename tag (and is sorted).
    assert _spp_scales_of("weird_name.npz", scales_hint=[4, 1, 2]) == [1, 2, 4]


# --------------------------------------------------------------------------- #
# _scale_block
# --------------------------------------------------------------------------- #
def test_scale_block_slices_each_scale():
    arr = _multiscale(7, [1, 2, 4])           # widths 5, 20, 80 → total 105
    b1 = _scale_block(arr, [1, 2, 4], 1)
    b2 = _scale_block(arr, [1, 2, 4], 2)
    b4 = _scale_block(arr, [1, 2, 4], 4)
    assert b1.shape == (7, 1 * 1 * C) and (b1 == 1).all()
    assert b2.shape == (7, 2 * 2 * C) and (b2 == 2).all()
    assert b4.shape == (7, 4 * 4 * C) and (b4 == 4).all()


def test_scale_block_single_scale_file():
    arr = _block(3, 2, 2)                      # a per-scale spp2 file
    out = _scale_block(arr, [2], 2)
    assert out.shape == (3, 4 * C) and (out == 2).all()


def test_scale_block_bad_width_raises():
    with pytest.raises(CastleDataError):
        _scale_block(np.zeros((4, 7), dtype=np.float32), [1, 2], 1)  # 7 % (1+4) != 0


# --------------------------------------------------------------------------- #
# _combine_scales_per_video — both file layouts
# --------------------------------------------------------------------------- #
def _save(path, arr, scales):
    save_latent_with_metadata(
        path, arr, video_name=os.path.basename(path), roi_id=1,
        model_name="m", tags={"pooling_method": "multiscale", "pooling_scales": scales},
    )


def _agg(scales):
    return SimpleNamespace(scales=scales, notify=lambda *a, **k: None)


def test_combine_from_legacy_combined_file():
    with tempfile.TemporaryDirectory() as d:
        fn = "vidA_ROI_1_m_spp1x2x4.npz"
        _save(os.path.join(d, fn), _multiscale(10, [1, 2, 4]), [1, 2, 4])
        sel = [(fn, "vidA", [1, 2, 4])]
        # subset {1,2}: width = (1+4)*C, block order ascending
        out = LatentAggregator._combine_scales_per_video(_agg([1, 2]), sel, d)
        assert len(out) == 1
        vid, mat = out[0]
        assert vid == "vidA" and mat.shape == (10, (1 + 4) * C)
        assert (mat[:, :C] == 1).all() and (mat[:, C:] == 2).all()
        # default (None) → all scales
        out_all = LatentAggregator._combine_scales_per_video(_agg(None), sel, d)
        assert out_all[0][1].shape == (10, (1 + 4 + 16) * C)


def test_combine_from_perscale_files_matches_combined():
    with tempfile.TemporaryDirectory() as d:
        for s in (1, 2, 4):
            fn = f"vidB_ROI_1_m_spp{s}.npz"
            _save(os.path.join(d, fn), _block(8, s, s), [s])
        sel = [(f"vidB_ROI_1_m_spp{s}.npz", "vidB", [s]) for s in (1, 2, 4)]
        out = LatentAggregator._combine_scales_per_video(_agg([1, 4]), sel, d)
        vid, mat = out[0]
        # only scales 1 and 4, in ascending order
        assert mat.shape == (8, (1 + 16) * C)
        assert (mat[:, :C] == 1).all() and (mat[:, C:] == 4).all()


def test_combine_skips_video_missing_a_requested_scale():
    with tempfile.TemporaryDirectory() as d:
        # vidD has scales {1,2}; vidE has only {1}. Requesting {1,2}: vidD is
        # combined, vidE is skipped (missing scale 2) rather than crashing.
        _save(os.path.join(d, "vidD_ROI_1_m_spp1x2.npz"), _multiscale(6, [1, 2]), [1, 2])
        _save(os.path.join(d, "vidE_ROI_1_m_spp1.npz"), _block(6, 1, 1), [1])
        sel = [
            ("vidD_ROI_1_m_spp1x2.npz", "vidD", [1, 2]),
            ("vidE_ROI_1_m_spp1.npz", "vidE", [1]),
        ]
        out = LatentAggregator._combine_scales_per_video(_agg([1, 2]), sel, d)
        vids = [v for v, _ in out]
        assert vids == ["vidD"]                       # vidE dropped, no crash
        assert out[0][1].shape == (6, (1 + 4) * C)


def test_combine_request_unavailable_scale_uses_available():
    with tempfile.TemporaryDirectory() as d:
        # Only scale 1 exists anywhere; requesting {1,2} → scale 2 is dropped
        # globally (no file has it), scale 1 is still combined.
        fn = "vidC_ROI_1_m_spp1.npz"
        _save(os.path.join(d, fn), _block(5, 1, 1), [1])
        out = LatentAggregator._combine_scales_per_video(_agg([1, 2]), [(fn, "vidC", [1])], d)
        assert len(out) == 1 and out[0][1].shape == (5, 1 * C)
