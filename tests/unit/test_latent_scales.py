"""Shared SPP-scale helpers: parse a latent file's scale list and slice one
scale's column block. These back the Prepare-time scale combination (see
test_prepare_scales.py); here we pin the low-level parsing/slicing contract.

C (base feature dim) is tiny here; real DINOv3 is 768. A multiscale latent's
columns are the concatenated per-scale blocks in ascending-scale order: scale s
occupies s²·C columns.
"""

import numpy as np
import pytest

from castle.core.latent_scales import _scale_block, _spp_scales_of
from castle.core.types import CastleDataError

C = 5


def _block(n, scale, fill):
    return np.full((n, scale * scale * C), float(fill), dtype=np.float32)


def _multiscale(n, scales):
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
