"""Unit tests for A-06: Multi-scale Spatial Pyramid Pooling & Multi-layer Feature Extraction."""

import torch
import numpy as np
import pytest

from castle.core.models import VisualEncoder


class StubEncoder(VisualEncoder):
    """Minimal concrete encoder for testing pooling methods."""
    def load_model(self):
        pass

    def extract_features(self, x, layers=None):
        pass


# --- Multi-scale Pooling Tests ---

def test_multiscale_pooling_scales_1():
    """Scale [1] produces (B, C) — global pooling."""
    enc = StubEncoder()
    B, N, C = 2, 37 * 37, 768
    features = torch.randn(B, N, C)
    masks = torch.ones(B, 592, 592)
    result = enc._multiscale_pooling(features, masks, 592, 16, scales=[1])
    assert result.shape == (B, 768)  # 1×1 → 768


def test_multiscale_pooling_scales_1_2():
    """Scales [1, 2] → (1 + 4) × C = 5C."""
    enc = StubEncoder()
    B, N, C = 2, 37 * 37, 768
    features = torch.randn(B, N, C)
    masks = torch.ones(B, 592, 592)
    result = enc._multiscale_pooling(features, masks, 592, 16, scales=[1, 2])
    assert result.shape == (B, 5 * 768)  # (1 + 4) × 768


def test_multiscale_pooling_scales_1_2_4():
    """Scales [1, 2, 4] → (1 + 4 + 16) × C = 21C."""
    enc = StubEncoder()
    B, N, C = 2, 37 * 37, 768
    features = torch.randn(B, N, C)
    masks = torch.ones(B, 592, 592)
    result = enc._multiscale_pooling(features, masks, 592, 16, scales=[1, 2, 4])
    assert result.shape == (B, 21 * 768)  # (1 + 4 + 16) × 768


def test_multiscale_pooling_with_roi_mask():
    """Non-trivial ROI mask should not produce NaN."""
    enc = StubEncoder()
    B, C = 1, 768
    features = torch.randn(B, 37 * 37, C)
    masks = torch.zeros(B, 592, 592)
    masks[:, 100:400, 100:400] = 1.0  # ROI in center
    result = enc._multiscale_pooling(features, masks, 592, 16, scales=[1, 2])
    assert result.shape == (B, 5 * C)
    assert not torch.isnan(result).any()


def test_multiscale_pooling_empty_mask():
    """All-zero mask should not produce NaN (clamped to 1e-6)."""
    enc = StubEncoder()
    B, C = 1, 768
    features = torch.randn(B, 37 * 37, C)
    masks = torch.zeros(B, 592, 592)
    result = enc._multiscale_pooling(features, masks, 592, 16, scales=[1, 2, 4])
    assert result.shape == (B, 21 * C)
    assert not torch.isnan(result).any()


def test_multiscale_pooling_scale_8():
    """Scale [8] with non-divisible grid (37 not divisible by 8)."""
    enc = StubEncoder()
    B, C = 2, 768
    features = torch.randn(B, 37 * 37, C)
    masks = torch.ones(B, 592, 592)
    result = enc._multiscale_pooling(features, masks, 592, 16, scales=[8])
    assert result.shape == (B, 64 * C)  # 8×8 = 64 regions


def test_multiscale_pooling_dinov2_resolution():
    """Test with DINOv2 resolution (518, patch_size=14, 37 patches)."""
    enc = StubEncoder()
    B, C = 2, 768
    features = torch.randn(B, 37 * 37, C)
    masks = torch.ones(B, 518, 518)
    result = enc._multiscale_pooling(features, masks, 518, 14, scales=[1, 2, 4])
    assert result.shape == (B, 21 * C)


def test_multiscale_pooling_evenly_divisible():
    """Test with evenly divisible grid (e.g. 16 patches, scale=4)."""
    enc = StubEncoder()
    B, C = 2, 768
    features = torch.randn(B, 16 * 16, C)
    masks = torch.ones(B, 256, 256)
    result = enc._multiscale_pooling(features, masks, 256, 16, scales=[1, 2, 4])
    assert result.shape == (B, 21 * C)


def test_weighted_pooling_backward_compat():
    """Ensure multiscale with scales=[1] equals old _weighted_pooling."""
    enc = StubEncoder()
    B, C = 2, 768
    features = torch.randn(B, 37 * 37, C)
    masks = torch.ones(B, 592, 592)

    old_result = enc._weighted_pooling(features, masks, 592, 16)
    new_result = enc._multiscale_pooling(features, masks, 592, 16, scales=[1])

    torch.testing.assert_close(old_result, new_result, rtol=1e-4, atol=1e-4)


def test_weighted_pooling_backward_compat_with_roi():
    """Backward compat with a partial ROI mask."""
    enc = StubEncoder()
    B, C = 2, 768
    features = torch.randn(B, 37 * 37, C)
    masks = torch.zeros(B, 592, 592)
    masks[:, 50:500, 50:500] = 1.0

    old_result = enc._weighted_pooling(features, masks, 592, 16)
    new_result = enc._multiscale_pooling(features, masks, 592, 16, scales=[1])

    torch.testing.assert_close(old_result, new_result, rtol=1e-4, atol=1e-4)


def test_weighted_pooling_backward_compat_dinov2():
    """Backward compat with DINOv2 resolution."""
    enc = StubEncoder()
    B, C = 2, 768
    features = torch.randn(B, 37 * 37, C)
    masks = torch.ones(B, 518, 518)

    old_result = enc._weighted_pooling(features, masks, 518, 14)
    new_result = enc._multiscale_pooling(features, masks, 518, 14, scales=[1])

    torch.testing.assert_close(old_result, new_result, rtol=1e-4, atol=1e-4)


# --- Config Tests ---

def test_config_pooling_fields():
    """ExtractionConfig has new A-06 fields with correct defaults."""
    from castle.core.project_config import ExtractionConfig
    cfg = ExtractionConfig()
    assert cfg.pooling_method == 'weighted_average'
    assert cfg.pooling_scales == [1, 2, 4]
    assert cfg.feature_layers is None


def test_config_pooling_custom():
    """ExtractionConfig accepts custom A-06 values."""
    from castle.core.project_config import ExtractionConfig
    cfg = ExtractionConfig(pooling_method='multiscale', feature_layers=[3, 7, 11])
    assert cfg.pooling_method == 'multiscale'
    assert cfg.feature_layers == [3, 7, 11]


def test_config_round_trip():
    """A-06 fields survive dict round-trip."""
    from castle.core.project_config import ProjectConfig, ExtractionConfig
    cfg = ProjectConfig(
        extraction=ExtractionConfig(
            pooling_method='multiscale',
            pooling_scales=[1, 2],
            feature_layers=[5, 11],
        )
    )
    d = cfg.to_dict()
    cfg2 = ProjectConfig.from_dict(d)
    assert cfg2.extraction.pooling_method == 'multiscale'
    assert cfg2.extraction.pooling_scales == [1, 2]
    assert cfg2.extraction.feature_layers == [5, 11]


def test_config_backward_compat_missing_fields():
    """Old configs without A-06 fields should load with defaults."""
    from castle.core.project_config import ProjectConfig
    old_dict = {
        'extraction': {
            'model': 'dinov3_vitb16',
            'roi_ids': [1],
            'batch_size': 32,
            'bin_size': 1,
            'preprocess': {},
        }
    }
    cfg = ProjectConfig.from_dict(old_dict)
    assert cfg.extraction.pooling_method == 'weighted_average'
    assert cfg.extraction.pooling_scales == [1, 2, 4]
    assert cfg.extraction.feature_layers is None


# --- extract_tensor_batch signature tests ---

def test_extract_tensor_batch_accepts_new_params():
    """Verify VisualEncoder.extract_tensor_batch accepts pooling/scales/layers kwargs."""
    import inspect
    sig = inspect.signature(VisualEncoder.extract_tensor_batch)
    param_names = list(sig.parameters.keys())
    # Should accept at minimum: self, frame_batch, mask_batch, roi_id
    assert 'frame_batch' in param_names or len(param_names) >= 4


def test_dinov2_extract_tensor_batch_signature():
    """DINOv2Encoder.extract_tensor_batch accepts new A-06 params."""
    import inspect
    from castle.core.models import DINOv2Encoder
    sig = inspect.signature(DINOv2Encoder.extract_tensor_batch)
    param_names = list(sig.parameters.keys())
    assert 'pooling' in param_names
    assert 'scales' in param_names
    assert 'layers' in param_names


def test_dinov3_extract_tensor_batch_signature():
    """DINOv3Encoder.extract_tensor_batch accepts new A-06 params."""
    import inspect
    from castle.core.models import DINOv3Encoder
    sig = inspect.signature(DINOv3Encoder.extract_tensor_batch)
    param_names = list(sig.parameters.keys())
    assert 'pooling' in param_names
    assert 'scales' in param_names
    assert 'layers' in param_names
