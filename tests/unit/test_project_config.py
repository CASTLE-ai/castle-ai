"""Unit tests for castle.core.project_config."""

import json
import tempfile
import os

from castle.core.project_config import (
    ProjectConfig, TrackingConfig, ExtractionConfig,
    ClusterConfig, UMAPConfig, PreprocessConfig,
)


def test_default_config():
    cfg = ProjectConfig()
    assert cfg.tracking.model == 'r50_deaotl'
    assert cfg.extraction.model == 'dinov3_vitb16'
    assert cfg.clustering.eps == 1.0


def test_default_tracking():
    t = TrackingConfig()
    assert t.model == 'r50_deaotl'
    assert t.smart_filter_ratio == 0.1
    assert t.batch_size == 16


def test_default_extraction():
    e = ExtractionConfig()
    assert e.model == 'dinov3_vitb16'
    assert e.roi_ids == [1]
    assert e.batch_size == 32
    assert e.bin_size == 1


def test_default_preprocess():
    p = PreprocessConfig()
    assert p.center_roi is False
    assert p.crop_width == 300
    assert p.remove_background is False


def test_default_umap():
    u = UMAPConfig()
    assert u.n_neighbors == 100
    assert u.min_dist == 0.0
    assert u.n_components == 2
    assert u.n_epochs == 5000


def test_round_trip_dict():
    cfg = ProjectConfig()
    d = cfg.to_dict()
    cfg2 = ProjectConfig.from_dict(d)
    assert cfg2.tracking.model == cfg.tracking.model
    assert cfg2.extraction.preprocess.center_roi == cfg.extraction.preprocess.center_roi
    assert cfg2.clustering.umap_stages[0].n_neighbors == cfg.clustering.umap_stages[0].n_neighbors


def test_round_trip_json():
    cfg = ProjectConfig()
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        path = f.name
    try:
        cfg.save(path)
        cfg2 = ProjectConfig.load(path)
        assert cfg2.extraction.batch_size == cfg.extraction.batch_size
        assert cfg2.tracking.model == cfg.tracking.model
    finally:
        os.unlink(path)


def test_custom_config():
    cfg = ProjectConfig(
        tracking=TrackingConfig(model='swinb_deaotl', smart_filter_ratio=0.2),
        extraction=ExtractionConfig(model='dinov2_vitb14', batch_size=64),
    )
    assert cfg.tracking.smart_filter_ratio == 0.2
    assert cfg.extraction.batch_size == 64


def test_from_dict_ignores_unknown_keys():
    """from_dict should silently ignore unknown keys for forward compat."""
    d = ProjectConfig().to_dict()
    d['unknown_future_key'] = 'whatever'
    cfg = ProjectConfig.from_dict(d)
    assert cfg.tracking.model == 'r50_deaotl'


def test_from_dict_missing_key_uses_default():
    """from_dict should use defaults for missing keys."""
    d = {'tracking': {'model': 'swinb_deaotl'}}
    cfg = ProjectConfig.from_dict(d)
    assert cfg.tracking.model == 'swinb_deaotl'
    assert cfg.tracking.batch_size == 16  # default
    assert cfg.extraction.model == 'dinov3_vitb16'  # default


def test_to_preprocess():
    """Test ProjectConfig.to_preprocess() bridges to castle.core.data.Preprocess.
    
    We mock the Preprocess import since it pulls in torch.
    """
    from unittest.mock import patch, MagicMock

    mock_preprocess_cls = MagicMock()
    mock_preprocess_instance = MagicMock()
    mock_preprocess_instance.center_roi_switch = True
    mock_preprocess_instance.center_roi_crop_width = 400
    mock_preprocess_cls.return_value = mock_preprocess_instance

    cfg = ProjectConfig()
    cfg.extraction.preprocess.center_roi = True
    cfg.extraction.preprocess.crop_width = 400

    with patch.dict('sys.modules', {'castle.core.data': MagicMock(Preprocess=mock_preprocess_cls)}):
        preprocess = cfg.to_preprocess()

    mock_preprocess_cls.assert_called_once_with(
        center_roi_switch=True,
        center_roi_id=1,
        center_roi_crop_width=400,
        center_roi_crop_height=300,
        rotate_roi_tail_switch=False,
        rotate_roi_tail_id=2,
        remove_background_switch=False,
    )


def test_nested_umap_stages():
    cfg = ProjectConfig(
        clustering=ClusterConfig(
            umap_stages=[
                UMAPConfig(n_neighbors=30, n_components=5),
                UMAPConfig(n_neighbors=50, n_components=2),
            ]
        )
    )
    d = cfg.to_dict()
    cfg2 = ProjectConfig.from_dict(d)
    assert len(cfg2.clustering.umap_stages) == 2
    assert cfg2.clustering.umap_stages[0].n_neighbors == 30
    assert cfg2.clustering.umap_stages[1].n_neighbors == 50
