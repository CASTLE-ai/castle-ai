"""Unit tests for castle.utils.latent_explorer (Latent / LocalLatent)."""

import numpy as np
import pytest

from castle.utils.latent_explorer import Latent, LocalLatent


def test_latent_init():
    raw = np.random.randn(100, 768).astype(np.float32)
    lat = Latent(raw, time_window=1, device='cpu')
    assert lat.data.shape == (100, 768)
    assert len(lat.cluster) == 100
    assert lat.cluster_meta[0]['name'] == 'init'


def test_latent_time_window():
    raw = np.random.randn(100, 768).astype(np.float32)
    lat = Latent(raw, time_window=5, device='cpu')
    assert lat.data.shape == (20, 768 * 5)


def test_latent_time_window_truncates():
    """Non-divisible length should be truncated."""
    raw = np.random.randn(103, 768).astype(np.float32)
    lat = Latent(raw, time_window=5, device='cpu')
    # 103 // 5 = 20, so 100 frames used
    assert lat.data.shape == (20, 768 * 5)


def test_latent_nan_handling():
    """NaN frames should get cluster=-1."""
    raw = np.random.randn(50, 768).astype(np.float32)
    raw[10, :] = np.nan  # One NaN frame
    lat = Latent(raw, time_window=1, device='cpu')
    assert lat.cluster[10] == -1
    assert lat.cluster[0] == 0


def test_latent_select():
    raw = np.random.randn(100, 768).astype(np.float32)
    lat = Latent(raw, time_window=1, device='cpu')
    local = lat.select('init')
    assert isinstance(local, LocalLatent)
    assert len(local.data) == 100


def test_latent_select_by_id():
    raw = np.random.randn(100, 768).astype(np.float32)
    lat = Latent(raw, time_window=1, device='cpu')
    local = lat.select(0)  # Select by cluster ID
    assert isinstance(local, LocalLatent)
    assert len(local.data) == 100


def test_local_latent_label():
    data = np.random.randn(50, 768).astype(np.float32)
    mask = np.ones(100, dtype=bool)
    mask[50:] = False
    local = LocalLatent(data, mask, color_avoid=set(), device='cpu')
    local.cluster = np.array([0] * 25 + [1] * 25)
    local.label_cluster(0, "running")
    local.label_cluster(1, "walking")
    assert local.export[0]['name'] == 'running'
    assert local.export[1]['name'] == 'walking'


def test_local_latent_clean_label():
    data = np.random.randn(20, 768).astype(np.float32)
    mask = np.ones(20, dtype=bool)
    local = LocalLatent(data, mask, color_avoid=set(), device='cpu')
    local.label_cluster(0, "test")
    assert 0 in local.export
    local.clean_label()
    assert len(local.export) == 0


def test_local_latent_merge():
    data = np.random.randn(30, 768).astype(np.float32)
    mask = np.ones(30, dtype=bool)
    local = LocalLatent(data, mask, color_avoid=set(), device='cpu')
    local.cluster = np.array([0] * 10 + [1] * 10 + [2] * 10)
    local.merge([1, 2])
    # Both 1 and 2 should become 1 (min)
    assert np.all(local.cluster[10:] == 1)


def test_latent_merge():
    raw = np.random.randn(60, 768).astype(np.float32)
    lat = Latent(raw, time_window=1, device='cpu')
    # Manually set clusters
    lat.cluster[:20] = 1
    lat.cluster[20:40] = 2
    lat.cluster[40:] = 3
    lat.merge([2, 3])
    assert np.all(lat.cluster[20:] == 2)


def test_latent_import_local_latent():
    raw = np.random.randn(100, 768).astype(np.float32)
    lat = Latent(raw, time_window=1, device='cpu')

    local = lat.select('init')
    local.cluster = np.array([0] * 50 + [1] * 50)
    local.label_cluster(0, "running")
    local.label_cluster(1, "walking")

    lat.import_local_latent(local)

    assert 'running' in lat.behavior_name2cluster_id
    assert 'walking' in lat.behavior_name2cluster_id
    assert lat.num_cluster == 3  # init + running + walking


def test_latent_palette():
    raw = np.random.randn(10, 768).astype(np.float32)
    lat = Latent(raw, time_window=1, device='cpu')
    assert lat.palette(0) == 'grey'  # init color
    assert lat.palette(999) == 'grey'  # unknown


def test_local_latent_palette():
    data = np.random.randn(10, 768).astype(np.float32)
    mask = np.ones(10, dtype=bool)
    local = LocalLatent(data, mask, color_avoid=set(), device='cpu')
    assert local.palette(-1) == '#DDDDDD'
    color = local.palette(0)
    assert color.startswith('#')


# ---- build_embedding progress_callback (C-05) ----

def test_build_embedding_progress_callback():
    """progress_callback should be invoked once per UMAP stage."""
    from unittest.mock import MagicMock, patch

    data = np.random.randn(30, 10).astype(np.float32)
    mask = np.ones(30, dtype=bool)
    local = LocalLatent(data, mask, color_avoid=set(), device='cpu')

    # Mock UMAP to avoid actual computation
    mock_umap_instance = MagicMock()
    mock_umap_instance.fit_transform.return_value = np.random.randn(30, 2)
    mock_umap_cls = MagicMock(return_value=mock_umap_instance)

    callback = MagicMock()

    with patch('umap.UMAP', mock_umap_cls):
        local.build_embedding(
            [{"n_neighbors": 5}, {"n_neighbors": 5}],
            progress_callback=callback,
        )

    assert callback.call_count == 2
    callback.assert_any_call(0, 2)
    callback.assert_any_call(1, 2)
    assert hasattr(local, 'embedding')


# ---- BUG-13: UMAP n_neighbors lower bound ----

def test_build_embedding_raises_when_too_few_samples():
    """BUG-13: <10 samples should hit the lower-bound guard."""
    from castle.core.types import InsufficientDataError

    data = np.random.randn(6, 10).astype(np.float32)  # 6 < 2*5
    mask = np.ones(6, dtype=bool)
    local = LocalLatent(data, mask, color_avoid=set(), device='cpu')
    with pytest.raises(InsufficientDataError, match="Need at least"):
        local.build_embedding([{"n_neighbors": 5}])


def test_build_embedding_raises_when_n_neighbors_too_small():
    """BUG-13: n_neighbors < 5 → InsufficientDataError."""
    from castle.core.types import InsufficientDataError

    data = np.random.randn(30, 10).astype(np.float32)
    mask = np.ones(30, dtype=bool)
    local = LocalLatent(data, mask, color_avoid=set(), device='cpu')
    with pytest.raises(InsufficientDataError, match="below minimum"):
        local.build_embedding([{"n_neighbors": 2}])


def test_build_embedding_raises_when_n_neighbors_exceeds_samples():
    """BUG-13: n_neighbors >= n_samples → InsufficientDataError."""
    from castle.core.types import InsufficientDataError

    data = np.random.randn(20, 10).astype(np.float32)
    mask = np.ones(20, dtype=bool)
    local = LocalLatent(data, mask, color_avoid=set(), device='cpu')
    with pytest.raises(InsufficientDataError, match="must be < n_samples"):
        local.build_embedding([{"n_neighbors": 20}])
