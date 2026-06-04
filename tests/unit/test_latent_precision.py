"""Tests for the latent storage-precision option (float32/float16):
- save_latent_with_metadata honors `dtype` and records it in metadata.
- the clustering aggregator upcasts any stored precision (incl. mixed) to float32
  so UMAP/DBSCAN are precision-agnostic.
"""

import tempfile

import numpy as np

from castle.core.cluster import _aggregate_latents
from castle.utils.latent_metadata import save_latent_with_metadata, load_latent_metadata


def _save(arr, dtype=None, tmp_path=None):
    p = str((tmp_path / "lat.npz")) if tmp_path else tempfile.mktemp(suffix=".npz")
    save_latent_with_metadata(p, arr, video_name="v.mp4", roi_id=1,
                              model_name="dinov3_vitb16", dtype=dtype)
    return p


def test_save_default_is_float32(tmp_path):
    arr = np.random.rand(20, 768).astype(np.float32)
    p = _save(arr, dtype=None, tmp_path=tmp_path)
    with np.load(p) as d:
        assert d["latent"].dtype == np.float32
    assert load_latent_metadata(p)["tags"]["latent_dtype"] == "float32"


def test_save_float16_halves_and_tags(tmp_path):
    import os
    arr = np.random.rand(64, 768).astype(np.float32)
    (tmp_path / "a").mkdir(); (tmp_path / "b").mkdir()
    p32 = _save(arr, dtype=np.float32, tmp_path=tmp_path / "a")
    p16 = _save(arr, dtype=np.float16, tmp_path=tmp_path / "b")
    with np.load(p16) as d:
        stored = d["latent"]
    assert stored.dtype == np.float16
    assert load_latent_metadata(p16)["tags"]["latent_dtype"] == "float16"
    # fp16 npz is meaningfully smaller than fp32 (dense float features).
    assert os.path.getsize(p16) < os.path.getsize(p32)
    # Values preserved within fp16 precision (~0.1% relative).
    np.testing.assert_allclose(stored.astype(np.float32), arr, rtol=0, atol=2e-3)


def test_aggregator_upcasts_float16_to_float32():
    chunk16 = np.random.rand(30, 768).astype(np.float16)
    out = _aggregate_latents([chunk16.copy()], cache_dir=tempfile.mkdtemp(), notify=lambda *a, **k: None)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, chunk16.astype(np.float32), rtol=0, atol=0)


def test_aggregator_upcasts_mixed_precision():
    # A project split across machines may have some fp16 and some fp32 latents.
    a = np.random.rand(10, 768).astype(np.float16)
    b = np.random.rand(15, 768).astype(np.float32)
    out = _aggregate_latents([a.copy(), b.copy()], cache_dir=tempfile.mkdtemp(), notify=lambda *a, **k: None)
    assert out.dtype == np.float32 and out.shape == (25, 768)
