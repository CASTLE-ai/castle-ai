"""Prepare-time SPP scale selection: the requested scale blocks are sliced from
a single combined latent file and column-concatenated BEFORE L2/PCA, so each
scale-combo is its own cache.

C (base feature dim) is tiny here; real DINOv3 is 768. A multiscale latent's
columns are the concatenated per-scale blocks in ascending order (scale s →
s²·C columns).
"""

import os
import tempfile

import numpy as np

from castle.core.prepare import (
    SourceSpec,
    _load_scale_combined_latent,
    _source_geometry,
    compute_prepare_id,
    load_meta,
    load_prepare,
    run_prepare,
)
from castle.utils.latent_metadata import save_latent_with_metadata

C = 4


def _block(n, scale, fill):
    return np.full((n, scale * scale * C), float(fill), dtype=np.float32)


def _save_combined(d, video, scales, n):
    """Write ONE combined multiscale npz (blocks ascending, scale s filled s).
    Returns ``(path, sorted_scales)``."""
    scales = sorted(scales)
    arr = np.hstack([_block(n, s, s) for s in scales])
    path = os.path.join(d, f"{video}_ROI_1_m_spp{'x'.join(map(str, scales))}.npz")
    save_latent_with_metadata(
        path, arr, video_name=f"{video}.mp4", roi_id=1, model_name="m",
        tags={"pooling_method": "multiscale", "pooling_scales": scales},
    )
    return path, scales


def _spec(path, file_scales, req):
    return SourceSpec(
        key=path, npz_path=path, video_name="vid.mp4", raw_fps=30.0, roi=1,
        file_scales=sorted(file_scales),
        req_scales=(sorted(req) if req else None),
    )


def test_load_scale_combined_latent_concats_ascending():
    with tempfile.TemporaryDirectory() as d:
        path, fs = _save_combined(d, "vid", [1, 2, 4], n=9)
        out = _load_scale_combined_latent(_spec(path, fs, [1, 4]))
        assert out.shape == (9, (1 + 16) * C)
        assert (out[:, :C] == 1).all() and (out[:, C:] == 4).all()


def test_source_geometry_combined_width():
    with tempfile.TemporaryDirectory() as d:
        path, fs = _save_combined(d, "vid", [1, 2, 4], n=11)
        n_orig, nf = _source_geometry(_spec(path, fs, [1, 2]))
        assert n_orig == 11 and nf == (1 + 4) * C


def test_compute_prepare_id_distinct_per_scale_combo():
    with tempfile.TemporaryDirectory() as d:
        path, fs = _save_combined(d, "vid", [1, 2, 4], n=5)
        kw = dict(downsample=False, target_fps_cap=60.0, normalize="none",
                  pca=False, K=16, fit_fraction=1.0, model_name="m")
        id1 = compute_prepare_id([_spec(path, fs, [1])], **kw)
        id12 = compute_prepare_id([_spec(path, fs, [1, 2])], **kw)
        id_whole = compute_prepare_id([_spec(path, fs, None)], **kw)  # req_scales=None
        assert len({id1, id12, id_whole}) == 3                        # distinct caches
        assert compute_prepare_id([_spec(path, fs, [1, 2])], **kw) == id12  # stable


def test_run_prepare_builds_cache_on_subset_scales():
    with tempfile.TemporaryDirectory() as d:
        path, fs = _save_combined(d, "vid", [1, 2, 4], n=20)
        out_dir = os.path.join(d, "cache")
        # normalize=none + pca=off + no downsample → reduced == the sliced input
        run_prepare(
            out_dir, [_spec(path, fs, [1, 2])],
            downsample=False, normalize="none", pca=False, model_name="m",
        )
        meta = load_meta(out_dir)
        assert meta["n_features"] == (1 + 4) * C
        assert meta["scales"] == [1, 2]                # provenance recorded
        pd = load_prepare(out_dir)
        assert pd.reduced.shape == (20, (1 + 4) * C)
        assert (pd.reduced[:, :C] == 1).all() and (pd.reduced[:, C:] == 2).all()


def test_run_prepare_whole_file_records_full_scales():
    with tempfile.TemporaryDirectory() as d:
        path, fs = _save_combined(d, "vid", [1, 2, 4], n=12)
        out_dir = os.path.join(d, "cache")
        run_prepare(
            out_dir, [_spec(path, fs, None)],   # whole file (no slicing)
            downsample=False, normalize="none", pca=False, model_name="m",
        )
        meta = load_meta(out_dir)
        assert meta["n_features"] == (1 + 4 + 16) * C
        assert meta["scales"] == [1, 2, 4]      # full set from file_scales
