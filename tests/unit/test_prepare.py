"""Tests for the clustering Prepare step (castle.core.prepare + prepare_service)."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from castle.core import prepare as P
from castle.core.prepare import (
    FrameIndexMap,
    SourceSpec,
    build_legacy_index_map,
    compute_prepare_id,
    decimate_indices,
    is_stale,
    k_prime_for_variance,
    l2_normalize_rows,
    load_prepare,
    run_prepare,
)


# --------------------------------------------------------------------------- #
# Decimation                                                                  #
# --------------------------------------------------------------------------- #
def test_decimate_noop_when_at_or_below_target():
    assert np.array_equal(decimate_indices(1000, 24.0, 24.0), np.arange(1000))
    assert np.array_equal(decimate_indices(1000, 24.0, None), np.arange(1000))  # downsample off


def test_decimate_integer_ratio_is_exact_stride():
    idx = decimate_indices(1000, 120.0, 60.0)
    assert len(idx) == 500
    assert np.array_equal(idx, np.arange(0, 1000, 2))


def test_decimate_noninteger_ratio_nearest_and_monotonic():
    idx = decimate_indices(1000, 100.0, 60.0)  # ratio 1.667
    assert len(idx) == 600
    assert np.all(np.diff(idx) >= 0)
    assert idx[0] == 0 and idx[-1] <= 999
    assert idx[1] == 2 and idx[2] == 3  # round([0, 1.667, 3.33, ...]) = 0, 2, 3, ...


# --------------------------------------------------------------------------- #
# L2                                                                          #
# --------------------------------------------------------------------------- #
def test_l2_unit_norm_and_zero_row_guard():
    out = l2_normalize_rows(np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32))
    assert np.allclose(np.linalg.norm(out[0]), 1.0)
    assert np.allclose(out[1], 0.0)  # zero row stays zero (eps-guard, no nan/inf)
    assert np.isfinite(out).all()


def test_l2_passes_nan_rows_through():
    out = l2_normalize_rows(np.array([[1.0, 1.0], [np.nan, np.nan]], dtype=np.float32))
    assert np.isfinite(out[0]).all()
    assert np.isnan(out[1]).all()


# --------------------------------------------------------------------------- #
# FrameIndexMap / window-aware view                                           #
# --------------------------------------------------------------------------- #
def test_window_aware_map_dp_to_orig_and_expand():
    fim = FrameIndexMap(
        video_names=["v.mp4"],
        dp_offsets=np.array([0, 6], dtype=np.int64),
        orig_frame_idx=np.array([0, 2, 4, 6, 8, 10], dtype=np.int64),
        raw_fps=np.array([120.0]),
        n_orig_frames=np.array([12], dtype=np.int64),
        source_roi=np.array([1], dtype=np.int64),
    )
    w = fim.for_window(2)  # 3 windows: [0,2],[4,6],[8,10]
    assert w.n_windows == 3
    v, orig = w.dp_to_orig_frame(1)  # centre decimated row = 0 + 1*2 + 1 = 3 -> orig 6
    assert (v, orig) == (0, 6)
    out = w.expand_labels_to_orig(np.array([10, 20, 30]), 0)
    assert out.shape == (12,)
    assert out[0] == 10 and out[3] == 10
    assert out[4] == 20 and out[7] == 20
    assert out[8] == 30 and out[11] == 30


def test_legacy_index_map_reproduces_bin_contract():
    fim = build_legacy_index_map([(3, "a.mp4"), (2, "b.mp4")], bin_size=4)
    w = fim.for_window(1)
    assert w.n_windows == 5
    assert w.dp_to_orig_frame(0) == (0, 2)  # bin 0 -> centre frame 0*4 + 2
    out = w.expand_labels_to_orig(np.array([7, 8, 9]), 0)
    assert out.shape == (12,)
    assert out[0] == 7 and out[3] == 7 and out[4] == 8 and out[11] == 9


# --------------------------------------------------------------------------- #
# run_prepare: pipeline + toggles + NaN + PCA                                 #
# --------------------------------------------------------------------------- #
def _write_latent(path: str, arr: np.ndarray) -> None:
    np.savez(path, latent=arr.astype(np.float32))


def _src(path: str, video: str, fps: float = 120.0, roi: int = 1) -> SourceSpec:
    return SourceSpec(key=os.path.basename(path), npz_path=path, video_name=video, raw_fps=fps, roi=roi)


@pytest.fixture
def two_videos(tmp_path):
    rng = np.random.default_rng(0)
    p1 = str(tmp_path / "a.npz")
    p2 = str(tmp_path / "b.npz")
    _write_latent(p1, rng.standard_normal((200, 32)))
    _write_latent(p2, rng.standard_normal((120, 32)))
    return [_src(p1, "a.mp4"), _src(p2, "b.mp4")]


def test_run_prepare_full_pipeline_shapes(tmp_path, two_videos):
    out = str(tmp_path / "cache")
    meta = run_prepare(out, two_videos, downsample=True, target_fps_cap=60.0,
                       normalize="l2", pca=True, K=8, fit_fraction=1.0, model_name="m")
    # 200->100, 120->60 datapoints
    assert meta["n_dp_total"] == 160
    assert meta["width"] == 8
    pd = load_prepare(out)
    assert pd.reduced.shape == (160, 8)
    assert pd.index_map.n_videos == 2
    assert pd.index_map.dp_offsets.tolist() == [0, 100, 160]
    assert len(meta["pca"]["explained_variance_ratio"]) == 8


@pytest.mark.parametrize("downsample,normalize,pca", [
    (False, "none", False),  # all off = raw passthrough
    (True, "none", False),
    (True, "l2", False),
    (False, "l2", True),
])
def test_run_prepare_toggle_combos(tmp_path, two_videos, downsample, normalize, pca):
    out = str(tmp_path / f"c_{downsample}_{normalize}_{pca}")
    meta = run_prepare(out, two_videos, downsample=downsample, target_fps_cap=60.0,
                       normalize=normalize, pca=pca, K=8, model_name="m")
    expected_dp = 160 if downsample else 320
    assert meta["n_dp_total"] == expected_dp
    assert meta["width"] == (8 if pca else 32)
    assert load_prepare(out).reduced.shape == (expected_dp, 8 if pca else 32)


def test_run_prepare_nan_passthrough(tmp_path):
    arr = np.random.default_rng(1).standard_normal((100, 16)).astype(np.float32)
    arr[4] = np.nan  # whole-row tracking-loss; even index -> survives stride-2 decimation
    p = str(tmp_path / "n.npz")
    _write_latent(p, arr)
    out = str(tmp_path / "cache")
    meta = run_prepare(out, [_src(p, "n.mp4")], downsample=True, target_fps_cap=60.0,
                       normalize="l2", pca=True, K=4, model_name="m")
    pd = load_prepare(out)
    # 100 frames -> 50 decimated (stride 2); orig frame 4 -> decimated row 2.
    assert pd.reduced.shape[0] == 50
    assert np.isnan(pd.reduced[2]).all()
    assert np.isfinite(pd.reduced[0]).all()
    assert meta["pca"]["n_finite_fit_rows"] == 49  # 50 decimated - 1 nan


def test_run_prepare_k_greater_than_rank_clamps(tmp_path, two_videos):
    out = str(tmp_path / "cache")
    meta = run_prepare(out, two_videos, downsample=False, normalize="none",
                       pca=True, K=50, model_name="m")  # n_features=32 < 50
    assert meta["pca"]["n_components_kept"] == 32
    assert meta["pca"]["rank_limited"] is True
    assert meta["width"] == 32


def test_k_prime_for_variance(tmp_path, two_videos):
    out = str(tmp_path / "cache")
    meta = run_prepare(out, two_videos, pca=True, K=8, model_name="m")
    k = k_prime_for_variance(meta, 0.95)
    assert 1 <= k <= 8
    assert k_prime_for_variance({"pca": {}, "width": 32}, 0.95) == 32  # PCA off -> full width


# --------------------------------------------------------------------------- #
# prepare_id determinism + staleness                                          #
# --------------------------------------------------------------------------- #
def test_prepare_id_deterministic_and_sensitive(two_videos):
    kw = dict(downsample=True, target_fps_cap=60.0, normalize="l2", pca=True,
              K=8, fit_fraction=1.0, model_name="m")
    a = compute_prepare_id(two_videos, **kw)
    b = compute_prepare_id(two_videos, **kw)
    assert a == b
    assert compute_prepare_id(two_videos, **{**kw, "K": 16}) != a
    assert compute_prepare_id(two_videos, **{**kw, "normalize": "none"}) != a


def test_is_stale_on_source_change(tmp_path, two_videos):
    out = str(tmp_path / "cache")
    run_prepare(out, two_videos, pca=True, K=8, model_name="m")
    resolve = {s.key: s.npz_path for s in two_videos}.get
    assert is_stale(out, resolve) is False
    # mutate one source's content (new size + mtime)
    bigger = np.random.default_rng(9).standard_normal((201, 32)).astype(np.float32)
    _write_latent(two_videos[0].npz_path, bigger)
    assert is_stale(out, resolve) is True
    assert is_stale(out, lambda k: None) is True  # missing source -> stale


# --------------------------------------------------------------------------- #
# Phase 2: prepared aggregator wiring (service build -> LatentAggregator)      #
# --------------------------------------------------------------------------- #
def _make_project(tmp_path, model="dinov3_vitb16", n_frames=200, n_feat=32):
    proj = "P"
    pdir = tmp_path / proj
    (pdir / "latent" / model).mkdir(parents=True)
    (pdir / "sources").mkdir()
    rng = np.random.default_rng(0)
    keys, latent_map = [], {}
    for name in ("a", "b"):
        arr = rng.standard_normal((n_frames, n_feat)).astype(np.float32)
        fn = f"{name}_ROI_1_ROI_1_{model}_rmbg_spp1x2x4_pre-deadbeef.npz"
        np.savez(pdir / "latent" / model / fn, latent=arr)
        key = f"deadbeef/{fn}"
        latent_map[key] = f"{name}_ROI_1.mp4"
        keys.append(key)
    (pdir / "config.json").write_text(json.dumps({"source": [], "latent": latent_map}))
    return str(tmp_path), proj, model, keys


def test_build_prepare_then_aggregator_prepared_mode(tmp_path):
    storage, proj, model, keys = _make_project(tmp_path)
    from castle.service import prepare_service
    from castle.core.cluster import LatentAggregator

    # fps unreadable (no real mp4) -> 30.0 fallback; target min(30,60)=30 -> no decimation.
    pid = prepare_service.build_prepare(
        storage, proj, model, keys, downsample=True, target_fps_cap=60.0,
        normalize="l2", pca=True, K=8, fit_fraction=1.0, notify=lambda *a: None,
    )
    # registry recorded it
    assert pid in prepare_service.list_prepared(storage, proj)

    W = 4
    agg = LatentAggregator(storage, proj, 1, W, model_name=model,
                           notify=lambda *a, **k: None, prepare_id=pid, k_prime=8)
    assert agg._prepared is True
    # 200 frames/video, no decimation, W=4 -> 50 windows/video; 2 videos -> 100.
    assert agg.latents.shape == (100, 8 * W)
    assert [c for c, _ in agg.videos_meta] == [50, 50]
    assert agg.frame_index_map is not None and agg.frame_index_map.n_windows == 100

    lat = agg.get_latent_object()
    assert lat.time_window == 1 and len(lat.cluster) == 100  # already windowed

    # frame mapping: datapoint -> (video, original frame) within bounds
    v, orig = agg.frame_index_map.dp_to_orig_frame(60)  # in video b (windows 50..99)
    assert v == 1 and 0 <= orig < 200


def test_prepared_reuse_returns_same_id(tmp_path):
    storage, proj, model, keys = _make_project(tmp_path)
    from castle.service import prepare_service
    kw = dict(downsample=True, target_fps_cap=60.0, normalize="l2", pca=True,
              K=8, fit_fraction=1.0, notify=lambda *a: None)
    pid1 = prepare_service.build_prepare(storage, proj, model, keys, **kw)
    pid2 = prepare_service.build_prepare(storage, proj, model, keys, **kw)  # reuse
    assert pid1 == pid2
