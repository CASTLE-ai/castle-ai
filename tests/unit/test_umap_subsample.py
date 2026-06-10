"""Memory-aware UMAP subsampling + label propagation (feature #2).

These tests pin the contracts the design review flagged as load-bearing:
- the whole multi-stage chain runs on the sample; the embedding stays length-M;
- subsample state is invalidated between runs (no stale propagation);
- the auto-cap refuses (rather than silently degrading) when the budget can't
  fit a UMAP-viable sample (S below the n_neighbors floor);
- transfer-model export trains only on the truly-sampled rows.

All runs use the CPU (deterministic) path so layouts are reproducible.
"""

import os
import tempfile

import numpy as np
import pytest

from castle.core.types import CastleDataError
from castle.utils.latent_explorer import (
    Latent,
    LocalLatent,
    _knn_sampled,
)
from castle.service.clustering_service import run_umap_on_cluster, run_dbscan_on_local
from castle.ui.embedding_scatter import EmbeddingScatterPlot


def _two_blobs(n: int, width: int = 8, sep: float = 8.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n // 2, width)).astype(np.float32) + sep
    b = rng.standard_normal((n - n // 2, width)).astype(np.float32) - sep
    return np.vstack([a, b])


def _local(data: np.ndarray) -> LocalLatent:
    return LocalLatent(data, np.ones(len(data), dtype=bool), color_avoid=set(), device="cpu")


_CFG1 = [{"n_neighbors": 15, "min_dist": 0.0, "n_components": 2, "n_epochs": 200}]
_CFG2 = [
    {"n_neighbors": 15, "min_dist": 0.0, "n_components": 5, "n_epochs": 200},
    {"n_neighbors": 15, "min_dist": 0.0, "n_components": 2, "n_epochs": 200},
]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def test_knn_sampled_cpu_euclidean():
    samp = np.array([[0.0, 0.0], [100.0, 100.0]], dtype=np.float32)
    q = np.array([[1.0, 1.0], [99.0, 99.0]], dtype=np.float32)
    idx, _ = _knn_sampled(samp, q, np.array([0, 1]), k=1, metric="euclidean", use_gpu=False)
    assert list(idx.ravel()) == [0, 1]


# --------------------------------------------------------------------------- #
# build_embedding / build_cluster
# --------------------------------------------------------------------------- #
def test_subsample_keeps_length_m_and_labels_all():
    data = _two_blobs(400)
    ll = _local(data)
    ll.build_embedding(_CFG1, base_seed=42, deterministic=True, max_points=150)
    assert ll.embedding.shape == (400, 2)
    assert np.isfinite(ll.embedding).all()           # non-sampled interpolated
    assert ll._subsample_idx is not None and len(ll._subsample_idx) == 150
    assert len(ll._prop_nonsampled_idx) == 250
    ll.build_cluster(method="dbscan", configs={"eps": 1.5})
    assert ll.cluster.shape == (400,)                # all points labelled
    assert len(set(ll.cluster.tolist()) - {-1}) == 2  # two blobs recovered


def test_nonsampled_points_snap_onto_sampled_positions():
    # Every non-sampled point must land EXACTLY on some sampled point's 2D
    # position (nearest-snap), never in a gap — guards against the weighted-
    # average collapse that piled points onto the global centroid.
    data = _two_blobs(400)
    ll = _local(data)
    ll.build_embedding(_CFG1, base_seed=9, deterministic=True, max_points=150)
    sampled_pos = ll.embedding[ll._subsample_idx]
    ns_pos = ll.embedding[ll._prop_nonsampled_idx]
    # each non-sampled position equals one of the sampled positions exactly
    sampled_set = {tuple(np.round(p, 6)) for p in sampled_pos}
    assert all(tuple(np.round(p, 6)) in sampled_set for p in ns_pos)
    # spread of non-sampled positions ~ spread of sampled (no centroid collapse)
    assert ns_pos.std(0).min() > 0.3 * sampled_pos.std(0).min()


def test_nonsampled_label_matches_its_2d_position_cluster():
    # Regression: a non-sampled point's cluster label MUST be the label of the
    # sampled point it was snapped onto — otherwise it shows as a wrong-coloured
    # dot sitting inside another cluster (confetti-over-the-embedding bug).
    data = _two_blobs(400)
    ll = _local(data)
    ll.build_embedding(_CFG1, base_seed=4, deterministic=True, max_points=150)
    ll.build_cluster(method="dbscan", configs={"eps": 1.5})
    sub = ll._subsample_idx
    ns = ll._prop_nonsampled_idx
    # the sampled point each non-sampled point shares its 2D position with
    sampled_pos = {tuple(np.round(ll.embedding[i], 6)): int(ll.cluster[i]) for i in sub}
    for j in ns:
        pos = tuple(np.round(ll.embedding[j], 6))
        assert int(ll.cluster[j]) == sampled_pos[pos]   # label == co-located sample's label


def test_subsample_is_reproducible_with_seed():
    data = _two_blobs(400)
    a, b = _local(data), _local(data)
    a.build_embedding(_CFG1, base_seed=7, deterministic=True, max_points=150)
    b.build_embedding(_CFG1, base_seed=7, deterministic=True, max_points=150)
    assert np.array_equal(a._subsample_idx, b._subsample_idx)
    assert np.allclose(a.embedding, b.embedding)


def test_multistage_subsample_runs_whole_chain_on_sample():
    data = _two_blobs(400)
    ll = _local(data)
    ll.build_embedding(_CFG2, base_seed=1, deterministic=True, max_points=150)
    # final stage is 2-D and length-M despite the 5-D intermediate stage
    assert ll.embedding.shape == (400, 2)
    assert ll._embedding_sampled.shape == (150, 2)
    assert np.isfinite(ll.embedding).all()
    ll.build_cluster(method="dbscan", configs={"eps": 1.5})
    assert ll.cluster.shape == (400,)


def test_no_subsample_when_max_points_ge_m():
    ll = _local(_two_blobs(200))
    ll.build_embedding(_CFG1, base_seed=3, deterministic=True, max_points=10_000)
    assert ll._subsample_idx is None
    assert ll.embedding.shape == (200, 2)


def test_stale_subsample_state_cleared_on_rerun():
    data = _two_blobs(400)
    ll = _local(data)
    ll.build_embedding(_CFG1, base_seed=1, deterministic=True, max_points=150)
    assert ll._subsample_idx is not None
    # second run does NOT subsample -> state must be cleared
    ll.build_embedding(_CFG1, base_seed=1, deterministic=True, max_points=None)
    assert ll._subsample_idx is None
    assert ll._embedding_sampled is None
    ll.build_cluster(method="dbscan", configs={"eps": 1.5})
    assert ll.cluster.shape == (400,)


def test_min_samples_scaled_to_subsample():
    # DBSCAN runs on the S-point subsample; a full-scale min_samples must be
    # scaled by S/M, else it is ~M/S times stricter and clusters collapse to
    # noise. Two separated blobs (2000), 10% sample, min_samples=100 (5% of full
    # = sensible; 50% of the 200 sample = far too strict unless scaled).
    data = _two_blobs(2000, sep=9.0)
    ll = _local(data)
    ll.build_embedding(_CFG1, base_seed=1, deterministic=True, max_points=200)
    ll.build_cluster(method="dbscan", configs={"eps": 1.0, "min_samples": 100})
    n_clusters = len(set(ll.cluster.tolist()) - {-1})
    noise = int((ll.cluster == -1).sum())
    assert n_clusters >= 2          # the two blobs still form (scaling worked)
    assert noise < 2000 * 0.5       # not collapsed to mostly-noise


def test_redbscan_after_subsample_reuses_sample():
    ll = _local(_two_blobs(400))
    ll.build_embedding(_CFG1, base_seed=3, deterministic=True, max_points=150)
    ll.build_cluster(method="dbscan", configs={"eps": 1.0})
    assert ll.cluster.shape == (400,)
    ll.build_cluster(method="dbscan", configs={"eps": 3.0})  # no re-UMAP
    assert ll._subsample_idx is not None and ll.cluster.shape == (400,)


# --------------------------------------------------------------------------- #
# service: manual subsample % + refusal when % is below the UMAP minimum
# --------------------------------------------------------------------------- #
def test_subsample_pct_below_nneighbors_floor_refuses():
    lat = Latent(_two_blobs(500), time_window=1, device="cpu")
    # need_min = max(2*5, n_neighbors+1) = 16; 3% of 500 = 15 < 16
    with pytest.raises(CastleDataError, match="below the UMAP minimum"):
        run_umap_on_cluster(
            lat, "init", _CFG1, base_seed=1, deterministic=True,
            subsample=True, subsample_pct=3,
        )


def test_subsample_pct_one_percent_refuses():
    lat = Latent(_two_blobs(500), time_window=1, device="cpu")
    with pytest.raises(CastleDataError, match="below the UMAP minimum"):
        run_umap_on_cluster(
            lat, "init", _CFG1, base_seed=1, deterministic=True,
            subsample=True, subsample_pct=1,
        )


def test_service_subsample_notes_and_propagates():
    lat = Latent(_two_blobs(500), time_window=1, device="cpu")
    res = run_umap_on_cluster(
        lat, "init", _CFG1, base_seed=11, deterministic=True,
        subsample=True, subsample_pct=40,   # 40% of 500 = 200
    )
    assert "sample" in res.status_text.lower()
    ll = res.local_latents
    assert ll._subsample_idx is not None and len(ll._subsample_idx) == 200
    assert ll.embedding.shape == (500, 2)
    run_dbscan_on_local(ll, eps=1.5)
    assert ll.cluster.shape == (500,)


def test_umap_host_bytes_counts_only_marginal_allocations():
    # The estimate must NOT include the already-resident n_total x d matrix
    # (select() materialises it before the guard, so MemAvailable already
    # excludes it — counting it again falsely refuses runs). Only the subsample
    # draw, the embedding output, and (optionally) a conversion copy.
    from castle.core.clustering_backends import umap_host_bytes
    M, S, d = 1000, 100, 50
    no_sub = umap_host_bytes(M, M, d, 2)                    # S == M, no copy
    sub = umap_host_bytes(M, S, d, 2)                       # + sampled draw
    with_copy = umap_host_bytes(M, M, d, 2, full_copy=True)  # + f32 conversion
    assert no_sub == M * 4 * 2                               # embedding output only
    assert sub == S * 4 * d + M * 4 * 2                      # + sampled draw
    assert with_copy == M * 4 * d + M * 4 * 2                # + full conversion copy
    assert sub > no_sub and with_copy > no_sub
    # the resident matrix (M*4*d ~ 200 KB here) is NOT in the no-copy estimate
    assert no_sub < M * 4 * d


def test_free_cuda_memory_pools_is_safe_to_call():
    # Must be a harmless no-op on the CPU/no-cupy path and never raise; on a GPU
    # box it drains cupy's per-device pools back to the driver after a fit.
    from castle.core.clustering_backends import free_cuda_memory_pools
    assert free_cuda_memory_pools() is None
    free_cuda_memory_pools()  # idempotent


def test_gpu_path_refuses_when_host_ram_too_tight(monkeypatch):
    # On the GPU path the VRAM guard never sees the host side. Simulate a GPU
    # with ample VRAM but a host with almost no free RAM: the run must refuse
    # with a host-RAM error (not silently proceed into an OS-killing OOM). Wide
    # blobs so the subsample draw alone exceeds the (tiny) free-RAM budget.
    import castle.core.clustering_backends as cb
    import castle.core.runtime_env as rt
    lat = Latent(_two_blobs(500, width=200), time_window=1, device="cpu")
    monkeypatch.setattr(cb, "target_cuda_free_bytes", lambda *a, **k: 40 * 10**9)
    monkeypatch.setattr(rt, "available_ram_bytes", lambda: 100_000)
    with pytest.raises(CastleDataError, match="host RAM"):
        run_umap_on_cluster(
            lat, "init", _CFG1, base_seed=1, deterministic=False,
            subsample=True, subsample_pct=40,
        )


def test_service_no_subsample_uses_all_points():
    lat = Latent(_two_blobs(500), time_window=1, device="cpu")
    res = run_umap_on_cluster(
        lat, "init", _CFG1, base_seed=2, deterministic=True, subsample=False,
    )
    ll = res.local_latents
    assert ll._subsample_idx is None             # fit every point
    assert ll.embedding.shape == (500, 2)


# --------------------------------------------------------------------------- #
# export: training rows are the sampled rows only
# --------------------------------------------------------------------------- #
def test_saved_npz_is_sampled_marks_only_sampled_rows():
    lat = Latent(_two_blobs(500), time_window=1, device="cpu")
    res = run_umap_on_cluster(
        lat, "init", _CFG1, base_seed=5, deterministic=True,
        subsample=True, subsample_pct=40,
    )
    ll = res.local_latents
    run_dbscan_on_local(ll, eps=1.5)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "cluster_x_.npz")
        EmbeddingScatterPlot(ll).save_named_embedding(p)
        z = np.load(p, allow_pickle=True)
        assert "is_sampled" in z.files
        assert z["is_sampled"].dtype == bool
        assert int(z["is_sampled"].sum()) == 200   # only the sampled rows
        assert z["emb"].shape == (500, 2) and z["cls"].shape == (500,)
        # the legacy export filter trains on sampled rows only
        frame_sampled = np.repeat(z["is_sampled"], 1)
        valid = ~np.isnan(z["emb"]).any(axis=1)
        assert int((valid & frame_sampled).sum()) == 200


def test_saved_npz_all_sampled_when_no_subsample():
    ll = _local(_two_blobs(60))
    ll.build_embedding(_CFG1, base_seed=2, deterministic=True, max_points=None)
    ll.build_cluster(method="dbscan", configs={"eps": 1.5})
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "cluster_y_.npz")
        EmbeddingScatterPlot(ll).save_named_embedding(p)
        z = np.load(p, allow_pickle=True)
        assert int(z["is_sampled"].sum()) == 60   # all rows are real members
