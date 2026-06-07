"""Tests for P1 reproducibility / science-quality fixes (2026-06 audit).

* P1-1  pairwise_distance preserves input dtype on both backends.
* P1-2  UMAP feature standardization (default ON), toggleable, recorded.
* P1-3  build_embedding raises on non-finite input instead of silent garbage.
* P1-5  safe_zscore / safe_minmax are NaN-tolerant (don't poison whole columns).
"""

import numpy as np
import pytest


# --------------------------------------------------------------------------- #
# P1-1: pairwise_distance dtype contract / device consistency
# --------------------------------------------------------------------------- #

def test_pairwise_distance_preserves_input_dtype():
    from castle.utils.distance import pairwise_distance

    rng = np.random.default_rng(0)
    A = rng.random((5, 4))   # float64
    B = rng.random((3, 4))

    d64 = pairwise_distance(A, B, device="cpu")
    assert d64.dtype == np.float64

    d32 = pairwise_distance(A.astype(np.float32), B.astype(np.float32), device="cpu")
    assert d32.dtype == np.float32


def test_pairwise_distance_cpu_gpu_consistent_float64():
    import torch

    from castle.utils.distance import pairwise_distance

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    rng = np.random.default_rng(1)
    A = rng.random((8, 6))   # float64
    B = rng.random((5, 6))

    d_cpu = pairwise_distance(A, B, device="cpu")
    d_gpu = pairwise_distance(A, B, device="cuda")

    assert d_gpu.dtype == np.float64           # not silently downcast to float32
    # Both compute in float64 now → agree far better than the old float32 GPU path.
    np.testing.assert_allclose(d_cpu, d_gpu, rtol=1e-6, atol=1e-6)


# --------------------------------------------------------------------------- #
# P1-2: UMAP standardization (default ON), toggle, recorded
# --------------------------------------------------------------------------- #

class _RecordingReducer:
    """Captures the matrix passed to fit_transform; returns a dummy 2-D layout."""

    last_X = None

    def __init__(self, cfg):
        self.cfg = cfg

    def fit_transform(self, X, *, random_state):
        _RecordingReducer.last_X = np.asarray(X).copy()
        return np.zeros((len(X), 2), dtype=float)


def _make_local_latent(data):
    from castle.utils.latent_explorer import LocalLatent
    return LocalLatent(data, np.ones(len(data), dtype=bool), set(), "cpu")


def test_umap_no_feature_standardization():
    # Per-feature z-score was removed (it never existed on main and amplified
    # noise dims for distance-based UMAP/DBSCAN). build_embedding must pass raw
    # features through untouched even if a legacy ``standardize`` key is present.
    rng = np.random.default_rng(0)
    data = rng.random((50, 4)) * np.array([1.0, 500.0, 1.0, 500.0])

    ll = _make_local_latent(data)
    ll.build_embedding(
        [{"n_neighbors": 10, "min_dist": 0.0, "n_components": 2, "standardize": True}],
        base_seed=42,
        reducer_factory=lambda cfg: _RecordingReducer(cfg),
    )

    # Raw features reach the reducer un-standardized despite standardize=True.
    assert np.allclose(_RecordingReducer.last_X, data)


def test_umapreducer_drops_standardize_kwarg():
    from castle.core.clustering_backends import UMAPReducer

    r = UMAPReducer(
        {"n_neighbors": 10, "min_dist": 0.0, "standardize": True, "random_state": 5},
        device="cpu",
    )
    # These are not UMAP constructor kwargs and must not reach the backend.
    assert "standardize" not in r.cfg
    assert "random_state" not in r.cfg
    assert r.cfg["n_neighbors"] == 10


# --------------------------------------------------------------------------- #
# P1-3: NaN/Inf guard before UMAP
# --------------------------------------------------------------------------- #

def test_build_embedding_rejects_nonfinite():
    from castle.core.types import CastleDataError

    rng = np.random.default_rng(0)
    data = rng.random((20, 4))
    data[3, 1] = np.nan

    ll = _make_local_latent(data)
    with pytest.raises(CastleDataError):
        ll.build_embedding(
            [{"n_neighbors": 5, "min_dist": 0.0, "n_components": 2}],
            base_seed=1,
            reducer_factory=lambda cfg: _RecordingReducer(cfg),
        )


# --------------------------------------------------------------------------- #
# P1-5: NaN-tolerant numeric_safe
# --------------------------------------------------------------------------- #

def test_safe_zscore_nan_does_not_poison_column():
    from castle.utils.numeric_safe import safe_zscore

    x = np.array([[1.0, 10.0], [2.0, np.nan], [3.0, 30.0], [4.0, 40.0]])
    z = safe_zscore(x)

    assert abs(float(np.nanmean(z[:, 0]))) < 1e-9      # fully-finite col: mean 0
    col1 = z[:, 1]
    assert np.isnan(col1[1])                            # the NaN entry stays NaN
    assert np.all(np.isfinite(col1[[0, 2, 3]]))         # other entries are valid


def test_safe_minmax_nan_does_not_poison_column():
    from castle.utils.numeric_safe import safe_minmax

    x = np.array([[1.0, 10.0], [2.0, np.nan], [3.0, 30.0]])
    m = safe_minmax(x)

    assert np.isnan(m[1, 1])
    assert np.all(np.isfinite(m[[0, 2]][:, 1]))
    assert np.all((m[[0, 2]][:, 1] >= 0) & (m[[0, 2]][:, 1] <= 1))
