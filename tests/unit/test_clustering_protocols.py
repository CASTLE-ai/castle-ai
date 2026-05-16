"""Unit tests for :mod:`castle.core.clustering_protocols` (ARCH-02 / P2-C)."""

from __future__ import annotations

import numpy as np

from castle.core.clustering_protocols import (
    Clusterer,
    DimensionReducer,
    LatentData,
)


def test_latent_data_minimal_construction() -> None:
    rng = np.random.default_rng(0)
    raw = rng.standard_normal((50, 8)).astype(np.float32)
    data = LatentData(raw=raw)

    assert data.n_samples == 50
    assert data.n_features == 8
    assert data.has_embedding() is False
    assert data.has_clusters() is False
    assert data.cluster_meta == {}


def test_latent_data_with_embedding_and_clusters() -> None:
    rng = np.random.default_rng(1)
    raw = rng.standard_normal((20, 4)).astype(np.float32)
    emb = rng.standard_normal((20, 2)).astype(np.float32)
    labels = np.zeros(20, dtype=int)
    labels[10:] = 1
    data = LatentData(
        raw=raw,
        embedding=emb,
        cluster_ids=labels,
        cluster_meta={0: {"name": "rest", "color": "#000"}, 1: {"name": "walk", "color": "#fff"}},
    )

    assert data.has_embedding()
    assert data.has_clusters()
    assert data.cluster_meta[1]["name"] == "walk"
    np.testing.assert_array_equal(data.cluster_ids, labels)


class _StubReducer:
    """Stub reducer that just keeps the first ``n_components`` cols."""

    def __init__(self, n_components: int = 2):
        self.n_components = n_components
        self.last_seed: int | None = None

    def fit_transform(self, X: np.ndarray, *, random_state: int) -> np.ndarray:
        self.last_seed = random_state
        return X[:, : self.n_components]


class _StubClusterer:
    """Stub clusterer that splits on the sign of the first dim."""

    def __init__(self):
        self.last_seed: int | None = None

    def fit_predict(self, X: np.ndarray, *, random_state: int) -> np.ndarray:
        self.last_seed = random_state
        labels = np.where(X[:, 0] >= 0, 0, 1)
        return labels.astype(int)


def test_dimension_reducer_runtime_check() -> None:
    reducer = _StubReducer()
    assert isinstance(reducer, DimensionReducer)
    X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    out = reducer.fit_transform(X, random_state=42)
    assert out.shape == (2, 2)
    assert reducer.last_seed == 42


def test_clusterer_runtime_check() -> None:
    cl = _StubClusterer()
    assert isinstance(cl, Clusterer)
    X = np.array([[-1.0, 0.0], [1.0, 0.0], [2.0, 0.0], [-3.0, 0.0]])
    labels = cl.fit_predict(X, random_state=7)
    np.testing.assert_array_equal(labels, [1, 0, 0, 1])
    assert cl.last_seed == 7


def test_non_conforming_object_fails_protocol_check() -> None:
    class _NotAReducer:
        def transform(self, X):  # wrong method name
            return X

    assert not isinstance(_NotAReducer(), DimensionReducer)
    assert not isinstance(_NotAReducer(), Clusterer)
