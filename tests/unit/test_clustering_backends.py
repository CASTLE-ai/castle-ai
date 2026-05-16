"""Tests for the Protocol adapters in :mod:`castle.core.clustering_backends`
(ARCH-02 / P4).

These adapters are the concrete other side of the Protocol seam — every
reducer / clusterer CASTLE currently uses, wrapped as a class that
satisfies :class:`DimensionReducer` / :class:`Clusterer`. The tests
below verify:

* Each adapter satisfies its Protocol structurally.
* The default UMAP / DBSCAN paths produce well-formed output on small
  synthetic data.
* :func:`build_default_clusterer` bridges the legacy ``method`` string.
* :class:`HDBSCANClusterer` skips cleanly when ``hdbscan`` isn't
  installed.
"""

from __future__ import annotations

import numpy as np
import pytest

from castle.core.clustering_protocols import Clusterer, DimensionReducer


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_umap_reducer_conforms_protocol() -> None:
    from castle.core.clustering_backends import UMAPReducer

    r = UMAPReducer({'n_neighbors': 5}, device='cpu')
    assert isinstance(r, DimensionReducer)


def test_dbscan_clusterer_conforms_protocol() -> None:
    from castle.core.clustering_backends import DBSCANClusterer

    c = DBSCANClusterer(eps=0.5)
    assert isinstance(c, Clusterer)


# ---------------------------------------------------------------------------
# Backend class resolution
# ---------------------------------------------------------------------------


def test_resolve_umap_class_cpu() -> None:
    from castle.core.clustering_backends import resolve_umap_class

    cls = resolve_umap_class('cpu')
    # umap-learn is the canonical CPU backend; cls should be umap.UMAP
    assert cls.__module__.startswith('umap')


def test_resolve_umap_class_unsupported() -> None:
    from castle.core.clustering_backends import resolve_umap_class

    with pytest.raises(ValueError, match="Unsupported device"):
        resolve_umap_class('tpu')


def test_resolve_dbscan_class_cpu() -> None:
    from castle.core.clustering_backends import resolve_dbscan_class

    cls = resolve_dbscan_class('cpu')
    assert cls.__module__.startswith('sklearn')


# ---------------------------------------------------------------------------
# UMAPReducer fit_transform
# ---------------------------------------------------------------------------


def test_umap_reducer_strips_random_state_from_init_cfg() -> None:
    """random_state passed via cfg should NOT survive — the kwarg owns it."""
    from castle.core.clustering_backends import UMAPReducer

    r = UMAPReducer({'n_neighbors': 5, 'random_state': 99}, device='cpu')
    assert 'random_state' not in r.cfg


def test_umap_reducer_fit_transform_returns_2d_array() -> None:
    from castle.core.clustering_backends import UMAPReducer

    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 8)).astype(np.float32)
    r = UMAPReducer({'n_neighbors': 5, 'min_dist': 0.1, 'n_components': 2}, device='cpu')
    Z = r.fit_transform(X, random_state=42)
    assert Z.shape == (40, 2)
    assert np.all(np.isfinite(Z))


def test_umap_reducer_same_seed_reproduces() -> None:
    from castle.core.clustering_backends import UMAPReducer

    rng = np.random.default_rng(1)
    X = rng.standard_normal((40, 8)).astype(np.float32)
    cfg = {'n_neighbors': 5, 'min_dist': 0.1, 'n_components': 2}
    Z1 = UMAPReducer(cfg, device='cpu').fit_transform(X, random_state=42)
    Z2 = UMAPReducer(cfg, device='cpu').fit_transform(X, random_state=42)
    np.testing.assert_array_equal(Z1, Z2)


# ---------------------------------------------------------------------------
# DBSCANClusterer fit_predict
# ---------------------------------------------------------------------------


def test_dbscan_clusterer_assigns_integer_labels() -> None:
    from castle.core.clustering_backends import DBSCANClusterer

    # Two well-separated 2D clusters
    X = np.vstack([
        np.random.default_rng(0).standard_normal((20, 2)),
        np.random.default_rng(1).standard_normal((20, 2)) + 10.0,
    ])
    c = DBSCANClusterer(eps=1.0, min_samples=3)
    labels = c.fit_predict(X, random_state=0)
    assert labels.dtype.kind == 'i'
    assert labels.shape == (40,)
    # Should find at least one non-noise cluster
    assert (labels >= 0).sum() > 0


def test_dbscan_clusterer_ignores_random_state() -> None:
    """DBSCAN is deterministic — different random_states must yield same labels."""
    from castle.core.clustering_backends import DBSCANClusterer

    X = np.random.default_rng(0).standard_normal((30, 3))
    c = DBSCANClusterer(eps=1.0)
    l1 = c.fit_predict(X, random_state=0)
    l2 = c.fit_predict(X, random_state=999)
    np.testing.assert_array_equal(l1, l2)


def test_dbscan_clusterer_forwards_min_samples() -> None:
    from castle.core.clustering_backends import DBSCANClusterer

    X = np.random.default_rng(0).standard_normal((10, 2))
    # min_samples=20 with 10 points → every point becomes noise
    c = DBSCANClusterer(eps=0.1, min_samples=20)
    labels = c.fit_predict(X, random_state=0)
    assert (labels == -1).all()


# ---------------------------------------------------------------------------
# build_default_clusterer
# ---------------------------------------------------------------------------


def test_build_default_clusterer_dbscan_returns_dbscan() -> None:
    from castle.core.clustering_backends import build_default_clusterer, DBSCANClusterer

    c = build_default_clusterer('dbscan', {'eps': 0.7}, device='cpu')
    assert isinstance(c, DBSCANClusterer)
    assert c.eps == pytest.approx(0.7)


def test_build_default_clusterer_unknown_method_raises() -> None:
    from castle.core.clustering_backends import build_default_clusterer

    with pytest.raises(ValueError, match="Unknown clusterer method"):
        build_default_clusterer('kmeans', {}, device='cpu')


# ---------------------------------------------------------------------------
# HDBSCANClusterer — gated on hdbscan availability
# ---------------------------------------------------------------------------


def test_hdbscan_clusterer_skips_when_hdbscan_missing() -> None:
    pytest.importorskip('hdbscan')
    from castle.core.clustering_backends import HDBSCANClusterer

    c = HDBSCANClusterer(min_cluster_size=5)
    assert isinstance(c, Clusterer)


def test_hdbscan_clusterer_fit_predict() -> None:
    pytest.importorskip('hdbscan')
    from castle.core.clustering_backends import HDBSCANClusterer

    X = np.vstack([
        np.random.default_rng(0).standard_normal((30, 2)),
        np.random.default_rng(1).standard_normal((30, 2)) + 10.0,
    ])
    c = HDBSCANClusterer(min_cluster_size=5)
    labels = c.fit_predict(X, random_state=0)
    assert labels.dtype.kind == 'i'
    assert labels.shape == (60,)


def test_hdbscan_import_raises_clear_message_when_missing() -> None:
    """If hdbscan is genuinely absent, the constructor raises with a hint."""
    try:
        import hdbscan  # noqa: F401
        pytest.skip("hdbscan IS installed; cannot test the import-error branch")
    except ImportError:
        pass

    from castle.core.clustering_backends import HDBSCANClusterer
    with pytest.raises(ImportError, match="pip install hdbscan"):
        HDBSCANClusterer(min_cluster_size=5)
