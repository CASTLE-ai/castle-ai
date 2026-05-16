"""Tests for cluster representatives export (P1 / UX-02)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

import numpy as np
import pytest


class _AggStub:
    """Aggregator stub that returns a synthesized frame per global bin."""

    def __init__(self):
        self.calls: List[int] = []

    def get_frame(self, idx: int) -> Optional[np.ndarray]:
        self.calls.append(int(idx))
        frame = np.zeros((16, 16, 3), dtype=np.uint8)
        # Encode the bin index in the red channel so we can verify identity.
        frame[..., 2] = int(idx) % 256
        return frame


def _make_latents(cluster_labels: np.ndarray, feature_dim: int = 4):
    """Build a duck-typed Latent stand-in for the representatives helper."""
    rng = np.random.default_rng(0)
    data = rng.standard_normal((cluster_labels.size, feature_dim)).astype(np.float32)
    ids = sorted({int(c) for c in cluster_labels})
    cluster_meta = {
        cid: {"name": f"cluster_{cid}", "color": "#ff0000"}
        for cid in ids if cid != -1
    }
    # Mimic the real Latent's initial 'init' placeholder so we can assert it's skipped.
    cluster_meta.setdefault(0, {"name": "init", "color": "grey"})
    return SimpleNamespace(
        data=data,
        cluster=cluster_labels.astype(int),
        cluster_meta=cluster_meta,
    )


def test_export_writes_pngs_and_grid_per_cluster(tmp_path) -> None:
    from castle.service.representatives_service import export_cluster_representatives

    labels = np.array([0, 0, 1, 1, 1, 2, 2, -1, 2])
    latents = _make_latents(labels)
    # Cluster 0 has 'init' placeholder name in cluster_meta; export must skip it.
    latents.cluster_meta[0]['name'] = 'init'

    agg = _AggStub()
    written = export_cluster_representatives(
        latents, agg,
        output_dir=tmp_path,
        n_per_cluster=2,
        selection="medoid",
    )
    # Only clusters 1 and 2 (-1 is noise, 0 is 'init')
    assert set(written.keys()) == {1, 2}
    for cid, paths in written.items():
        # 2 PNGs + 1 grid PNG
        assert len(paths) == 3
        # Last one is the grid
        assert "grid" in paths[-1].name
        for p in paths:
            assert p.exists()
            assert p.suffix == ".png"


def test_export_random_selection_is_reproducible(tmp_path) -> None:
    from castle.service.representatives_service import export_cluster_representatives

    labels = np.array([1] * 20 + [2] * 20)
    latents = _make_latents(labels)
    agg = _AggStub()

    out1 = export_cluster_representatives(
        latents, agg, output_dir=tmp_path / "run1",
        n_per_cluster=3, selection="random", seed=7,
    )
    out2 = export_cluster_representatives(
        latents, agg, output_dir=tmp_path / "run2",
        n_per_cluster=3, selection="random", seed=7,
    )
    names1 = [p.name for p in out1[1]]
    names2 = [p.name for p in out2[1]]
    assert names1 == names2, "Same seed must reproduce the same picks"


def test_export_skips_empty_clusters(tmp_path) -> None:
    from castle.service.representatives_service import export_cluster_representatives

    labels = np.array([1, 1, 1])
    latents = _make_latents(labels)
    # Add a cluster id that nothing maps to
    latents.cluster_meta[99] = {"name": "ghost", "color": "#000000"}

    written = export_cluster_representatives(
        latents, _AggStub(), output_dir=tmp_path,
        n_per_cluster=2, selection="medoid",
    )
    assert 99 not in written
    assert 1 in written


def test_export_invalid_selection_raises(tmp_path) -> None:
    from castle.service.representatives_service import export_cluster_representatives

    labels = np.array([1, 1, 1])
    latents = _make_latents(labels)
    with pytest.raises(ValueError, match="Unknown selection method"):
        export_cluster_representatives(
            latents, _AggStub(), output_dir=tmp_path,
            n_per_cluster=1, selection="not_a_method",
        )
