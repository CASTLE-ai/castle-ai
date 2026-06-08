"""Unit tests for the pure algorithmic helpers in clustering_service
(ARCH-01 / P4).

These functions used to live inside Gradio handlers; the tests below
exercise them directly with stub objects (no Gradio import required) so
the algorithm path is reachable from CLI / notebook / PyQt callers too.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, List, Optional

import numpy as np
import pytest

from castle.core.types import InsufficientDataError


# ---------------------------------------------------------------------------
# Lightweight stubs — mimic the LocalLatent / Latent / Aggregator surface
# without spinning up an actual project.
# ---------------------------------------------------------------------------


class _StubLocalLatent:
    """Minimal LocalLatent stand-in for service-helper tests."""

    def __init__(self, data: np.ndarray):
        self.data = data
        self.index_mask = np.ones(len(data), dtype=bool)
        self.cluster: Optional[np.ndarray] = None
        self.embedding: Optional[np.ndarray] = None
        self.export: dict = {}
        self.label_calls: List[tuple] = []

    def build_embedding(
        self,
        cfg: Any,
        *,
        progress_callback=None,
        base_seed: Optional[int] = None,
        log_path=None,
        deterministic: bool = False,
        max_points: Optional[int] = None,
    ) -> List[int]:
        self.max_points_seen = max_points
        stages = cfg if isinstance(cfg, list) else [cfg]
        seeds: List[int] = []
        for i, _ in enumerate(stages):
            if progress_callback is not None:
                progress_callback(i, len(stages))
            seeds.append(int(base_seed) + i if base_seed is not None else 1000 + i)
        self.embedding = np.zeros((len(self.data), 2), dtype=np.float32)
        return seeds

    def build_cluster(self, *, method: str, configs: dict) -> None:
        self.cluster = np.zeros(len(self.data), dtype=int)

    def label_cluster(self, cluster_id: int, name: str, color: str = '') -> None:
        self.label_calls.append((int(cluster_id), name, color))


class _StubLatent:
    """Minimal Latent stand-in. Returns a fresh _StubLocalLatent on select()."""

    def __init__(self):
        self.data = np.random.default_rng(0).standard_normal((30, 4)).astype(np.float32)
        self.cluster = np.zeros(30, dtype=int)
        self.cluster_meta = {0: {'name': 'init', 'color': 'grey'}}
        self.behavior_name2cluster_id = {'init': 0}
        self.used_palette: set = set()
        self.device = 'cpu'
        self.time_window = 1
        self.num_cluster = 1
        self._next_select: Optional[_StubLocalLatent] = None

    def select(self, *, selected_cluster: Any) -> _StubLocalLatent:
        if self._next_select is not None:
            ll = self._next_select
            self._next_select = None
            return ll
        return _StubLocalLatent(self.data)

    def import_local_latent(self, local: _StubLocalLatent) -> None:
        for cid, meta in local.export.items():
            new_cid = max(self.cluster_meta) + 1
            self.cluster_meta[new_cid] = meta


# ---------------------------------------------------------------------------
# run_umap_on_cluster
# ---------------------------------------------------------------------------


def test_run_umap_on_cluster_returns_resolved_seeds() -> None:
    from castle.service.clustering_service import run_umap_on_cluster

    latents = _StubLatent()
    result = run_umap_on_cluster(
        latents, 'init', {'n_neighbors': 5}, base_seed=42,
    )
    assert result.resolved_seeds == [42]
    assert result.local_latents.embedding is not None


def test_run_umap_on_cluster_raises_on_empty_cluster() -> None:
    from castle.service.clustering_service import run_umap_on_cluster

    latents = _StubLatent()
    latents._next_select = _StubLocalLatent(np.empty((0, 4)))
    with pytest.raises(InsufficientDataError, match="no data points"):
        run_umap_on_cluster(latents, 'init', {'n_neighbors': 5})


def test_run_umap_on_cluster_invokes_progress_callback() -> None:
    from castle.service.clustering_service import run_umap_on_cluster

    latents = _StubLatent()
    calls: List[tuple] = []
    run_umap_on_cluster(
        latents, 'init', [{'n_neighbors': 5}, {'n_neighbors': 5}],
        base_seed=10,
        progress_callback=lambda i, n: calls.append((i, n)),
    )
    assert calls == [(0, 2), (1, 2)]


def test_run_umap_on_cluster_status_text_mentions_seed() -> None:
    from castle.service.clustering_service import run_umap_on_cluster

    latents = _StubLatent()
    result = run_umap_on_cluster(latents, 'init', {'n_neighbors': 5}, base_seed=777)
    assert "777" in result.status_text


# ---------------------------------------------------------------------------
# run_dbscan_on_local
# ---------------------------------------------------------------------------


def test_run_dbscan_on_local_requires_embedding() -> None:
    from castle.service.clustering_service import run_dbscan_on_local

    ll = _StubLocalLatent(np.zeros((10, 4)))
    # No embedding attribute → raise
    with pytest.raises(InsufficientDataError, match="No embedding"):
        run_dbscan_on_local(ll, 0.5)


def test_run_dbscan_on_local_mutates_cluster() -> None:
    from castle.service.clustering_service import run_dbscan_on_local

    ll = _StubLocalLatent(np.zeros((10, 4)))
    ll.embedding = np.zeros((10, 2))
    run_dbscan_on_local(ll, 0.5)
    assert ll.cluster is not None
    assert ll.cluster.shape == (10,)


# ---------------------------------------------------------------------------
# auto_label_local_clusters
# ---------------------------------------------------------------------------


def test_auto_label_local_clusters_skips_noise() -> None:
    from castle.service.clustering_service import auto_label_local_clusters

    ll = _StubLocalLatent(np.zeros((10, 4)))
    ll.cluster = np.array([-1, -1, 0, 0, 1, 1, 1, 2, 2, 2])
    n = auto_label_local_clusters(ll, parent_name='init')
    assert n == 3  # clusters 0, 1, 2 — noise -1 skipped
    labelled_ids = [call[0] for call in ll.label_calls]
    assert -1 not in labelled_ids
    assert set(labelled_ids) == {0, 1, 2}


def test_auto_label_local_clusters_raises_without_cluster() -> None:
    from castle.service.clustering_service import auto_label_local_clusters

    ll = _StubLocalLatent(np.zeros((5, 4)))
    # ll has no `cluster` attribute set
    with pytest.raises(InsufficientDataError, match="No clusters"):
        auto_label_local_clusters(ll, parent_name='init')


# ---------------------------------------------------------------------------
# UMAPRunArtifacts dataclass
# ---------------------------------------------------------------------------


def test_umap_run_artifacts_carries_all_fields() -> None:
    from castle.service.clustering_service import UMAPRunArtifacts

    art = UMAPRunArtifacts(
        local_latents=object(),
        resolved_seeds=[1, 2, 3],
        status_text="UMAP done.",
    )
    assert art.resolved_seeds == [1, 2, 3]
    assert art.status_text == "UMAP done."


def test_submit_artifacts_dataclass_fields() -> None:
    from castle.service.clustering_service import SubmitArtifacts

    art = SubmitArtifacts(
        syllables_fig=object(),
        cluster_choices=[("init", 0)],
        id_csv_path="/tmp/id.csv",
        time_series_paths=["/tmp/ts1.csv"],
        subtitle_paths=["/tmp/sub1.srt"],
        local_latents=object(),
        embedding_path="/tmp/emb.npz",
    )
    assert art.id_csv_path == "/tmp/id.csv"
    assert art.cluster_choices == [("init", 0)]


def test_restored_session_artifacts_carries_optional_fields() -> None:
    from castle.service.clustering_service import RestoredSessionArtifacts

    art = RestoredSessionArtifacts(
        aggregator=object(),
        latents=object(),
        syllables_fig=object(),
        cluster_choices=[],
        id_csv_path="/tmp/id.csv",
        time_series_paths=[],
        local_latents=None,
        embedding_array=None,
    )
    assert art.local_latents is None
    assert art.embedding_array is None


# ---------------------------------------------------------------------------
# clip_service helpers
# ---------------------------------------------------------------------------


def test_get_bin_video_info_walks_videos() -> None:
    from castle.service.clip_service import get_bin_video_info

    agg = SimpleNamespace(
        videos_meta=[(100, "vid_a.mp4"), (50, "vid_b.mp4")],
        bin_size=1,
    )
    name, frame_idx = get_bin_video_info(agg, 0)
    assert name == "vid_a.mp4"
    assert frame_idx == 0

    name, frame_idx = get_bin_video_info(agg, 120)
    assert name == "vid_b.mp4"
    assert frame_idx == 20


def test_get_bin_video_info_returns_none_past_end() -> None:
    from castle.service.clip_service import get_bin_video_info

    agg = SimpleNamespace(videos_meta=[(10, "vid.mp4")], bin_size=1)
    name, frame_idx = get_bin_video_info(agg, 999)
    assert name is None
    assert frame_idx is None


def test_get_bin_video_info_respects_bin_size() -> None:
    from castle.service.clip_service import get_bin_video_info

    agg = SimpleNamespace(videos_meta=[(10, "vid.mp4")], bin_size=5)
    name, frame_idx = get_bin_video_info(agg, 3)
    # frame_idx = bin_idx * bin_size + bin_size//2
    assert name == "vid.mp4"
    assert frame_idx == 3 * 5 + 2


def test_save_project_cluster_model_bin_alignment(tmp_path):
    """PR2 Stage 6: latent .npz rows are per-FRAME but the embedding/labels are
    per-BIN. The saved transfer model must label each FRAME by the bin it
    belongs to (frame f -> bin f // bin_size), not pair the i-th frame with the
    i-th bin's label (which mis-aligned every example when bin_size > 1)."""
    import numpy as np
    import pandas as pd
    from castle.service.clustering_service import save_project_cluster_model
    from castle.core.cluster_transfer import load_cluster_model

    proj = tmp_path / "proj"
    (proj / "cluster").mkdir(parents=True)
    (proj / "latent" / "model").mkdir(parents=True)

    F, n_bins, bs = 8, 5, 2
    n_frames = n_bins * bs  # 10

    pd.DataFrame(
        [{"Id": 0, "Name": "a", "Color": "#111111"},
         {"Id": 1, "Name": "b", "Color": "#222222"}]
    ).to_csv(proj / "cluster" / "id.csv", index=False)

    emb = np.random.RandomState(0).randn(n_bins, 2).astype(np.float64)  # bin-res, no NaN
    cls = np.array([0, 0, 1, 1, 0], dtype=np.int16)                     # per-bin label
    np.savez(proj / "cluster" / "cluster_a_b_.npz", emb=emb, cls=cls)

    # frame-res latent: frame f -> a row whose values are all f (so we can check
    # which frame ended up where).
    latent = (np.arange(n_frames)[:, None] * np.ones((1, F))).astype(np.float32)
    np.savez(proj / "latent" / "model" / "v.npz", latent=latent)

    out = save_project_cluster_model(str(proj), output_path=str(proj / "m.npz"))
    model = load_cluster_model(out)

    # feature dim stays frame-resolution F (apply path is also per-frame)
    assert model.training_features.shape == (n_frames, F)
    assert len(model.cluster_labels) == n_frames
    for f in range(n_frames):
        assert np.allclose(model.training_features[f], float(f))   # frame f kept in place
        assert model.cluster_labels[f] == cls[f // bs]             # labelled by its bin
