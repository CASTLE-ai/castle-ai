"""Latent space exploration: Latent and LocalLatent classes."""

from __future__ import annotations

import datetime as _datetime
import json as _json
import logging as _logging
import secrets
from pathlib import Path
from typing import TYPE_CHECKING, Callable, List, Optional, Union

import numpy as np

from castle.core.environment import get_device
from castle.core.config import PALETTE_HEX
from castle.core.types import InsufficientDataError

if TYPE_CHECKING:
    from castle.core.clustering_protocols import Clusterer, DimensionReducer

# BUG-13: lower bound for UMAP n_neighbors. Below ~5 UMAP becomes
# numerically pathological (k-NN graph too sparse to identify manifold
# structure). 5 is also UMAP's documented minimum.
_UMAP_MIN_N_NEIGHBORS = 5

_logger = _logging.getLogger(__name__)

DEFAULT_DEVICE = get_device()

_palette = PALETTE_HEX * 5


def generate_distinct_color(index, saturation=0.7, value=0.9):
    """Generate a distinct color using golden ratio for even distribution in HSV space.

    This ensures an unlimited number of visually distinct colors can be generated,
    preventing clusters from becoming grey when the fixed palette is exhausted.
    """
    import colorsys
    golden_ratio = 0.618033988749895
    hue = (index * golden_ratio) % 1.0
    rgb = colorsys.hsv_to_rgb(hue, saturation, value)
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))


def generate_palette(avoid):
    res = [c for c in _palette if c not in avoid]
    return res or _palette


def _resolve_umap_seed(cfg: dict, base_seed: Optional[int], stage: int) -> tuple:
    """Pick the seed to use for one UMAP stage.

    Args:
        cfg: The stage's UMAP config dict.
        base_seed: Optional deterministic base; stage ``i`` uses ``base_seed + i``.
        stage: 0-indexed stage number.

    Returns:
        (seed, source) where source ∈ {"user", "base+offset", "drawn"}.
    """
    if cfg.get('random_state') is not None:
        return int(cfg['random_state']), 'user'
    if base_seed is not None:
        return int(base_seed) + int(stage), 'base+offset'
    return secrets.randbits(32), 'drawn'


def _append_umap_log(
    log_path: Union[str, Path], *, stage: int, seed: int, source: str, cfg: dict
) -> None:
    """Append one JSONL entry recording a UMAP stage's seed and config.

    Failure to write is logged at debug level but never raised — logging is
    advisory and must not break the UMAP run.
    """
    try:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": _datetime.datetime.now(_datetime.timezone.utc).isoformat(),
            "stage": int(stage),
            "seed": int(seed),
            "source": source,
            "config": {k: v for k, v in cfg.items() if _json_safe(v)},
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(_json.dumps(record) + "\n")
    except Exception as exc:
        _logger.debug("Failed to append umap_log.jsonl at %s: %s", log_path, exc)


def _json_safe(value) -> bool:
    """Best-effort check that ``value`` is JSON-serializable."""
    try:
        _json.dumps(value)
        return True
    except (TypeError, ValueError):
        return False



class Latent:
    """Top-level latent space manager for behavioral clustering.

    Holds temporally-windowed latent data, cluster assignments, and metadata.
    Supports hierarchical split/merge via LocalLatent and visualization
    delegation to ``castle.visualization.embedding_plots``.

    Attributes:
        data: (T, F) array of latent features after temporal windowing.
        cluster: (T,) integer array of cluster IDs (-1 = NaN/invalid).
        cluster_meta: Dict mapping cluster ID to {name, color}.
        time_window: Number of consecutive frames concatenated per sample.
    """

    def __init__(self, raw, time_window=1, device=''):
        if not device:
            device = DEFAULT_DEVICE
        time_window = int(time_window)
        n = (len(raw) // time_window) * time_window
        num_feature = raw.shape[-1]
        self.time_window = time_window
        self.data = raw[:n].reshape((-1,  num_feature * time_window))
        self.cluster = np.zeros(len(self.data)).astype(int)
        self.cluster[np.isnan(self.data.sum(axis=1))] = -1
        self.cluster_meta = dict()
        self.behavior_name2cluster_id = dict()
        
        self.cluster_meta[0] = {
            'name': 'init',
            'color': 'grey'
        }
        self.behavior_name2cluster_id['init'] = 0
        self.num_cluster = 1
        self.need_maintain_key_frames = True
        self.device=device
        
        self.used_palette = set()
        
    def get_time_window(self):
        return self.time_window

    def select(self, selected_cluster):
        if isinstance(selected_cluster, str):
            selected_cluster = self.behavior_name2cluster_id[selected_cluster]
        return LocalLatent(self.data[self.cluster == selected_cluster], self.cluster == selected_cluster, color_avoid=self.used_palette, device=self.device)
    
    def merge(self, cluster_ids):
        cluster_ids = np.array(cluster_ids)
        mi = cluster_ids.min()

        for it in cluster_ids:
            self.cluster[self.cluster == it] = mi

        self.need_maintain_key_frames = True

    def maintain_key_frames(self):
        if hasattr(self, 'key_frames'):
            delattr(self, 'key_frames')
        n = len(self.data)
        self.key_frames = [0] + [i + 1 for i in range(n - 1) if self.cluster[i] != self.cluster[i + 1]] + [n - 1]
        self.need_maintain_key_frames = False

    def palette(self, c):
        if c in self.cluster_meta:
            return self.cluster_meta[c]['color']
        else:
            return 'grey'

    def plot_syllables(self):
        """Plot behavioral syllables timeline. Delegates to castle.visualization."""
        if self.need_maintain_key_frames:
            self.maintain_key_frames()

        from castle.visualization.embedding_plots import plot_syllables as _plot_syllables
        _plot_syllables(self.cluster, self.key_frames, self.cluster_meta, palette_fn=self.palette)



    def import_local_latent(self, local_latent):
        assert hasattr(local_latent, 'cluster')
        cluster = local_latent.cluster
        index_mask = local_latent.index_mask
        old_cluster = self.cluster[index_mask]

        # Check Name used?
        # for _, it in local_latent.export.items():
            # assert not it['name'] in self.behavior_name2cluster_id, 'new name be used'

        for cluster_local_id, it in local_latent.export.items():
            incoming_name = it['name']
            if incoming_name in self.behavior_name2cluster_id:
                # Auto-rename with a numeric suffix to avoid silent data loss
                suffix = 1
                while f"{incoming_name}_{suffix}" in self.behavior_name2cluster_id:
                    suffix += 1
                incoming_name = f"{incoming_name}_{suffix}"
            cluster_id = self.num_cluster
            self.num_cluster += 1

            old_cluster[cluster == cluster_local_id] = cluster_id
            self.cluster_meta[cluster_id] = {
                'name': incoming_name,
                'color': it['color']
            }
            self.behavior_name2cluster_id[incoming_name] = cluster_id
            self.used_palette.add(it['color'])

        self.cluster[index_mask] = old_cluster

        self.need_maintain_key_frames = True


class LocalLatent:
    """Focused latent subset for one cluster, supporting UMAP + DBSCAN sub-clustering.

    Created by ``Latent.select()`` with the data and boolean index mask for a
    single cluster. Provides multi-stage UMAP embedding, DBSCAN clustering,
    cluster labeling, and visualization.

    Attributes:
        data: (N, F) latent features for the selected cluster.
        index_mask: (T,) boolean mask into the parent Latent's data array.
        embedding: (N, D) UMAP embedding (set after ``build_embedding()``).
        cluster: (N,) DBSCAN labels (set after ``build_cluster()``).
        export: Dict mapping cluster_id to {name, color} for labeled clusters.
    """

    def __init__(self, data, index_mask, color_avoid, device):
        self.data = data
        self.index_mask = index_mask
        self.device = device
        self.color_avoid = color_avoid
        self._palette = generate_palette(color_avoid)
        self.export = dict()
        

    def build_embedding(
        self,
        configs,
        progress_callback=None,
        *,
        base_seed: Optional[int] = None,
        log_path: Optional[Union[str, Path]] = None,
        reducer_factory: Optional[Callable[[dict], "DimensionReducer"]] = None,
    ) -> List[int]:
        """Run multi-stage UMAP dimensionality reduction.

        Each stage's UMAP receives a ``random_state``. The seed source for
        stage ``i`` is resolved as follows (first match wins):

        1. ``configs[i]['random_state']`` if the user supplied one explicitly.
        2. ``base_seed + i`` if ``base_seed`` was passed.
        3. A fresh ``secrets.randbits(32)`` draw (the "re-roll" path).

        The actually-used seed for every stage is returned, also stored on
        ``self.umap_seeds`` for later inspection (e.g. UI status bar).

        Args:
            configs: Single UMAP config dict or list of dicts for multi-stage.
                Each dict's keys are passed straight to ``UMAP(**dict)``.
            progress_callback: Optional ``(stage_index, total_stages) -> None``
                callable invoked before each UMAP stage, useful for Gradio
                progress bars.
            base_seed: If provided, derive each stage's seed deterministically
                as ``base_seed + i``. Useful for reproducing exact prior runs.
            log_path: If provided, append one JSON line per stage to this path
                with ``{timestamp, stage, seed, source, config}``. Created if
                missing; parent directory must already exist.
            reducer_factory: Optional ``cfg → DimensionReducer`` factory. The
                returned protocol object's ``fit_transform(X, *, random_state)``
                is called once per stage. When ``None`` (default), a
                device-appropriate :class:`UMAPReducer` is used — preserving
                the legacy umap-learn / cuml / myumap fallback chain.

                Pass a custom factory to plug in HDBSCAN, GMM, or any other
                Protocol-conforming reducer without modifying ``LocalLatent``
                (ARCH-02).

        Returns:
            List of integers — the seed actually used at each stage (length
            equals ``len(configs)``). Also stored on ``self.umap_seeds``.

        Example:
            >>> # Reproducible run
            >>> seeds = local.build_embedding(cfg_list, base_seed=42)
            >>> # Re-roll (fresh draw per stage)
            >>> seeds = local.build_embedding(cfg_list)
            >>> # Plug a different reducer (ARCH-02)
            >>> from my_pkg import PCAReducer
            >>> seeds = local.build_embedding(
            ...     cfg_list,
            ...     reducer_factory=lambda cfg: PCAReducer(**cfg),
            ... )
        """
        if reducer_factory is None:
            # Default: device-aware UMAP via the Protocol adapter (ARCH-02).
            from castle.core.clustering_backends import UMAPReducer
            device = self.device
            reducer_factory = lambda cfg: UMAPReducer(cfg, device=device)

        Z = self.data
        if hasattr(self, 'embedding'):
            delattr(self, 'embedding')

        if not isinstance(configs, list):
            configs = [configs]

        # BUG-13: catch pathological configs before UMAP raises a cryptic
        # internal error. UMAP requires n_neighbors < n_samples and
        # n_neighbors >= 5 (its documented minimum). Surface a CastleError
        # with hints instead.
        n_samples = int(Z.shape[0])
        if n_samples < 2 * _UMAP_MIN_N_NEIGHBORS:
            raise InsufficientDataError(
                f"Only {n_samples} samples available for UMAP. "
                f"Need at least {2 * _UMAP_MIN_N_NEIGHBORS}. "
                f"Hint: check whether pre-scan dropped most frames "
                f"(rotate_roi_tail missing on most masks?) or pick a "
                f"larger ROI."
            )
        for i, raw_cfg in enumerate(configs):
            nn = raw_cfg.get('n_neighbors')
            if nn is None:
                continue
            nn = int(nn)
            if nn < _UMAP_MIN_N_NEIGHBORS:
                raise InsufficientDataError(
                    f"UMAP stage {i}: n_neighbors={nn} below minimum "
                    f"{_UMAP_MIN_N_NEIGHBORS}. UMAP's k-NN graph becomes "
                    f"degenerate at very small k."
                )
            if nn >= n_samples:
                raise InsufficientDataError(
                    f"UMAP stage {i}: n_neighbors={nn} must be < n_samples="
                    f"{n_samples}. Reduce n_neighbors or supply more frames."
                )

        resolved_seeds: List[int] = []
        resolved_configs: List[dict] = []
        total_stages = len(configs)
        for i, raw_cfg in enumerate(configs):
            seed, source = _resolve_umap_seed(raw_cfg, base_seed, i)
            stage_cfg = dict(raw_cfg)
            stage_cfg['random_state'] = seed
            resolved_configs.append(stage_cfg)
            resolved_seeds.append(seed)

            _logger.info("UMAP stage %d: seed=%d (source=%s)", i, seed, source)
            if log_path is not None:
                _append_umap_log(log_path, stage=i, seed=seed, source=source, cfg=stage_cfg)

            if progress_callback is not None:
                progress_callback(i, total_stages)

            reducer = reducer_factory(raw_cfg)
            Z = reducer.fit_transform(Z, random_state=seed)

        self.embedding = np.array(Z)
        self.configs = resolved_configs
        self.umap_seeds = resolved_seeds
        return resolved_seeds



    def build_cluster(
        self,
        method: str = 'dbscan',
        configs: Optional[dict] = None,
        *,
        clusterer: Optional["Clusterer"] = None,
        random_state: int = 0,
    ) -> None:
        """Run a clusterer on the current embedding.

        Args:
            method: Legacy method-name switch. ``'dbscan'`` (default) builds
                a :class:`DBSCANClusterer`. Any other value raises unless
                ``clusterer`` is passed explicitly.
            configs: Method-specific config dict (e.g. ``{'eps': 1.0}`` for
                DBSCAN). Ignored when ``clusterer`` is passed directly.
            clusterer: Optional :class:`Clusterer` Protocol instance. When
                provided, ``method`` and ``configs`` are ignored — this is
                the ARCH-02 injection point that lets callers plug HDBSCAN /
                GMM / spectral without modifying :class:`LocalLatent`.
            random_state: Seed for stochastic clusterers (KMeans-style).
                Density-based clusterers (DBSCAN, HDBSCAN) accept the kwarg
                for API uniformity but ignore it.

        Raises:
            AssertionError: No embedding has been built yet.
            ValueError: Unknown ``method`` and no ``clusterer`` was passed.

        Example:
            >>> # Legacy default — DBSCAN
            >>> local.build_cluster(method='dbscan', configs={'eps': 0.5})
            >>> # ARCH-02 injection — HDBSCAN without touching LocalLatent
            >>> from castle.core.clustering_backends import HDBSCANClusterer
            >>> local.build_cluster(clusterer=HDBSCANClusterer(min_cluster_size=20))
        """
        assert hasattr(self, 'embedding'), (
            "Embedding not built yet — call build_embedding() before build_cluster()."
        )
        if hasattr(self, 'cluster'):
            delattr(self, 'cluster')

        if clusterer is None:
            from castle.core.clustering_backends import build_default_clusterer
            clusterer = build_default_clusterer(method, configs or {}, device=self.device)

        self.cluster = clusterer.fit_predict(self.embedding, random_state=random_state)

    def palette(self, x):
        if x == -1:
            return '#DDDDDD'
        return self._palette[x % len(self._palette)]

    
    def plot_embedding(self, dims=None):
        """Plot embedding scatter colored by cluster. Delegates to castle.visualization."""
        if dims is None:
            dims = [0, 1]
        assert hasattr(self, 'embedding')
        from castle.visualization.embedding_plots import plot_embedding as _plot_embedding
        cluster = self.cluster if hasattr(self, 'cluster') else None
        _plot_embedding(self.embedding, cluster=cluster, palette_fn=self.palette, dims=dims)
    
    def plot_name_embedding(self, dims=None):
        """Plot embedding scatter colored by named labels. Delegates to castle.visualization."""
        if dims is None:
            dims = [0, 1]
        assert hasattr(self, 'embedding')
        from castle.visualization.embedding_plots import plot_named_embedding as _plot_named
        cluster = self.cluster if hasattr(self, 'cluster') else None
        _plot_named(self.embedding, cluster=cluster, export=self.export, 
                    palette_fn=self.palette, dims=dims)



    def merge(self, cluster_ids):
        cluster_ids = np.array(cluster_ids)
        mi = cluster_ids.min()

        for it in cluster_ids:
            self.cluster[self.cluster == it] = mi


    def label_cluster(self, cluster_id, cluster_name, cluster_color=''):
        self.export[cluster_id] = {
            'name': cluster_name,
            'color': cluster_color or self._palette[cluster_id % len(self._palette)],
        }
    
    def clean_label(self):
        self.export = dict()


