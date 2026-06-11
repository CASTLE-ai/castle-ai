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
from castle.core.types import CastleDataError, InsufficientDataError

if TYPE_CHECKING:
    from castle.core.clustering_protocols import Clusterer, DimensionReducer

# BUG-13: lower bound for UMAP n_neighbors. Below ~5 UMAP becomes
# numerically pathological (k-NN graph too sparse to identify manifold
# structure). 5 is also UMAP's documented minimum.
_UMAP_MIN_N_NEIGHBORS = 5

# Non-sampled points are placed AND labelled by their single nearest sampled
# point (nearest-prototype) when UMAP subsampling is active — see build_embedding
# / build_cluster. One neighbour keeps the 2D position and the cluster label
# mutually consistent (a k>1 majority vote diverges from the snapped position on
# UMAP's non-linear layout and scatters labels).
_PROP_K = 1

_logger = _logging.getLogger(__name__)


def _knn_sampled(sampled, query_source, query_rows, k, metric, use_gpu,
                 chunk: int = 32768):
    """k nearest *sampled* rows for each queried row, in FEATURE space.

    ``sampled`` is the (S, width) fit set; the queries are
    ``query_source[query_rows]`` evaluated in chunks (so the (Nq, width) fancy-
    index copy is never materialised whole — it would be GBs at the scale that
    forces subsampling in the first place). Returns ``(idx (Nq, k), dist (Nq,
    k))`` with ``idx`` indexing into ``sampled`` (0..S-1).

    GPU (cuML) when ``use_gpu`` and cuML/cupy import; else sklearn (CPU). The
    cuML index lives on S rows (small) — NOT the M-row nn_descent graph that OOMs
    UMAP — so this fits VRAM where the full UMAP did not. metric matches what
    UMAP actually consumed (euclidean unless a cfg pins otherwise).
    """
    k = int(min(k, len(sampled)))
    n_q = len(query_rows)
    idx_out = np.empty((n_q, k), dtype=np.int64)
    dist_out = np.empty((n_q, k), dtype=np.float64)
    if n_q == 0:
        return idx_out, dist_out

    if use_gpu:
        try:
            import cupy as _cp  # noqa: PLC0415
            from castle.core.clustering_backends import _cuda_device_ctx
            if k == 1 and metric == 'euclidean':
                # Exact 1-NN via a cuBLAS GEMM: ||q-x||^2 = ||q||^2 + ||x||^2 -
                # 2 q·x, argmin over x. ~3x faster than cuML's brute NN and
                # bit-identical (verified 100% agreement). This is the label/
                # position propagation path (_PROP_K == 1) — the dominant cost on
                # big clusters (e.g. ~34s -> ~12s propagating 296k pts). cupy
                # only; no cuML import on the hot path.
                with _cuda_device_ctx('cuda'):
                    Xg = _cp.asarray(np.ascontiguousarray(sampled, dtype=np.float32))
                    Xn = (Xg * Xg).sum(axis=1)                       # ||x||^2, (S,)
                    S = int(Xg.shape[0])
                    # Bound the (gchunk x S) distance block to ~512 MB of VRAM.
                    gchunk = max(1, min(int(chunk), (512 * 1024 * 1024) // max(1, S * 4)))
                    for s in range(0, n_q, gchunk):
                        rows = query_rows[s:s + gchunk]
                        q = _cp.asarray(np.ascontiguousarray(query_source[rows], dtype=np.float32))
                        d = (q * q).sum(axis=1)[:, None] + Xn[None, :] - 2.0 * (q @ Xg.T)
                        nn1 = d.argmin(axis=1)
                        idx_out[s:s + gchunk, 0] = _cp.asnumpy(nn1)
                        dmin = _cp.take_along_axis(d, nn1[:, None], axis=1)[:, 0]
                        dist_out[s:s + gchunk, 0] = _cp.asnumpy(_cp.sqrt(_cp.maximum(dmin, 0.0)))
                        del q, d, nn1, dmin
                return idx_out, dist_out
            # General k>1 / non-euclidean: cuML brute NN.
            from cuml.neighbors import NearestNeighbors as _cuNN  # noqa: PLC0415
            with _cuda_device_ctx('cuda'):
                nn = _cuNN(n_neighbors=k, metric=metric)
                nn.fit(_cp.asarray(np.ascontiguousarray(sampled, dtype=np.float32)))
                for s in range(0, n_q, chunk):
                    rows = query_rows[s:s + chunk]
                    q = _cp.asarray(np.ascontiguousarray(query_source[rows], dtype=np.float32))
                    d, i = nn.kneighbors(q)
                    idx_out[s:s + chunk] = _cp.asnumpy(i)
                    dist_out[s:s + chunk] = _cp.asnumpy(d)
            # Pool drained once at the end of build_embedding, not here.
            return idx_out, dist_out
        except Exception as exc:  # noqa: BLE001 — cuML/cupy absent or OOM → CPU
            _logger.info("GPU k-NN propagation unavailable (%s); using CPU.", exc)

    from sklearn.neighbors import NearestNeighbors  # noqa: PLC0415
    nn = NearestNeighbors(n_neighbors=k, metric=metric)
    nn.fit(np.ascontiguousarray(sampled, dtype=np.float32))
    for s in range(0, n_q, chunk):
        rows = query_rows[s:s + chunk]
        d, i = nn.kneighbors(np.ascontiguousarray(query_source[rows], dtype=np.float32))
        idx_out[s:s + chunk] = i
        dist_out[s:s + chunk] = d
    return idx_out, dist_out


DEFAULT_DEVICE = get_device()

_palette = PALETTE_HEX * 5


def generate_distinct_color(index, saturation=0.7):
    """A visually distinct colour for cluster ``index`` via the golden-ratio hue
    sequence — unlimited non-repeating colours (vs the old fixed 62-colour
    palette, which repeated and, after ancestor-colour avoidance, collapsed to a
    few near-identical pales). Lightness cycles over a few levels so even many
    clusters with near-equal hues stay separable. ``-1`` is handled by callers.
    """
    import colorsys
    i = int(index)
    hue = (i * 0.618033988749895) % 1.0
    value = (0.95, 0.75, 0.87, 0.68)[i % 4]   # vary lightness to split near hues
    rgb = colorsys.hsv_to_rgb(hue, saturation, value)
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))


def _color_for_name(name: str) -> str:
    """Stable distinct colour derived from a cluster's NAME.

    Used identically by the live labelling scatter
    (:meth:`LocalLatent.label_cluster`) and the persisted tree / ethogram
    (:meth:`Latent.import_local_latent`) so a cluster keeps the SAME colour from
    the moment it is labelled through Submit. The hierarchical name is unique
    across the tree, so this also keeps siblings distinct — unlike colouring by
    the per-node local DBSCAN id (which restarts at 0 under every parent).
    """
    import hashlib
    idx = int(hashlib.md5(name.encode("utf-8")).hexdigest()[:8], 16)
    return generate_distinct_color(idx)


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
        # Exclude any non-finite row (NaN OR +/-Inf). np.isnan alone missed Inf
        # (np.isnan(inf) is False), so Inf rows slipped through to the embedding
        # and crashed the whole session instead of being marked -1 (contract C-4).
        self.cluster[~np.isfinite(self.data.sum(axis=1))] = -1
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
        import numpy as np
        if isinstance(selected_cluster, str):
            name = selected_cluster
            if name in self.behavior_name2cluster_id:
                # Exact match — may be leaf or a submitted parent whose frames
                # have all moved to descendants.
                cid = self.behavior_name2cluster_id[name]
                mask = self.cluster == cid
                if not mask.any():
                    # Parent was submitted: fall through to prefix match so the
                    # user can re-UMAP a node whose children exist but whose own
                    # cid now has 0 frames.
                    prefix = name + '_'
                    child_ids = [
                        cid2 for n2, cid2 in self.behavior_name2cluster_id.items()
                        if n2.startswith(prefix)
                    ]
                    if child_ids:
                        mask = np.isin(self.cluster, child_ids)
            else:
                # Prefix match — synthetic parent node: select all descendants.
                prefix = name + '_'
                child_ids = [
                    cid for n, cid in self.behavior_name2cluster_id.items()
                    if n == name or n.startswith(prefix)
                ]
                if not child_ids:
                    raise KeyError(
                        f"Cluster '{name}' not found and has no children. "
                        f"Known clusters: {list(self.behavior_name2cluster_id)}"
                    )
                mask = np.isin(self.cluster, child_ids)
        else:
            mask = self.cluster == selected_cluster
        return LocalLatent(self.data[mask], mask, color_avoid=self.used_palette, device=self.device)
    
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



    def remove_cluster_subtree(self, parent_name: str) -> List[str]:
        """Remove all descendants of ``parent_name`` and reset their frames.

        Used during overwrite-submit: clears the old sub-clustering before
        importing fresh clusters. The parent node itself is kept; only its
        descendants are removed. Frames that were assigned to descendants are
        reset to the parent cluster ID so ``import_local_latent`` can
        re-assign them to new cluster IDs.

        Args:
            parent_name: Name of the parent cluster (e.g. ``'init'``).

        Returns:
            Sorted list of descendant names that were removed.
        """
        parent_cid = self.behavior_name2cluster_id.get(parent_name)
        if parent_cid is None:
            return []

        prefix = parent_name + '_'
        descendant_names = sorted([
            n for n in list(self.behavior_name2cluster_id.keys())
            if n.startswith(prefix)
        ])
        if not descendant_names:
            return []

        descendant_cids = [self.behavior_name2cluster_id[n] for n in descendant_names]

        # Reset cluster array: all descendant frames → parent_cid
        desc_mask = np.isin(self.cluster, descendant_cids)
        self.cluster[desc_mask] = parent_cid

        # Remove descendants from metadata
        for name in descendant_names:
            cid = self.behavior_name2cluster_id.pop(name, None)
            if cid is not None:
                meta = self.cluster_meta.pop(cid, None)
                if meta:
                    self.used_palette.discard(meta.get('color', ''))

        self.need_maintain_key_frames = True
        return descendant_names

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
            # Colour by the cluster NAME, identically to LocalLatent.label_cluster,
            # so the persisted tree/ethogram colour matches what the user saw in
            # the live labelling scatter (cross-view continuity). The name is
            # hierarchical and tree-wide unique, so siblings stay distinct. A
            # user-set custom colour (differing from the name default) is kept;
            # if the name was auto-suffixed to avoid a collision, the new name's
            # colour is used (a genuinely different cluster).
            name_default = _color_for_name(it['name'])
            color = it['color'] if it.get('color') and it['color'] != name_default \
                else _color_for_name(incoming_name)
            self.cluster_meta[cluster_id] = {
                'name': incoming_name,
                'color': color,
            }
            self.behavior_name2cluster_id[incoming_name] = cluster_id
            self.used_palette.add(color)

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
        # UMAP-subsampling state (None unless build_embedding subsampled this run).
        # _embedding_sampled is the S-row UMAP output DBSCAN clusters on; the
        # neighbour arrays place + label the remaining (M-S) points. Reset
        # unconditionally at every build_embedding entry so a later no-subsample
        # run can never propagate against a stale sample (see build_cluster).
        self._subsample_idx: Optional[np.ndarray] = None
        self._embedding_sampled: Optional[np.ndarray] = None
        self._prop_neighbor_idx: Optional[np.ndarray] = None
        self._prop_nonsampled_idx: Optional[np.ndarray] = None


    def build_embedding(
        self,
        configs,
        progress_callback=None,
        *,
        base_seed: Optional[int] = None,
        log_path: Optional[Union[str, Path]] = None,
        reducer_factory: Optional[Callable[[dict], "DimensionReducer"]] = None,
        deterministic: bool = False,
        max_points: Optional[int] = None,
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
            deterministic: When ``True``, forces CPU umap-learn even on CUDA
                machines.  cuML GPU UMAP is non-deterministic (GPU parallelism
                reorders floating-point ops); CPU umap-learn with the same
                ``random_state`` produces bit-identical embeddings. Ignored
                when ``reducer_factory`` is provided.

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
            # When `deterministic=True` (user supplied an explicit seed to
            # reproduce a layout), force CPU umap-learn.  cuML UMAP is
            # non-deterministic even with a fixed random_state because GPU
            # parallelism produces non-reproducible floating-point orderings.
            from castle.core.clustering_backends import UMAPReducer
            device = 'cpu' if deterministic else self.device
            reducer_factory = lambda cfg: UMAPReducer(cfg, device=device)

        # UMAP backends (cuML / umap-learn) expect float32. self.data from
        # select() is already a fresh contiguous float32 copy in the common case,
        # so only re-copy when it isn't — avoids a full duplicate of the whole
        # selected set (a host-RAM hotspot on large caches).
        Z = (
            self.data
            if (self.data.dtype == np.float32 and self.data.flags['C_CONTIGUOUS'])
            else np.ascontiguousarray(self.data, dtype=np.float32)
        )
        if hasattr(self, 'embedding'):
            delattr(self, 'embedding')
        # Invalidation contract: always clear any subsample state from a PRIOR
        # build_embedding so a later no-subsample run (smaller cluster / bigger
        # budget) can't leave build_cluster propagating against a stale sample.
        # Set again below only on the subsampled path.
        self._subsample_idx = None
        self._embedding_sampled = None
        self._prop_neighbor_idx = None
        self._prop_nonsampled_idx = None

        # P1-3: fail loud on non-finite input instead of letting UMAP build a
        # silent garbage embedding. Normal flow can't hit this — Latent marks
        # NaN rows as cluster -1 (latent_explorer Latent.__init__) and select()
        # excludes them — so this only catches genuine misuse (e.g. select(-1)
        # or a hand-built LocalLatent fed unfiltered NaN/Inf rows).
        if not np.all(np.isfinite(Z)):
            n_bad = int((~np.isfinite(np.asarray(Z))).sum())
            raise CastleDataError(
                f"UMAP input contains {n_bad} non-finite value(s). Latent rows "
                f"with NaN/Inf features are excluded from clustering (cluster -1); "
                f"a non-finite input here means data reached UMAP unfiltered."
            )

        if not isinstance(configs, list):
            configs = [configs]

        # Draw ONE master seed when none supplied — all stages use master+i, so
        # the user only needs to remember a single value to reproduce any run.
        # Resolved BEFORE the subsample draw so the sample is reproducible too.
        if base_seed is None:
            base_seed = secrets.randbits(32)
            _logger.info("UMAP master seed drawn: %d", base_seed)

        # Memory-aware subsampling: when max_points caps the M selected points,
        # run the WHOLE multi-stage UMAP chain on a seeded S-row sample, then
        # place + label the remaining (M-S) points by k-NN to the sample in the
        # ORIGINAL feature space (stage-0 input). This is the only way to fit a
        # cluster too large for UMAP's k-NN graph; labels stay complete for the
        # time_series / ethogram. n_neighbors is validated against the EFFECTIVE
        # fit size (S), not M.
        M = int(Z.shape[0])
        if max_points is not None and M > int(max_points):
            S = int(max_points)
            rng = np.random.default_rng(int(base_seed))
            sub_idx = np.sort(rng.choice(M, size=S, replace=False))
        else:
            S = M
            sub_idx = None

        # BUG-13: catch pathological configs before UMAP raises a cryptic
        # internal error. UMAP requires n_neighbors < n_samples and
        # n_neighbors >= 5 (its documented minimum). Validate against S (the
        # actual UMAP fit size). Surface a CastleError with hints instead.
        if S < 2 * _UMAP_MIN_N_NEIGHBORS:
            raise InsufficientDataError(
                f"Only {S} samples available for UMAP. "
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
            if nn >= S:
                raise InsufficientDataError(
                    f"UMAP stage {i}: n_neighbors={nn} must be < n_samples="
                    f"{S}. Reduce n_neighbors or supply more frames."
                )

        # NOTE: per-feature z-score standardization was intentionally removed
        # (it never existed on main; it amplified low-variance / noise feature
        # dimensions for distance-based UMAP/DBSCAN). Normalisation, when wanted,
        # is now a per-sample L2 step inside the Prepare cache
        # (:mod:`castle.core.prepare`), applied before PCA. A legacy
        # ``"standardize"`` key in a saved config is harmless — UMAPReducer
        # drops it before constructing UMAP.
        resolved_seeds: List[int] = []
        resolved_configs: List[dict] = []
        total_stages = len(configs)
        # Subsample ONCE up front; the full chain runs on the sampled rows only.
        Z_fit = Z[sub_idx] if sub_idx is not None else Z
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
            Z_fit = reducer.fit_transform(Z_fit, random_state=seed)

        if sub_idx is None:
            self.embedding = np.array(Z_fit)
        else:
            # Z_fit is the S-row final embedding. Place the (M-S) non-sampled
            # points into a length-M embedding via a SINGLE feature-space nearest
            # neighbour in the sample (original stage-0 features `Z`, NOT the
            # discarded intermediate UMAP spaces). SNAP each to that nearest
            # sampled point's 2D position — do NOT distance-weight-average
            # several neighbours: UMAP (esp. min_dist=0) is non-linear, so a
            # point's feature neighbours land in DIFFERENT 2D clumps, and
            # averaging drags it into the empty gap between them — collapsing most
            # points onto the global centroid. Snapping keeps every point on a
            # real clump. build_cluster then labels each point with the SAME
            # nearest sampled point's cluster (one neighbour for both position and
            # label, so colour and location can't disagree).
            emb_sampled = np.asarray(Z_fit)
            metric = (configs[0].get('metric', 'euclidean')
                      if isinstance(configs[0], dict) else 'euclidean')
            samp_mask = np.zeros(M, dtype=bool)
            samp_mask[sub_idx] = True
            ns_idx = np.where(~samp_mask)[0]
            use_gpu = (isinstance(self.device, str)
                       and self.device.startswith('cuda') and not deterministic)
            nbr_idx, _ = _knn_sampled(
                Z[sub_idx], Z, ns_idx, _PROP_K, metric, use_gpu,
            )
            embedding = np.empty((M, emb_sampled.shape[1]), dtype=np.float64)
            embedding[sub_idx] = emb_sampled
            embedding[ns_idx] = emb_sampled[nbr_idx[:, 0]]   # snap to nearest sampled
            self.embedding = embedding
            self._subsample_idx = sub_idx
            self._embedding_sampled = emb_sampled
            self._prop_neighbor_idx = nbr_idx
            self._prop_nonsampled_idx = ns_idx
            _logger.info(
                "UMAP subsampled: fit on %d of %d points; %d propagated via "
                "%d-NN in feature space (metric=%s, gpu=%s).",
                S, M, len(ns_idx), min(_PROP_K, S), metric, use_gpu,
            )
        self.configs = resolved_configs
        self.umap_seeds = resolved_seeds
        # Release the GPU memory pool ONCE, after the whole run (all UMAP stages
        # + k-NN propagation). cuML/cupy cache freed device blocks in the pool
        # rather than returning them to the driver, so without this the VRAM
        # stays pinned (nvidia-smi / the pre-flight guard see it as used). Doing
        # it here — not per stage — keeps the pool warm DURING the run (no
        # mid-pipeline re-cudaMalloc) and frees it only when we're done.
        if (isinstance(self.device, str) and self.device.startswith('cuda')
                and not deterministic):
            from castle.core.clustering_backends import free_cuda_memory_pools
            free_cuda_memory_pools()
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

        if self._subsample_idx is None:
            if clusterer is None:
                from castle.core.clustering_backends import build_default_clusterer
                clusterer = build_default_clusterer(method, configs or {}, device=self.device)
            self.cluster = clusterer.fit_predict(self.embedding, random_state=random_state)
            return

        # Subsampled run: DBSCAN clusters the S sampled points (cheap, so eps
        # tuning re-runs fast), then each non-sampled point inherits the majority
        # label of the same sampled neighbours that placed it in 2D — keeping the
        # full-length labels consistent with the density model that produced them.
        assert self._embedding_sampled is not None and self._prop_neighbor_idx is not None, (
            "Subsample state is inconsistent — re-run build_embedding."
        )
        assert len(self._embedding_sampled) == len(self._subsample_idx), (
            "Sampled embedding length does not match the subsample index."
        )
        M = len(self.embedding)
        S = len(self._embedding_sampled)
        assert int(self._subsample_idx.max(initial=-1)) < M, (
            "Subsample index points outside the current selection — stale state."
        )
        # DBSCAN runs on the S-point subsample, but min_samples (a neighbour
        # COUNT) is interpreted by the user at full-data scale. A uniform sample
        # keeps the eps-radius density but scales counts by S/M, so feeding the
        # full-scale min_samples straight in makes it ~M/S times stricter (e.g.
        # 467 on a 1% sample needs ~10% of neighbours -> almost everything noise).
        # Scale min_samples by S/M so "min points" behaves the same at any
        # subsample %. eps is a 2D radius set by UMAP's layout (~N-invariant) and
        # is NOT scaled. Only the default (config-built) DBSCAN is rescaled; an
        # injected clusterer is used verbatim.
        if clusterer is None:
            from castle.core.clustering_backends import build_default_clusterer
            cfg = dict(configs or {})
            _ms = cfg.get('min_samples')
            if _ms:
                scaled = max(1, int(round(int(_ms) * S / M)))
                if scaled != int(_ms):
                    _logger.info(
                        "DBSCAN min_samples %d -> %d (scaled by subsample S/M = %d/%d)",
                        int(_ms), scaled, S, M,
                    )
                cfg['min_samples'] = scaled
            clusterer = build_default_clusterer(method, cfg, device=self.device)
        labels_sampled = np.asarray(clusterer.fit_predict(
            self._embedding_sampled, random_state=random_state,
        ))
        cluster = np.empty(M, dtype=labels_sampled.dtype)
        cluster[self._subsample_idx] = labels_sampled
        if self._prop_nonsampled_idx is not None and len(self._prop_nonsampled_idx):
            # Each non-sampled point inherits the cluster label of the SAME
            # nearest sampled point that build_embedding snapped its 2D position
            # to. Label and position must come from one neighbour: a k-NN
            # majority vote (in feature space) routinely disagrees with the
            # single nearest neighbour's 2D location (UMAP is non-linear, so a
            # point's feature neighbours span different 2D clusters), which paints
            # a point INSIDE cluster A with cluster B's colour — scattered
            # confetti over the embedding. Nearest-prototype keeps colour and
            # position consistent: the point sits on a sampled point and shares
            # its label.
            cluster[self._prop_nonsampled_idx] = labels_sampled[
                self._prop_neighbor_idx[:, 0]
            ]
        self.cluster = cluster

    def palette(self, x):
        if x == -1:
            return '#DDDDDD'
        # Golden-ratio distinct colour per local cluster id — never repeats and
        # doesn't shrink as ancestor clusters consume palette entries.
        return generate_distinct_color(int(x))

    
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
        # Colour by NAME (not local id) so the live scatter matches the colour
        # the persisted tree/ethogram will use after Submit (import_local_latent
        # colours by the same name-derived key). A user-set colour wins.
        self.export[cluster_id] = {
            'name': cluster_name,
            'color': cluster_color or _color_for_name(cluster_name),
        }
    
    def clean_label(self):
        self.export = dict()


