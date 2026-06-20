"""
castle/service/clustering_service.py
Service layer for UMAP clustering and behavioral annotation.

Provides a ClusteringSession class that manages the full clustering workflow
without depending on Gradio.

No gradio imports.
"""

import os
import json
import logging
import time
from dataclasses import dataclass
from typing import List, Optional, Callable, Any, Tuple

import numpy as np
import pandas as pd

from castle.core import runtime_env
from castle.core.cluster import LatentAggregator, auto_generate_cluster_name
from castle.core.logging_config import setup_logger
from castle.core.types import CastleDataError, InsufficientDataError
from castle.service.session_manager import SessionManager
# The cluster_*.npz filename grammar + node-meta sidecar lookup were extracted to
# castle/service/cluster_npz.py (this module was a 1876-line god-module). Re-export
# them so internal callers and `from castle.service.clustering_service import ...`
# keep working unchanged.
from castle.service.cluster_npz import (  # noqa: F401
    load_node_meta,
    _parent_from_cluster_filename,
    find_cluster_npz_for_parent,
    _embedding_npz_files,
    find_latest_cluster_npz,
    _extract_child_names_from_filename,
)
# Heuristic param suggester also extracted; re-exported for the CLI importer.
from castle.service.cluster_params import (  # noqa: F401
    ClusteringParamSuggestion,
    suggest_clustering_params,
)
# Cluster-transfer model save/apply extracted to cluster_persistence; re-exported
# so the UI/CLI/test importers (`from clustering_service import ...`) keep working.
from castle.service.cluster_persistence import (  # noqa: F401
    save_project_cluster_model,
    apply_cluster_model_to_project,
)
from castle.utils.latent_explorer import LocalLatent
# Session/local-latent restore-from-disk extracted to cluster_restore.py (further
# god-module split). Re-exported so `from clustering_service import ...` keeps working.
from castle.service.cluster_restore import (  # noqa: F401
    RestoredSessionArtifacts,
    restore_local_latent_from_npz,
    restore_session_from_disk,
)

# setup_logger attaches an INFO StreamHandler (the module previously used a bare
# getLogger, so its INFO lines — incl. the [UMAP timing] log — were dropped when
# the app left root logging at the default WARNING).
logger = setup_logger(__name__)


def build_timeseries_meta(fps: float, cluster_id_to_name: dict, n_frames: int) -> dict:
    """Self-describing sidecar metadata for a per-video ``time_series_*.csv``.

    The CSV has one row per ORIGINAL video frame with columns ``behavior``
    (per-frame cluster id; ``-1`` = unclustered) and ``exclude_reason`` (a
    per-frame exclusion-reason code). On its own the CSV has no frame→time
    mapping and no cluster-id→name lookup, so it isn't usable as a standalone
    supplementary-data file. This sidecar records fps, the name map, and the
    CASTLE version so the CSV is interpretable independently.
    """
    import castle
    return {
        "schema_version": 1,
        "castle_version": getattr(castle, "__version__", "unknown"),
        "fps": float(fps),
        "n_frames": int(n_frames),
        "time_seconds": "frame_index / fps",
        "columns": {
            "behavior": "per-frame cluster id (-1 = unclustered)",
            "exclude_reason": "per-frame exclusion-reason code",
        },
        "cluster_id_to_name": {int(k): str(v) for k, v in cluster_id_to_name.items()},
    }


def _write_id_csv(cluster_meta: dict, cluster_path: str) -> str:
    """Write the cluster ``id.csv`` (``Id``/``Name``/``Color``) from cluster_meta.

    Shared by both submit paths (CLI ``ClusteringSession.submit`` and UI
    ``submit_local_to_global``) so the cluster table is identical and cannot
    diverge. Returns the written path.
    """
    df = pd.DataFrame({
        'Id': [k for k in cluster_meta],
        'Name': [v['name'] for v in cluster_meta.values()],
        'Color': [v['color'] for v in cluster_meta.values()],
    })
    id_csv_path = os.path.join(cluster_path, 'id.csv')
    df.to_csv(id_csv_path, index=False)
    return id_csv_path


def _write_timeseries_csvs(latents, aggregator, cluster_path: str) -> list:
    """Write per-video ``time_series_*.csv`` (``behavior`` + ``exclude_reason``)
    plus a self-describing ``.meta.json`` sidecar for each video.

    Bins are expanded to original frames via the aggregator's FrameIndexMap (the
    legacy ``for_window(1)`` map reproduces ``np.repeat(., bin_size)``; the
    prepared map handles decimation + windowing). Shared by BOTH submit paths —
    ``ClusteringSession.submit`` (CLI) and ``submit_local_to_global`` (UI) — so the
    two frontends emit byte-identical time_series artifacts and cannot re-diverge.

    Args:
        latents: global Latent (provides ``cluster``, ``data``, ``cluster_meta``,
            ``time_window``).
        aggregator: LatentAggregator (provides ``videos_meta``,
            ``frame_index_map``, ``fps_per_video``, ``fps``).
        cluster_path: ``<project>/cluster/`` output directory.

    Returns:
        List of written ``time_series_*.csv`` paths (one per video).
    """
    from castle.core.ethogram import derive_exclude_reason
    try:
        reason_bins = derive_exclude_reason(latents.cluster, latents.data)
    except Exception as exc:  # never block submit; fall back to "not excluded" (0)
        logger.warning(
            "submit: could not derive exclude_reason (%s); defaulting to 0.", exc
        )
        reason_bins = np.zeros(len(latents.cluster), dtype=np.int8)

    fim = aggregator.frame_index_map
    ts_paths: list = []
    cum = 0
    for video_idx, (vn, v) in enumerate(aggregator.videos_meta):
        video_cluster = latents.cluster[cum:cum + vn]
        video_reason = reason_bins[cum:cum + vn]
        if fim is not None:
            video_frames = fim.expand_labels_to_orig(video_cluster, video_idx)
            video_reason_frames = fim.expand_labels_to_orig(video_reason, video_idx)
        else:
            video_frames = np.repeat(video_cluster, latents.time_window)
            video_reason_frames = np.repeat(video_reason, latents.time_window)
        df2 = pd.DataFrame({
            'behavior': video_frames,
            'exclude_reason': video_reason_frames,
        })
        video_basename = os.path.splitext(os.path.basename(v))[0]
        ts_path = os.path.join(cluster_path, f'time_series_{video_basename}.csv')
        df2.to_csv(ts_path, index=False)
        ts_paths.append(ts_path)
        # Self-describing sidecar (fps/units/cluster-name map). Best-effort.
        try:
            video_fps = aggregator.fps_per_video.get(v, aggregator.fps)
            meta = build_timeseries_meta(
                video_fps,
                {cid: m.get('name', f'cluster_{cid}')
                 for cid, m in latents.cluster_meta.items()},
                len(video_frames),
            )
            meta_path = os.path.join(
                cluster_path, f'time_series_{video_basename}.meta.json'
            )
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
        except (OSError, AttributeError) as exc:
            logger.warning("Could not write time_series meta sidecar: %s", exc)
        cum += vn
    return ts_paths


class ClusteringSession:
    """
    Manages a complete clustering session: loading latents, UMAP, DBSCAN,
    labeling, and exporting results.
    
    This is the Gradio-independent equivalent of the state managed by
    cluster_page_ui.py's event handlers and gr.State objects.
    
    Usage:
        session = ClusteringSession(storage_path, project, roi=1, bin_size=1, model='dinov3_vitb16')
        session.run_umap('init', [{'n_neighbors': 100, 'min_dist': 0.0, 'n_components': 2, 'n_epochs': 5000}])
        session.run_dbscan(eps=1.0)
        session.label_cluster(0, 'walking')
        session.label_cluster(1, 'rearing')
        result = session.submit()
    """
    
    def __init__(self, storage_path: str, project_name: str, roi: int,
                 bin_size: int, model: str,
                 notify: Optional[Callable] = None,
                 prepare_id: Optional[str] = None, k_prime: Optional[int] = None):
        """
        Initialize clustering session by loading and aggregating latents.
        
        Args:
            storage_path: Root storage directory
            project_name: Project name
            roi: ROI ID to cluster
            bin_size: Temporal binning (frames per bin)
            model: Visual model name used for extraction
            notify: Optional notification callback(message, level)
        """
        self.storage_path = storage_path
        self.project_name = project_name
        self.roi = roi
        self.bin_size = bin_size
        self.model = model
        # Prepared-cache provenance — persisted into the session manifest so a
        # CLI run restores in the UI with the identical PCA slice (see submit()).
        self._prepare_id = prepare_id
        self._k_prime = k_prime
        self._notify = notify or (lambda msg, level='info': logger.log(
            logging.WARNING if level == 'error' else logging.INFO, msg))

        # Initialize aggregator (prepared cache when prepare_id is given)
        self.aggregator = LatentAggregator(
            storage_path, project_name, roi, bin_size,
            model_name=model,
            notify=self._notify,
            prepare_id=prepare_id,
            k_prime=k_prime,
        )

        # Create Latent explorer object
        self.latents = self.aggregator.get_latent_object()

        # Working state
        self.local_latents: Optional[LocalLatent] = None
        self._current_cluster_name: Optional[str] = None
        # Run parameters captured so submit() can write the same node_*_meta.json
        # sidecar the UI writes (lets the UI restore umap_config/eps/seed).
        self._last_umap_config: Any = None
        self._last_umap_seeds: List[int] = []
        self._last_eps: Optional[float] = None
    
    @property
    def cluster_names(self) -> List[str]:
        """List of all named clusters."""
        if hasattr(self.latents, 'behavior_name2cluster_id'):
            return list(self.latents.behavior_name2cluster_id.keys())
        return []
    
    @property
    def videos_meta(self) -> List:
        """Video metadata from the aggregator."""
        return self.aggregator.videos_meta
    
    def run_umap(
        self,
        cluster_name: str,
        umap_config: Any,
        *,
        base_seed: Optional[int] = None,
        log_path: Optional[str] = None,
    ) -> dict:
        """Select a cluster and run UMAP dimensionality reduction.

        Args:
            cluster_name: Name of the cluster to focus on (e.g., 'init').
            umap_config: UMAP config — either a dict, list of dicts, or JSON
                string.
            base_seed: If provided, every UMAP stage uses ``base_seed + stage_i``
                as ``random_state``; otherwise a fresh ``secrets.randbits(32)``
                is drawn per stage. The resolved seeds are returned in the
                ``umap_seeds`` key.
            log_path: Optional absolute path to a ``umap_log.jsonl`` file; if
                given, one JSON line per stage is appended. The parent dir is
                created on demand.

        Returns:
            dict with keys:
                'n_points': int — number of points in the selected cluster
                'embedding_shape': tuple — shape of the embedding
                'umap_seeds': List[int] — seeds used per stage (re-run with
                    the last value as ``base_seed`` to reproduce)
                'success': bool
        """
        if isinstance(umap_config, str):
            umap_config = json.loads(umap_config)

        self._current_cluster_name = cluster_name
        self._last_umap_config = umap_config
        self.local_latents = self.latents.select(selected_cluster=cluster_name)

        if len(self.local_latents.data) == 0:
            return {'n_points': 0, 'embedding_shape': (0, 0),
                    'umap_seeds': [], 'success': False,
                    'error': 'Selected cluster is empty'}

        resolved_seeds = self.local_latents.build_embedding(
            umap_config, base_seed=base_seed, log_path=log_path,
        )
        self._last_umap_seeds = list(resolved_seeds)

        return {
            'n_points': len(self.local_latents.data),
            'embedding_shape': self.local_latents.embedding.shape,
            'umap_seeds': list(resolved_seeds),
            'success': True,
        }
    
    def run_dbscan(self, eps: float) -> dict:
        """
        Run DBSCAN clustering on the current embedding.
        
        Args:
            eps: DBSCAN epsilon-neighborhood radius
        
        Returns:
            dict with keys:
                'n_clusters': int — number of clusters found (excluding noise)
                'cluster_ids': list — unique cluster IDs
                'noise_count': int — number of noise points (-1)
                'success': bool
        """
        if self.local_latents is None or not hasattr(self.local_latents, 'embedding'):
            return {'success': False, 'error': 'Run UMAP first'}

        self._last_eps = eps
        self.local_latents.build_cluster(method='dbscan', configs={'eps': eps})
        
        unique = np.unique(self.local_latents.cluster)
        n_clusters = len(unique[unique >= 0])
        noise_count = int(np.sum(self.local_latents.cluster == -1))
        
        return {
            'n_clusters': n_clusters,
            'cluster_ids': unique.tolist(),
            'noise_count': noise_count,
            'success': True,
        }
    
    def label_cluster(self, cluster_id: int, name: str, color: str = '') -> None:
        """
        Assign a name (and optional color) to a local cluster.
        
        Args:
            cluster_id: Cluster ID from DBSCAN output
            name: Human-readable behavior name
            color: Optional hex color. Auto-assigned if empty.
        """
        if self.local_latents is None:
            raise ValueError("No local latents — run UMAP + DBSCAN first")
        self.local_latents.label_cluster(cluster_id, name, color)
    
    def auto_label_all(self, parent_name: Optional[str] = None) -> int:
        """
        Auto-label all clusters with hierarchical names.
        
        Args:
            parent_name: Parent cluster name for hierarchical naming.
                         If None, uses the current cluster name.
        
        Returns:
            Number of clusters labeled.
        """
        if self.local_latents is None or not hasattr(self.local_latents, 'cluster'):
            raise ValueError("No clusters — run UMAP + DBSCAN first")
        
        if parent_name is None:
            parent_name = self._current_cluster_name
        
        unique_clusters = np.unique(self.local_latents.cluster)
        count = 0
        for cid in unique_clusters:
            if cid == -1:
                continue
            name = auto_generate_cluster_name(parent_name, cid)
            self.local_latents.label_cluster(cid, name)
            count += 1
        
        return count

    def start_new_session(self, *, variance_pct: Optional[float] = None,
                          name: str = "") -> str:
        """Begin a fresh, UI-restorable clustering session.

        Mirrors the UI's ``init_aggregator``: clears the ``cluster/`` root of any
        stale session files, then registers a :class:`SessionManager` session +
        ``manifest.json`` so a completed CLI run appears in — and can be restored
        from — the Behavior Microscope UI. Call once before ``run_umap``.

        Returns:
            The new ``session_id`` (e.g. ``"session_003"``).
        """
        mgr = SessionManager(self.storage_path, self.project_name)
        mgr._clear_cluster_root()
        total_frames = (
            len(self.aggregator.latents)
            if self.aggregator.latents is not None else 0
        )
        info = mgr.create_session(
            model=self.model,
            roi_id=int(self.roi) if self.roi else 1,
            bin_size=int(self.bin_size),
            total_frames=total_frames,
            name=name,
            prepare_id=self._prepare_id,
            k_prime=self._k_prime,
            variance_pct=variance_pct,
        )
        return info.session_id

    def _write_node_meta(self, cluster_path: str, embedding_path: str) -> None:
        """Write the ``node_{parent}_meta.json`` sidecar the UI uses on restore.

        Payload is byte-for-byte the shape produced by the UI submit path
        (:func:`submit_local_to_global`) so a CLI-produced node restores its
        umap_config / eps / seed when reclicked in the Behavior Microscope.
        """
        parent = self._current_cluster_name
        if not parent:
            return
        umap_config_str = (
            json.dumps(self._last_umap_config)
            if self._last_umap_config is not None else None
        )
        meta_path = os.path.join(cluster_path, f'node_{parent}_meta.json')
        meta_payload = {
            'parent_cluster_name': parent,
            'umap_config': umap_config_str,
            'eps': self._last_eps,
            'min_samples': None,  # CLI run_dbscan uses eps only
            'preset': None,
            'umap_seed': self._last_umap_seeds[0] if self._last_umap_seeds else None,
            'embedding_npz': (
                os.path.basename(embedding_path) if embedding_path else None
            ),
        }
        try:
            with open(meta_path, 'w') as f:
                json.dump(meta_payload, f, indent=2)
        except OSError as e:
            logger.warning("Failed to persist node meta sidecar %s: %s",
                           meta_path, e)

    def _snapshot_to_session(self) -> Optional[str]:
        """Snapshot the current ``cluster/`` artifacts into the active session.

        If no session is active (e.g. ``submit`` was called via the service API
        without :meth:`start_new_session`), one is created on the fly so the run
        is still UI-restorable. Returns the session id used, or ``None`` on error.
        """
        try:
            mgr = SessionManager(self.storage_path, self.project_name)
            active_id = mgr.get_active_session_id()
            if not active_id:
                total_frames = (
                    len(self.aggregator.latents)
                    if self.aggregator.latents is not None else 0
                )
                active_id = mgr.create_session(
                    model=self.model,
                    roi_id=int(self.roi) if self.roi else 1,
                    bin_size=int(self.bin_size),
                    total_frames=total_frames,
                    prepare_id=self._prepare_id,
                    k_prime=self._k_prime,
                ).session_id
            mgr.snapshot_to_session(active_id)
            n_clusters = len([
                k for k in self.latents.cluster_meta
                if self.latents.cluster_meta[k].get('name') != 'init'
            ])
            mgr.save_session_state(active_id, n_clusters)
            return active_id
        except Exception as e:  # noqa: BLE001 — snapshot is best-effort, never fail submit
            logger.warning("Failed to snapshot CLI session: %s", e)
            return None

    def submit(self) -> dict:
        """
        Import labeled local clusters into the global latent and export results.
        
        Returns:
            dict with keys:
                'id_csv_path': str — path to the cluster ID CSV
                'time_series_paths': list[str] — per-video time series CSVs
                'srt_paths': list[str] — per-video SRT subtitle files
                'embedding_path': str — path to saved embedding NPZ
                'success': bool
        """
        if self.local_latents is None:
            return {'success': False, 'error': 'No local latents'}
        
        # Import local clusters into global latent
        self.latents.import_local_latent(self.local_latents)
        
        # Save ID CSV
        cluster_path = os.path.join(self.storage_path, self.project_name, 'cluster')
        os.makedirs(cluster_path, exist_ok=True)
        
        id_csv_path = _write_id_csv(self.latents.cluster_meta, cluster_path)
        
        # Generate per-video time_series CSVs at ORIGINAL-frame resolution.
        # Per-video time_series_*.csv (+ self-describing meta sidecar), via the
        # shared writer used by the UI path too (keeps the two paths identical).
        ts_paths = _write_timeseries_csvs(self.latents, self.aggregator, cluster_path)
        
        # Generate subtitles
        srt_paths = self.aggregator.generate_subtitles(
            self.latents.cluster, self.latents.cluster_meta
        )
        
        # Save embedding
        emb_name = ''
        # Encode child names in sorted-cluster-ID order so restore can re-pair
        # them to cluster IDs deterministically (label order is not necessarily
        # ID order, which mis-mapped historic names on restore).
        for cid in sorted(self.local_latents.export):
            emb_name += self.local_latents.export[cid]['name'] + '_'
        emb_path = os.path.join(cluster_path, f'cluster_{emb_name}.npz')
        
        index_mask = self.local_latents.index_mask
        masked_emb = self.local_latents.embedding
        masked_cls = self.local_latents.cluster
        config = self.local_latents.configs
        n_samples = len(index_mask)
        n_features = masked_emb.shape[-1]

        emb_full = np.zeros((n_samples, n_features)) + np.nan
        emb_full[index_mask] = masked_emb
        cls_full = np.zeros(n_samples, dtype=np.int16) - 1
        cls_full[index_mask] = masked_cls

        # is_sampled: True DBSCAN members vs k-NN-propagated rows (UMAP subsample).
        # Export trains only on sampled rows; missing key => all-True (legacy).
        local_sampled = np.ones(len(masked_cls), dtype=bool)
        _sub_idx = getattr(self.local_latents, '_subsample_idx', None)
        if _sub_idx is not None:
            local_sampled = np.zeros(len(masked_cls), dtype=bool)
            local_sampled[_sub_idx] = True
        is_sampled = np.zeros(n_samples, dtype=bool)
        is_sampled[index_mask] = local_sampled

        # Record the resolved software/hardware stack (device, cuML vs CPU, lib
        # versions) into the artifact so a non-reproduction can be told apart
        # from a backend mismatch (cuML-GPU and CPU UMAP give different
        # embeddings). Additive key — existing loaders read emb/cls/config only.
        from castle.core.environment import collect_run_environment
        np.savez_compressed(
            emb_path, emb=emb_full, cls=cls_full, config=config,
            is_sampled=is_sampled,
            run_environment=np.array([json.dumps(collect_run_environment())]),
        )

        # #5: node-meta sidecar (parent = the clustered node, e.g. 'init') so the
        # UI restores umap_config/eps/seed when the node is reclicked. #4: snapshot
        # the cluster/ artifacts into a SessionManager session so this CLI run is
        # listed in and restorable from the Behavior Microscope UI.
        self._write_node_meta(cluster_path, emb_path)
        session_id = self._snapshot_to_session()

        return {
            'id_csv_path': id_csv_path,
            'time_series_paths': ts_paths,
            'srt_paths': srt_paths,
            'embedding_path': emb_path,
            'session_id': session_id,
            'success': True,
        }
    
    def restore(self) -> dict:
        """
        Restore a previous clustering session from saved CSV files.
        
        Returns:
            dict with keys:
                'cluster_count': int — number of clusters restored
                'id_csv_path': str — path to the ID CSV
                'time_series_paths': list[str]
                'success': bool
        """
        cluster_path = os.path.join(self.storage_path, self.project_name, 'cluster')
        id_csv_path = os.path.join(cluster_path, 'id.csv')
        
        if not os.path.exists(id_csv_path):
            return {'success': False, 'error': 'No previous session found'}
        
        # Restore cluster_meta from id.csv
        id_df = pd.read_csv(id_csv_path)
        for _, row in id_df.iterrows():
            cluster_id = int(row['Id'])
            # An engine-default colour is stored as an empty cell, which pandas
            # reads back as NaN — coerce to '' so it resolves live by name.
            raw = row.get('Color', '')
            color = raw if isinstance(raw, str) and raw.strip().lower() != 'nan' else ''
            self.latents.cluster_meta[cluster_id] = {
                'name': row['Name'], 'color': color
            }
            self.latents.behavior_name2cluster_id[row['Name']] = cluster_id
            if color and color != 'grey':
                self.latents.used_palette.add(color)
        self.latents.num_cluster = len(id_df)
        
        # Restore cluster assignments from time_series CSVs. Prepared sessions
        # recover per-window GLOBAL labels through the window map (see
        # restore_session_from_disk); legacy keeps the bin downsample.
        fim = getattr(self.aggregator, "frame_index_map", None)
        prepared = bool(getattr(self.aggregator, "_prepared", False)) and fim is not None
        ts_paths = []
        cum = 0
        for video_idx, (vn, v) in enumerate(self.aggregator.videos_meta):
            video_basename = os.path.splitext(os.path.basename(v))[0]
            ts_path = os.path.join(cluster_path, f'time_series_{video_basename}.csv')
            if os.path.exists(ts_path):
                ts_df = pd.read_csv(ts_path)
                behavior = ts_df['behavior'].values
                if prepared:
                    assert fim is not None  # `prepared` implies a window map
                    bin_clusters = fim.windowed_labels_from_orig(behavior, video_idx)
                else:
                    bin_clusters = behavior[::self.latents.time_window][:vn]
                if len(bin_clusters) != vn:
                    raise CastleDataError(
                        f"Session restore: {os.path.basename(ts_path)} downsamples "
                        f"to {len(bin_clusters)} bins but video {v!r} expects {vn}. "
                        f"The time_series CSV is likely truncated/corrupt. Assigning "
                        f"it would mislabel this and every subsequent video — refusing. "
                        f"Re-save the session or delete the corrupt CSV and re-cluster."
                    )
                # A NaN/empty cell makes pandas read the whole column as float;
                # assigning that into the int cluster array would silently
                # truncate / produce garbage IDs. Refuse instead.
                if not np.isfinite(np.asarray(bin_clusters, dtype=np.float64)).all():
                    raise CastleDataError(
                        f"Session restore: {os.path.basename(ts_path)} has non-finite "
                        f"(NaN/inf) behavior values; refusing to coerce them into "
                        f"integer cluster labels. Fix or delete the CSV and re-cluster."
                    )
                self.latents.cluster[cum:cum + vn] = bin_clusters.astype(
                    self.latents.cluster.dtype
                )
                ts_paths.append(ts_path)
            cum += vn
        
        return {
            'cluster_count': self.latents.num_cluster,
            'id_csv_path': id_csv_path,
            'time_series_paths': ts_paths,
            'success': True,
        }
    
    def get_frame(self, global_bin_index: int) -> Optional[np.ndarray]:
        """
        Retrieve a representative frame for a given global bin index.

        Args:
            global_bin_index: Global bin index across all videos

        Returns:
            Frame as numpy array (H, W, 3) or None
        """
        return self.aggregator.get_frame(global_bin_index)


# ---------------------------------------------------------------------------
# Clustering hyper-parameter suggester (PERF-06 / P3-D)
# ---------------------------------------------------------------------------


# (ClusteringParamSuggestion + suggest_clustering_params — the heuristic
# param suggester — now live in castle/service/cluster_params.py and are
# imported at the top of this module.)


# ---------------------------------------------------------------------------
# Session restore helpers (ARCH-01 / P2-D)
# ---------------------------------------------------------------------------

# (The cluster_*.npz filename grammar + node-meta sidecar lookup —
# load_node_meta / _parent_from_cluster_filename / find_cluster_npz_for_parent /
# _embedding_npz_files / find_latest_cluster_npz / _extract_child_names_from_filename —
# now live in castle/service/cluster_npz.py and are imported at the top of this
# module.)




# ---------------------------------------------------------------------------
# Cluster-transfer helpers (project-level)
# ---------------------------------------------------------------------------

# (Prepared-model / cluster-transfer persistence — _save_prepared_cluster_model,
# save_project_cluster_model, apply_cluster_model_to_project — extracted to
# castle/service/cluster_persistence.py and re-exported at the top of this module.)


# ---------------------------------------------------------------------------
# Pure algorithmic helpers used by both ClusteringSession + cluster_handlers
# (ARCH-01 / P4)
#
# These functions encapsulate "what the algorithm actually does" without any
# UI coupling. They take the relevant Latent / LocalLatent /
# LatentAggregator objects explicitly and return structured results.
#
# Gradio handlers call them and translate exceptions into ``gr.Info`` /
# ``gr.Warning``. The CLI calls ``ClusteringSession`` (which in turn
# wraps these helpers).
# ---------------------------------------------------------------------------


@dataclass
class UMAPRunArtifacts:
    """Pure result of running UMAP on a single cluster.

    Attributes:
        local_latents: The freshly built :class:`LocalLatent` (caller stores
            this in Gradio state).
        resolved_seeds: Per-stage seeds actually used (length =
            ``len(cfg_list)`` where the input was multi-stage).
        status_text: User-facing one-line summary suitable for a status bar.
    """
    local_latents: Any
    resolved_seeds: List[int]
    status_text: str


def run_umap_on_cluster(
    latents: Any,
    cluster_name: str,
    cfg: Any,
    *,
    base_seed: Optional[int] = None,
    deterministic: bool = False,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    log_path: Optional[str] = None,
    subsample: bool = False,
    subsample_pct: float = 30.0,
) -> UMAPRunArtifacts:
    """Select a cluster and build its UMAP embedding.

    Args:
        latents: Parent :class:`castle.utils.latent_explorer.Latent`.
        cluster_name: Cluster name from the UI tree (e.g. ``'init'``).
        cfg: Single UMAP config dict or list of dicts (multi-stage UMAP).
            Passed through to :meth:`LocalLatent.build_embedding`.
        base_seed: If provided, stage ``i`` uses ``base_seed + i`` for
            ``random_state``; otherwise a fresh ``secrets.randbits(32)``
            is drawn (re-roll path).
        deterministic: When True, forces CPU umap-learn regardless of the
            device detected by the environment. cuML UMAP is non-deterministic
            even with a fixed ``random_state`` (GPU parallelism); CPU
            umap-learn with the same seed produces bit-identical embeddings.
            Set this when the caller supplied an explicit ``base_seed``.
        progress_callback: Optional ``(stage_index, total_stages) -> None``
            callable invoked before each UMAP stage. Gradio uses this to
            drive its progress bar.
        log_path: Optional absolute path to ``umap_log.jsonl`` — one JSON
            line per stage will be appended.

    Returns:
        :class:`UMAPRunArtifacts` with the new LocalLatent and resolved
        seeds.

    Raises:
        InsufficientDataError: The selected cluster is empty (caller
            should surface this to the user — already does so via
            ``gr.Info`` in the Gradio path).
    """
    _t0 = time.perf_counter()
    local_latents = latents.select(selected_cluster=cluster_name)
    _t_select = time.perf_counter()
    if len(local_latents.data) == 0:
        raise InsufficientDataError(
            f"Cluster '{cluster_name}' has no data points. Select a "
            f"different cluster or re-cluster with adjusted parameters."
        )

    # Subsampling is MANUAL (the "Subsample UMAP" toggle + "% of points"). When
    # on, UMAP fits a seeded S = round(pct% * M)-row sample and labels propagate
    # to all M points in build_cluster (so time_series / ethogram stay complete);
    # this is the speed lever on huge selections. When off, UMAP fits all M. A
    # pre-flight guard estimates peak memory either way (cuML's CUDA OOM is a bare
    # std::bad_alloc that pins the GPU) and refuses with a concrete next step —
    # rather than silently auto-capping behind the user's back.
    from castle.core.clustering_backends import (
        target_cuda_free_bytes,
        umap_peak_bytes,
        umap_host_bytes,
    )
    from castle.utils.latent_explorer import _UMAP_MIN_N_NEIGHBORS

    _first_cfg = cfg[0] if isinstance(cfg, (list, tuple)) and cfg else cfg
    _nn = int(_first_cfg.get('n_neighbors', 15)) if isinstance(_first_cfg, dict) else 15
    _M = len(local_latents.data)
    _width = int(local_latents.data.shape[1]) if local_latents.data.ndim == 2 else 1

    # Resolve the requested fit size.
    if subsample:
        _pct = min(100.0, max(1.0, float(subsample_pct)))
        _S = min(_M, max(1, int(round(_pct / 100.0 * _M))))
        _need_min = max(2 * _UMAP_MIN_N_NEIGHBORS, _nn + 1)
        if _S < _need_min:
            raise CastleDataError(
                f"Subsample {_pct:.0f}% of {_M:,} points = only {_S:,}, below the "
                f"UMAP minimum ({_need_min}) for n_neighbors={_nn}. Raise the % (or "
                f"reduce n_neighbors)."
            )
    else:
        _S = _M

    # Pre-flight memory guards. umap_peak_bytes / umap_host_bytes are raw 1x
    # estimates, so the free-memory fraction is the only safety margin. Two
    # independent constraints, each refused with a concrete fix:
    #   • device peak — UMAP's working buffers (VRAM on GPU, RAM on the CPU/
    #     reproducible path).
    #   • host RAM    — on the GPU path the buffers are on the device, but the
    #     latent matrix + its transient copies (select()'s slice, the float32
    #     conversion, the sampled draw) + the embedding output still live in
    #     host RAM. The VRAM guard never sees these, so add a host-RAM floor.
    # Both fractions default to 0.85 (~1.18x headroom). Env overrides:
    # CASTLE_UMAP_VRAM_FRACTION, CASTLE_UMAP_RAM_FRACTION.
    def _env_frac(name: str, default: float) -> float:
        try:
            raw = os.environ.get(name)
            return float(raw) if raw is not None else default
        except ValueError:
            return default

    _frac_vram = _env_frac('CASTLE_UMAP_VRAM_FRACTION', 0.85)
    _frac_ram = _env_frac('CASTLE_UMAP_RAM_FRACTION', 0.85)
    _ncomp = int(cfg[-1].get('n_components', 2)) if (
        isinstance(cfg, (list, tuple)) and cfg and isinstance(cfg[-1], dict)
    ) else (int(_first_cfg.get('n_components', 2)) if isinstance(_first_cfg, dict) else 2)
    _fix = (
        f"Lower the 'UMAP % of points' (currently {_pct:.0f}%)."
        if subsample else
        "Turn on 'Subsample UMAP' and set a % of points."
    )

    def _ram_free() -> Optional[float]:
        try:
            return runtime_env.available_ram_bytes()
        except Exception:  # noqa: BLE001
            return None

    _peak = umap_peak_bytes(_S, _width, _nn)
    _vram_free: Optional[float] = (
        None if deterministic else target_cuda_free_bytes('cuda')
    )
    if _vram_free:
        # GPU path: UMAP buffers in VRAM; matrix + copies + embedding in host RAM.
        if _peak > _frac_vram * float(_vram_free):
            raise CastleDataError(
                f"UMAP on cluster '{cluster_name}' ({_S:,} points x {_width} dims, "
                f"n_neighbors={_nn}) needs ~{_peak / 1e9:.1f} GB GPU memory but only "
                f"~{_vram_free / 1e9:.1f} GB is free (safe limit {_frac_vram:.0%}). {_fix}"
            )
        # build_embedding only re-copies the matrix when it isn't already
        # float32 + C-contiguous (mirrors latent_explorer.build_embedding).
        _dat = local_latents.data
        _full_copy = not (
            getattr(_dat, 'dtype', None) == np.float32
            and getattr(_dat, 'flags', None) is not None
            and _dat.flags['C_CONTIGUOUS']
        )
        _host = umap_host_bytes(_M, _S, _width, _ncomp, full_copy=_full_copy)
        _rf = _ram_free()
        if _rf and _host > _frac_ram * float(_rf):
            raise CastleDataError(
                f"UMAP on cluster '{cluster_name}' ({_M:,} points x {_width} dims) "
                f"needs ~{_host / 1e9:.1f} GB additional host RAM for its working "
                f"copies but only ~{_rf / 1e9:.1f} GB is free (safe limit "
                f"{_frac_ram:.0%}). {_fix}"
            )
    else:
        # CPU / no-GPU path: everything is in host RAM; umap_peak models it.
        _rf = _ram_free()
        if _rf and _peak > _frac_ram * float(_rf):
            raise CastleDataError(
                f"UMAP on cluster '{cluster_name}' ({_S:,} points x {_width} dims, "
                f"n_neighbors={_nn}) needs ~{_peak / 1e9:.1f} GB RAM but only "
                f"~{_rf / 1e9:.1f} GB is free (safe limit {_frac_ram:.0%}). {_fix}"
            )

    _cap = _S if (subsample and _S < _M) else None
    sub_note = ""
    if _cap is not None:
        sub_note = (
            f"Subsampled: UMAP on {_cap:,} of {_M:,} pts ({_pct:.0f}%); labels "
            f"propagated to all {_M:,}. "
        )
        logger.info(
            "UMAP subsample: M=%d -> S=%d (%.0f%%) width=%d nn=%d",
            _M, _cap, _pct, _width, _nn,
        )

    _t_guard = time.perf_counter()
    resolved_seeds = local_latents.build_embedding(
        cfg,
        progress_callback=progress_callback,
        base_seed=base_seed,
        deterministic=deterministic,
        log_path=log_path,
        max_points=_cap,
    )
    _t_build = time.perf_counter()
    logger.info(
        "[UMAP timing] select=%.2fs guard+setup=%.2fs build_embedding(UMAP+prop)=%.2fs "
        "total=%.2fs (M=%d S=%d width=%d nn=%d device=%s)",
        _t_select - _t0, _t_guard - _t_select, _t_build - _t_guard, _t_build - _t0,
        _M, _S, _width, _nn, "cpu" if deterministic else "cuda",
    )

    mode_note = " (CPU, reproducible)" if deterministic else ""
    status_text = (
        f"✅ UMAP done{mode_note}. {sub_note}seed={resolved_seeds[0]}. "
        f"Paste `{resolved_seeds[0]}` into the seed box to reproduce."
    )
    return UMAPRunArtifacts(
        local_latents=local_latents,
        resolved_seeds=list(resolved_seeds),
        status_text=status_text,
    )


def run_dbscan_on_local(
    local_latents: Any, eps: float, min_samples: Optional[int] = None,
) -> None:
    """Run DBSCAN in place on an existing :class:`LocalLatent`.

    Args:
        local_latents: A LocalLatent with an embedding already built.
        eps: DBSCAN epsilon. The function mutates ``local_latents.cluster``.
        min_samples: DBSCAN min_samples (core-point neighbour count). ``None``
            keeps the backend default (5). Larger → more points become noise and
            only denser regions form clusters.

    Raises:
        InsufficientDataError: No embedding has been built yet.
    """
    if local_latents is None:
        raise InsufficientDataError(
            "No embedding available. Run UMAP before clustering."
        )
    embedding = getattr(local_latents, 'embedding', None)
    if embedding is None:
        raise InsufficientDataError(
            "No embedding available. Run UMAP before clustering."
        )
    configs: dict = {'eps': eps}
    if min_samples is not None:
        configs['min_samples'] = int(min_samples)
    local_latents.build_cluster(method='dbscan', configs=configs)


@dataclass
class SubmitArtifacts:
    """File paths produced by :func:`submit_local_to_global`."""
    syllables_fig: Any
    cluster_choices: List[Tuple[str, int]]
    id_csv_path: str
    time_series_paths: List[str]
    subtitle_paths: List[str]
    local_latents: Any
    embedding_path: Optional[str]


def submit_local_to_global(
    latents: Any,
    local_latents: Any,
    aggregator: Any,
    *,
    storage_path: str,
    project_name: str,
    parent_cluster_name: Optional[str] = None,
    umap_config_str: Optional[str] = None,
    eps_value: Optional[float] = None,
    min_samples_value: Optional[int] = None,
    preset_value: Optional[str] = None,
    umap_seed: Optional[int] = None,
    overwrite: bool = False,
) -> SubmitArtifacts:
    """Merge local clusters into the global ``Latent`` and persist artefacts.

    Mirrors what the legacy handler ``import_info_from_local_latent`` did,
    minus the Gradio coupling. The caller wraps the result for Gradio
    ``outputs=...``.

    Args:
        latents: Parent :class:`Latent`. Mutated — local clusters are
            imported into it.
        local_latents: The LocalLatent whose labels should be merged.
        aggregator: :class:`LatentAggregator` (for ``videos_meta``,
            ``generate_subtitles``, ``time_window``).
        storage_path: Root storage directory.
        project_name: Project name (used to compose the ``cluster/`` path).

    Returns:
        :class:`SubmitArtifacts`.

    Raises:
        CastleError: Re-raised from ``import_local_latent`` failure.
    """
    # Plotting lives in plotting_service (no Gradio dep). The "choice tuple"
    # shaper is in cluster_handlers because Gradio shapes it.
    from castle.service.plotting_service import plot_syllables_per_video
    from castle.ui.cluster_handlers import update_select_cluster_list

    cluster_path = os.path.join(storage_path, project_name, 'cluster')

    if overwrite and parent_cluster_name:
        # Delete old embedding npz referenced by the existing sidecar before
        # adding new clusters so orphaned files do not accumulate.
        old_meta = load_node_meta(cluster_path, parent_cluster_name)
        if old_meta and old_meta.get('embedding_npz'):
            old_npz = os.path.join(cluster_path, old_meta['embedding_npz'])
            try:
                os.unlink(old_npz)
                logger.debug("Deleted old embedding npz on overwrite: %s", old_npz)
            except OSError as e:
                logger.debug("Could not delete old npz %s: %s", old_npz, e)
        # Remove all descendants from the global latent so import starts fresh.
        latents.remove_cluster_subtree(parent_cluster_name)

    latents.import_local_latent(local_latents)
    fig = plot_syllables_per_video(latents, aggregator)
    cluster_choices = update_select_cluster_list(latents)
    os.makedirs(cluster_path, exist_ok=True)

    id_csv_path = _write_id_csv(latents.cluster_meta, cluster_path)

    # Per-video time_series_*.csv (+ self-describing meta sidecar) via the shared
    # writer (same as the CLI submit() path, so both frontends emit identical
    # artifacts and the two paths cannot re-diverge).
    df2_paths = _write_timeseries_csvs(latents, aggregator, cluster_path)

    subtitle_paths = aggregator.generate_subtitles(
        latents.cluster, latents.cluster_meta,
    )

    embedding_path: Optional[str] = None
    if (local_latents is not None
            and hasattr(local_latents, 'embedding')
            and local_latents.embedding is not None):
        from castle.visualization.embedding_scatter import EmbeddingScatterPlot

        Z_plt = EmbeddingScatterPlot(local_latents)
        cluster_name = ''
        # Sorted-cluster-ID order (see submit) so restore can re-pair names.
        for cid in sorted(local_latents.export):
            cluster_name += local_latents.export[cid]['name'] + '_'
        embedding_path = os.path.join(cluster_path, f'cluster_{cluster_name}.npz')
        Z_plt.save_named_embedding(save_path=embedding_path)

    # Sidecar metadata indexed by parent cluster name so the UI can restore
    # umap_config / eps / preset / seed / embedding npz when the user
    # reclicks the node.
    if parent_cluster_name:
        # Pull the resolved seed from local_latents when caller didn't pass
        # one explicitly — generate_embedding writes umap_seeds onto the
        # LocalLatent after build_embedding completes.
        resolved_seed = umap_seed
        if resolved_seed is None and local_latents is not None:
            seeds = getattr(local_latents, 'umap_seeds', None)
            if seeds:
                resolved_seed = int(seeds[0])
            else:
                cfgs = getattr(local_latents, 'configs', None)
                if cfgs:
                    first = cfgs[0] if isinstance(cfgs, list) else cfgs
                    if isinstance(first, dict) and first.get('random_state') is not None:
                        resolved_seed = int(first['random_state'])

        meta_path = os.path.join(
            cluster_path, f'node_{parent_cluster_name}_meta.json'
        )
        meta_payload = {
            'parent_cluster_name': parent_cluster_name,
            'umap_config': umap_config_str,
            'eps': eps_value,
            'min_samples': min_samples_value,
            'preset': preset_value,
            'umap_seed': resolved_seed,
            'embedding_npz': (
                os.path.basename(embedding_path) if embedding_path else None
            ),
        }
        try:
            with open(meta_path, 'w') as f:
                json.dump(meta_payload, f, indent=2)
        except OSError as e:
            logger.warning("Failed to persist node meta sidecar %s: %s",
                           meta_path, e)

    return SubmitArtifacts(
        syllables_fig=fig,
        cluster_choices=cluster_choices,
        id_csv_path=id_csv_path,
        time_series_paths=df2_paths,
        subtitle_paths=subtitle_paths,
        local_latents=local_latents,
        embedding_path=embedding_path,
    )


def auto_label_local_clusters(
    local_latents: Any,
    parent_name: str,
) -> int:
    """Auto-label every non-noise local cluster with a hierarchical name.

    Args:
        local_latents: LocalLatent whose ``cluster`` array carries DBSCAN
            output. Mutated in place via ``label_cluster``.
        parent_name: Name of the parent cluster (e.g. ``'init'``). Used
            as a prefix for ``auto_generate_cluster_name``.

    Returns:
        Number of clusters labelled (i.e. excluding the noise label
        ``-1``).
    """
    cluster_arr = getattr(local_latents, 'cluster', None)
    if cluster_arr is None:
        raise InsufficientDataError(
            "No clusters available. Run DBSCAN before submitting."
        )
    unique_clusters = np.unique(cluster_arr)
    count = 0
    for cluster_id in unique_clusters:
        if cluster_id == -1:
            continue
        name = auto_generate_cluster_name(parent_name, cluster_id)
        local_latents.label_cluster(cluster_id, name)
        count += 1
    return count






@dataclass
class InitAggregatorArtifacts:
    """Pure result of initialising a fresh clustering aggregator."""
    aggregator: Any
    latents: Any


def init_clustering_aggregator(
    storage_path: str,
    project_name: str,
    *,
    select_roi_id: Any,
    bin_size: Any,
    select_model: str,
    notify: Optional[Callable[[str, str], None]] = None,
    prepare_id: Optional[str] = None,
    variance_pct: Optional[float] = None,
    pooling: str = 'auto',
) -> InitAggregatorArtifacts:
    """Build a :class:`LatentAggregator` + record a new session row.

    Pure (Gradio-free) version of the legacy ``init_mulvideo`` handler.

    Args:
        storage_path: Root storage directory.
        project_name: Project name.
        select_roi_id: ROI ID.
        bin_size: Temporal bin size.
        select_model: Model name (e.g. ``'dinov3_vitb16'``).
        notify: ``(msg, level)`` callback for LatentAggregator progress
            messages.
        prepare_id: Prepared-cache id, or None for the legacy raw path.
        variance_pct: Explained-variance target (percent) the user entered for
            the prepared path; resolved here against the cache's evr to the
            concrete PCA-dim count ``k_prime`` that is fed to UMAP. ``None`` →
            the 95% default. Ignored on the legacy path.

    Returns:
        :class:`InitAggregatorArtifacts` carrying the freshly built
        aggregator and its associated ``Latent`` object.
    """
    from castle.core.cluster import LatentAggregator

    # Resolve the explained-variance % the user typed into the concrete PCA-dim
    # count k' (the unit UMAP / windowing actually slice). Done here, where the
    # cache meta is loadable, so _init_prepared keeps taking a plain dim count.
    # Persist BOTH: variance_pct = intent, k_prime = frozen width for restore.
    k_prime: Optional[int] = None
    persisted_variance_pct: Optional[float] = None
    if prepare_id:
        from castle.core.prepare import (
            k_prime_for_variance,
            load_prepare,
            variance_pct_to_fraction,
        )
        frac = variance_pct_to_fraction(variance_pct)
        persisted_variance_pct = frac * 100.0
        prep_dir = os.path.join(
            storage_path, project_name, 'cluster', 'prepared', prepare_id
        )
        try:
            k_prime = k_prime_for_variance(load_prepare(prep_dir).meta, frac)
        except Exception as exc:  # noqa: BLE001 — fall back to the cache default
            logger.warning(
                "Could not resolve variance %%->k' for %s (%s); using cache default.",
                prepare_id, exc,
            )
            k_prime = None

    # Clear cluster/ root before starting a new session so no files from a
    # previous session can pollute this one.  Do this before creating the
    # LatentAggregator so that any memmap cache from the old session is gone.
    mgr = SessionManager(storage_path, project_name)
    mgr._clear_cluster_root()

    aggregator = LatentAggregator(
        storage_path, project_name, select_roi_id, int(bin_size),
        model_name=select_model,
        notify=notify,
        prepare_id=prepare_id,
        k_prime=k_prime,
        pooling=pooling,
    )
    latents = aggregator.get_latent_object()

    mgr.create_session(
        model=select_model,
        roi_id=int(select_roi_id) if select_roi_id else 1,
        bin_size=int(bin_size),
        total_frames=(
            len(aggregator.latents)
            if aggregator.latents is not None else 0
        ),
        prepare_id=prepare_id,
        k_prime=k_prime,
        variance_pct=persisted_variance_pct,
    )

    return InitAggregatorArtifacts(aggregator=aggregator, latents=latents)
