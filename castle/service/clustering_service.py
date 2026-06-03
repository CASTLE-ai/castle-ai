"""
castle/service/clustering_service.py
Service layer for UMAP clustering and behavioral annotation.

Provides a ClusteringSession class that manages the full clustering workflow
without depending on Gradio.

No gradio imports.
"""

import glob
import os
import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Tuple

import numpy as np
import pandas as pd

from castle.core.cluster import LatentAggregator, auto_generate_cluster_name
from castle.core.types import CastleDataError, InsufficientDataError
from castle.service.session_manager import SessionManager
from castle.utils.latent_explorer import LocalLatent

logger = logging.getLogger(__name__)


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
                 notify: Optional[Callable] = None):
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
        self._notify = notify or (lambda msg, level='info': logger.log(
            logging.WARNING if level == 'error' else logging.INFO, msg))
        
        # Initialize aggregator
        self.aggregator = LatentAggregator(
            storage_path, project_name, roi, bin_size,
            model_name=model,
            notify=self._notify,
        )
        
        # Create Latent explorer object
        self.latents = self.aggregator.get_latent_object()
        
        # Working state
        self.local_latents: Optional[LocalLatent] = None
        self._current_cluster_name: Optional[str] = None
    
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
        self.local_latents = self.latents.select(selected_cluster=cluster_name)

        if len(self.local_latents.data) == 0:
            return {'n_points': 0, 'embedding_shape': (0, 0),
                    'umap_seeds': [], 'success': False,
                    'error': 'Selected cluster is empty'}

        resolved_seeds = self.local_latents.build_embedding(
            umap_config, base_seed=base_seed, log_path=log_path,
        )

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
        
        df1 = pd.DataFrame({
            'Id': [k for k in self.latents.cluster_meta],
            'Name': [v['name'] for v in self.latents.cluster_meta.values()],
            'Color': [v['color'] for v in self.latents.cluster_meta.values()],
        })
        id_csv_path = os.path.join(cluster_path, 'id.csv')
        df1.to_csv(id_csv_path, index=False)
        
        # Generate per-video time_series CSVs
        ts_paths = []
        cum = 0
        for vn, v in self.aggregator.videos_meta:
            video_cluster = self.latents.cluster[cum:cum + vn]
            video_frames = np.repeat(video_cluster, self.latents.time_window)
            df2 = pd.DataFrame({'behavior': video_frames})
            
            video_basename = os.path.basename(v).split('.')[0]
            ts_path = os.path.join(cluster_path, f'time_series_{video_basename}.csv')
            df2.to_csv(ts_path, index=False)
            ts_paths.append(ts_path)
            cum += vn
        
        # Generate subtitles
        srt_paths = self.aggregator.generate_subtitles(
            self.latents.cluster, self.latents.cluster_meta
        )
        
        # Save embedding
        emb_name = ''
        for _, it in self.local_latents.export.items():
            emb_name += it['name'] + '_'
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
        
        np.savez_compressed(emb_path, emb=emb_full, cls=cls_full, config=config)
        
        return {
            'id_csv_path': id_csv_path,
            'time_series_paths': ts_paths,
            'srt_paths': srt_paths,
            'embedding_path': emb_path,
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
            color = row.get('Color', 'grey')
            self.latents.cluster_meta[cluster_id] = {
                'name': row['Name'], 'color': color
            }
            self.latents.behavior_name2cluster_id[row['Name']] = cluster_id
            if color != 'grey':
                self.latents.used_palette.add(color)
        self.latents.num_cluster = len(id_df)
        
        # Restore cluster assignments from time_series CSVs
        ts_paths = []
        cum = 0
        for vn, v in self.aggregator.videos_meta:
            video_basename = os.path.basename(v).split('.')[0]
            ts_path = os.path.join(cluster_path, f'time_series_{video_basename}.csv')
            if os.path.exists(ts_path):
                ts_df = pd.read_csv(ts_path)
                bin_clusters = ts_df['behavior'].values[::self.latents.time_window][:vn]
                if len(bin_clusters) != vn:
                    raise CastleDataError(
                        f"Session restore: {os.path.basename(ts_path)} downsamples "
                        f"to {len(bin_clusters)} bins but video {v!r} expects {vn}. "
                        f"The time_series CSV is likely truncated/corrupt. Assigning "
                        f"it would mislabel this and every subsequent video — refusing. "
                        f"Re-save the session or delete the corrupt CSV and re-cluster."
                    )
                self.latents.cluster[cum:cum + vn] = bin_clusters
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


@dataclass
class ClusteringParamSuggestion:
    """Heuristic clustering parameters for first-time users.

    Attributes:
        n_samples: Sample count the suggestion was computed for.
        min_cluster_size: HDBSCAN ``min_cluster_size`` suggestion. Sized
            so the smallest accepted cluster represents ~0.5% of the
            data (a B-SOiD / MoSeq convention).
        min_samples: HDBSCAN ``min_samples`` suggestion. Always smaller
            than ``min_cluster_size``.
        eps_range: DBSCAN ``eps`` values worth sweeping interactively.
    """

    n_samples: int
    min_cluster_size: int
    min_samples: int
    eps_range: List[float] = field(default_factory=list)


def suggest_clustering_params(n_samples: int) -> ClusteringParamSuggestion:
    """Suggest HDBSCAN/DBSCAN starting parameters for ``n_samples`` bins.

    Args:
        n_samples: Total number of latent samples (bins) the user is
            about to cluster.

    Returns:
        :class:`ClusteringParamSuggestion`. Values are heuristics —
        researchers should sweep ``eps_range`` interactively in the
        Behavior Microscope rather than trust the suggestion blindly.

    Notes:
        Rationale: ``min_cluster_size = max(10, n//200)`` keeps the
        smallest cluster ≥ 0.5% of the data. ``min_samples = max(5,
        n//500)`` keeps DBSCAN's k-neighbour requirement lower than
        ``min_cluster_size`` (HDBSCAN expects this). The eps sweep is
        anchored at 1.0 (the global default, see
        :data:`castle.defaults.DBSCAN_EPS`) and brackets two octaves on
        each side.
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}.")
    return ClusteringParamSuggestion(
        n_samples=int(n_samples),
        min_cluster_size=max(10, n_samples // 200),
        min_samples=max(5, n_samples // 500),
        eps_range=[0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
    )


# ---------------------------------------------------------------------------
# Session restore helpers (ARCH-01 / P2-D)
# ---------------------------------------------------------------------------

def load_node_meta(cluster_path: str, parent_cluster_name: str) -> Optional[dict]:
    """Return the persisted sidecar metadata for a parent cluster node, or None.

    The sidecar is written by :func:`submit_local_to_global` when the UI
    submits a fresh round of clustering against a parent node. It holds the
    UMAP config string and DBSCAN eps used at that submission, plus the
    basename of the associated ``cluster_*.npz``.

    Args:
        cluster_path: Directory typically ``<project>/cluster/``.
        parent_cluster_name: Name of the parent cluster (e.g. ``'init_a0'``).

    Returns:
        Parsed dict, or ``None`` if the file is missing or malformed.
    """
    if not parent_cluster_name:
        return None
    meta_path = os.path.join(
        cluster_path, f'node_{parent_cluster_name}_meta.json'
    )
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, 'r') as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Could not read node meta %s: %s", meta_path, e)
        return None


def _parent_from_cluster_filename(
    basename: str,
    parent_cluster_name: str,
) -> bool:
    """Return True iff ``basename`` is the embedding npz for the supplied
    parent node.

    The filename is built in :func:`submit_local_to_global` as
    ``cluster_{c1}_{c2}_..._{ck}_.npz`` where every ``c_i`` is an immediate
    child of the parent and therefore has ``parent_depth + 1``
    underscore-segments. Parsing the filename works even after deeper
    splits have evicted intermediate nodes from ``cluster_meta`` (which
    is why an export-name based check breaks for non-deepest parents).
    """
    if not basename.startswith('cluster_') or not basename.endswith('.npz'):
        return False
    if basename == 'cluster_model.npz':
        return False
    core = basename[len('cluster_'):-len('.npz')]
    if not core.endswith('_'):
        return False
    segments = core.rstrip('_').split('_')
    parent_depth = len(parent_cluster_name.split('_'))
    seg_per_child = parent_depth + 1
    if seg_per_child <= 0 or len(segments) % seg_per_child != 0:
        return False
    child_count = len(segments) // seg_per_child
    if child_count < 1:
        return False
    parent_segs = parent_cluster_name.split('_')
    for i in range(child_count):
        chunk = segments[i * seg_per_child:(i + 1) * seg_per_child]
        if chunk[:parent_depth] != parent_segs:
            return False
    return True


def find_cluster_npz_for_parent(
    cluster_path: str,
    parent_cluster_name: str,
    latents: Any,
) -> Optional[str]:
    """Fallback locator: pick the ``cluster_*.npz`` produced when
    ``parent_cluster_name`` was last submitted.

    Used when a node has no ``node_{parent}_meta.json`` sidecar (e.g.
    submissions made before the sidecar feature landed) or when the
    sidecar points at a missing file. The parent is identified by parsing
    the canonical filename ``cluster_{c1}_..._{ck}_.npz`` — see
    :func:`_parent_from_cluster_filename`. When several files match we
    return the most recently modified one.

    Args:
        cluster_path: Directory typically ``<project>/cluster/``.
        parent_cluster_name: Parent node name (e.g. ``'init'``).
        latents: Unused; kept for backwards-compatible call sites.

    Returns:
        Absolute path to the best-matching npz, or ``None``.
    """
    del latents  # filename-only matching no longer needs cluster_meta
    if not parent_cluster_name:
        return None

    candidates = glob.glob(os.path.join(cluster_path, 'cluster_*.npz'))
    best: Optional[str] = None
    best_mtime = -1.0
    for npz in candidates:
        if not _parent_from_cluster_filename(
            os.path.basename(npz), parent_cluster_name,
        ):
            continue
        mt = os.path.getmtime(npz)
        if mt > best_mtime:
            best = npz
            best_mtime = mt
    return best


def find_latest_cluster_npz(cluster_path: str) -> Optional[str]:
    """Return the most recently modified ``cluster_*.npz`` in ``cluster_path``.

    Args:
        cluster_path: Directory typically ``<project>/cluster/``.

    Returns:
        Absolute path to the newest matching file, or ``None`` if no
        ``cluster_*.npz`` exists.
    """
    npz_files = glob.glob(os.path.join(cluster_path, 'cluster_*.npz'))
    if not npz_files:
        return None
    npz_files.sort(key=os.path.getmtime, reverse=True)
    return npz_files[0]


def _extract_child_names_from_filename(
    basename: str,
    parent_cluster_name: str,
) -> List[str]:
    """Parse child cluster names from a ``cluster_*.npz`` filename.

    The file is named ``cluster_{c1}_{c2}_..._{ck}_.npz`` where each ``c_i``
    is an immediate child of ``parent_cluster_name``.  Because children have
    exactly ``parent_depth + 1`` underscore-segments we can recover the
    ordered list without touching ``cluster_meta``.

    Args:
        basename: Filename (no directory), e.g. ``cluster_init_a0_init_a1_.npz``.
        parent_cluster_name: Parent node name, e.g. ``'init'``.

    Returns:
        Ordered list of child names (empty list on parse failure).
    """
    if not basename.startswith('cluster_') or not basename.endswith('.npz'):
        return []
    core = basename[len('cluster_'):-len('.npz')]
    if not core.endswith('_'):
        return []
    segments = core.rstrip('_').split('_')
    parent_depth = len(parent_cluster_name.split('_'))
    seg_per_child = parent_depth + 1
    if seg_per_child <= 0 or len(segments) % seg_per_child != 0:
        return []
    child_count = len(segments) // seg_per_child
    return [
        '_'.join(segments[i * seg_per_child:(i + 1) * seg_per_child])
        for i in range(child_count)
    ]


def restore_local_latent_from_npz(
    npz_path: str,
    latents: Any,
    parent_cluster_name: Optional[str] = None,
) -> Tuple[Optional[Any], Optional[np.ndarray]]:
    """Reconstruct a :class:`LocalLatent` from a saved cluster ``.npz``.

    The npz holds three arrays:

    * ``emb``    — ``(N, 2)`` embedding with NaN for non-selected rows.
    * ``cls``    — ``(N,)`` integer labels with ``-1`` for non-selected rows.
    * ``config`` — UMAP config used to produce ``emb``.

    Args:
        npz_path: Path to the saved ``cluster_*.npz``.
        latents: Parent :class:`castle.utils.latent_explorer.Latent`
            object (provides ``data``, ``used_palette``, ``device``,
            ``cluster``, ``cluster_meta``).
        parent_cluster_name: Name of the parent node (e.g. ``'init'``).
            When supplied, the filename is parsed as a fallback to recover
            the original child names for any local cluster IDs whose current
            global counterpart has since been evicted from ``cluster_meta``
            due to deeper splits.

    Returns:
        ``(local_latents, embedding_array)`` or ``(None, None)`` on
        failure. ``embedding_array`` is the ``(M, 2)`` masked embedding
        ready to hand to ``EmbeddingScatterPlot``.

    Notes:
        The function is intentionally exception-tolerant — clustering
        sessions sometimes carry partially-written npz files from
        crashed runs, and restoring a session should fall back to "no
        embedding restored" rather than refusing to open the UI.
    """
    try:
        data = np.load(npz_path, allow_pickle=True)
        # Other npz artefacts (e.g. cluster_model.npz from save_cluster_model)
        # live in the same directory but use a different schema. Bail out
        # quietly rather than logging a full traceback.
        required_keys = {'emb', 'cls', 'config'}
        if not required_keys.issubset(set(data.files)):
            logger.debug(
                "Skipping %s: missing required keys (have %s)",
                npz_path, list(data.files),
            )
            return None, None
        emb_full = data['emb']
        cls_full = data['cls']
        config = data['config']

        valid_mask = ~np.isnan(emb_full[:, 0])
        masked_emb = emb_full[valid_mask]
        masked_cls = cls_full[valid_mask]

        # latents.data is the FULL latent (N rows); emb_full is the LOCAL
        # subset (M rows, M ≤ N). valid_mask has M entries, so indexing
        # latents.data with it fails when N ≠ M.  Use the embedding itself
        # as a stand-in when the sizes do not match.
        local_data = (
            latents.data[valid_mask]
            if hasattr(latents, 'data') and latents.data.shape[0] == emb_full.shape[0]
            else masked_emb
        )
        local_latents = LocalLatent(
            data=local_data,
            index_mask=valid_mask,
            color_avoid=latents.used_palette,
            device=latents.device,
        )
        local_latents.embedding = masked_emb
        local_latents.cluster = masked_cls
        local_latents.configs = config.tolist() if hasattr(config, 'tolist') else config

        # Step 1: try to recover historic child names from the filename.
        # The file is named cluster_{c1}_{c2}_..._{ck}_.npz in submission
        # order (c_i corresponds to local cluster ID i).  This gives us the
        # correct names even when deeper splits have evicted the original
        # children from cluster_meta.
        basename = os.path.basename(npz_path)
        filename_child_names: List[str] = []
        if parent_cluster_name:
            filename_child_names = _extract_child_names_from_filename(
                basename, parent_cluster_name,
            )

        # Step 2: build export — prefer filename-derived historic names;
        # fall back to current cluster_meta for any ID not covered.
        # Build a name→color lookup from current cluster_meta so we can
        # assign colours to historic clusters whose descendants are still live.
        name_to_color: Dict[str, str] = {
            meta['name']: meta['color']
            for meta in latents.cluster_meta.values()
        }

        def _find_color_for_historic(child_name: str) -> str:
            """Return color for a historic cluster, walking to descendants."""
            if child_name in name_to_color:
                return name_to_color[child_name]
            prefix = child_name + '_'
            for nm, col in name_to_color.items():
                if nm.startswith(prefix) and col:
                    return col
            return '#888888'  # neutral grey; plot_named_embedding also guards against ''

        for cid_local in np.unique(masked_cls):
            if cid_local == -1:
                continue
            # Prefer historic name from filename when available.
            if cid_local < len(filename_child_names):
                child_name = filename_child_names[cid_local]
                local_latents.export[cid_local] = {
                    'name': child_name,
                    'color': _find_color_for_historic(child_name),
                }
                continue
            # Fallback: map via current global cluster_meta.
            global_indices = np.where(valid_mask)[0]
            global_cluster_vals = latents.cluster[global_indices]
            local_mask = masked_cls == cid_local
            if not np.any(local_mask):
                continue
            global_ids = global_cluster_vals[local_mask]
            global_id = Counter(global_ids.tolist()).most_common(1)[0][0]
            if global_id in latents.cluster_meta:
                meta = latents.cluster_meta[global_id]
                local_latents.export[cid_local] = {
                    'name': meta['name'],
                    'color': meta['color'],
                }
        return local_latents, masked_emb
    except Exception:
        logger.exception("Failed to restore local latent from %s", npz_path)
        return None, None


# ---------------------------------------------------------------------------
# Cluster-transfer helpers (project-level)
# ---------------------------------------------------------------------------

def save_project_cluster_model(
    project_path: str,
    output_path: Optional[str] = None,
    model_name: str = "",
    k: int = 5,
) -> str:
    """Save a project's clustering model for transfer.

    Loads the UMAP embedding, cluster labels, and original latent features
    from the project's ``cluster/`` directory, then persists them as a
    ``.npz`` file that can be applied to new data.

    Args:
        project_path: Absolute path to the project directory.
        output_path: Where to write the model file.  Defaults to
            ``<project_path>/cluster/cluster_model.npz``.
        model_name: Descriptive name saved in the metadata.
        k: Number of neighbours for k-NN at apply time.

    Returns:
        Absolute path to the saved model file.

    Raises:
        FileNotFoundError: If required cluster/embedding files are missing.
    """
    from castle.core.cluster_transfer import save_cluster_model
    import glob

    cluster_dir = os.path.join(project_path, "cluster")
    if not os.path.isdir(cluster_dir):
        raise FileNotFoundError(f"No cluster directory found: {cluster_dir}")

    # --- Load id.csv for cluster names ---
    id_csv_path = os.path.join(cluster_dir, "id.csv")
    if not os.path.exists(id_csv_path):
        raise FileNotFoundError(f"No id.csv found: {id_csv_path}")

    id_df = pd.read_csv(id_csv_path)
    cluster_names = {int(row["Id"]): row["Name"] for _, row in id_df.iterrows()}

    # --- Load embedding .npz (most recently modified, not arbitrary glob order) ---
    emb_files = glob.glob(os.path.join(cluster_dir, "cluster_*.npz"))
    if not emb_files:
        raise FileNotFoundError(f"No embedding .npz found in {cluster_dir}")
    emb_path = max(emb_files, key=os.path.getmtime)
    emb_data = np.load(emb_path, allow_pickle=True)
    emb_full = emb_data["emb"]        # (N, 2) with NaN for masked-out points
    cls_full = emb_data["cls"]        # (N,) with -1 for masked-out points

    # --- Load latent features from latent/ directory ---
    latent_dir = os.path.join(project_path, "latent")
    if not os.path.isdir(latent_dir):
        raise FileNotFoundError(f"No latent directory found: {latent_dir}")

    # Pick most-recently-modified model sub-directory (matches user's latest extraction).
    model_dirs = [
        os.path.join(latent_dir, d) for d in os.listdir(latent_dir)
        if os.path.isdir(os.path.join(latent_dir, d))
    ]
    if not model_dirs:
        raise FileNotFoundError(f"No model sub-directories in {latent_dir}")
    model_subdir = max(model_dirs, key=os.path.getmtime)

    # Concatenate latent files in the same order as the project config
    latent_files = sorted(glob.glob(os.path.join(model_subdir, "*.npz")))
    if not latent_files:
        raise FileNotFoundError(f"No latent .npz files in {model_subdir}")

    latent_chunks = []
    for lf in latent_files:
        loaded = np.load(lf)
        latent_chunks.append(loaded["latent"])
    all_features = np.concatenate(latent_chunks, axis=0)

    # --- Build valid mask (non-NaN embedding rows) ---
    valid_mask = ~np.isnan(emb_full).any(axis=1)
    umap_embedding = emb_full[valid_mask]
    cluster_labels = cls_full[valid_mask]

    # Align features: the embedding was built from the same number of bins
    n_emb = len(emb_full)
    if len(all_features) > n_emb:
        all_features = all_features[:n_emb]
    training_features = all_features[valid_mask]

    if output_path is None:
        output_path = os.path.join(cluster_dir, "cluster_model.npz")

    # Determine fps from project config if available
    fps = 30.0
    config_path = os.path.join(project_path, "castle_config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path) as f:
                cfg = json.load(f)
            fps = cfg.get("fps", fps)
        except Exception:
            pass

    return save_cluster_model(
        output_path=output_path,
        umap_embedding=umap_embedding,
        training_features=training_features,
        cluster_labels=cluster_labels,
        cluster_names=cluster_names,
        model_name=model_name,
        fps=fps,
        k=k,
    )


def apply_cluster_model_to_project(
    model_path: str,
    project_path: str,
    method: str = "knn_feature",
) -> dict:
    """Apply a saved cluster model to a new project's latent features.

    Loads latent features from *project_path*, classifies them with the
    saved model, and writes ``transferred_labels.csv`` + ``id.csv`` into
    the project's ``cluster/`` directory.

    Args:
        model_path: Path to the saved model ``.npz``.
        project_path: Absolute path to the target project directory.
        method: ``"knn_feature"`` or ``"knn_umap"``.

    Returns:
        A dict with ``labels``, ``confidence``, ``cluster_names``,
        ``output_csv``, and ``n_frames``.
    """
    from castle.core.cluster_transfer import load_cluster_model, apply_cluster_model
    import glob

    model = load_cluster_model(model_path)

    # --- Load latent features from target project ---
    latent_dir = os.path.join(project_path, "latent")
    if not os.path.isdir(latent_dir):
        raise FileNotFoundError(f"No latent directory found: {latent_dir}")

    # Pick most-recently-modified model sub-directory (matches user's latest extraction).
    model_dirs = [
        os.path.join(latent_dir, d) for d in os.listdir(latent_dir)
        if os.path.isdir(os.path.join(latent_dir, d))
    ]
    if not model_dirs:
        raise FileNotFoundError(f"No model sub-directories in {latent_dir}")
    model_subdir = max(model_dirs, key=os.path.getmtime)

    latent_files = sorted(glob.glob(os.path.join(model_subdir, "*.npz")))
    if not latent_files:
        raise FileNotFoundError(f"No latent .npz files in {model_subdir}")

    latent_chunks = []
    for lf in latent_files:
        loaded = np.load(lf)
        latent_chunks.append(loaded["latent"])
    new_features = np.concatenate(latent_chunks, axis=0)

    # --- Apply ---
    result = apply_cluster_model(model, new_features, method=method)

    # --- Write results ---
    cluster_dir = os.path.join(project_path, "cluster")
    os.makedirs(cluster_dir, exist_ok=True)

    # id.csv (from model cluster names)
    id_rows = sorted(result["cluster_names"].items())
    id_df = pd.DataFrame(
        [{"Id": cid, "Name": cname, "Color": "grey"} for cid, cname in id_rows]
    )
    id_csv_path = os.path.join(cluster_dir, "id.csv")
    id_df.to_csv(id_csv_path, index=False)

    # transferred_labels.csv
    labels_df = pd.DataFrame({
        "behavior": result["labels"],
        "confidence": result["confidence"],
    })
    labels_csv_path = os.path.join(cluster_dir, "transferred_labels.csv")
    labels_df.to_csv(labels_csv_path, index=False)

    logger.info(
        "Applied cluster model to %s: %d frames, %d unique labels",
        project_path,
        len(result["labels"]),
        len(np.unique(result["labels"])),
    )

    return {
        "labels": result["labels"],
        "confidence": result["confidence"],
        "cluster_names": result["cluster_names"],
        "output_csv": labels_csv_path,
        "id_csv": id_csv_path,
        "n_frames": len(result["labels"]),
        "mean_confidence": float(result["confidence"].mean()) if len(result["confidence"]) else 0.0,
    }


# ---------------------------------------------------------------------------
# Pure algorithmic helpers used by both ClusteringSession + cluster_handlers
# (ARCH-01 / P4)
#
# These functions encapsulate "what the algorithm actually does" without any
# Gradio / PyQt coupling. They take the relevant Latent / LocalLatent /
# LatentAggregator objects explicitly and return structured results.
#
# Gradio handlers call them and translate exceptions into ``gr.Info`` /
# ``gr.Warning``; PyQt panels can call the same functions and translate
# into Qt signals. The CLI calls ``ClusteringSession`` (which in turn
# wraps these helpers).
# ---------------------------------------------------------------------------


@dataclass
class UMAPRunArtifacts:
    """Pure result of running UMAP on a single cluster.

    Attributes:
        local_latents: The freshly built :class:`LocalLatent` (caller stores
            this in Gradio state / PyQt model).
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
    local_latents = latents.select(selected_cluster=cluster_name)
    if len(local_latents.data) == 0:
        raise InsufficientDataError(
            f"Cluster '{cluster_name}' has no data points. Select a "
            f"different cluster or re-cluster with adjusted parameters."
        )

    resolved_seeds = local_latents.build_embedding(
        cfg,
        progress_callback=progress_callback,
        base_seed=base_seed,
        deterministic=deterministic,
        log_path=log_path,
    )

    mode_note = " (CPU, reproducible)" if deterministic else ""
    status_text = (
        f"✅ UMAP done{mode_note}. seed={resolved_seeds[0]}. "
        f"Paste `{resolved_seeds[0]}` into the seed box to reproduce."
    )
    return UMAPRunArtifacts(
        local_latents=local_latents,
        resolved_seeds=list(resolved_seeds),
        status_text=status_text,
    )


def run_dbscan_on_local(local_latents: Any, eps: float) -> None:
    """Run DBSCAN in place on an existing :class:`LocalLatent`.

    Args:
        local_latents: A LocalLatent with an embedding already built.
        eps: DBSCAN epsilon. The function mutates ``local_latents.cluster``.

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
    local_latents.build_cluster(method='dbscan', configs={'eps': eps})


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
    preset_value: Optional[str] = None,
    umap_seed: Optional[int] = None,
    overwrite: bool = False,
) -> SubmitArtifacts:
    """Merge local clusters into the global ``Latent`` and persist artefacts.

    Mirrors what the legacy handler ``import_info_from_local_latent`` did,
    minus the Gradio coupling. The caller wraps the result for Gradio
    ``outputs=...`` or PyQt slots.

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

    df1 = pd.DataFrame({
        'Id': [k for k in latents.cluster_meta],
        'Name': [v['name'] for v in latents.cluster_meta.values()],
        'Color': [v['color'] for v in latents.cluster_meta.values()],
    })
    id_csv_path = os.path.join(cluster_path, 'id.csv')
    df1.to_csv(id_csv_path, index=False)

    df2_paths: List[str] = []
    cum = 0
    for vn, v in aggregator.videos_meta:
        video_cluster = latents.cluster[cum:cum + vn]
        video_frames = np.repeat(video_cluster, latents.time_window)
        df2 = pd.DataFrame({'behavior': video_frames})
        video_basename = os.path.basename(v).split('.')[0]
        df2_path = os.path.join(cluster_path, f'time_series_{video_basename}.csv')
        df2.to_csv(df2_path, index=False)
        df2_paths.append(df2_path)
        cum += vn

    subtitle_paths = aggregator.generate_subtitles(
        latents.cluster, latents.cluster_meta,
    )

    embedding_path: Optional[str] = None
    if (local_latents is not None
            and hasattr(local_latents, 'embedding')
            and local_latents.embedding is not None):
        from castle.ui.embedding_scatter import EmbeddingScatterPlot

        Z_plt = EmbeddingScatterPlot(local_latents)
        cluster_name = ''
        for _, it in local_latents.export.items():
            cluster_name += it['name'] + '_'
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
class RestoredSessionArtifacts:
    """Pure result of restoring a clustering session from disk."""
    aggregator: Any
    latents: Any
    syllables_fig: Any
    cluster_choices: List[Tuple[str, int]]
    id_csv_path: str
    time_series_paths: List[str]
    local_latents: Optional[Any]
    embedding_array: Optional[np.ndarray]


def restore_session_from_disk(
    storage_path: str,
    project_name: str,
    *,
    select_roi_id: Any,
    bin_size: Any,
    select_model: str,
    session_id: Optional[str] = None,
    notify: Optional[Callable[[str, str], None]] = None,
) -> RestoredSessionArtifacts:
    """Restore a clustering session — Gradio-free version of ``_do_restore_session``.

    Replaces the dual responsibility of "build LatentAggregator + reload
    cluster_meta from id.csv + reload assignments from per-video CSVs +
    optionally restore UMAP embedding from npz" with a single typed call.

    Args:
        storage_path: Root storage directory.
        project_name: Project to restore.
        select_roi_id: ROI ID currently selected in the UI (may be
            overridden by the session's stored value).
        bin_size: Bin size from UI (may be overridden).
        select_model: Model name from UI (may be overridden).
        session_id: Explicit session to restore; if None, picks the most
            recently updated.
        notify: ``(msg, level)`` callback for LatentAggregator progress
            messages.

    Returns:
        :class:`RestoredSessionArtifacts` ready to map into Gradio state /
        PyQt model fields.
    """
    from castle.core.cluster import LatentAggregator
    from castle.service.plotting_service import plot_syllables_per_video
    from castle.ui.cluster_handlers import update_select_cluster_list

    mgr = SessionManager(storage_path, project_name)
    session_info = None
    if session_id:
        session_info = mgr.get_session(session_id)
        mgr.activate_session(session_id)
    else:
        sessions = mgr.list_sessions()
        if sessions:
            session_info = sessions[0]
            mgr.activate_session(sessions[0].session_id)

    if session_info:
        select_model = session_info.model or select_model
        select_roi_id = (str(session_info.roi_id)
                         if session_info.roi_id else select_roi_id)
        bin_size = session_info.bin_size if session_info.bin_size else bin_size

    aggregator = LatentAggregator(
        storage_path, project_name, select_roi_id, int(bin_size),
        model_name=select_model,
        notify=notify,
    )
    latents = aggregator.get_latent_object()

    cluster_path = os.path.join(storage_path, project_name, 'cluster')

    id_csv_path = os.path.join(cluster_path, 'id.csv')
    df2_paths: List[str] = []
    if os.path.exists(id_csv_path):
        id_df = pd.read_csv(id_csv_path)
        for _, row in id_df.iterrows():
            cluster_id = int(row['Id'])
            color = row.get('Color', 'grey')
            latents.cluster_meta[cluster_id] = {'name': row['Name'], 'color': color}
            latents.behavior_name2cluster_id[row['Name']] = cluster_id
            if color != 'grey':
                latents.used_palette.add(color)
        latents.num_cluster = len(id_df)

        cum = 0
        for vn, v in aggregator.videos_meta:
            video_basename = os.path.basename(v).split('.')[0]
            ts_path = os.path.join(cluster_path, f'time_series_{video_basename}.csv')
            if os.path.exists(ts_path):
                ts_df = pd.read_csv(ts_path)
                bin_clusters = ts_df['behavior'].values[::latents.time_window][:vn]
                if len(bin_clusters) != vn:
                    raise CastleDataError(
                        f"Session restore: {os.path.basename(ts_path)} downsamples "
                        f"to {len(bin_clusters)} bins but video {v!r} expects {vn}. "
                        f"The time_series CSV is likely truncated/corrupt. Assigning "
                        f"it would mislabel this and every subsequent video — refusing. "
                        f"Re-save the session or delete the corrupt CSV and re-cluster."
                    )
                latents.cluster[cum:cum + vn] = bin_clusters
                df2_paths.append(ts_path)
            cum += vn
    else:
        logger.info(
            "No id.csv at %s — restoring aggregator only; cluster_meta is empty.",
            id_csv_path,
        )

    restored_local_latents: Optional[Any] = None
    embedding_array: Optional[np.ndarray] = None

    npz_path = find_latest_cluster_npz(cluster_path)
    if npz_path:
        restored_local_latents, embedding_array = restore_local_latent_from_npz(
            npz_path, latents,
        )

    fig = plot_syllables_per_video(latents, aggregator)
    choices = update_select_cluster_list(latents)

    return RestoredSessionArtifacts(
        aggregator=aggregator,
        latents=latents,
        syllables_fig=fig,
        cluster_choices=choices,
        id_csv_path=id_csv_path,
        time_series_paths=df2_paths,
        local_latents=restored_local_latents,
        embedding_array=embedding_array,
    )


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

    Returns:
        :class:`InitAggregatorArtifacts` carrying the freshly built
        aggregator and its associated ``Latent`` object.
    """
    from castle.core.cluster import LatentAggregator

    # Clear cluster/ root before starting a new session so no files from a
    # previous session can pollute this one.  Do this before creating the
    # LatentAggregator so that any memmap cache from the old session is gone.
    mgr = SessionManager(storage_path, project_name)
    mgr._clear_cluster_root()

    aggregator = LatentAggregator(
        storage_path, project_name, select_roi_id, int(bin_size),
        model_name=select_model,
        notify=notify,
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
    )

    return InitAggregatorArtifacts(aggregator=aggregator, latents=latents)
