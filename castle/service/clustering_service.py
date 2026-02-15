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
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable, Any

from castle.core.cluster import LatentAggregator, auto_generate_cluster_name
from castle.utils.latent_explorer import Latent, LocalLatent

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
    
    def run_umap(self, cluster_name: str, umap_config: Any) -> dict:
        """
        Select a cluster and run UMAP dimensionality reduction.
        
        Args:
            cluster_name: Name of the cluster to focus on (e.g., 'init')
            umap_config: UMAP config — either a dict, list of dicts, or JSON string
        
        Returns:
            dict with keys:
                'n_points': int — number of points in the selected cluster
                'embedding_shape': tuple — shape of the embedding
                'success': bool
        """
        if isinstance(umap_config, str):
            umap_config = json.loads(umap_config)
        
        self._current_cluster_name = cluster_name
        self.local_latents = self.latents.select(selected_cluster=cluster_name)
        
        if len(self.local_latents.data) == 0:
            return {'n_points': 0, 'embedding_shape': (0, 0), 'success': False,
                    'error': 'Selected cluster is empty'}
        
        self.local_latents.build_embedding(umap_config)
        
        return {
            'n_points': len(self.local_latents.data),
            'embedding_shape': self.local_latents.embedding.shape,
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
            df2.to_csv(ts_path)
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
                bin_clusters = ts_df['behavior'].values[::self.aggregator.bin_size][:vn]
                self.latents.cluster[cum:cum + len(bin_clusters)] = bin_clusters
                ts_paths.append(ts_path)
            cum += vn
        
        return {
            'cluster_count': self.latents.num_cluster - 1,
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
