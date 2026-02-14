"""
CASTLE Desktop - Background Worker Threads

QThread-based workers for long-running computations:
- UMAP embedding generation
- DBSCAN clustering
- Latent extraction

These prevent the UI from freezing during computation.
"""

from PySide6.QtCore import QThread, Signal


class UMAPWorker(QThread):
    """Background thread for UMAP computation."""
    
    finished = Signal(object)  # Emits LocalLatent
    progress = Signal(str)     # Progress message
    error = Signal(str)        # Error message
    
    def __init__(self, latents, cluster_name, config):
        super().__init__()
        self._latents = latents
        self._cluster_name = cluster_name
        self._config = config
    
    def run(self):
        try:
            self.progress.emit(f"Selecting cluster '{self._cluster_name}'...")
            local_latents = self._latents.select(self._cluster_name)
            
            if len(local_latents.data) == 0:
                self.error.emit("Selected cluster is empty.")
                return
            
            self.progress.emit("Running UMAP embedding...")
            local_latents.build_embedding(self._config)
            
            self.finished.emit(local_latents)
            
        except Exception as e:
            self.error.emit(str(e))


class ClusterWorker(QThread):
    """Background thread for DBSCAN clustering."""
    
    finished = Signal(object)  # Emits LocalLatent with cluster
    error = Signal(str)
    
    def __init__(self, local_latents, method, config):
        super().__init__()
        self._local_latents = local_latents
        self._method = method
        self._config = config
    
    def run(self):
        try:
            self._local_latents.build_cluster(
                method=self._method,
                configs=self._config
            )
            self.finished.emit(self._local_latents)
        except Exception as e:
            self.error.emit(str(e))


class ExtractionWorker(QThread):
    """Background thread for latent extraction."""
    
    finished = Signal(str)     # Emits output path
    progress = Signal(float, str)  # (fraction, description)
    error = Signal(str)
    
    def __init__(self, storage_path, project_name, video_name, roi_id,
                 model_name, batch_size, preprocess_config, skip_existing):
        super().__init__()
        self._storage_path = storage_path
        self._project_name = project_name
        self._video_name = video_name
        self._roi_id = roi_id
        self._model_name = model_name
        self._batch_size = batch_size
        self._preprocess_config = preprocess_config
        self._skip_existing = skip_existing
    
    def run(self):
        try:
            from castle.core.extractor import extract_roi_latent_from_video
            
            def progress_callback(p, desc=None):
                self.progress.emit(p, desc or "")
            
            path = extract_roi_latent_from_video(
                storage_path=self._storage_path,
                project_name=self._project_name,
                video_name=self._video_name,
                roi_id=self._roi_id,
                model_name=self._model_name,
                batch_size=self._batch_size,
                preprocess_config=self._preprocess_config,
                skip_existing=self._skip_existing,
                progress_callback=progress_callback
            )
            
            self.finished.emit(path or "")
            
        except Exception as e:
            self.error.emit(str(e))
