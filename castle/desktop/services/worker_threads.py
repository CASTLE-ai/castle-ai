"""
CASTLE Desktop - Background Worker Threads

Generic QThread-based worker for running service layer functions
without freezing the UI.
"""

from PyQt6.QtCore import QThread, pyqtSignal


class ServiceWorker(QThread):
    """Generic background thread for any callable.

    Usage:
        worker = ServiceWorker(some_service_fn, arg1, arg2, key=val)
        worker.finished.connect(on_done)
        worker.error.connect(on_error)
        worker.start()
    """

    progress = pyqtSignal(int, str)    # (percent 0-100, message)
    finished = pyqtSignal(object)      # result of fn()
    error = pyqtSignal(str)            # error message

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class TrackingWorker(QThread):
    """Worker thread for tracking a single video."""

    progress = pyqtSignal(int, str)
    finished = pyqtSignal(str)       # status string
    error = pyqtSignal(str)

    def __init__(self, storage_path, project_name, video_name, model,
                 skip_existing=True):
        super().__init__()
        self._storage = storage_path
        self._project = project_name
        self._video = video_name
        self._model = model
        self._skip = skip_existing

    def run(self):
        try:
            from castle.service.tracking_service import track_video
            result = track_video(
                self._storage, self._project, self._video,
                model=self._model,
                skip_existing=self._skip,
            )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class ExtractionWorker(QThread):
    """Worker thread for latent extraction."""

    progress = pyqtSignal(int, str)
    finished = pyqtSignal(str)       # semicolon-separated paths
    error = pyqtSignal(str)

    def __init__(self, storage_path, project_name, video_name, model,
                 roi, batch_size=32, preprocess_config=None,
                 skip_existing=True):
        super().__init__()
        self._storage = storage_path
        self._project = project_name
        self._video = video_name
        self._model = model
        self._roi = roi
        self._batch = batch_size
        self._preprocess = preprocess_config
        self._skip = skip_existing

    def run(self):
        try:
            from castle.service.extraction_service import extract_latent

            def _progress(frac, desc=None):
                pct = int(frac * 100) if frac <= 1.0 else int(frac)
                self.progress.emit(pct, desc or "")

            result = extract_latent(
                self._storage, self._project, self._video,
                model=self._model,
                roi=self._roi,
                batch_size=self._batch,
                preprocess_config=self._preprocess,
                skip_existing=self._skip,
                progress_callback=_progress,
            )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class ClusteringSessionWorker(QThread):
    """Worker thread for initializing a ClusteringSession."""

    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object)      # ClusteringSession instance
    error = pyqtSignal(str)

    def __init__(self, storage_path, project_name, roi, bin_size, model):
        super().__init__()
        self._storage = storage_path
        self._project = project_name
        self._roi = roi
        self._bin = bin_size
        self._model = model

    def run(self):
        try:
            from castle.service.clustering_service import ClusteringSession
            session = ClusteringSession(
                self._storage, self._project,
                roi=self._roi, bin_size=self._bin, model=self._model,
            )
            self.finished.emit(session)
        except Exception as e:
            self.error.emit(str(e))


class UMAPWorker(QThread):
    """Worker thread for UMAP computation."""

    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object)    # result dict from session.run_umap()
    error = pyqtSignal(str)

    def __init__(self, session, cluster_name, umap_config):
        super().__init__()
        self._session = session
        self._cluster_name = cluster_name
        self._config = umap_config

    def run(self):
        try:
            result = self._session.run_umap(self._cluster_name, self._config)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))
