"""
CASTLE Desktop - Tracking Panel (Tab 2)

ROI tracking using the service layer.
"""


from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QGroupBox, QListWidget, QProgressBar,
    QCheckBox, QTextEdit, QMessageBox
)

from castle.service.project_service import get_project_info as svc_get_project_info
from castle.service.tracking_service import get_tracking_status as svc_tracking_status
from castle.desktop.services.worker_threads import TrackingWorker


class TrackingPanel(QWidget):
    """Panel for ROI tracking."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._storage_path = None
        self._project_name = None
        self._workers = []   # keep references to running workers
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # --- Config ---
        config_group = QGroupBox("Tracking Configuration")
        cfg_layout = QHBoxLayout(config_group)

        cfg_layout.addWidget(QLabel("Model:"))
        self._model_combo = QComboBox()
        self._model_combo.addItems(["r50_deaotl", "swinb_deaotl"])
        cfg_layout.addWidget(self._model_combo)

        self._skip_cb = QCheckBox("Skip existing")
        self._skip_cb.setChecked(True)
        cfg_layout.addWidget(self._skip_cb)

        cfg_layout.addStretch()

        self._track_btn = QPushButton("🎯 Track Selected Video")
        self._track_btn.setObjectName("primaryButton")
        self._track_btn.clicked.connect(self._track_selected)
        cfg_layout.addWidget(self._track_btn)

        self._track_all_btn = QPushButton("Track All Videos")
        self._track_all_btn.clicked.connect(self._track_all)
        cfg_layout.addWidget(self._track_all_btn)

        layout.addWidget(config_group)

        # --- Video list with tracking status ---
        list_group = QGroupBox("Videos & Tracking Status")
        list_layout = QVBoxLayout(list_group)

        self._video_list = QListWidget()
        self._video_list.setAlternatingRowColors(True)
        list_layout.addWidget(self._video_list)

        self._refresh_btn = QPushButton("Refresh Status")
        self._refresh_btn.clicked.connect(self._refresh)
        list_layout.addWidget(self._refresh_btn)

        layout.addWidget(list_group, stretch=1)

        # --- Progress ---
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)

        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)
        progress_layout.addWidget(self._progress_bar)

        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setMaximumHeight(150)
        self._log_text.setPlaceholderText("Tracking log...")
        progress_layout.addWidget(self._log_text)

        layout.addWidget(progress_group)

    # --- Public API ---

    def set_project(self, storage_path: str, project_name: str):
        self._storage_path = storage_path
        self._project_name = project_name
        self._refresh()

    def _refresh(self):
        self._video_list.clear()
        if not self._storage_path or not self._project_name:
            return
        try:
            info = svc_get_project_info(self._storage_path, self._project_name)
            for vname in info.get('videos', []):
                status = svc_tracking_status(
                    self._storage_path, self._project_name, vname
                )
                if status['tracked']:
                    label = f"✅ {vname}  ({status['n_rois']} ROIs, {status['n_frames']} frames)"
                else:
                    label = f"⬜ {vname}  (not tracked)"
                self._video_list.addItem(label)
        except Exception as e:
            self._log_text.append(f"Error refreshing: {e}")

    def _get_selected_video_name(self) -> str:
        """Extract video name from list item text."""
        item = self._video_list.currentItem()
        if not item:
            return ""
        text = item.text()
        # Strip status prefix
        for prefix in ("✅ ", "⬜ "):
            if text.startswith(prefix):
                text = text[len(prefix):]
                break
        # Take everything before the first parenthesis
        return text.split("(")[0].strip()

    def _track_selected(self):
        vname = self._get_selected_video_name()
        if not vname:
            QMessageBox.warning(self, "Error", "Select a video first.")
            return
        self._run_tracking([vname])

    def _track_all(self):
        if not self._storage_path or not self._project_name:
            return
        info = svc_get_project_info(self._storage_path, self._project_name)
        videos = info.get('videos', [])
        if not videos:
            QMessageBox.warning(self, "Error", "No videos in project.")
            return
        self._run_tracking(videos)

    def _run_tracking(self, video_names: list):
        self._progress_bar.setVisible(True)
        self._progress_bar.setRange(0, len(video_names))
        self._progress_bar.setValue(0)
        self._track_btn.setEnabled(False)
        self._track_all_btn.setEnabled(False)
        self._pending_videos = list(video_names)
        self._completed_count = 0
        self._track_next()

    def _track_next(self):
        if not self._pending_videos:
            self._on_all_done()
            return

        vname = self._pending_videos.pop(0)
        self._log_text.append(f"Tracking: {vname}...")

        worker = TrackingWorker(
            self._storage_path, self._project_name, vname,
            model=self._model_combo.currentText(),
            skip_existing=self._skip_cb.isChecked(),
        )
        worker.finished.connect(lambda result: self._on_video_done(vname, result))
        worker.error.connect(lambda err: self._on_video_error(vname, err))
        self._workers.append(worker)
        worker.start()

    def _on_video_done(self, vname, result):
        self._completed_count += 1
        self._progress_bar.setValue(self._completed_count)
        self._log_text.append(f"  {vname}: {result}")
        self._track_next()

    def _on_video_error(self, vname, err):
        self._completed_count += 1
        self._progress_bar.setValue(self._completed_count)
        self._log_text.append(f"  ❌ {vname}: {err}")
        self._track_next()

    def _on_all_done(self):
        self._progress_bar.setVisible(False)
        self._track_btn.setEnabled(True)
        self._track_all_btn.setEnabled(True)
        self._log_text.append("✅ Tracking complete.")
        self._refresh()
