"""
CASTLE Desktop - Tracking Panel (Tab 2)

ROI tracking and stabilized camera preprocessing using the service layer.
"""


from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QGroupBox, QListWidget, QProgressBar,
    QCheckBox, QTextEdit, QMessageBox, QTabWidget,
    QSpinBox, QDoubleSpinBox, QFormLayout,
)
from PyQt6.QtCore import QThread, pyqtSignal

from castle.service.project_service import get_project_info as svc_get_project_info
from castle.service.tracking_service import get_tracking_status as svc_tracking_status
from castle.desktop.services.worker_threads import TrackingWorker


class _PreprocessWorker(QThread):
    """Background worker for stabilized camera preprocessing."""

    progress = pyqtSignal(int, str)    # (percent 0-100, message)
    finished = pyqtSignal(dict)        # result dict
    error = pyqtSignal(str)            # error message

    def __init__(
        self,
        storage_path: str,
        project_name: str,
        video_name: str,
        body_roi_id: int,
        head_roi_id: int,
        fc: float = 0.25,
        order: int = 2,
        margin: int = 75,
        min_crop: int = 300,
        output_size: int = 518,
        preview_duration: float = 10.0,
    ) -> None:
        super().__init__()
        self._storage = storage_path
        self._project = project_name
        self._video = video_name
        self._body_roi = body_roi_id
        self._head_roi = head_roi_id
        self._fc = fc
        self._order = order
        self._margin = margin
        self._min_crop = min_crop
        self._output_size = output_size
        self._preview_duration = preview_duration

    def run(self) -> None:  # noqa: D102
        try:
            from castle.service.preprocessing_service import preprocess_stabilized_camera

            def _cb(fraction: float, description: str = "") -> None:
                pct = max(0, min(100, int(fraction * 100)))
                self.progress.emit(pct, description or "Processing…")

            result = preprocess_stabilized_camera(
                storage_path=self._storage,
                project_name=self._project,
                video_name=self._video,
                body_roi_id=self._body_roi,
                head_roi_id=self._head_roi,
                fc=self._fc,
                order=self._order,
                margin=self._margin,
                min_crop=self._min_crop,
                output_size=self._output_size,
                preview_duration=self._preview_duration,
                progress_callback=_cb,
            )
            self.finished.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))


class _PreprocessPanel(QWidget):
    """Stabilized camera preprocessing sub-panel inside TrackingPanel."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._storage_path: str | None = None
        self._project_name: str | None = None
        self._workers: list[QThread] = []
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # ---- ROI / Video selection ----
        roi_group = QGroupBox("Input")
        roi_form = QFormLayout(roi_group)

        self._video_combo = QComboBox()
        roi_form.addRow("Video:", self._video_combo)

        self._body_roi_spin = QSpinBox()
        self._body_roi_spin.setMinimum(1)
        self._body_roi_spin.setValue(1)
        roi_form.addRow("Body ROI ID:", self._body_roi_spin)

        self._head_roi_spin = QSpinBox()
        self._head_roi_spin.setMinimum(1)
        self._head_roi_spin.setValue(2)
        roi_form.addRow("Head ROI ID:", self._head_roi_spin)

        layout.addWidget(roi_group)

        # ---- Advanced parameters ----
        adv_group = QGroupBox("Filter / Crop Parameters")
        adv_form = QFormLayout(adv_group)

        self._fc_spin = QDoubleSpinBox()
        self._fc_spin.setDecimals(3)
        self._fc_spin.setMinimum(0.001)
        self._fc_spin.setMaximum(50.0)
        self._fc_spin.setValue(0.25)
        self._fc_spin.setSuffix(" Hz")
        self._fc_spin.setToolTip(
            "Butterworth low-pass filter cutoff frequency. Default: 0.25 Hz.\n"
            "Lower values = smoother camera movement (more filtering).\n"
            "Higher values = preserves more rapid motion."
        )
        adv_form.addRow("Low-pass cutoff (fc):", self._fc_spin)

        self._order_spin = QSpinBox()
        self._order_spin.setMinimum(1)
        self._order_spin.setMaximum(10)
        self._order_spin.setValue(2)
        self._order_spin.setToolTip(
            "Butterworth filter order. Default: 2.\n"
            "Higher orders give a sharper cutoff but may introduce ringing artifacts."
        )
        adv_form.addRow("Filter order:", self._order_spin)

        self._margin_spin = QSpinBox()
        self._margin_spin.setMinimum(0)
        self._margin_spin.setMaximum(2000)
        self._margin_spin.setValue(75)
        self._margin_spin.setSuffix(" px")
        self._margin_spin.setToolTip(
            "Extra padding pixels added around the crop region. Default: 75 px.\n"
            "Increase if the animal moves to the edge of the visible area."
        )
        adv_form.addRow("Crop margin:", self._margin_spin)

        self._min_crop_spin = QSpinBox()
        self._min_crop_spin.setMinimum(64)
        self._min_crop_spin.setMaximum(4096)
        self._min_crop_spin.setValue(300)
        self._min_crop_spin.setSuffix(" px")
        self._min_crop_spin.setToolTip(
            "Minimum crop size in pixels. Default: 300 px.\n"
            "Prevents the crop from becoming too small when the animal is stationary."
        )
        adv_form.addRow("Min crop size:", self._min_crop_spin)

        self._output_size_spin = QSpinBox()
        self._output_size_spin.setMinimum(64)
        self._output_size_spin.setMaximum(4096)
        self._output_size_spin.setValue(518)
        self._output_size_spin.setSuffix(" px")
        self._output_size_spin.setToolTip(
            "Output frame side length in pixels. Default: 518 px\n"
            "(optimal for DINOv2 ViT-B/14). Use 224 for smaller/faster models."
        )
        adv_form.addRow("Output frame size:", self._output_size_spin)

        self._preview_dur_spin = QDoubleSpinBox()
        self._preview_dur_spin.setMinimum(1.0)
        self._preview_dur_spin.setMaximum(3600.0)
        self._preview_dur_spin.setValue(10.0)
        self._preview_dur_spin.setSuffix(" s")
        adv_form.addRow("Preview duration:", self._preview_dur_spin)

        layout.addWidget(adv_group)

        # ---- Run button ----
        btn_row = QHBoxLayout()
        self._run_btn = QPushButton("▶ Run Stabilized Camera")
        self._run_btn.setObjectName("primaryButton")
        self._run_btn.clicked.connect(self._run)
        btn_row.addWidget(self._run_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        # ---- Progress & log ----
        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)
        layout.addWidget(self._progress_bar)

        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setPlaceholderText("Preprocessing log…")
        layout.addWidget(self._log_text, stretch=1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_project(self, storage_path: str, project_name: str) -> None:
        """Update project context and refresh the video list."""
        self._storage_path = storage_path
        self._project_name = project_name
        self._refresh_videos()

    def _refresh_videos(self) -> None:
        self._video_combo.clear()
        if not self._storage_path or not self._project_name:
            return
        try:
            info = svc_get_project_info(self._storage_path, self._project_name)
            for vname in info.get("videos", []):
                self._video_combo.addItem(vname)
        except Exception as exc:
            self._log_text.append(f"Error loading videos: {exc}")

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _run(self) -> None:
        if not self._storage_path or not self._project_name:
            QMessageBox.warning(
                self,
                "No Project Loaded",
                "No project loaded. Please open a project from the Project panel first.",
            )
            return
        video_name = self._video_combo.currentText()
        if not video_name:
            QMessageBox.warning(
                self,
                "No Video Selected",
                "No video selected. Please select a video from the list.",
            )
            return
        body_roi = self._body_roi_spin.value()
        head_roi = self._head_roi_spin.value()
        if body_roi == head_roi:
            QMessageBox.warning(
                self,
                "Invalid ROI IDs",
                "Body ROI ID and Head ROI ID cannot be the same. "
                "Please enter a different value for each.",
            )
            return

        self._progress_bar.setVisible(True)
        self._progress_bar.setValue(0)
        self._run_btn.setEnabled(False)
        self._log_text.append(f"Starting preprocessing for: {video_name}")

        worker = _PreprocessWorker(
            storage_path=self._storage_path,
            project_name=self._project_name,
            video_name=video_name,
            body_roi_id=body_roi,
            head_roi_id=head_roi,
            fc=self._fc_spin.value(),
            order=self._order_spin.value(),
            margin=self._margin_spin.value(),
            min_crop=self._min_crop_spin.value(),
            output_size=self._output_size_spin.value(),
            preview_duration=self._preview_dur_spin.value(),
        )
        worker.progress.connect(self._on_progress)
        worker.finished.connect(self._on_done)
        worker.error.connect(self._on_error)
        self._workers.append(worker)
        worker.start()

    def _on_progress(self, pct: int, msg: str) -> None:
        self._progress_bar.setValue(pct)
        if msg:
            self._log_text.append(f"  [{pct}%] {msg}")

    def _on_done(self, result: dict) -> None:
        self._progress_bar.setValue(100)
        self._progress_bar.setVisible(False)
        self._run_btn.setEnabled(True)
        diag = result.get("diagnostics", {})
        self._log_text.append("✅ Preprocessing complete!")
        self._log_text.append(f"  Video   : {result.get('preprocessed_video_path', '')}")
        self._log_text.append(f"  Preview : {result.get('preview_path', '')}")
        self._log_text.append(f"  Frames  : {result.get('n_frames', 0)}")
        self._log_text.append(
            f"  HP residual RMS : {diag.get('hp_residual_rms', float('nan')):.2f} px  |  "
            f"% at min_crop : {diag.get('pct_at_min_crop', float('nan')):.1f}%  |  "
            f"speed-crop r : {diag.get('speed_crop_correlation', float('nan')):.3f}"
        )

    def _on_error(self, err: str) -> None:
        self._progress_bar.setVisible(False)
        self._run_btn.setEnabled(True)
        self._log_text.append(f"❌ Error: {err}")
        QMessageBox.critical(
            self,
            "Preprocessing Error",
            f"Preprocessing failed.\n\n{err}\n\n"
            "Verify that ROI tracking has been completed for this video (Step 2) "
            "and that the ROI IDs are correct.",
        )


class TrackingPanel(QWidget):
    """Panel for ROI tracking and preprocessing (Tab 2)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._storage_path = None
        self._project_name = None
        self._workers = []   # keep references to running workers
        self._setup_ui()

    def _setup_ui(self):
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        self._sub_tabs = QTabWidget()
        self._sub_tabs.setDocumentMode(True)

        # ---- Tracking sub-tab ----
        tracking_widget = QWidget()
        layout = QVBoxLayout(tracking_widget)
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

        self._sub_tabs.addTab(tracking_widget, "ROI Tracking")

        # ---- Preprocessing sub-tab ----
        self._preprocess_panel = _PreprocessPanel(self)
        self._sub_tabs.addTab(self._preprocess_panel, "Preprocessing")

        outer_layout.addWidget(self._sub_tabs)

    # --- Public API ---

    def set_project(self, storage_path: str, project_name: str):
        self._storage_path = storage_path
        self._project_name = project_name
        self._refresh()
        self._preprocess_panel.set_project(storage_path, project_name)

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
            QMessageBox.warning(
                self,
                "No Video Selected",
                "No video selected. Please select a video from the tracking list.",
            )
            return
        self._run_tracking([vname])

    def _track_all(self):
        if not self._storage_path or not self._project_name:
            return
        info = svc_get_project_info(self._storage_path, self._project_name)
        videos = info.get('videos', [])
        if not videos:
            QMessageBox.warning(
                self,
                "No Videos",
                "No videos found in this project. "
                "Please add videos in the Source panel first.",
            )
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
