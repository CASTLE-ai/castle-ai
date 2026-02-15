"""
CASTLE Desktop - Extraction Panel (Tab 3)

Latent feature extraction using the service layer.
"""

import os

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QGroupBox, QCheckBox, QSpinBox,
    QProgressBar, QTextEdit, QMessageBox
)
from PyQt6.QtCore import pyqtSignal

from castle.service.project_service import get_project_info as svc_get_project_info
from castle.service.extraction_service import make_preprocess_config
from castle.desktop.services.worker_threads import ExtractionWorker


class ExtractPanel(QWidget):
    """Panel for latent feature extraction."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._storage_path = None
        self._project_name = None
        self._worker = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # === Model & ROI config ===
        config_group = QGroupBox("Extraction Configuration")
        cfg = QHBoxLayout(config_group)

        cfg.addWidget(QLabel("Model:"))
        self._model_combo = QComboBox()
        self._model_combo.addItems([
            "dinov2_vitb14_reg4_pretrain",
            "dinov3_vitb16",
            "dinov3_vitl16",
        ])
        cfg.addWidget(self._model_combo)

        cfg.addWidget(QLabel("ROI:"))
        self._roi_spin = QSpinBox()
        self._roi_spin.setMinimum(1)
        self._roi_spin.setMaximum(99)
        self._roi_spin.setValue(1)
        cfg.addWidget(self._roi_spin)

        cfg.addWidget(QLabel("Batch:"))
        self._batch_spin = QSpinBox()
        self._batch_spin.setMinimum(1)
        self._batch_spin.setMaximum(256)
        self._batch_spin.setValue(32)
        cfg.addWidget(self._batch_spin)

        self._skip_cb = QCheckBox("Skip existing")
        self._skip_cb.setChecked(True)
        cfg.addWidget(self._skip_cb)

        cfg.addStretch()
        layout.addWidget(config_group)

        # === Video selection ===
        video_group = QGroupBox("Target Video")
        video_layout = QHBoxLayout(video_group)

        video_layout.addWidget(QLabel("Video:"))
        self._video_combo = QComboBox()
        video_layout.addWidget(self._video_combo, stretch=1)

        self._count_label = QLabel("0 videos")
        video_layout.addWidget(self._count_label)

        layout.addWidget(video_group)

        # === Preprocessing ===
        preprocess_group = QGroupBox("Preprocessing Options")
        pp_layout = QVBoxLayout(preprocess_group)

        row1 = QHBoxLayout()
        self._center_cb = QCheckBox("Center ROI")
        row1.addWidget(self._center_cb)

        row1.addWidget(QLabel("Center ROI ID:"))
        self._center_id_spin = QSpinBox()
        self._center_id_spin.setMinimum(1)
        self._center_id_spin.setMaximum(99)
        self._center_id_spin.setValue(1)
        row1.addWidget(self._center_id_spin)

        row1.addWidget(QLabel("Crop W:"))
        self._crop_w_spin = QSpinBox()
        self._crop_w_spin.setMinimum(50)
        self._crop_w_spin.setMaximum(1000)
        self._crop_w_spin.setValue(300)
        row1.addWidget(self._crop_w_spin)

        row1.addWidget(QLabel("H:"))
        self._crop_h_spin = QSpinBox()
        self._crop_h_spin.setMinimum(50)
        self._crop_h_spin.setMaximum(1000)
        self._crop_h_spin.setValue(300)
        row1.addWidget(self._crop_h_spin)

        row1.addStretch()
        pp_layout.addLayout(row1)

        row2 = QHBoxLayout()
        self._rotate_cb = QCheckBox("Rotate (tail)")
        row2.addWidget(self._rotate_cb)

        row2.addWidget(QLabel("Tail ROI ID:"))
        self._tail_id_spin = QSpinBox()
        self._tail_id_spin.setMinimum(1)
        self._tail_id_spin.setMaximum(99)
        self._tail_id_spin.setValue(2)
        row2.addWidget(self._tail_id_spin)

        self._rmbg_cb = QCheckBox("Remove Background")
        row2.addWidget(self._rmbg_cb)

        row2.addStretch()
        pp_layout.addLayout(row2)

        layout.addWidget(preprocess_group)

        # === Action buttons ===
        btn_layout = QHBoxLayout()

        self._extract_btn = QPushButton("🧬 Extract Latent")
        self._extract_btn.setObjectName("primaryButton")
        self._extract_btn.clicked.connect(self._run_extraction)
        btn_layout.addWidget(self._extract_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # === Progress ===
        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)
        layout.addWidget(self._progress_bar)

        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setMaximumHeight(200)
        self._log_text.setPlaceholderText("Extraction log...")
        layout.addWidget(self._log_text)

        layout.addStretch()

    # --- Public API ---

    def set_project(self, storage_path: str, project_name: str):
        self._storage_path = storage_path
        self._project_name = project_name
        self._refresh_videos()

    def _refresh_videos(self):
        self._video_combo.clear()
        if not self._storage_path or not self._project_name:
            return
        try:
            info = svc_get_project_info(self._storage_path, self._project_name)
            videos = info.get('videos', [])
            self._video_combo.addItem("All")
            for v in videos:
                self._video_combo.addItem(v)
            self._count_label.setText(f"{len(videos)} videos")
        except Exception as e:
            self._log_text.append(f"Error: {e}")

    def _build_preprocess(self):
        return make_preprocess_config(
            center_roi_switch=self._center_cb.isChecked(),
            center_roi_id=self._center_id_spin.value(),
            center_roi_crop_width=self._crop_w_spin.value(),
            center_roi_crop_height=self._crop_h_spin.value(),
            rotate_roi_tail_switch=self._rotate_cb.isChecked(),
            rotate_roi_tail_id=self._tail_id_spin.value(),
            remove_background_switch=self._rmbg_cb.isChecked(),
        )

    def _run_extraction(self):
        if not self._storage_path or not self._project_name:
            QMessageBox.warning(self, "Error", "Open a project first.")
            return

        video_name = self._video_combo.currentText()
        if not video_name:
            QMessageBox.warning(self, "Error", "Select a video.")
            return

        self._extract_btn.setEnabled(False)
        self._progress_bar.setVisible(True)
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._log_text.append(f"Starting extraction: {video_name}...")

        preprocess = self._build_preprocess()

        self._worker = ExtractionWorker(
            self._storage_path, self._project_name, video_name,
            model=self._model_combo.currentText(),
            roi=self._roi_spin.value(),
            batch_size=self._batch_spin.value(),
            preprocess_config=preprocess,
            skip_existing=self._skip_cb.isChecked(),
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_progress(self, pct, msg):
        self._progress_bar.setValue(pct)
        if msg:
            self._log_text.append(f"  {msg}")

    def _on_finished(self, paths_str):
        self._progress_bar.setVisible(False)
        self._extract_btn.setEnabled(True)
        if paths_str:
            for p in paths_str.split(';'):
                self._log_text.append(f"✅ Saved: {os.path.basename(p)}")
        else:
            self._log_text.append("✅ Extraction complete (no new files).")

    def _on_error(self, err):
        self._progress_bar.setVisible(False)
        self._extract_btn.setEnabled(True)
        self._log_text.append(f"❌ Error: {err}")
        QMessageBox.critical(self, "Extraction Error", err)
