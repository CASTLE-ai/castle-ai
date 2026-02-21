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
        self._roi_spin.setToolTip(
            "ROI (Region of Interest) ID for feature extraction.\n"
            "Default: 1 (the animal body). Must match the ROI tracked in Step 2."
        )
        cfg.addWidget(self._roi_spin)

        cfg.addWidget(QLabel("Batch:"))
        self._batch_spin = QSpinBox()
        self._batch_spin.setMinimum(1)
        self._batch_spin.setMaximum(256)
        self._batch_spin.setValue(32)
        self._batch_spin.setToolTip(
            "Frames processed simultaneously. Default: 32.\n"
            "Larger values are faster but use more GPU memory.\n"
            "Reduce to 8 or 16 if you get out-of-memory errors."
        )
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
        self._center_id_spin.setToolTip(
            "ROI ID used to center the crop region. Default: 1 (body centroid)."
        )
        row1.addWidget(self._center_id_spin)

        row1.addWidget(QLabel("Crop W:"))
        self._crop_w_spin = QSpinBox()
        self._crop_w_spin.setMinimum(50)
        self._crop_w_spin.setMaximum(1000)
        self._crop_w_spin.setValue(300)
        self._crop_w_spin.setToolTip(
            "Crop region width in pixels. Default: 300.\n"
            "Larger values include more context but increase processing time."
        )
        row1.addWidget(self._crop_w_spin)

        row1.addWidget(QLabel("H:"))
        self._crop_h_spin = QSpinBox()
        self._crop_h_spin.setMinimum(50)
        self._crop_h_spin.setMaximum(1000)
        self._crop_h_spin.setValue(300)
        self._crop_h_spin.setToolTip(
            "Crop region height in pixels. Default: 300.\n"
            "Keep width and height equal for square crops."
        )
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
        self._tail_id_spin.setToolTip(
            "ROI ID for the tail/reference point used to compute body orientation.\n"
            "Default: 2. Requires the tail to be tracked in Step 2."
        )
        row2.addWidget(self._tail_id_spin)

        self._rmbg_cb = QCheckBox("Remove Background")
        row2.addWidget(self._rmbg_cb)

        row2.addStretch()
        pp_layout.addLayout(row2)

        layout.addWidget(preprocess_group)

        # === Advanced Extraction Options (A-06) ===
        adv_group = QGroupBox("Advanced Extraction Options (A-06)")
        adv_layout = QVBoxLayout(adv_group)

        adv_row1 = QHBoxLayout()
        adv_row1.addWidget(QLabel("Pooling:"))
        self._pooling_combo = QComboBox()
        self._pooling_combo.addItems(["weighted_average", "multiscale"])
        adv_row1.addWidget(self._pooling_combo)
        adv_row1.addStretch()
        adv_layout.addLayout(adv_row1)

        adv_row2 = QHBoxLayout()
        adv_row2.addWidget(QLabel("Scales:"))
        self._scale_1_cb = QCheckBox("1 (global)")
        self._scale_1_cb.setChecked(True)
        adv_row2.addWidget(self._scale_1_cb)
        self._scale_2_cb = QCheckBox("2 (2×2)")
        self._scale_2_cb.setChecked(True)
        adv_row2.addWidget(self._scale_2_cb)
        self._scale_4_cb = QCheckBox("4 (4×4)")
        self._scale_4_cb.setChecked(True)
        adv_row2.addWidget(self._scale_4_cb)
        self._scale_8_cb = QCheckBox("8 (8×8)")
        adv_row2.addWidget(self._scale_8_cb)
        adv_row2.addStretch()
        adv_layout.addLayout(adv_row2)

        adv_row3 = QHBoxLayout()
        adv_row3.addWidget(QLabel("Feature Layers:"))
        from PyQt6.QtWidgets import QLineEdit
        self._layers_edit = QLineEdit()
        self._layers_edit.setPlaceholderText("e.g. 3,7,11 (empty=last only)")
        adv_row3.addWidget(self._layers_edit)
        adv_row3.addStretch()
        adv_layout.addLayout(adv_row3)

        layout.addWidget(adv_group)

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
            QMessageBox.warning(
                self,
                "No Project Open",
                "No project open. Please open or create a project from the Project panel first.",
            )
            return

        video_name = self._video_combo.currentText()
        if not video_name:
            QMessageBox.warning(
                self,
                "No Video Selected",
                "No video selected. Please select a target video from the dropdown.",
            )
            return

        self._extract_btn.setEnabled(False)
        self._progress_bar.setVisible(True)
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._log_text.append(f"Starting extraction: {video_name}...")

        preprocess = self._build_preprocess()

        # A-06: Collect advanced extraction options
        pooling_method = self._pooling_combo.currentText()
        pooling_scales = []
        if self._scale_1_cb.isChecked():
            pooling_scales.append(1)
        if self._scale_2_cb.isChecked():
            pooling_scales.append(2)
        if self._scale_4_cb.isChecked():
            pooling_scales.append(4)
        if self._scale_8_cb.isChecked():
            pooling_scales.append(8)
        feature_layers = None
        layers_text = self._layers_edit.text().strip()
        if layers_text:
            try:
                feature_layers = [int(x.strip()) for x in layers_text.split(',') if x.strip()]
            except ValueError:
                QMessageBox.warning(
                    self,
                    "Invalid Layer Format",
                    f"Invalid feature layer format: '{layers_text}'.\n"
                    "Please use comma-separated integers, e.g. '3,7,11'.",
                )
                return

        self._worker = ExtractionWorker(
            self._storage_path, self._project_name, video_name,
            model=self._model_combo.currentText(),
            roi=self._roi_spin.value(),
            batch_size=self._batch_spin.value(),
            preprocess_config=preprocess,
            skip_existing=self._skip_cb.isChecked(),
            pooling_method=pooling_method,
            pooling_scales=pooling_scales if pooling_method == 'multiscale' else None,
            feature_layers=feature_layers,
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
        QMessageBox.critical(
            self,
            "Extraction Error",
            f"Feature extraction failed.\n\n{err}\n\n"
            "Check that ROI tracking is complete and that you have "
            "sufficient GPU memory (try reducing Batch size).",
        )
