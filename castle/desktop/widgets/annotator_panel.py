"""
CASTLE Desktop - Annotator Panel (sub-tab inside 4. Behavior Microscope)

Cluster annotation with behavior labeling, classification scheme management,
optional comment field, and a grid video preview of representative bouts.
Mirrors the Gradio annotator_ui tab.
"""

import datetime
import logging
import os

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QComboBox, QLineEdit, QListWidget, QListWidgetItem,
    QTextEdit, QProgressBar, QMessageBox, QSplitter, QAbstractItemView,
    QSpinBox, QSlider, QSizePolicy,
)
from PyQt6.QtCore import Qt, pyqtSlot, QUrl
from PyQt6.QtGui import QDesktopServices

from castle.desktop.services.worker_threads import ServiceWorker

logger = logging.getLogger(__name__)


class AnnotatorPanel(QWidget):
    """Cluster annotation panel — load cluster data and assign behavior labels."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self._storage_path: str | None = None
        self._project_name: str | None = None
        self._annotator_data = None   # AnnotatorData instance
        self._annotations: dict = {}  # {cluster_name: annotation_dict}
        self._session_id: str | None = None
        self._worker: ServiceWorker | None = None
        self._grid_worker: ServiceWorker | None = None
        self._grid_video_path: str | None = None

        self._setup_ui()

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Top: session selector + load button
        layout.addWidget(self._create_load_group())

        # Status label
        self._status_label = QLabel("Status: Not loaded")
        layout.addWidget(self._status_label)

        # Main content splitter: left = cluster list, right = annotation + preview
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._create_cluster_group())

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self._create_annotation_group())
        right_layout.addWidget(self._create_grid_video_group())
        splitter.addWidget(right_widget)

        splitter.setSizes([350, 700])
        layout.addWidget(splitter, stretch=1)

        # Progress bar
        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)
        layout.addWidget(self._progress_bar)

    def _create_load_group(self) -> QGroupBox:
        group = QGroupBox("Session")
        layout = QHBoxLayout(group)

        layout.addWidget(QLabel("Session:"))
        self._session_combo = QComboBox()
        self._session_combo.setMinimumWidth(300)
        layout.addWidget(self._session_combo, stretch=1)

        self._refresh_sessions_btn = QPushButton("🔄 Refresh")
        self._refresh_sessions_btn.clicked.connect(self._refresh_sessions)
        layout.addWidget(self._refresh_sessions_btn)

        self._load_btn = QPushButton("📂 Load Cluster Data")
        self._load_btn.setObjectName("primaryButton")
        self._load_btn.clicked.connect(self._load_cluster_data)
        layout.addWidget(self._load_btn)

        return group

    def _create_cluster_group(self) -> QGroupBox:
        group = QGroupBox("Clusters")
        layout = QVBoxLayout(group)

        self._cluster_list = QListWidget()
        self._cluster_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._cluster_list.currentItemChanged.connect(self._on_cluster_selected)
        layout.addWidget(self._cluster_list)

        self._cluster_info_label = QLabel("Selected: None")
        self._cluster_info_label.setWordWrap(True)
        layout.addWidget(self._cluster_info_label)

        return group

    def _create_annotation_group(self) -> QGroupBox:
        group = QGroupBox("Annotation")
        layout = QVBoxLayout(group)

        # Classification scheme
        scheme_row = QHBoxLayout()
        scheme_row.addWidget(QLabel("Scheme:"))
        self._scheme_combo = QComboBox()
        self._scheme_combo.currentTextChanged.connect(self._on_scheme_changed)
        scheme_row.addWidget(self._scheme_combo, stretch=1)
        layout.addLayout(scheme_row)

        # Behavior label
        layout.addWidget(QLabel("Behavior Label:"))
        self._label_combo = QComboBox()
        layout.addWidget(self._label_combo)

        # Comment
        layout.addWidget(QLabel("Comment (optional):"))
        self._comment_edit = QLineEdit()
        self._comment_edit.setPlaceholderText("e.g. mostly grooming with some head movement")
        layout.addWidget(self._comment_edit)

        # Save button
        self._save_btn = QPushButton("💾 Save Annotation")
        self._save_btn.setObjectName("primaryButton")
        self._save_btn.clicked.connect(self._save_annotation)
        self._save_btn.setEnabled(False)
        layout.addWidget(self._save_btn)

        layout.addStretch()

        # Custom scheme section
        custom_group = QGroupBox("Custom Scheme")
        custom_layout = QVBoxLayout(custom_group)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Name:"))
        self._custom_scheme_name = QLineEdit()
        self._custom_scheme_name.setPlaceholderText("my-custom-scheme")
        name_row.addWidget(self._custom_scheme_name)
        custom_layout.addLayout(name_row)

        custom_layout.addWidget(QLabel("Labels (one per line):"))
        self._custom_labels_edit = QTextEdit()
        self._custom_labels_edit.setMaximumHeight(100)
        self._custom_labels_edit.setPlaceholderText("Running\nWalking\nImmobile\n...")
        custom_layout.addWidget(self._custom_labels_edit)

        self._save_scheme_btn = QPushButton("Save Scheme")
        self._save_scheme_btn.clicked.connect(self._save_custom_scheme)
        custom_layout.addWidget(self._save_scheme_btn)

        layout.addWidget(custom_group)

        return group

    def _create_grid_video_group(self) -> QGroupBox:
        """Create the grid video preview group box."""
        group = QGroupBox("Grid Video Preview")
        layout = QVBoxLayout(group)

        # Controls row
        controls_row = QHBoxLayout()

        controls_row.addWidget(QLabel("Grid Size:"))
        self._grid_size_spin = QSpinBox()
        self._grid_size_spin.setRange(1, 5)
        self._grid_size_spin.setValue(3)
        self._grid_size_spin.setSuffix("×N")
        self._grid_size_spin.setToolTip(
            "Number of behavior clips per row/column in the preview grid.\n"
            "Default: 3 (3×3 = 9 clips). Larger grids show more variety "
            "but take longer to generate."
        )
        controls_row.addWidget(self._grid_size_spin)

        controls_row.addSpacing(16)

        controls_row.addWidget(QLabel("Speed:"))
        self._speed_slider = QSlider(Qt.Orientation.Horizontal)
        self._speed_slider.setRange(1, 20)   # 0.1x → 2.0x in 0.1 steps
        self._speed_slider.setValue(10)       # default 1.0x
        self._speed_slider.setMaximumWidth(120)
        self._speed_slider.setToolTip(
            "Video playback speed multiplier. Default: 1.0× (normal speed).\n"
            "Increase to quickly scan through fast behaviors.\n"
            "Range: 0.1× (slow motion) to 2.0× (double speed)."
        )
        self._speed_slider.valueChanged.connect(self._on_speed_changed)
        controls_row.addWidget(self._speed_slider)

        self._speed_label = QLabel("1.0×")
        self._speed_label.setMinimumWidth(40)
        controls_row.addWidget(self._speed_label)

        controls_row.addStretch()

        self._generate_preview_btn = QPushButton("🎬 Generate Preview")
        self._generate_preview_btn.setObjectName("primaryButton")
        self._generate_preview_btn.clicked.connect(self._generate_grid_preview)
        self._generate_preview_btn.setEnabled(False)
        controls_row.addWidget(self._generate_preview_btn)

        layout.addLayout(controls_row)

        # Video path / status display
        self._video_status_label = QLabel("No preview generated.")
        self._video_status_label.setWordWrap(True)
        self._video_status_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        layout.addWidget(self._video_status_label)

        # Open video button (hidden until a video is ready)
        open_row = QHBoxLayout()
        self._open_video_btn = QPushButton("▶ Open Video")
        self._open_video_btn.setVisible(False)
        self._open_video_btn.clicked.connect(self._open_grid_video)
        open_row.addWidget(self._open_video_btn)
        open_row.addStretch()
        layout.addLayout(open_row)

        # Progress bar for grid generation
        self._grid_progress = QProgressBar()
        self._grid_progress.setVisible(False)
        layout.addWidget(self._grid_progress)

        return group

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_project(self, storage_path: str, project_name: str):
        """Called when a project is opened."""
        self._storage_path = storage_path
        self._project_name = project_name
        self._annotator_data = None
        self._annotations = {}
        self._session_id = None
        self._grid_video_path = None
        self._cluster_list.clear()
        self._status_label.setText("Status: Project loaded — click 'Load Cluster Data'")
        self._save_btn.setEnabled(False)
        self._generate_preview_btn.setEnabled(False)
        self._open_video_btn.setVisible(False)
        self._video_status_label.setText("No preview generated.")
        self._refresh_sessions()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    @pyqtSlot()
    def _refresh_sessions(self):
        if not self._storage_path or not self._project_name:
            return
        try:
            from castle.service.session_manager import SessionManager
            mgr = SessionManager(self._storage_path, self._project_name)
            sessions = mgr.list_sessions()
            self._session_combo.clear()
            if not sessions:
                self._session_combo.addItem("(no sessions)", "")
            else:
                for s in sessions:
                    label = f"{s.name} — {s.n_clusters} clusters, bin_size={s.bin_size}"
                    self._session_combo.addItem(label, s.session_id)
                # Auto-select active session
                active_id = mgr.get_active_session_id()
                if active_id:
                    for i in range(self._session_combo.count()):
                        if self._session_combo.itemData(i) == active_id:
                            self._session_combo.setCurrentIndex(i)
                            break
        except Exception as exc:
            logger.warning("Failed to refresh sessions: %s", exc)

    @pyqtSlot()
    def _load_cluster_data(self):
        if not self._storage_path or not self._project_name:
            QMessageBox.warning(
                self,
                "No Project Selected",
                "No project selected. Please open a project from the Project panel first.",
            )
            return

        session_id = self._session_combo.currentData() or None
        self._session_id = session_id

        self._progress_bar.setVisible(True)
        self._progress_bar.setRange(0, 0)
        self._load_btn.setEnabled(False)
        self._status_label.setText("Status: Loading cluster data…")

        from castle.service.annotator_loader import load_annotator_data
        self._worker = ServiceWorker(
            load_annotator_data,
            self._storage_path,
            self._project_name,
            session_id=session_id,
        )
        self._worker.finished.connect(self._on_data_loaded)
        self._worker.error.connect(self._on_load_error)
        self._worker.start()

    @pyqtSlot(object)
    def _on_data_loaded(self, annotator_data):
        self._annotator_data = annotator_data
        self._progress_bar.setVisible(False)
        self._load_btn.setEnabled(True)

        # Load existing annotations
        from castle.service.annotation_service import load_annotations
        self._annotations = load_annotations(
            self._storage_path, self._project_name, session_id=self._session_id
        )

        # Populate cluster list
        self._populate_cluster_list()

        # Populate scheme dropdown
        self._populate_schemes()

        n_clusters = len(annotator_data.cluster_meta)
        n_bins = len(annotator_data.cluster)
        self._status_label.setText(
            f"Loaded: {n_clusters} clusters, {n_bins} bins "
            f"(bin_size={annotator_data.bin_size}, fps={annotator_data.fps:.1f})"
        )
        self._save_btn.setEnabled(True)
        self._generate_preview_btn.setEnabled(True)

    @pyqtSlot(str)
    def _on_load_error(self, err: str):
        self._progress_bar.setVisible(False)
        self._load_btn.setEnabled(True)
        self._status_label.setText(f"Error: {err}")
        QMessageBox.critical(
            self,
            "Load Error",
            f"Failed to load cluster data.\n\n{err}\n\nPlease complete the clustering step (Step 3) first.",
        )

    def _populate_cluster_list(self):
        self._cluster_list.clear()
        if self._annotator_data is None:
            return
        for cid, meta in sorted(
            self._annotator_data.cluster_meta.items(),
            key=lambda x: x[1]["name"],
        ):
            name = meta["name"]
            if name == "init":
                continue
            prefix = "✅ " if name in self._annotations else ""
            item = QListWidgetItem(f"{prefix}{name}")
            item.setData(Qt.ItemDataRole.UserRole, name)
            self._cluster_list.addItem(item)

    def _populate_schemes(self):
        self._scheme_combo.blockSignals(True)
        try:
            from castle.service.annotation_service import list_schemes
            schemes = list_schemes(
                self._storage_path, self._project_name, session_id=self._session_id
            )
            self._scheme_combo.clear()
            for name in schemes:
                self._scheme_combo.addItem(name)
            # Default to 10-class
            idx = self._scheme_combo.findText("10-class")
            if idx >= 0:
                self._scheme_combo.setCurrentIndex(idx)
        finally:
            self._scheme_combo.blockSignals(False)
        self._on_scheme_changed(self._scheme_combo.currentText())

    @pyqtSlot(QListWidgetItem, QListWidgetItem)
    def _on_cluster_selected(self, current, previous):
        if current is None:
            self._cluster_info_label.setText("Selected: None")
            return
        cluster_name = current.data(Qt.ItemDataRole.UserRole)
        self._cluster_info_label.setText(f"Selected: {cluster_name}")

        # Pre-fill existing annotation if available
        if cluster_name in self._annotations:
            ann = self._annotations[cluster_name]
            label = ann.get("behavior_label", "")
            comment = ann.get("comment", "")
            scheme = ann.get("scheme", "")

            # Set scheme
            if scheme:
                idx = self._scheme_combo.findText(scheme)
                if idx >= 0:
                    self._scheme_combo.setCurrentIndex(idx)

            # Set label
            idx = self._label_combo.findText(label)
            if idx >= 0:
                self._label_combo.setCurrentIndex(idx)

            self._comment_edit.setText(comment)

        # Reset grid video status when cluster changes
        self._grid_video_path = None
        self._open_video_btn.setVisible(False)
        self._video_status_label.setText("No preview generated for this cluster.")

    @pyqtSlot(str)
    def _on_scheme_changed(self, scheme_name: str):
        if not scheme_name or not self._storage_path:
            return
        from castle.service.annotation_service import get_scheme_labels
        labels = get_scheme_labels(
            self._storage_path, self._project_name, scheme_name,
            session_id=self._session_id,
        )
        self._label_combo.clear()
        self._label_combo.addItem("")  # blank default
        self._label_combo.addItems(labels)

    @pyqtSlot(int)
    def _on_speed_changed(self, value: int):
        speed = value / 10.0
        self._speed_label.setText(f"{speed:.1f}×")

    @pyqtSlot()
    def _save_annotation(self):
        if self._annotator_data is None:
            QMessageBox.warning(
                self,
                "Data Not Loaded",
                "Cluster data not loaded. Please click 'Load Cluster Data' before annotating.",
            )
            return

        current = self._cluster_list.currentItem()
        if current is None:
            QMessageBox.warning(
                self,
                "No Cluster Selected",
                "No cluster selected. Please select a cluster from the list on the left.",
            )
            return

        cluster_name = current.data(Qt.ItemDataRole.UserRole)
        behavior_label = self._label_combo.currentText().strip()
        scheme_name = self._scheme_combo.currentText()
        comment = self._comment_edit.text().strip()

        if not behavior_label:
            QMessageBox.warning(
                self,
                "No Label Selected",
                "No behavior label selected. Please choose a behavior label from the dropdown.",
            )
            return

        self._annotations[cluster_name] = {
            "behavior_label": behavior_label,
            "scheme": scheme_name,
            "comment": comment,
            "annotator": "user",
            "timestamp": datetime.datetime.now().isoformat(),
        }

        from castle.service.annotation_service import save_annotations
        save_annotations(
            self._storage_path, self._project_name, self._annotations,
            session_id=self._session_id,
        )

        self._status_label.setText(f"Saved: {cluster_name} → {behavior_label}")

        # Refresh cluster list to show ✅ marks
        current_row = self._cluster_list.currentRow()
        self._populate_cluster_list()
        if 0 <= current_row < self._cluster_list.count():
            self._cluster_list.setCurrentRow(current_row)

    @pyqtSlot()
    def _save_custom_scheme(self):
        if not self._storage_path or not self._project_name:
            QMessageBox.warning(
                self,
                "No Project Selected",
                "No project selected. Please open a project from the Project panel first.",
            )
            return

        name = self._custom_scheme_name.text().strip()
        raw_labels = self._custom_labels_edit.toPlainText().strip()
        if not name or not raw_labels:
            QMessageBox.warning(
                self,
                "Missing Input",
                "Scheme name or labels missing. Please enter a scheme name and at least one label.",
            )
            return

        labels = [ln.strip() for ln in raw_labels.split("\n") if ln.strip()]
        if not labels:
            QMessageBox.warning(
                self,
                "No Valid Labels",
                "No valid labels found. Add at least one label (one per line) in the Labels field.",
            )
            return

        from castle.service.annotation_service import save_scheme
        save_scheme(
            self._storage_path, self._project_name, name, labels,
            session_id=self._session_id,
        )
        self._populate_schemes()
        idx = self._scheme_combo.findText(name)
        if idx >= 0:
            self._scheme_combo.setCurrentIndex(idx)
        self._status_label.setText(f"Saved scheme '{name}' with {len(labels)} labels")

    # ------------------------------------------------------------------
    # Grid video
    # ------------------------------------------------------------------

    @pyqtSlot()
    def _generate_grid_preview(self):
        """Generate a grid video for the currently selected cluster."""
        if self._annotator_data is None:
            QMessageBox.warning(
                self,
                "Data Not Loaded",
                "Cluster data not loaded. Please click 'Load Cluster Data' before generating a preview.",
            )
            return

        current = self._cluster_list.currentItem()
        if current is None:
            QMessageBox.warning(
                self,
                "No Cluster Selected",
                "No cluster selected. Please select a cluster from the list on the left.",
            )
            return

        cluster_name = current.data(Qt.ItemDataRole.UserRole)

        # Find cluster_id from cluster_meta
        cluster_id = None
        for cid, meta in self._annotator_data.cluster_meta.items():
            if meta.get("name") == cluster_name:
                cluster_id = cid
                break

        if cluster_id is None:
            QMessageBox.warning(
                self,
                "Cluster Not Found",
                f"Cluster '{cluster_name}' not found. Try reloading the cluster data.",
            )
            return

        grid_cols = self._grid_size_spin.value()

        # Disable controls during generation
        self._generate_preview_btn.setEnabled(False)
        self._open_video_btn.setVisible(False)
        self._grid_progress.setVisible(True)
        self._grid_progress.setRange(0, 0)
        self._video_status_label.setText(
            f"Generating {grid_cols}×{grid_cols} grid for '{cluster_name}'…"
        )

        from castle.service.bout_service import generate_grid_video

        self._grid_worker = ServiceWorker(
            generate_grid_video,
            self._annotator_data,
            cluster_id,
            grid_cols=grid_cols,
        )
        self._grid_worker.finished.connect(self._on_grid_video_ready)
        self._grid_worker.error.connect(self._on_grid_video_error)
        self._grid_worker.start()

    @pyqtSlot(object)
    def _on_grid_video_ready(self, video_path):
        self._grid_progress.setVisible(False)
        self._generate_preview_btn.setEnabled(True)

        if not video_path or not os.path.exists(video_path):
            self._video_status_label.setText(
                "⚠ Grid video generation returned no output (check logs)."
            )
            return

        self._grid_video_path = video_path
        self._video_status_label.setText(f"✅ Ready: {video_path}")
        self._open_video_btn.setVisible(True)

        # Auto-open with system player
        self._open_grid_video()

    @pyqtSlot(str)
    def _on_grid_video_error(self, err: str):
        self._grid_progress.setVisible(False)
        self._generate_preview_btn.setEnabled(True)
        self._video_status_label.setText(f"Error: {err}")
        logger.error("Grid video generation failed: %s", err)

    @pyqtSlot()
    def _open_grid_video(self):
        """Open the generated grid video with the system default player."""
        if not self._grid_video_path or not os.path.exists(self._grid_video_path):
            QMessageBox.warning(
                self,
                "No Grid Video",
                "No grid video available. Generate a preview first by clicking '🎬 Generate Preview'.",
            )
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(self._grid_video_path))
