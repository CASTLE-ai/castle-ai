"""
CASTLE Desktop - Clustering Panel (Tab 4)

The main workspace: interactive UMAP embedding, DBSCAN clustering,
behavior labeling, and export.  Uses ClusteringSession from the service layer.
"""

import json
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QGroupBox, QLabel, QPushButton, QComboBox, QSpinBox,
    QDoubleSpinBox, QTextEdit, QLineEdit, QProgressBar,
    QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSlot

from castle.desktop.components.embedding_view import EmbeddingWidget
from castle.desktop.components.video_player import FrameViewer
from castle.desktop.components.cluster_tree import ClusterTreeWidget
from castle.desktop.components.syllable_bar import SyllableBarWidget
from castle.desktop.services.worker_threads import (
    ClusteringSessionWorker, UMAPWorker,
)


class ClusterPanel(QWidget):
    """Behavior clustering panel with interactive embedding visualization."""

    def __init__(self, parent=None):
        super().__init__(parent)

        # State
        self._storage_path = None
        self._project_name = None
        self._session = None          # ClusteringSession
        self._worker = None           # keep worker alive
        self._setup_ui()

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)

        # Top controls
        main_layout.addWidget(self._create_controls())

        # Main content: splitter with 3 panes
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._create_left_panel())
        splitter.addWidget(self._create_center_panel())
        splitter.addWidget(self._create_right_panel())
        splitter.setSizes([280, 600, 320])
        main_layout.addWidget(splitter, stretch=1)

        # Bottom: syllable bar
        self._syllable_bar = SyllableBarWidget()
        main_layout.addWidget(self._syllable_bar)

        # Progress bar
        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)
        main_layout.addWidget(self._progress_bar)

    def _create_controls(self) -> QWidget:
        group = QGroupBox("Initialization")
        layout = QHBoxLayout(group)

        layout.addWidget(QLabel("Model:"))
        self._model_combo = QComboBox()
        self._model_combo.addItems([
            "dinov2_vitb14_reg4_pretrain",
            "dinov3_vitb16",
            "dinov3_vitl16",
        ])
        self._model_combo.setCurrentIndex(1)
        layout.addWidget(self._model_combo)

        layout.addWidget(QLabel("ROI:"))
        self._roi_spin = QSpinBox()
        self._roi_spin.setMinimum(1)
        self._roi_spin.setMaximum(99)
        self._roi_spin.setValue(1)
        layout.addWidget(self._roi_spin)

        layout.addWidget(QLabel("Bin:"))
        self._bin_spin = QSpinBox()
        self._bin_spin.setMinimum(1)
        self._bin_spin.setMaximum(100)
        self._bin_spin.setValue(1)
        layout.addWidget(self._bin_spin)

        layout.addStretch()

        self._init_btn = QPushButton("Initialize")
        self._init_btn.setObjectName("primaryButton")
        self._init_btn.clicked.connect(self._initialize)
        layout.addWidget(self._init_btn)

        self._restore_btn = QPushButton("Restore Session")
        self._restore_btn.clicked.connect(self._restore_session)
        self._restore_btn.setEnabled(False)
        layout.addWidget(self._restore_btn)

        return group

    def _create_left_panel(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        # --- Cluster tree ---
        tree_group = QGroupBox("Cluster Hierarchy")
        tree_layout = QVBoxLayout(tree_group)
        self._cluster_tree = ClusterTreeWidget()
        self._cluster_tree.cluster_selected.connect(self._on_cluster_selected)
        tree_layout.addWidget(self._cluster_tree)
        layout.addWidget(tree_group, stretch=1)

        # --- UMAP config ---
        umap_group = QGroupBox("UMAP Configuration")
        umap_layout = QVBoxLayout(umap_group)

        umap_layout.addWidget(QLabel("Preset:"))
        self._preset_combo = QComboBox()
        self._preset_combo.addItems([
            "Low-mag 100", "Low-mag 50", "Low-mag 25",
            "Intermediate (100, 50)", "Intermediate (50, 25)",
            "High (100, 50)", "High (50, 25)",
        ])
        self._preset_combo.currentTextChanged.connect(self._on_preset_changed)
        umap_layout.addWidget(self._preset_combo)

        # n_neighbors
        nn_layout = QHBoxLayout()
        nn_layout.addWidget(QLabel("n_neighbors:"))
        self._nn_spin = QSpinBox()
        self._nn_spin.setMinimum(2)
        self._nn_spin.setMaximum(500)
        self._nn_spin.setValue(100)
        self._nn_spin.setToolTip(
            "UMAP n_neighbors: controls balance between local and global structure.\n"
            "Default: 100. Higher = more global clusters; lower = more local detail.\n"
            "Typical range: 25–200 depending on dataset size."
        )
        nn_layout.addWidget(self._nn_spin)
        umap_layout.addLayout(nn_layout)

        # min_dist
        md_layout = QHBoxLayout()
        md_layout.addWidget(QLabel("min_dist:"))
        self._md_spin = QDoubleSpinBox()
        self._md_spin.setMinimum(0.0)
        self._md_spin.setMaximum(1.0)
        self._md_spin.setSingleStep(0.05)
        self._md_spin.setValue(0.0)
        self._md_spin.setToolTip(
            "UMAP min_dist: minimum distance between points in the 2D embedding.\n"
            "Default: 0.0. Higher values spread points more evenly; lower values "
            "create tighter clusters."
        )
        md_layout.addWidget(self._md_spin)
        umap_layout.addLayout(md_layout)

        # metric
        metric_layout = QHBoxLayout()
        metric_layout.addWidget(QLabel("metric:"))
        self._metric_combo = QComboBox()
        self._metric_combo.addItems(["euclidean", "cosine", "manhattan"])
        metric_layout.addWidget(self._metric_combo)
        umap_layout.addLayout(metric_layout)

        # n_components
        nc_layout = QHBoxLayout()
        nc_layout.addWidget(QLabel("n_components:"))
        self._nc_spin = QSpinBox()
        self._nc_spin.setMinimum(2)
        self._nc_spin.setMaximum(50)
        self._nc_spin.setValue(2)
        nc_layout.addWidget(self._nc_spin)
        umap_layout.addLayout(nc_layout)

        # Full JSON config editor (advanced)
        self._umap_config_edit = QTextEdit()
        self._umap_config_edit.setMaximumHeight(100)
        self._umap_config_edit.setPlaceholderText("Advanced: edit UMAP JSON config...")
        self._sync_umap_json()
        umap_layout.addWidget(self._umap_config_edit)

        self._run_umap_btn = QPushButton("Generate Embedding")
        self._run_umap_btn.setObjectName("primaryButton")
        self._run_umap_btn.clicked.connect(self._run_umap)
        self._run_umap_btn.setEnabled(False)
        umap_layout.addWidget(self._run_umap_btn)

        layout.addWidget(umap_group, stretch=1)

        # --- DBSCAN ---
        dbscan_group = QGroupBox("Clustering (DBSCAN)")
        dbscan_layout = QVBoxLayout(dbscan_group)

        eps_layout = QHBoxLayout()
        eps_layout.addWidget(QLabel("Epsilon:"))
        self._eps_spin = QDoubleSpinBox()
        self._eps_spin.setMinimum(0.01)
        self._eps_spin.setMaximum(20.0)
        self._eps_spin.setSingleStep(0.1)
        self._eps_spin.setValue(1.0)
        self._eps_spin.setToolTip(
            "DBSCAN epsilon: neighborhood search radius.\n"
            "Default: 1.0. Larger values = fewer, bigger clusters; "
            "smaller values = more, finer-grained clusters.\n"
            "Adjust based on the density visible in the embedding scatter plot."
        )
        eps_layout.addWidget(self._eps_spin)
        dbscan_layout.addLayout(eps_layout)

        self._run_cluster_btn = QPushButton("Generate Clusters")
        self._run_cluster_btn.clicked.connect(self._run_clustering)
        self._run_cluster_btn.setEnabled(False)
        dbscan_layout.addWidget(self._run_cluster_btn)

        self._cluster_info_label = QLabel("")
        dbscan_layout.addWidget(self._cluster_info_label)

        layout.addWidget(dbscan_group)

        # --- Labeling ---
        label_group = QGroupBox("Label Cluster")
        label_layout = QVBoxLayout(label_group)

        id_layout = QHBoxLayout()
        id_layout.addWidget(QLabel("Cluster ID:"))
        self._cluster_id_spin = QSpinBox()
        self._cluster_id_spin.setMinimum(-1)
        self._cluster_id_spin.setMaximum(999)
        id_layout.addWidget(self._cluster_id_spin)
        label_layout.addLayout(id_layout)

        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Name:"))
        self._cluster_name_input = QLineEdit()
        self._cluster_name_input.setPlaceholderText("Behavior name...")
        name_layout.addWidget(self._cluster_name_input)
        label_layout.addLayout(name_layout)

        btn_row = QHBoxLayout()
        self._label_btn = QPushButton("Label")
        self._label_btn.clicked.connect(self._label_cluster)
        self._label_btn.setEnabled(False)
        btn_row.addWidget(self._label_btn)

        self._auto_label_btn = QPushButton("Auto-Label All")
        self._auto_label_btn.clicked.connect(self._auto_label)
        self._auto_label_btn.setEnabled(False)
        btn_row.addWidget(self._auto_label_btn)
        label_layout.addLayout(btn_row)

        submit_row = QHBoxLayout()
        self._submit_btn = QPushButton("Submit & Export")
        self._submit_btn.setObjectName("primaryButton")
        self._submit_btn.clicked.connect(self._submit_all)
        self._submit_btn.setEnabled(False)
        submit_row.addWidget(self._submit_btn)
        label_layout.addLayout(submit_row)

        layout.addWidget(label_group)

        return widget

    def _create_center_panel(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        self._embedding_widget = EmbeddingWidget()
        self._embedding_widget.point_clicked.connect(self._on_embedding_click)
        layout.addWidget(self._embedding_widget)

        return widget

    def _create_right_panel(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        # Frame viewer
        self._frame_viewer = FrameViewer()
        layout.addWidget(self._frame_viewer, stretch=1)

        # Status
        self._status_text = QTextEdit()
        self._status_text.setReadOnly(True)
        self._status_text.setMaximumHeight(120)
        self._status_text.setPlaceholderText("Status & info...")
        layout.addWidget(self._status_text)

        return widget

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_project(self, storage_path: str, project_name: str):
        """Called when a project is opened."""
        self._storage_path = storage_path
        self._project_name = project_name
        self._session = None
        self._cluster_tree.clear()
        self._embedding_widget.clear()
        self._frame_viewer.clear()
        self._status_text.clear()
        self._enable_clustering_controls(False)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    @pyqtSlot()
    def _initialize(self):
        if not self._project_name:
            QMessageBox.warning(
                self,
                "No Project Selected",
                "No project selected. Please open a project from the Project panel first.",
            )
            return

        self._progress_bar.setVisible(True)
        self._progress_bar.setRange(0, 0)
        self._init_btn.setEnabled(False)
        self._status_text.append("Initializing clustering session...")

        self._worker = ClusteringSessionWorker(
            self._storage_path, self._project_name,
            roi=self._roi_spin.value(),
            bin_size=self._bin_spin.value(),
            model=self._model_combo.currentText(),
        )
        self._worker.finished.connect(self._on_init_finished)
        self._worker.error.connect(self._on_init_error)
        self._worker.start()

    @pyqtSlot(object)
    def _on_init_finished(self, session):
        self._session = session
        self._progress_bar.setVisible(False)
        self._init_btn.setEnabled(True)

        # Populate cluster tree
        self._cluster_tree.set_latent(session.latents)
        self._run_umap_btn.setEnabled(True)
        self._restore_btn.setEnabled(True)

        n = len(session.latents.data) if hasattr(session.latents, 'data') else 0
        self._status_text.append(f"✅ Session initialized. {n} data points loaded.")
        self._status_text.append(f"Clusters: {session.cluster_names}")

    @pyqtSlot(str)
    def _on_init_error(self, err):
        self._progress_bar.setVisible(False)
        self._init_btn.setEnabled(True)
        self._status_text.append(f"❌ Init error: {err}")
        QMessageBox.critical(
            self,
            "Initialization Error",
            f"Session initialization failed.\n\n{err}\n\n"
            "Please check that latent features have been extracted (Step 3) "
            "and the ROI ID is correct.",
        )

    @pyqtSlot()
    def _restore_session(self):
        if self._session is None:
            QMessageBox.warning(
                self,
                "Not Initialized",
                "Session not initialized. Please click 'Initialize' before restoring a session.",
            )
            return
        try:
            result = self._session.restore()
            if result.get('success'):
                self._cluster_tree.set_latent(self._session.latents)
                self._status_text.append(
                    f"✅ Session restored: {result['cluster_count']} clusters"
                )
                self._enable_clustering_controls(True)
            else:
                self._status_text.append(f"Restore failed: {result.get('error', 'unknown')}")
        except Exception as e:
            QMessageBox.critical(
                self,
                "Restore Failed",
                f"Failed to restore session.\n\n{e}\n\nTry reinitializing instead.",
            )

    @pyqtSlot(str)
    def _on_cluster_selected(self, cluster_name: str):
        self._status_text.append(f"Selected cluster: {cluster_name}")

    @pyqtSlot()
    def _run_umap(self):
        if self._session is None:
            return

        selected = self._cluster_tree.selected_cluster_name()
        if not selected:
            QMessageBox.warning(
                self,
                "No Cluster Selected",
                "No cluster selected. Please select a cluster from the Cluster Hierarchy tree.",
            )
            return

        # Try reading JSON config, else build from spinboxes
        try:
            config_text = self._umap_config_edit.toPlainText().strip()
            if config_text:
                umap_config = json.loads(config_text)
            else:
                umap_config = self._build_umap_config()
        except json.JSONDecodeError:
            umap_config = self._build_umap_config()

        self._progress_bar.setVisible(True)
        self._progress_bar.setRange(0, 0)
        self._run_umap_btn.setEnabled(False)
        self._status_text.append(f"Running UMAP on '{selected}'...")

        self._worker = UMAPWorker(self._session, selected, umap_config)
        self._worker.finished.connect(self._on_umap_finished)
        self._worker.error.connect(self._on_umap_error)
        self._worker.start()

    @pyqtSlot(object)
    def _on_umap_finished(self, result):
        self._progress_bar.setVisible(False)
        self._run_umap_btn.setEnabled(True)

        if result.get('success'):
            self._run_cluster_btn.setEnabled(True)
            ll = self._session.local_latents
            self._embedding_widget.set_embedding(ll.embedding)
            self._status_text.append(
                f"✅ UMAP done: {result['n_points']} points, "
                f"shape {result['embedding_shape']}"
            )
        else:
            self._status_text.append(f"UMAP failed: {result.get('error', '')}")

    @pyqtSlot(str)
    def _on_umap_error(self, err):
        self._progress_bar.setVisible(False)
        self._run_umap_btn.setEnabled(True)
        self._status_text.append(f"❌ UMAP error: {err}")
        QMessageBox.critical(
            self,
            "UMAP Error",
            f"UMAP dimensionality reduction failed.\n\n{err}\n\n"
            "Try adjusting n_neighbors, selecting a different preset, "
            "or choosing a different cluster.",
        )

    @pyqtSlot()
    def _run_clustering(self):
        if self._session is None or self._session.local_latents is None:
            return

        eps = self._eps_spin.value()
        self._status_text.append(f"Running DBSCAN (eps={eps})...")

        try:
            result = self._session.run_dbscan(eps)
            if result.get('success'):
                ll = self._session.local_latents
                self._embedding_widget.set_embedding(
                    ll.embedding, cluster=ll.cluster,
                )
                self._cluster_info_label.setText(
                    f"{result['n_clusters']} clusters, "
                    f"{result['noise_count']} noise points"
                )
                self._enable_clustering_controls(True)
                self._status_text.append(
                    f"✅ DBSCAN: {result['n_clusters']} clusters found"
                )
            else:
                self._status_text.append(f"DBSCAN failed: {result.get('error', '')}")
        except Exception as e:
            QMessageBox.critical(
                self,
                "DBSCAN Error",
                f"DBSCAN clustering failed.\n\n{e}\n\nTry adjusting the Epsilon value.",
            )

    @pyqtSlot(int, float, float)
    def _on_embedding_click(self, local_index: int, x: float, y: float):
        if self._session is None:
            return
        ll = self._session.local_latents
        if ll is None:
            return

        # Map local index to global index
        global_index = np.arange(len(ll.index_mask))[ll.index_mask][local_index]
        frame = self._session.get_frame(int(global_index))
        if frame is not None:
            self._frame_viewer.show_frame(frame)

        # If clusters exist, update cluster ID spin
        if hasattr(ll, 'cluster') and ll.cluster is not None:
            cid = int(ll.cluster[local_index])
            self._cluster_id_spin.setValue(cid)

    @pyqtSlot()
    def _label_cluster(self):
        if self._session is None or self._session.local_latents is None:
            return
        cid = self._cluster_id_spin.value()
        name = self._cluster_name_input.text().strip()
        if not name:
            QMessageBox.warning(
                self,
                "No Cluster Name",
                "Please enter a name for the cluster before clicking Label.",
            )
            return
        try:
            self._session.label_cluster(cid, name)
            ll = self._session.local_latents
            self._embedding_widget.set_embedding(
                ll.embedding, cluster=ll.cluster, labels=ll.export,
            )
            self._status_text.append(f"Labeled cluster {cid} → '{name}'")
        except Exception as e:
            QMessageBox.critical(
                self,
                "Label Error",
                f"Failed to label cluster {cid}.\n\n{e}",
            )

    @pyqtSlot()
    def _auto_label(self):
        if self._session is None:
            return
        try:
            count = self._session.auto_label_all()
            ll = self._session.local_latents
            self._embedding_widget.set_embedding(
                ll.embedding, cluster=ll.cluster, labels=ll.export,
            )
            self._status_text.append(f"Auto-labeled {count} clusters")
        except Exception as e:
            QMessageBox.critical(
                self,
                "Auto-Label Error",
                f"Auto-labeling failed.\n\n{e}",
            )

    @pyqtSlot()
    def _submit_all(self):
        if self._session is None:
            return
        try:
            result = self._session.submit()
            if result.get('success'):
                self._cluster_tree.set_latent(self._session.latents)

                # Update syllable bar
                if hasattr(self._session, 'aggregator'):
                    agg = self._session.aggregator
                    self._syllable_bar.set_data(
                        self._session.latents.cluster,
                        self._session.latents.cluster_meta,
                        agg.videos_meta,
                        self._session.bin_size,
                        agg.fps if hasattr(agg, 'fps') else 30.0,
                    )

                self._status_text.append("✅ Submitted & exported.")
                self._status_text.append(f"  ID CSV: {result.get('id_csv_path', '')}")
                self._status_text.append(
                    f"  {len(result.get('time_series_paths', []))} time series files"
                )
                QMessageBox.information(self, "Success", "Clusters submitted and exported.")
            else:
                self._status_text.append(f"Submit failed: {result.get('error', '')}")
        except Exception as e:
            QMessageBox.critical(
                self,
                "Submit Error",
                f"Failed to submit clusters.\n\n{e}\n\nCheck the status log for details.",
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_umap_config(self) -> list:
        return [{
            "n_neighbors": self._nn_spin.value(),
            "min_dist": self._md_spin.value(),
            "metric": self._metric_combo.currentText(),
            "n_components": self._nc_spin.value(),
            "n_epochs": 5000,
        }]

    def _sync_umap_json(self):
        config = self._build_umap_config()
        self._umap_config_edit.setPlainText(json.dumps(config, indent=2))

    def _on_preset_changed(self, text: str):
        import re
        numbers = re.findall(r'\d+', text)
        if 'Low-mag' in text and numbers:
            config = [{
                "n_neighbors": int(numbers[0]), "min_dist": 0.0,
                "n_components": 2, "n_epochs": 5000,
            }]
        elif 'Intermediate' in text and len(numbers) >= 2:
            config = [
                {"n_neighbors": int(numbers[0]), "min_dist": 0.0,
                 "n_components": 5, "n_epochs": 5000},
                {"n_neighbors": int(numbers[1]), "min_dist": 0.0,
                 "n_components": 2, "n_epochs": 5000},
            ]
        elif 'High' in text and len(numbers) >= 2:
            config = [
                {"n_neighbors": int(numbers[0]), "min_dist": 0.0,
                 "n_components": 10, "n_epochs": 5000},
                {"n_neighbors": int(numbers[1]), "min_dist": 0.0,
                 "n_components": 2, "n_epochs": 5000},
            ]
        else:
            return

        self._umap_config_edit.setPlainText(json.dumps(config, indent=2))
        # Also sync spin boxes to the first stage
        if config:
            first = config[0]
            self._nn_spin.setValue(first.get('n_neighbors', 100))
            self._md_spin.setValue(first.get('min_dist', 0.0))
            self._nc_spin.setValue(first.get('n_components', 2))

    def _enable_clustering_controls(self, enabled: bool):
        self._label_btn.setEnabled(enabled)
        self._auto_label_btn.setEnabled(enabled)
        self._submit_btn.setEnabled(enabled)
