"""
CASTLE Desktop - Behavior Microscope Panel (Stage 4)

Interactive UMAP embedding visualization and cluster annotation.
This is the primary panel where PyQt's advantages over Gradio are most apparent.

Features:
- pyqtgraph ScatterPlotItem for real-time embedding interaction
- Click-to-inspect: click on point → show corresponding video frame
- Color-coded clusters with legend
- Hierarchical cluster tree navigation
- Async UMAP computation via QThread
"""

import os
import json
import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QGroupBox, QLabel, QPushButton, QComboBox, QSpinBox,
    QDoubleSpinBox, QTextEdit, QTreeWidget, QTreeWidgetItem,
    QLineEdit, QProgressBar, QMessageBox, QFrame
)
from PySide6.QtCore import Qt, Signal, Slot, QThread, QSize
from PySide6.QtGui import QImage, QPixmap

import pyqtgraph as pg

from castle.desktop.components.embedding_view import EmbeddingWidget
from castle.desktop.components.video_player import FrameViewer
from castle.desktop.components.cluster_tree import ClusterTreeWidget
from castle.desktop.components.syllable_bar import SyllableBarWidget
from castle.desktop.services.worker_threads import UMAPWorker, ClusterWorker


class MicroscopePanel(QWidget):
    """Behavior Microscope panel with interactive embedding visualization."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # State
        self._storage_path = None
        self._project_name = None
        self._aggregator = None
        self._latents = None
        self._local_latents = None
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Build the panel layout."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        
        # Top controls
        controls = self._create_controls()
        main_layout.addWidget(controls)
        
        # Main content area (splitter)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left: Cluster tree + config
        left_panel = self._create_left_panel()
        splitter.addWidget(left_panel)
        
        # Center: Embedding plot
        center_panel = self._create_center_panel()
        splitter.addWidget(center_panel)
        
        # Right: Frame viewer
        right_panel = self._create_right_panel()
        splitter.addWidget(right_panel)
        
        # Set splitter proportions
        splitter.setSizes([250, 600, 350])
        main_layout.addWidget(splitter, stretch=1)
        
        # Bottom: Syllable bar
        self._syllable_bar = SyllableBarWidget()
        main_layout.addWidget(self._syllable_bar)
        
        # Progress bar
        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)
        main_layout.addWidget(self._progress_bar)
    
    def _create_controls(self) -> QWidget:
        """Create the top control bar."""
        group = QGroupBox("Configuration")
        layout = QHBoxLayout(group)
        
        # Model selection
        layout.addWidget(QLabel("Model:"))
        self._model_combo = QComboBox()
        self._model_combo.addItems([
            "dinov2_vitb14_reg4_pretrain",
            "dinov3_vitb16",
            "dinov3_vitl16"
        ])
        self._model_combo.setCurrentIndex(1)
        layout.addWidget(self._model_combo)
        
        # ROI ID
        layout.addWidget(QLabel("ROI ID:"))
        self._roi_spin = QSpinBox()
        self._roi_spin.setMinimum(1)
        self._roi_spin.setMaximum(99)
        self._roi_spin.setValue(1)
        layout.addWidget(self._roi_spin)
        
        # Bin size
        layout.addWidget(QLabel("Bin Size:"))
        self._bin_spin = QSpinBox()
        self._bin_spin.setMinimum(1)
        self._bin_spin.setMaximum(100)
        self._bin_spin.setValue(1)
        layout.addWidget(self._bin_spin)
        
        layout.addStretch()
        
        # Initialize button
        self._init_btn = QPushButton("Initialize")
        self._init_btn.setObjectName("primaryButton")
        self._init_btn.clicked.connect(self._initialize)
        layout.addWidget(self._init_btn)
        
        # Restore button
        self._restore_btn = QPushButton("Restore Session")
        self._restore_btn.clicked.connect(self._restore_session)
        self._restore_btn.setEnabled(False)
        layout.addWidget(self._restore_btn)
        
        return group
    
    def _create_left_panel(self) -> QWidget:
        """Create the left panel with cluster tree and UMAP config."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Cluster tree
        tree_group = QGroupBox("Cluster Hierarchy")
        tree_layout = QVBoxLayout(tree_group)
        self._cluster_tree = ClusterTreeWidget()
        self._cluster_tree.cluster_selected.connect(self._on_cluster_selected)
        tree_layout.addWidget(self._cluster_tree)
        layout.addWidget(tree_group, stretch=1)
        
        # UMAP config
        umap_group = QGroupBox("UMAP Configuration")
        umap_layout = QVBoxLayout(umap_group)
        
        # Preset dropdown
        self._preset_combo = QComboBox()
        self._preset_combo.addItems([
            "Low-mag 100", "Low-mag 50", "Low-mag 25",
            "Intermediate (100, 50)", "Intermediate (50, 25)",
            "High (100, 50)", "High (50, 25)"
        ])
        self._preset_combo.currentTextChanged.connect(self._on_preset_changed)
        umap_layout.addWidget(QLabel("Preset:"))
        umap_layout.addWidget(self._preset_combo)
        
        # UMAP JSON config
        self._umap_config = QTextEdit()
        self._umap_config.setMaximumHeight(120)
        self._umap_config.setPlainText(json.dumps([{
            "n_neighbors": 100,
            "min_dist": 0.0,
            "n_components": 2,
            "n_epochs": 5000
        }], indent=2))
        umap_layout.addWidget(QLabel("Config:"))
        umap_layout.addWidget(self._umap_config)
        
        self._run_umap_btn = QPushButton("Generate Embedding")
        self._run_umap_btn.setObjectName("primaryButton")
        self._run_umap_btn.clicked.connect(self._run_umap)
        self._run_umap_btn.setEnabled(False)
        umap_layout.addWidget(self._run_umap_btn)
        
        layout.addWidget(umap_group, stretch=1)
        
        # DBSCAN config
        dbscan_group = QGroupBox("Clustering")
        dbscan_layout = QVBoxLayout(dbscan_group)
        
        dbscan_layout.addWidget(QLabel("Epsilon:"))
        self._eps_spin = QDoubleSpinBox()
        self._eps_spin.setMinimum(0.1)
        self._eps_spin.setMaximum(10.0)
        self._eps_spin.setSingleStep(0.1)
        self._eps_spin.setValue(1.0)
        dbscan_layout.addWidget(self._eps_spin)
        
        self._run_cluster_btn = QPushButton("Generate Clusters")
        self._run_cluster_btn.clicked.connect(self._run_clustering)
        self._run_cluster_btn.setEnabled(False)
        dbscan_layout.addWidget(self._run_cluster_btn)
        
        layout.addWidget(dbscan_group)
        
        # Naming
        naming_group = QGroupBox("Label Cluster")
        naming_layout = QVBoxLayout(naming_group)
        
        id_layout = QHBoxLayout()
        id_layout.addWidget(QLabel("ID:"))
        self._cluster_id_spin = QSpinBox()
        self._cluster_id_spin.setMinimum(0)
        id_layout.addWidget(self._cluster_id_spin)
        naming_layout.addLayout(id_layout)
        
        naming_layout.addWidget(QLabel("Name:"))
        self._cluster_name_input = QLineEdit()
        naming_layout.addWidget(self._cluster_name_input)
        
        btn_layout = QHBoxLayout()
        self._label_btn = QPushButton("Label")
        self._label_btn.clicked.connect(self._label_cluster)
        self._label_btn.setEnabled(False)
        btn_layout.addWidget(self._label_btn)
        
        self._submit_btn = QPushButton("Submit All")
        self._submit_btn.setObjectName("primaryButton")
        self._submit_btn.clicked.connect(self._submit_all)
        self._submit_btn.setEnabled(False)
        btn_layout.addWidget(self._submit_btn)
        naming_layout.addLayout(btn_layout)
        
        layout.addWidget(naming_group)
        
        return widget
    
    def _create_center_panel(self) -> QWidget:
        """Create the center panel with embedding plot."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self._embedding_widget = EmbeddingWidget()
        self._embedding_widget.point_clicked.connect(self._on_embedding_click)
        layout.addWidget(self._embedding_widget)
        
        return widget
    
    def _create_right_panel(self) -> QWidget:
        """Create the right panel with frame viewer."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self._frame_viewer = FrameViewer()
        layout.addWidget(self._frame_viewer)
        
        return widget
    
    # --- Public API ---
    
    def set_project(self, storage_path: str, project_name: str):
        """Called when a project is opened."""
        self._storage_path = storage_path
        self._project_name = project_name
        self._cluster_tree.clear()
        self._embedding_widget.clear()
        self._frame_viewer.clear()
    
    # --- Event Handlers ---
    
    @Slot()
    def _initialize(self):
        """Initialize the LatentAggregator."""
        if not self._project_name:
            QMessageBox.warning(self, "Error", "No project selected.")
            return
        
        try:
            from castle.core.cluster import LatentAggregator
            
            self._progress_bar.setVisible(True)
            self._progress_bar.setRange(0, 0)  # Indeterminate
            
            self._aggregator = LatentAggregator(
                self._storage_path,
                self._project_name,
                select_roi_id=self._roi_spin.value(),
                bin_size=self._bin_spin.value(),
                model_name=self._model_combo.currentText(),
            )
            
            self._latents = self._aggregator.get_latent_object()
            
            # Update cluster tree
            self._cluster_tree.set_latent(self._latents)
            
            # Enable UMAP controls
            self._run_umap_btn.setEnabled(True)
            
            self._progress_bar.setVisible(False)
            
            # Check for existing session
            from castle.ui.cluster_page_ui import check_session_exists
            session_info = check_session_exists(self._storage_path, self._project_name)
            self._restore_btn.setEnabled(session_info is not None)
            
        except Exception as e:
            self._progress_bar.setVisible(False)
            QMessageBox.critical(self, "Initialization Error", str(e))
    
    @Slot()
    def _restore_session(self):
        """Restore a previous clustering session."""
        # Placeholder for restore logic
        QMessageBox.information(self, "Info", "Session restore not yet implemented in desktop UI.")
    
    @Slot(str)
    def _on_cluster_selected(self, cluster_name: str):
        """Handle cluster selection from tree."""
        # This would trigger UMAP on the selected cluster
        pass
    
    @Slot()
    def _run_umap(self):
        """Run UMAP embedding computation."""
        if self._latents is None:
            return
        
        try:
            config_text = self._umap_config.toPlainText()
            config = json.loads(config_text)
        except json.JSONDecodeError:
            QMessageBox.warning(self, "Error", "Invalid UMAP config JSON.")
            return
        
        # Get selected cluster
        selected = self._cluster_tree.selected_cluster_name()
        if not selected:
            QMessageBox.warning(self, "Error", "Select a cluster first.")
            return
        
        self._progress_bar.setVisible(True)
        self._progress_bar.setRange(0, 0)
        self._run_umap_btn.setEnabled(False)
        
        # Run in background thread
        self._umap_worker = UMAPWorker(self._latents, selected, config)
        self._umap_worker.finished.connect(self._on_umap_finished)
        self._umap_worker.error.connect(self._on_umap_error)
        self._umap_worker.start()
    
    @Slot(object)
    def _on_umap_finished(self, local_latents):
        """Handle UMAP completion."""
        self._local_latents = local_latents
        self._progress_bar.setVisible(False)
        self._run_umap_btn.setEnabled(True)
        self._run_cluster_btn.setEnabled(True)
        
        # Update embedding plot
        self._embedding_widget.set_embedding(
            local_latents.embedding,
            cluster=getattr(local_latents, 'cluster', None)
        )
    
    @Slot(str)
    def _on_umap_error(self, error_msg: str):
        """Handle UMAP error."""
        self._progress_bar.setVisible(False)
        self._run_umap_btn.setEnabled(True)
        QMessageBox.critical(self, "UMAP Error", error_msg)
    
    @Slot()
    def _run_clustering(self):
        """Run DBSCAN clustering on current embedding."""
        if self._local_latents is None or not hasattr(self._local_latents, 'embedding'):
            return
        
        eps = self._eps_spin.value()
        config = {"eps": eps}
        self._local_latents.build_cluster(method='dbscan', configs=config)
        
        # Update plot with cluster colors
        self._embedding_widget.set_embedding(
            self._local_latents.embedding,
            cluster=self._local_latents.cluster
        )
        
        # Enable labeling
        self._label_btn.setEnabled(True)
        self._submit_btn.setEnabled(True)
    
    @Slot(int, float, float)
    def _on_embedding_click(self, index: int, x: float, y: float):
        """Handle click on embedding plot."""
        if self._aggregator is None or self._local_latents is None:
            return
        
        # Map local index to global index
        global_index = np.arange(len(self._local_latents.index_mask))[self._local_latents.index_mask][index]
        
        # Get frame from aggregator
        frame = self._aggregator.get_frame(global_index)
        if frame is not None:
            self._frame_viewer.show_frame(frame)
    
    @Slot()
    def _label_cluster(self):
        """Label a cluster with a name."""
        if self._local_latents is None:
            return
        
        cluster_id = self._cluster_id_spin.value()
        name = self._cluster_name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Error", "Please enter a cluster name.")
            return
        
        self._local_latents.label_cluster(cluster_id, name)
        
        # Update plot
        self._embedding_widget.set_embedding(
            self._local_latents.embedding,
            cluster=self._local_latents.cluster,
            labels=self._local_latents.export
        )
    
    @Slot()
    def _submit_all(self):
        """Submit all labeled clusters."""
        if self._local_latents is None or self._latents is None:
            return
        
        try:
            self._latents.import_local_latent(self._local_latents)
            self._cluster_tree.set_latent(self._latents)
            
            # Update syllable bar
            self._syllable_bar.set_data(
                self._latents.cluster,
                self._latents.cluster_meta,
                self._aggregator.videos_meta if self._aggregator else [],
                self._aggregator.bin_size if self._aggregator else 1,
                self._aggregator.fps if self._aggregator else 30.0
            )
            
            QMessageBox.information(self, "Success", "Clusters submitted successfully.")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
    
    def _on_preset_changed(self, text: str):
        """Update UMAP config based on preset selection."""
        import re
        numbers = re.findall(r'\d+', text)
        
        if 'Low-mag' in text and numbers:
            config = [{"n_neighbors": int(numbers[0]), "min_dist": 0.0, "n_components": 2, "n_epochs": 5000}]
        elif 'Intermediate' in text and len(numbers) >= 2:
            config = [
                {"n_neighbors": int(numbers[0]), "min_dist": 0.0, "n_components": 5, "n_epochs": 5000},
                {"n_neighbors": int(numbers[1]), "min_dist": 0.0, "n_components": 2, "n_epochs": 5000}
            ]
        elif 'High' in text and len(numbers) >= 2:
            config = [
                {"n_neighbors": int(numbers[0]), "min_dist": 0.0, "n_components": 10, "n_epochs": 5000},
                {"n_neighbors": int(numbers[1]), "min_dist": 0.0, "n_components": 2, "n_epochs": 5000}
            ]
        else:
            return
        
        self._umap_config.setPlainText(json.dumps(config, indent=2))
