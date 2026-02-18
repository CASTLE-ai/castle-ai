"""
CASTLE Desktop - Analysis Panel (Tab 6)

Ethogram analysis (transition matrix, bout stats) and clustering quality
metrics.  Mirrors the Gradio analysis_ui tab.

Uses matplotlib embedded in Qt via FigureCanvasQTAgg.
"""

import logging

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QComboBox, QTableWidget, QTableWidgetItem,
    QProgressBar, QMessageBox, QSplitter, QTabWidget,
    QHeaderView, QAbstractItemView,
)
from PyQt6.QtCore import Qt, pyqtSlot

from castle.desktop.services.worker_threads import ServiceWorker

logger = logging.getLogger(__name__)


def _make_canvas():
    """Create a blank matplotlib Figure + FigureCanvasQTAgg."""
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend before importing pyplot
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

    fig = Figure(tight_layout=True)
    canvas = FigureCanvasQTAgg(fig)
    return fig, canvas


class AnalysisPanel(QWidget):
    """Ethogram and quality-metrics analysis panel."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self._storage_path: str | None = None
        self._project_name: str | None = None
        self._annotator_data = None
        self._worker: ServiceWorker | None = None

        self._setup_ui()

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Session selector
        layout.addWidget(self._create_session_group())

        self._status_label = QLabel("Status: Not loaded")
        layout.addWidget(self._status_label)

        # Sub-tabs: Ethogram | Quality Metrics
        self._sub_tabs = QTabWidget()
        self._sub_tabs.addTab(self._create_ethogram_tab(), "📊 Ethogram")
        self._sub_tabs.addTab(self._create_metrics_tab(), "📐 Quality Metrics")
        layout.addWidget(self._sub_tabs, stretch=1)

        # Progress bar
        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)
        layout.addWidget(self._progress_bar)

    def _create_session_group(self) -> QGroupBox:
        group = QGroupBox("Session")
        layout = QHBoxLayout(group)

        layout.addWidget(QLabel("Session:"))
        self._session_combo = QComboBox()
        self._session_combo.setMinimumWidth(300)
        layout.addWidget(self._session_combo, stretch=1)

        self._refresh_btn = QPushButton("🔄 Refresh")
        self._refresh_btn.clicked.connect(self._refresh_sessions)
        layout.addWidget(self._refresh_btn)

        self._load_btn = QPushButton("📂 Load Data")
        self._load_btn.setObjectName("primaryButton")
        self._load_btn.clicked.connect(self._load_data)
        layout.addWidget(self._load_btn)

        return group

    # ---- Ethogram tab ----

    def _create_ethogram_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Generate button
        btn_row = QHBoxLayout()
        self._ethogram_btn = QPushButton("▶ Generate Ethogram")
        self._ethogram_btn.setObjectName("primaryButton")
        self._ethogram_btn.clicked.connect(self._generate_ethogram)
        self._ethogram_btn.setEnabled(False)
        btn_row.addWidget(self._ethogram_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        # Plots side-by-side
        plot_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Transition matrix canvas
        self._tm_fig, self._tm_canvas = _make_canvas()
        plot_splitter.addWidget(self._tm_canvas)

        # Raster canvas
        self._raster_fig, self._raster_canvas = _make_canvas()
        plot_splitter.addWidget(self._raster_canvas)

        layout.addWidget(plot_splitter, stretch=1)

        # Bout stats table
        layout.addWidget(QLabel("Bout Duration Statistics:"))
        self._bout_table = self._make_table([
            "Cluster", "N Bouts", "Freq (%)",
            "Mean Dur (s)", "Median Dur (s)", "Std Dur (s)", "CV", "Mean IBI (s)",
        ])
        layout.addWidget(self._bout_table)

        return widget

    # ---- Quality Metrics tab ----

    def _create_metrics_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        btn_row = QHBoxLayout()
        self._metrics_btn = QPushButton("▶ Compute Metrics")
        self._metrics_btn.setObjectName("primaryButton")
        self._metrics_btn.clicked.connect(self._compute_metrics)
        self._metrics_btn.setEnabled(False)
        btn_row.addWidget(self._metrics_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self._metrics_table = self._make_table(["Metric", "Value", "Note"])
        layout.addWidget(self._metrics_table, stretch=1)

        return widget

    @staticmethod
    def _make_table(headers: list) -> QTableWidget:
        table = QTableWidget(0, len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        return table

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_project(self, storage_path: str, project_name: str):
        self._storage_path = storage_path
        self._project_name = project_name
        self._annotator_data = None
        self._status_label.setText("Status: Project loaded — click 'Load Data'")
        self._ethogram_btn.setEnabled(False)
        self._metrics_btn.setEnabled(False)
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
                    label = (
                        f"{s.name} — {s.n_clusters} clusters, "
                        f"bin_size={s.bin_size} ({s.updated_at[:16]})"
                    )
                    self._session_combo.addItem(label, s.session_id)
                active_id = mgr.get_active_session_id()
                if active_id:
                    for i in range(self._session_combo.count()):
                        if self._session_combo.itemData(i) == active_id:
                            self._session_combo.setCurrentIndex(i)
                            break
        except Exception as exc:
            logger.warning("Failed to refresh sessions: %s", exc)

    @pyqtSlot()
    def _load_data(self):
        if not self._storage_path or not self._project_name:
            QMessageBox.warning(self, "Error", "No project selected.")
            return

        session_id = self._session_combo.currentData() or None
        self._progress_bar.setVisible(True)
        self._progress_bar.setRange(0, 0)
        self._load_btn.setEnabled(False)
        self._status_label.setText("Status: Loading…")

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
        self._ethogram_btn.setEnabled(True)
        self._metrics_btn.setEnabled(True)

        n_clusters = len(annotator_data.cluster_meta)
        n_bins = len(annotator_data.cluster)
        self._status_label.setText(
            f"Loaded: {n_clusters} clusters, {n_bins} bins "
            f"(bin_size={annotator_data.bin_size}, fps={annotator_data.fps:.1f})"
        )

    @pyqtSlot(str)
    def _on_load_error(self, err: str):
        self._progress_bar.setVisible(False)
        self._load_btn.setEnabled(True)
        self._status_label.setText(f"Error: {err}")
        QMessageBox.critical(self, "Load Error", err)

    # ---- Ethogram ----

    @pyqtSlot()
    def _generate_ethogram(self):
        if self._annotator_data is None:
            QMessageBox.warning(self, "Error", "Load data first.")
            return

        self._ethogram_btn.setEnabled(False)
        self._progress_bar.setVisible(True)
        self._progress_bar.setRange(0, 0)
        self._status_label.setText("Computing ethogram…")

        self._worker = ServiceWorker(self._compute_ethogram_worker, self._annotator_data)
        self._worker.finished.connect(self._on_ethogram_done)
        self._worker.error.connect(self._on_worker_error)
        self._worker.start()

    @staticmethod
    def _compute_ethogram_worker(annotator_data):
        """Run in background thread — returns (ethogram, heatmap_bytes, raster_bytes)."""
        import io
        import matplotlib
        matplotlib.use("Agg")

        cluster_labels = annotator_data.cluster
        fps = annotator_data.fps or 30.0
        cluster_names = {
            cid: meta["name"]
            for cid, meta in annotator_data.cluster_meta.items()
        }

        from castle.core.ethogram import compute_ethogram
        ethogram = compute_ethogram(cluster_labels, fps=fps, cluster_names=cluster_names)

        from castle.visualization.ethogram_plots import (
            plot_transition_heatmap,
            plot_ethogram_raster,
        )
        heatmap_fig = plot_transition_heatmap(ethogram.transition_matrix)
        raster_fig = plot_ethogram_raster(ethogram)

        # Render figures to PNG bytes so they are safely passed across threads
        heatmap_buf = io.BytesIO()
        heatmap_fig.savefig(heatmap_buf, format="png", bbox_inches="tight")
        heatmap_buf.seek(0)

        raster_buf = io.BytesIO()
        raster_fig.savefig(raster_buf, format="png", bbox_inches="tight")
        raster_buf.seek(0)

        import matplotlib.pyplot as plt
        plt.close(heatmap_fig)
        plt.close(raster_fig)

        return ethogram, heatmap_buf.read(), raster_buf.read()

    @pyqtSlot(object)
    def _on_ethogram_done(self, result):
        self._progress_bar.setVisible(False)
        self._ethogram_btn.setEnabled(True)

        ethogram, heatmap_png, raster_png = result

        # Display transition matrix PNG in the canvas
        self._tm_fig.clf()
        ax = self._tm_fig.add_subplot(111)
        import io
        from PIL import Image
        ax.imshow(Image.open(io.BytesIO(heatmap_png)))
        ax.axis("off")
        self._tm_canvas.draw()

        # Display raster PNG
        self._raster_fig.clf()
        ax2 = self._raster_fig.add_subplot(111)
        ax2.imshow(Image.open(io.BytesIO(raster_png)))
        ax2.axis("off")
        self._raster_canvas.draw()

        # Populate bout stats table
        self._bout_table.setRowCount(0)
        for cid in sorted(ethogram.bout_stats.keys()):
            bs = ethogram.bout_stats[cid]
            row = self._bout_table.rowCount()
            self._bout_table.insertRow(row)
            values = [
                bs.cluster_name,
                str(bs.n_bouts),
                f"{bs.frequency * 100:.1f}",
                f"{bs.mean_duration_s:.3f}",
                f"{bs.median_duration_s:.3f}",
                f"{bs.std_duration_s:.3f}",
                f"{bs.cv_duration:.3f}",
                f"{bs.mean_inter_bout_interval_s:.3f}",
            ]
            for col, val in enumerate(values):
                self._bout_table.setItem(row, col, QTableWidgetItem(val))

        self._status_label.setText("Ethogram generated.")

    # ---- Quality Metrics ----

    @pyqtSlot()
    def _compute_metrics(self):
        if self._annotator_data is None:
            QMessageBox.warning(self, "Error", "Load data first.")
            return

        self._metrics_btn.setEnabled(False)
        self._progress_bar.setVisible(True)
        self._progress_bar.setRange(0, 0)
        self._status_label.setText("Computing quality metrics…")

        self._worker = ServiceWorker(self._compute_metrics_worker, self._annotator_data)
        self._worker.finished.connect(self._on_metrics_done)
        self._worker.error.connect(self._on_worker_error)
        self._worker.start()

    @staticmethod
    def _compute_metrics_worker(annotator_data):
        """Run in background thread — returns list of (metric, value, note) rows."""
        cluster_labels = annotator_data.cluster
        embedding = annotator_data.embedding

        from castle.core.metrics import evaluate_clustering
        report = evaluate_clustering(
            labels=cluster_labels,
            embedding=(
                embedding
                if embedding is not None and len(embedding) > 0
                else None
            ),
            fps=annotator_data.fps or 30.0,
        )

        def _fmt(v):
            if v is None:
                return "N/A"
            if isinstance(v, float):
                return f"{v:.4f}"
            return str(v)

        rows = [
            ("Temporal Coherence", _fmt(report.temporal_coherence),
             "↑ better  (>0.95 = GOOD)"),
            ("Silhouette Score", _fmt(report.silhouette_sample),
             "↑ better  (>0 = clusters separated)"),
            ("Calinski-Harabasz", _fmt(report.calinski_harabasz),
             "↑ better  (higher = more compact)"),
            ("Davies-Bouldin", _fmt(report.davies_bouldin),
             "↓ better  (<1 = well separated)"),
            ("Single-Frame Bout Ratio", _fmt(report.single_frame_ratio),
             "↓ better  (<0.1 = stable bouts)"),
            ("Median Bout Duration (frames)", _fmt(report.median_bout_duration_frames), ""),
            ("Bout Duration CV", _fmt(report.bout_duration_cv), ""),
            ("Verdict", report.verdict, ""),
        ]
        if report.warnings:
            for w in report.warnings:
                rows.append(("⚠ Warning", w, ""))
        return rows

    @pyqtSlot(object)
    def _on_metrics_done(self, rows):
        self._progress_bar.setVisible(False)
        self._metrics_btn.setEnabled(True)

        self._metrics_table.setRowCount(0)
        for metric, value, note in rows:
            row = self._metrics_table.rowCount()
            self._metrics_table.insertRow(row)
            self._metrics_table.setItem(row, 0, QTableWidgetItem(metric))
            self._metrics_table.setItem(row, 1, QTableWidgetItem(value))
            self._metrics_table.setItem(row, 2, QTableWidgetItem(note))

        self._status_label.setText("Quality metrics computed.")

    @pyqtSlot(str)
    def _on_worker_error(self, err: str):
        self._progress_bar.setVisible(False)
        self._ethogram_btn.setEnabled(True)
        self._metrics_btn.setEnabled(True)
        self._status_label.setText(f"Error: {err}")
        QMessageBox.critical(self, "Error", err)
