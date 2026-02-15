"""
CASTLE Desktop - Embedding Visualization Widget

Uses pyqtgraph ScatterPlotItem for interactive UMAP embedding visualization.
Advantages over matplotlib-based Gradio approach:
- Real-time pan/zoom without server round-trips
- Click-to-inspect with instant response
- Handles 10K-100K points smoothly
"""

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QWidget, QVBoxLayout
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QColor


# Default palette (fallback if core.config unavailable)
_FALLBACK_PALETTE = [
    '#7AE4F0', '#FFD0EC', '#6EE368', '#C1B5EA', '#A7CCED',
    '#F0E07A', '#E3686E', '#68E3C1', '#EA9FB5', '#9FD4EA',
    '#D4EA9F', '#EA9FD4', '#9FEAD4', '#EAC19F', '#9FC1EA',
    '#C1EA9F', '#EA9FC1', '#9FEAC1',
]

try:
    from castle.core.config import PALETTE_HEX
except ImportError:
    PALETTE_HEX = _FALLBACK_PALETTE


class EmbeddingWidget(QWidget):
    """Interactive scatter plot for UMAP embeddings."""

    # Signal: (local_index, x, y)
    point_clicked = pyqtSignal(int, float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._data = None
        self._cluster = None
        self._selected_index = -1
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        pg.setConfigOptions(
            background='#1e1e2e',
            foreground='#cdd6f4',
            antialias=True,
        )

        self._plot_widget = pg.PlotWidget()
        self._plot_widget.setAspectLocked(False)
        self._plot_widget.showGrid(x=False, y=False)
        self._plot_widget.hideAxis('left')
        self._plot_widget.hideAxis('bottom')

        # Main scatter
        self._scatter = pg.ScatterPlotItem(
            size=5,
            pen=pg.mkPen(None),
            brush=pg.mkBrush(150, 150, 150, 120),
            hoverable=True,
            hoverSize=8,
            hoverBrush=pg.mkBrush(255, 255, 255, 200),
        )
        self._scatter.sigClicked.connect(self._on_scatter_clicked)
        self._plot_widget.addItem(self._scatter)

        # Selected point marker
        self._selected_marker = pg.ScatterPlotItem(
            size=12,
            pen=pg.mkPen('r', width=2),
            brush=pg.mkBrush(255, 0, 0, 200),
            symbol='x',
        )
        self._plot_widget.addItem(self._selected_marker)

        layout.addWidget(self._plot_widget)

    def set_embedding(self, embedding: np.ndarray, cluster=None, labels=None):
        """Update the scatter plot with new embedding data.

        Args:
            embedding: 2D array of shape (N, 2)
            cluster: Optional array of cluster IDs (N,)
            labels: Optional dict mapping cluster_id to {name, color}
        """
        self._data = embedding
        self._cluster = cluster

        if embedding is None or len(embedding) == 0:
            self._scatter.setData([])
            return

        if cluster is not None:
            brushes = []
            for c in cluster:
                if c == -1:
                    brushes.append(pg.mkBrush(100, 100, 100, 80))
                else:
                    color = self._get_cluster_color(c, labels)
                    brushes.append(pg.mkBrush(color))

            self._scatter.setData(
                x=embedding[:, 0], y=embedding[:, 1],
                brush=brushes, size=5, pen=pg.mkPen(None),
            )
        else:
            self._scatter.setData(
                x=embedding[:, 0], y=embedding[:, 1],
                brush=pg.mkBrush(150, 150, 150, 120),
                size=5, pen=pg.mkPen(None),
            )

        self._plot_widget.autoRange()

    def clear(self):
        """Clear the plot."""
        self._scatter.setData([])
        self._selected_marker.setData([])
        self._data = None
        self._cluster = None

    def _get_cluster_color(self, cluster_id: int, labels=None) -> QColor:
        if labels and cluster_id in labels:
            hex_color = labels[cluster_id].get('color', '')
            if hex_color:
                return QColor(hex_color)
        palette = PALETTE_HEX
        hex_c = palette[cluster_id % len(palette)]
        color = QColor(hex_c)
        color.setAlpha(180)
        return color

    def _on_scatter_clicked(self, plot, points, ev):
        if len(points) == 0:
            return
        point = points[0]
        pos = point.pos()
        x, y = pos.x(), pos.y()

        if self._data is not None:
            distances = np.sum((self._data - np.array([x, y])) ** 2, axis=1)
            index = int(np.argmin(distances))
            self._selected_index = index

            self._selected_marker.setData(
                x=[self._data[index, 0]],
                y=[self._data[index, 1]],
            )
            self.point_clicked.emit(index, float(x), float(y))
