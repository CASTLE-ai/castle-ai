"""
CASTLE Desktop - Syllable Bar Widget

Displays behavior assignments over time as colored bars.
Each video gets its own row.
"""

import os
from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QPainter, QColor, QPen, QFont


class SyllableBarWidget(QWidget):
    """Widget that displays behavior syllable bars for each video."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._data = None
        self._meta = None
        self._videos = []
        self._bin_size = 1
        self._fps = 30.0
        self.setMinimumHeight(60)
        self.setMaximumHeight(200)

    def set_data(self, cluster, cluster_meta, videos_meta, bin_size, fps):
        """Set the data to display.

        Args:
            cluster: Array of cluster IDs per bin
            cluster_meta: Dict mapping cluster_id to {name, color}
            videos_meta: List of (n_bins, video_name) tuples
            bin_size: Frames per bin
            fps: Frames per second
        """
        self._data = cluster
        self._meta = cluster_meta
        self._videos = videos_meta
        self._bin_size = bin_size
        self._fps = fps
        self.setMinimumHeight(max(60, 30 * len(videos_meta)))
        self.update()

    def paintEvent(self, event):
        if self._data is None or not self._videos:
            super().paintEvent(event)
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()
        n_videos = len(self._videos)
        row_height = h / n_videos if n_videos > 0 else h

        cum = 0
        for video_idx, (n_bins, video_name) in enumerate(self._videos):
            y_top = video_idx * row_height
            video_cluster = self._data[cum:cum + n_bins]

            if len(video_cluster) == 0:
                cum += n_bins
                continue

            # Find transitions
            n = len(video_cluster)
            key_frames = [0]
            for i in range(n - 1):
                if video_cluster[i] != video_cluster[i + 1]:
                    key_frames.append(i + 1)
            key_frames.append(n)

            # Draw bars
            for j in range(len(key_frames) - 1):
                start = key_frames[j]
                end = key_frames[j + 1]
                cid = video_cluster[start]
                color = self._get_color(cid)

                x1 = (start / n) * w
                x2 = (end / n) * w
                painter.fillRect(
                    QRectF(x1, y_top + 2, x2 - x1, row_height - 4),
                    color,
                )

            # Draw video name
            painter.setPen(QPen(QColor("#cdd6f4")))
            painter.setFont(QFont("Segoe UI", 8))
            short_name = os.path.splitext(os.path.basename(video_name))[0]
            painter.drawText(
                QRectF(4, y_top + 2, w - 8, row_height - 4),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                short_name,
            )
            cum += n_bins

        painter.end()

    def _get_color(self, cluster_id):
        if self._meta and cluster_id in self._meta:
            color_str = self._meta[cluster_id].get('color', 'grey')
            try:
                return QColor(color_str)
            except Exception:
                pass
        return QColor(100, 100, 100)
