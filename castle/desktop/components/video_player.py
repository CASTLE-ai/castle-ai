"""
CASTLE Desktop - Video Player / Frame Viewer Widget

Provides:
- FrameViewer: single-frame display (click on embedding → show frame)
- VideoPlayer: frame-by-frame player with play/pause/seek
"""

import numpy as np
import cv2

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSlider, QSpinBox
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap


def _numpy_to_pixmap(frame: np.ndarray, max_width: int = 0) -> QPixmap:
    """Convert an RGB numpy array to QPixmap."""
    if frame is None:
        return QPixmap()
    if frame.dtype != np.uint8:
        frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)

    h, w, ch = frame.shape
    bytes_per_line = ch * w
    fmt = QImage.Format.Format_RGB888 if ch == 3 else QImage.Format.Format_RGBA8888
    qimage = QImage(frame.data.tobytes(), w, h, bytes_per_line, fmt)
    pixmap = QPixmap.fromImage(qimage)

    if max_width > 0 and w > max_width:
        pixmap = pixmap.scaledToWidth(max_width, Qt.TransformationMode.SmoothTransformation)
    return pixmap


class FrameViewer(QWidget):
    """Widget for displaying a single video frame (numpy array)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._label = QLabel("Click a point to view its frame")
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setMinimumSize(280, 280)
        self._label.setStyleSheet(
            "QLabel { background-color: #181825; border: 1px solid #45475a; border-radius: 4px; }"
        )
        layout.addWidget(self._label)

    def show_frame(self, frame: np.ndarray):
        """Display an RGB numpy array."""
        if frame is None:
            self.clear()
            return
        pixmap = _numpy_to_pixmap(frame)
        scaled = pixmap.scaled(
            self._label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._label.setPixmap(scaled)

    def clear(self):
        self._label.clear()
        self._label.setText("Click a point to view its frame")


class VideoPlayer(QWidget):
    """Simple frame-by-frame video player using OpenCV + QTimer."""

    frame_changed = pyqtSignal(int)  # current frame index

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cap = None
        self._total_frames = 0
        self._fps = 30.0
        self._current_frame = 0
        self._playing = False
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._next_frame)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Frame display
        self._display = QLabel()
        self._display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._display.setMinimumSize(400, 300)
        self._display.setStyleSheet(
            "QLabel { background-color: #181825; border: 1px solid #45475a; }"
        )
        layout.addWidget(self._display, stretch=1)

        # Seek slider
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        self._slider.valueChanged.connect(self._on_slider)
        layout.addWidget(self._slider)

        # Controls
        ctrl_layout = QHBoxLayout()
        self._play_btn = QPushButton("▶ Play")
        self._play_btn.clicked.connect(self._toggle_play)
        ctrl_layout.addWidget(self._play_btn)

        self._frame_spin = QSpinBox()
        self._frame_spin.setPrefix("Frame: ")
        self._frame_spin.setMinimum(0)
        self._frame_spin.setMaximum(0)
        self._frame_spin.valueChanged.connect(self._seek_frame)
        ctrl_layout.addWidget(self._frame_spin)

        self._info_label = QLabel("No video loaded")
        ctrl_layout.addWidget(self._info_label)
        ctrl_layout.addStretch()

        layout.addLayout(ctrl_layout)

    # --- Public API ---

    def load_video(self, path: str):
        """Load a video file."""
        self.stop()
        if self._cap is not None:
            self._cap.release()

        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            self._info_label.setText(f"Failed to open: {path}")
            return

        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._current_frame = 0

        self._slider.setMaximum(max(0, self._total_frames - 1))
        self._frame_spin.setMaximum(max(0, self._total_frames - 1))
        self._info_label.setText(
            f"{self._total_frames} frames | {self._fps:.1f} fps"
        )
        self._show_current_frame()

    def stop(self):
        """Stop playback."""
        self._playing = False
        self._timer.stop()
        self._play_btn.setText("▶ Play")

    def get_frame(self, index: int) -> np.ndarray | None:
        """Read a specific frame as RGB numpy array."""
        if self._cap is None:
            return None
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = self._cap.read()
        if not ret:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # --- Private ---

    def _toggle_play(self):
        if self._cap is None:
            return
        if self._playing:
            self.stop()
        else:
            self._playing = True
            self._play_btn.setText("⏸ Pause")
            interval = max(1, int(1000 / self._fps))
            self._timer.start(interval)

    def _next_frame(self):
        if self._current_frame >= self._total_frames - 1:
            self.stop()
            return
        self._current_frame += 1
        self._slider.blockSignals(True)
        self._slider.setValue(self._current_frame)
        self._slider.blockSignals(False)
        self._frame_spin.blockSignals(True)
        self._frame_spin.setValue(self._current_frame)
        self._frame_spin.blockSignals(False)
        self._show_current_frame()
        self.frame_changed.emit(self._current_frame)

    def _on_slider(self, value):
        self._seek_frame(value)

    def _seek_frame(self, index):
        if self._cap is None:
            return
        self._current_frame = index
        # Sync slider and spin without re-triggering
        self._slider.blockSignals(True)
        self._slider.setValue(index)
        self._slider.blockSignals(False)
        self._frame_spin.blockSignals(True)
        self._frame_spin.setValue(index)
        self._frame_spin.blockSignals(False)
        self._show_current_frame()
        self.frame_changed.emit(index)

    def _show_current_frame(self):
        frame = self.get_frame(self._current_frame)
        if frame is not None:
            pixmap = _numpy_to_pixmap(frame)
            scaled = pixmap.scaled(
                self._display.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._display.setPixmap(scaled)

    def closeEvent(self, event):
        self.stop()
        if self._cap is not None:
            self._cap.release()
        super().closeEvent(event)
