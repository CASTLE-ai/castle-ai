"""
CASTLE Desktop - Frame Viewer Widget

Displays individual video frames (RGB numpy arrays).
Used for inspecting frames corresponding to clicked embedding points.
"""

import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap


class FrameViewer(QWidget):
    """Widget for displaying video frames."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self._label = QLabel("Click on a point in the embedding to view its frame")
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setMinimumSize(300, 300)
        self._label.setStyleSheet(
            "QLabel { background-color: #181825; border: 1px solid #45475a; border-radius: 4px; }"
        )
        layout.addWidget(self._label)
    
    def show_frame(self, frame: np.ndarray):
        """Display an RGB numpy array as an image.
        
        Args:
            frame: RGB image array of shape (H, W, 3), dtype uint8
        """
        if frame is None:
            self.clear()
            return
        
        # Ensure correct format
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
        
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        
        # Create QImage from numpy array
        qimage = QImage(
            frame.data.tobytes(),
            w, h,
            bytes_per_line,
            QImage.Format.Format_RGB888
        )
        
        # Scale to fit label while maintaining aspect ratio
        pixmap = QPixmap.fromImage(qimage)
        scaled = pixmap.scaled(
            self._label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        self._label.setPixmap(scaled)
    
    def clear(self):
        """Clear the displayed frame."""
        self._label.clear()
        self._label.setText("Click on a point in the embedding to view its frame")
