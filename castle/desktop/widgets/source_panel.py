"""
CASTLE Desktop - Source Panel (Stage 1)

Video upload and management.
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel


class SourcePanel(QWidget):
    """Panel for video source management."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        
        placeholder = QLabel(
            "<h2>📹 Upload Videos</h2>"
            "<p>This panel will allow you to:</p>"
            "<ul>"
            "<li>Upload local video files</li>"
            "<li>Import videos from a server directory</li>"
            "<li>Preview video metadata</li>"
            "</ul>"
            "<p><i>Coming soon in a future release.</i></p>"
        )
        placeholder.setWordWrap(True)
        layout.addWidget(placeholder)
        layout.addStretch()
