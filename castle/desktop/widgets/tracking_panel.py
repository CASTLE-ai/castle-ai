"""
CASTLE Desktop - Tracking Panel (Stage 2)

ROI annotation and tracking.
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel


class TrackingPanel(QWidget):
    """Panel for ROI tracking."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        
        placeholder = QLabel(
            "<h2>🎯 Tracking ROIs</h2>"
            "<p>This panel will allow you to:</p>"
            "<ul>"
            "<li>Annotate ROI prompts on video frames</li>"
            "<li>Run single-video and batch tracking</li>"
            "<li>View and refine tracking results</li>"
            "</ul>"
            "<p><i>Coming soon in a future release.</i></p>"
        )
        placeholder.setWordWrap(True)
        layout.addWidget(placeholder)
        layout.addStretch()
