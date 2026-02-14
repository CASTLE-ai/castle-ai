"""
CASTLE Desktop - Extract Panel (Stage 3)

Latent feature extraction from tracked ROIs.
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel


class ExtractPanel(QWidget):
    """Panel for latent feature extraction."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        
        placeholder = QLabel(
            "<h2>🧬 Extract Latent Features</h2>"
            "<p>This panel will allow you to:</p>"
            "<ul>"
            "<li>Configure visual model (DINOv2/DINOv3)</li>"
            "<li>Set preprocessing options</li>"
            "<li>Run batch latent extraction</li>"
            "<li>Monitor extraction progress</li>"
            "</ul>"
            "<p><i>Coming soon in a future release.</i></p>"
        )
        placeholder.setWordWrap(True)
        layout.addWidget(placeholder)
        layout.addStretch()
