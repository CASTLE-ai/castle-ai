"""
CASTLE Desktop Application - QApplication setup and launch.
"""

import sys
import os

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from castle.desktop.main_window import MainWindow


def main():
    """Launch the CASTLE Desktop application."""
    # High DPI support
    os.environ.setdefault('QT_ENABLE_HIGHDPI_SCALING', '1')
    
    app = QApplication(sys.argv)
    app.setApplicationName("CASTLE")
    app.setApplicationVersion("0.1.0")
    app.setOrganizationName("CASTLE Project")
    
    # Set default font
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    # Apply a clean stylesheet
    app.setStyleSheet(_get_stylesheet())
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


def _get_stylesheet() -> str:
    """Return the application stylesheet."""
    return """
    QMainWindow {
        background-color: #1e1e2e;
    }
    QTabWidget::pane {
        border: 1px solid #313244;
        background-color: #1e1e2e;
    }
    QTabBar::tab {
        background-color: #313244;
        color: #cdd6f4;
        padding: 10px 20px;
        margin-right: 2px;
        border-top-left-radius: 4px;
        border-top-right-radius: 4px;
        font-size: 13px;
    }
    QTabBar::tab:selected {
        background-color: #45475a;
        color: #f5e0dc;
        font-weight: bold;
    }
    QTabBar::tab:hover {
        background-color: #585b70;
    }
    QLabel {
        color: #cdd6f4;
    }
    QPushButton {
        background-color: #45475a;
        color: #cdd6f4;
        border: 1px solid #585b70;
        padding: 8px 16px;
        border-radius: 4px;
        font-size: 13px;
    }
    QPushButton:hover {
        background-color: #585b70;
    }
    QPushButton:pressed {
        background-color: #313244;
    }
    QPushButton:disabled {
        background-color: #313244;
        color: #6c7086;
    }
    QPushButton#primaryButton {
        background-color: #89b4fa;
        color: #1e1e2e;
        font-weight: bold;
    }
    QPushButton#primaryButton:hover {
        background-color: #74c7ec;
    }
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit {
        background-color: #313244;
        color: #cdd6f4;
        border: 1px solid #585b70;
        padding: 6px;
        border-radius: 4px;
    }
    QComboBox::drop-down {
        border: none;
    }
    QGroupBox {
        color: #a6adc8;
        border: 1px solid #45475a;
        border-radius: 4px;
        margin-top: 10px;
        padding-top: 15px;
        font-weight: bold;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        padding: 0 5px;
    }
    QStatusBar {
        background-color: #181825;
        color: #a6adc8;
    }
    QTreeWidget {
        background-color: #1e1e2e;
        color: #cdd6f4;
        border: 1px solid #45475a;
        alternate-background-color: #313244;
    }
    QTreeWidget::item:selected {
        background-color: #45475a;
    }
    QSplitter::handle {
        background-color: #45475a;
    }
    QScrollBar:vertical {
        background-color: #1e1e2e;
        width: 12px;
    }
    QScrollBar::handle:vertical {
        background-color: #45475a;
        border-radius: 4px;
        min-height: 20px;
    }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
        height: 0px;
    }
    """
