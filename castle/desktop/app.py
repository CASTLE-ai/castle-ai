"""
CASTLE Desktop Application - QApplication setup and launch.
"""

import sys
import os
import argparse

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QFont

from castle.desktop.main_window import MainWindow


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="CASTLE Desktop GUI")
    parser.add_argument("--storage", type=str, default="projects/",
                        help="Storage directory path (default: projects/)")
    parser.add_argument("--project", type=str, default=None,
                        help="Project name to open on launch")
    return parser.parse_args()


def main():
    """Launch the CASTLE Desktop application."""
    args = parse_args()

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

    window = MainWindow(storage_path=args.storage, project_name=args.project)
    window.show()

    sys.exit(app.exec())


def _get_stylesheet() -> str:
    """Return the application stylesheet (Catppuccin Mocha dark theme)."""
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
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit, QPlainTextEdit {
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
    QListWidget {
        background-color: #1e1e2e;
        color: #cdd6f4;
        border: 1px solid #45475a;
        alternate-background-color: #313244;
    }
    QListWidget::item:selected {
        background-color: #45475a;
    }
    QProgressBar {
        background-color: #313244;
        border: 1px solid #45475a;
        border-radius: 4px;
        text-align: center;
        color: #cdd6f4;
    }
    QProgressBar::chunk {
        background-color: #89b4fa;
        border-radius: 3px;
    }
    QSplitter::handle {
        background-color: #45475a;
    }
    QCheckBox {
        color: #cdd6f4;
    }
    QRadioButton {
        color: #cdd6f4;
    }
    QSlider::groove:horizontal {
        background-color: #313244;
        height: 6px;
        border-radius: 3px;
    }
    QSlider::handle:horizontal {
        background-color: #89b4fa;
        width: 14px;
        margin: -4px 0;
        border-radius: 7px;
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
    QScrollBar:horizontal {
        background-color: #1e1e2e;
        height: 12px;
    }
    QScrollBar::handle:horizontal {
        background-color: #45475a;
        border-radius: 4px;
        min-width: 20px;
    }
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
        width: 0px;
    }
    """
