"""
CASTLE Desktop - Main Window

Provides the main application window with tab navigation matching
the CASTLE pipeline stages.
"""

from PyQt6.QtWidgets import (
    QMainWindow, QTabWidget, QWidget, QVBoxLayout,
    QStatusBar, QLabel, QMessageBox
)
from PyQt6.QtCore import QSize
from PyQt6.QtGui import QAction

from castle.desktop.widgets.project_panel import ProjectPanel
from castle.desktop.widgets.source_panel import SourcePanel
from castle.desktop.widgets.tracking_panel import TrackingPanel
from castle.desktop.widgets.extract_panel import ExtractPanel
from castle.desktop.widgets.cluster_panel import ClusterPanel
from castle.desktop.widgets.annotator_panel import AnnotatorPanel
from castle.desktop.widgets.analysis_panel import AnalysisPanel
from castle.desktop.widgets.export_panel import ExportPanel


class MainWindow(QMainWindow):
    """Main application window for CASTLE Desktop."""

    def __init__(self, storage_path="projects/", project_name=None):
        super().__init__()
        self.setWindowTitle(
            "CASTLE — Combined Approach for Segmentation and Tracking "
            "with Latent Extraction"
        )
        self.setMinimumSize(QSize(1200, 800))
        self.resize(1400, 900)

        # State
        self._storage_path = storage_path
        self._project_name = None

        self._setup_menubar()
        self._setup_ui()
        self._setup_statusbar()

        # Auto-open project if specified
        if project_name:
            self._on_project_opened(storage_path, project_name)

    def _setup_menubar(self):
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        new_project_action = QAction("&New Project...", self)
        new_project_action.setShortcut("Ctrl+N")
        file_menu.addAction(new_project_action)

        open_project_action = QAction("&Open Project...", self)
        open_project_action.setShortcut("Ctrl+O")
        file_menu.addAction(open_project_action)

        file_menu.addSeparator()

        quit_action = QAction("&Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # Help menu
        help_menu = menubar.addMenu("&Help")
        about_action = QAction("&About CASTLE", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)

        # Tab widget
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.TabPosition.North)
        self.tabs.setDocumentMode(True)

        # Create panels
        self.project_panel = ProjectPanel(self, storage_path=self._storage_path)
        self.source_panel = SourcePanel(self)
        self.tracking_panel = TrackingPanel(self)
        self.extract_panel = ExtractPanel(self)
        self.cluster_panel = ClusterPanel(self)
        self.annotator_panel = AnnotatorPanel(self)
        self.analysis_panel = AnalysisPanel(self)
        self.export_panel = ExportPanel(self)

        # Add tabs
        self.tabs.addTab(self.project_panel, "0. Project")
        self.tabs.addTab(self.source_panel, "1. Upload Videos")
        self.tabs.addTab(self.tracking_panel, "2. Tracking ROIs")
        self.tabs.addTab(self.extract_panel, "3. Extract Latent")
        self.tabs.addTab(self.cluster_panel, "4. Behavior Microscope")
        self.tabs.addTab(self.annotator_panel, "5. Annotator")
        self.tabs.addTab(self.analysis_panel, "6. Analysis")
        self.tabs.addTab(self.export_panel, "7. Export")

        # Disable non-project tabs initially
        for i in range(1, self.tabs.count()):
            self.tabs.setTabEnabled(i, False)

        # Connect signals
        self.project_panel.project_opened.connect(self._on_project_opened)

        layout.addWidget(self.tabs)

    def _setup_statusbar(self):
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)

        self._status_label = QLabel("No project loaded")
        self.statusbar.addWidget(self._status_label)

        version_label = QLabel("CASTLE Desktop v0.1.0")
        self.statusbar.addPermanentWidget(version_label)

    def _on_project_opened(self, storage_path: str, project_name: str):
        """Handle project selection."""
        self._storage_path = storage_path
        self._project_name = project_name

        # Enable all tabs
        for i in range(1, self.tabs.count()):
            self.tabs.setTabEnabled(i, True)

        # Update status
        self._status_label.setText(f"Project: {project_name}")

        # Notify all panels
        self.source_panel.set_project(storage_path, project_name)
        self.tracking_panel.set_project(storage_path, project_name)
        self.extract_panel.set_project(storage_path, project_name)
        self.cluster_panel.set_project(storage_path, project_name)
        self.annotator_panel.set_project(storage_path, project_name)
        self.analysis_panel.set_project(storage_path, project_name)
        self.export_panel.set_project(storage_path, project_name)

        self.statusbar.showMessage(
            f"Project '{project_name}' loaded successfully", 3000
        )

    @property
    def storage_path(self) -> str:
        return self._storage_path

    @property
    def project_name(self) -> str:
        return self._project_name

    def _show_about(self):
        QMessageBox.about(
            self,
            "About CASTLE",
            "<h2>CASTLE Desktop</h2>"
            "<p>Combined Approach for Segmentation and Tracking "
            "with Latent Extraction</p>"
            "<p>Version 0.1.0</p>"
            "<p>An animal behavior analysis framework using AI-powered "
            "ROI tracking and unsupervised clustering.</p>"
        )
