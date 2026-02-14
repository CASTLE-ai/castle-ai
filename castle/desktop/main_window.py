"""
CASTLE Desktop - Main Window

Provides the main application window with tab navigation matching
the 5 stages of the CASTLE pipeline.
"""

from PySide6.QtWidgets import (
    QMainWindow, QTabWidget, QWidget, QVBoxLayout,
    QStatusBar, QLabel, QMenuBar, QMenu
)
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QAction

from castle.desktop.widgets.project_panel import ProjectPanel
from castle.desktop.widgets.source_panel import SourcePanel
from castle.desktop.widgets.tracking_panel import TrackingPanel
from castle.desktop.widgets.extract_panel import ExtractPanel
from castle.desktop.widgets.microscope_panel import MicroscopePanel


class MainWindow(QMainWindow):
    """Main application window for CASTLE Desktop."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CASTLE - Combined Approach for Segmentation and Tracking with Latent Extraction")
        self.setMinimumSize(QSize(1200, 800))
        self.resize(1400, 900)
        
        # State
        self._storage_path = "projects/"
        self._project_name = None
        
        self._setup_menubar()
        self._setup_ui()
        self._setup_statusbar()
    
    def _setup_menubar(self):
        """Create the menu bar."""
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
        
        # View menu
        view_menu = menubar.addMenu("&View")
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        about_action = QAction("&About CASTLE", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
    
    def _setup_ui(self):
        """Create the main UI layout with tab widget."""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Tab widget
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.TabPosition.North)
        self.tabs.setDocumentMode(True)
        
        # Create panels for each stage
        self.project_panel = ProjectPanel(self)
        self.source_panel = SourcePanel(self)
        self.tracking_panel = TrackingPanel(self)
        self.extract_panel = ExtractPanel(self)
        self.microscope_panel = MicroscopePanel(self)
        
        # Add tabs
        self.tabs.addTab(self.project_panel, "0. Project")
        self.tabs.addTab(self.source_panel, "1. Upload Videos")
        self.tabs.addTab(self.tracking_panel, "2. Tracking ROIs")
        self.tabs.addTab(self.extract_panel, "3. Extract Latent")
        self.tabs.addTab(self.microscope_panel, "4. Behavior Microscope")
        
        # Disable non-project tabs initially
        for i in range(1, self.tabs.count()):
            self.tabs.setTabEnabled(i, False)
        
        # Connect project selection
        self.project_panel.project_opened.connect(self._on_project_opened)
        
        layout.addWidget(self.tabs)
    
    def _setup_statusbar(self):
        """Create the status bar."""
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        
        self._status_label = QLabel("No project loaded")
        self.statusbar.addWidget(self._status_label)
        
        # Version label on the right
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
        
        # Notify panels
        self.microscope_panel.set_project(storage_path, project_name)
        
        # Switch to appropriate tab
        self.statusbar.showMessage(f"Project '{project_name}' loaded successfully", 3000)
    
    @property
    def storage_path(self) -> str:
        return self._storage_path
    
    @property
    def project_name(self) -> str:
        return self._project_name
    
    def _show_about(self):
        """Show about dialog."""
        from PySide6.QtWidgets import QMessageBox
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
