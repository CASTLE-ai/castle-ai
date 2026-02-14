"""
CASTLE Desktop - Project Panel (Stage 0)

Project creation, opening, and management.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QListWidget, QGroupBox, QMessageBox, QFileDialog
)
from PySide6.QtCore import Signal

from castle.utils.project_manager import (
    list_projects, create_project, delete_project,
    generate_default_project_name, initialize_storage
)


class ProjectPanel(QWidget):
    """Panel for project management."""
    
    # Signal emitted when a project is opened: (storage_path, project_name)
    project_opened = Signal(str, str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._storage_path = "projects/"
        self._setup_ui()
        self._refresh_projects()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        
        # Welcome section
        welcome = QLabel(
            "<h2>🏰 Welcome to CASTLE Desktop</h2>"
            "<p>Create or open a project to begin analyzing animal behavior.</p>"
        )
        welcome.setWordWrap(True)
        layout.addWidget(welcome)
        
        # Storage path
        storage_group = QGroupBox("Storage Location")
        storage_layout = QHBoxLayout(storage_group)
        self._storage_input = QLineEdit(self._storage_path)
        self._storage_browse_btn = QPushButton("Browse...")
        self._storage_browse_btn.clicked.connect(self._browse_storage)
        self._storage_input.textChanged.connect(self._on_storage_changed)
        storage_layout.addWidget(self._storage_input, stretch=4)
        storage_layout.addWidget(self._storage_browse_btn, stretch=1)
        layout.addWidget(storage_group)
        
        # Project list
        projects_group = QGroupBox("Existing Projects")
        projects_layout = QVBoxLayout(projects_group)
        
        self._project_list = QListWidget()
        self._project_list.setAlternatingRowColors(True)
        self._project_list.itemDoubleClicked.connect(self._open_selected_project)
        projects_layout.addWidget(self._project_list)
        
        btn_layout = QHBoxLayout()
        self._open_btn = QPushButton("Open Project")
        self._open_btn.setObjectName("primaryButton")
        self._open_btn.clicked.connect(self._open_selected_project)
        self._open_btn.setEnabled(False)
        
        self._delete_btn = QPushButton("Delete")
        self._delete_btn.clicked.connect(self._delete_selected_project)
        self._delete_btn.setEnabled(False)
        
        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.clicked.connect(self._refresh_projects)
        
        btn_layout.addWidget(self._open_btn)
        btn_layout.addWidget(self._delete_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(self._refresh_btn)
        projects_layout.addLayout(btn_layout)
        
        layout.addWidget(projects_group)
        
        # Create new project
        new_group = QGroupBox("Create New Project")
        new_layout = QHBoxLayout(new_group)
        self._new_name_input = QLineEdit()
        self._new_name_input.setPlaceholderText("Enter project name...")
        self._create_btn = QPushButton("Create")
        self._create_btn.setObjectName("primaryButton")
        self._create_btn.clicked.connect(self._create_project)
        new_layout.addWidget(self._new_name_input, stretch=4)
        new_layout.addWidget(self._create_btn, stretch=1)
        layout.addWidget(new_group)
        
        layout.addStretch()
        
        # Connect selection changes
        self._project_list.currentRowChanged.connect(self._on_selection_changed)
    
    def _on_selection_changed(self, row):
        has_selection = row >= 0
        self._open_btn.setEnabled(has_selection)
        self._delete_btn.setEnabled(has_selection)
    
    def _on_storage_changed(self, text):
        self._storage_path = text
        self._refresh_projects()
    
    def _browse_storage(self):
        path = QFileDialog.getExistingDirectory(self, "Select Storage Directory")
        if path:
            self._storage_input.setText(path + "/")
    
    def _refresh_projects(self):
        self._project_list.clear()
        try:
            self._storage_path = initialize_storage(self._storage_path)
            projects = list_projects(self._storage_path)
            self._project_list.addItems(projects)
        except Exception as e:
            pass  # Storage path might not exist yet
    
    def _open_selected_project(self):
        current_item = self._project_list.currentItem()
        if current_item:
            project_name = current_item.text()
            self.project_opened.emit(self._storage_path, project_name)
    
    def _create_project(self):
        name = self._new_name_input.text().strip()
        if not name:
            name = generate_default_project_name()
        
        try:
            create_project(self._storage_path, name)
            self._new_name_input.clear()
            self._refresh_projects()
            # Auto-select and open the new project
            items = self._project_list.findItems(name, self._project_list.model().match(
                self._project_list.model().index(0, 0), 0, name
            )[0] if False else 0)
            self.project_opened.emit(self._storage_path, name)
        except FileExistsError:
            QMessageBox.warning(self, "Error", f"Project '{name}' already exists.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create project: {e}")
    
    def _delete_selected_project(self):
        current_item = self._project_list.currentItem()
        if not current_item:
            return
        
        project_name = current_item.text()
        reply = QMessageBox.question(
            self, "Confirm Delete",
            f"Are you sure you want to delete project '{project_name}'?\n"
            "This action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            delete_project(self._storage_path, project_name)
            self._refresh_projects()
