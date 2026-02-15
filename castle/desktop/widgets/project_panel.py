"""
CASTLE Desktop - Project Panel (Tab 0 / part of Source tab)

Project creation, opening, and management — uses the service layer.
"""

import os

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QListWidget, QGroupBox, QMessageBox, QFileDialog,
    QTextEdit
)
from PyQt6.QtCore import pyqtSignal

from castle.service.project_service import (
    list_projects as svc_list_projects,
    create_project as svc_create_project,
    get_project_info as svc_get_project_info,
)


class ProjectPanel(QWidget):
    """Panel for project management."""

    # Signal emitted when a project is opened: (storage_path, project_name)
    project_opened = pyqtSignal(str, str)

    def __init__(self, parent=None, storage_path="projects/"):
        super().__init__(parent)
        self._storage_path = storage_path
        self._setup_ui()
        self._refresh_projects()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)

        # Welcome
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

        # Project info display
        info_group = QGroupBox("Project Info")
        info_layout = QVBoxLayout(info_group)
        self._info_text = QTextEdit()
        self._info_text.setReadOnly(True)
        self._info_text.setMaximumHeight(120)
        self._info_text.setPlaceholderText("Select a project to see details...")
        info_layout.addWidget(self._info_text)
        layout.addWidget(info_group)

        layout.addStretch()

        # Selection change
        self._project_list.currentRowChanged.connect(self._on_selection_changed)

    @property
    def storage_path(self) -> str:
        return self._storage_path

    def set_storage_path(self, path: str):
        self._storage_input.setText(path)

    def _on_selection_changed(self, row):
        has_selection = row >= 0
        self._open_btn.setEnabled(has_selection)
        self._delete_btn.setEnabled(has_selection)
        if has_selection:
            self._show_project_info()

    def _on_storage_changed(self, text):
        self._storage_path = text.strip()
        self._refresh_projects()

    def _browse_storage(self):
        path = QFileDialog.getExistingDirectory(self, "Select Storage Directory")
        if path:
            self._storage_input.setText(path + "/")

    def _refresh_projects(self):
        self._project_list.clear()
        self._info_text.clear()
        try:
            os.makedirs(self._storage_path, exist_ok=True)
            projects = svc_list_projects(self._storage_path)
            self._project_list.addItems(projects)
        except Exception:
            pass

    def _show_project_info(self):
        item = self._project_list.currentItem()
        if not item:
            return
        name = item.text()
        try:
            info = svc_get_project_info(self._storage_path, name)
            lines = [
                f"Name: {info['name']}",
                f"Path: {info['path']}",
                f"Videos: {info['video_count']}",
            ]
            if info.get('videos'):
                lines.append("Video files: " + ", ".join(info['videos'][:10]))
                if info['video_count'] > 10:
                    lines.append(f"  ... and {info['video_count'] - 10} more")
            self._info_text.setPlainText("\n".join(lines))
        except Exception as e:
            self._info_text.setPlainText(f"Error: {e}")

    def _open_selected_project(self):
        item = self._project_list.currentItem()
        if item:
            name = item.text()
            self.project_opened.emit(self._storage_path, name)

    def _create_project(self):
        name = self._new_name_input.text().strip()
        if not name:
            from datetime import datetime
            name = datetime.now().strftime("project_%Y%m%d_%H%M%S")

        try:
            svc_create_project(self._storage_path, name)
            self._new_name_input.clear()
            self._refresh_projects()
            self.project_opened.emit(self._storage_path, name)
        except FileExistsError:
            QMessageBox.warning(self, "Error", f"Project '{name}' already exists.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create project: {e}")

    def _delete_selected_project(self):
        item = self._project_list.currentItem()
        if not item:
            return
        name = item.text()
        reply = QMessageBox.question(
            self, "Confirm Delete",
            f"Are you sure you want to delete project '{name}'?\n"
            "This action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            import shutil
            project_path = os.path.join(self._storage_path, name)
            if os.path.isdir(project_path):
                shutil.rmtree(project_path)
            self._refresh_projects()
