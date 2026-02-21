"""
CASTLE Desktop - Source Panel (Tab 1)

Video upload, listing, and preview.
Uses the service layer for project / video management.
"""

import os

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QListWidget, QGroupBox, QMessageBox, QFileDialog,
    QLineEdit, QSplitter, QTextEdit
)
from PyQt6.QtCore import pyqtSignal, Qt

from castle.service.project_service import (
    add_videos as svc_add_videos,
    add_videos_from_directory as svc_add_from_dir,
    get_project_info as svc_get_project_info,
)
from castle.desktop.components.video_player import VideoPlayer


class SourcePanel(QWidget):
    """Panel for video source management."""

    videos_changed = pyqtSignal()  # emitted when video list changes

    def __init__(self, parent=None):
        super().__init__(parent)
        self._storage_path = None
        self._project_name = None
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # === Left: video list + add controls ===
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # Video list
        list_group = QGroupBox("Project Videos")
        list_layout = QVBoxLayout(list_group)

        self._video_list = QListWidget()
        self._video_list.setAlternatingRowColors(True)
        self._video_list.currentRowChanged.connect(self._on_video_selected)
        list_layout.addWidget(self._video_list)

        self._count_label = QLabel("0 videos")
        list_layout.addWidget(self._count_label)
        left_layout.addWidget(list_group, stretch=1)

        # Upload local files
        upload_group = QGroupBox("Add Videos")
        upload_layout = QVBoxLayout(upload_group)

        self._upload_btn = QPushButton("📁 Browse & Add Video Files...")
        self._upload_btn.setObjectName("primaryButton")
        self._upload_btn.clicked.connect(self._browse_videos)
        upload_layout.addWidget(self._upload_btn)

        # Server directory import
        dir_layout = QHBoxLayout()
        self._dir_input = QLineEdit()
        self._dir_input.setPlaceholderText("Server directory path...")
        dir_layout.addWidget(self._dir_input, stretch=4)

        self._dir_btn = QPushButton("Import All")
        self._dir_btn.clicked.connect(self._import_from_directory)
        dir_layout.addWidget(self._dir_btn, stretch=1)
        upload_layout.addLayout(dir_layout)

        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setMaximumHeight(100)
        self._log_text.setPlaceholderText("Import log...")
        upload_layout.addWidget(self._log_text)

        left_layout.addWidget(upload_group)
        splitter.addWidget(left)

        # === Right: video preview ===
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        preview_group = QGroupBox("Video Preview")
        preview_layout = QVBoxLayout(preview_group)

        self._player = VideoPlayer()
        preview_layout.addWidget(self._player)
        right_layout.addWidget(preview_group, stretch=1)

        splitter.addWidget(right)
        splitter.setSizes([350, 650])

        main_layout.addWidget(splitter)

    # --- Public API ---

    def set_project(self, storage_path: str, project_name: str):
        """Called when a project is opened."""
        self._storage_path = storage_path
        self._project_name = project_name
        self._refresh_video_list()

    def _refresh_video_list(self):
        self._video_list.clear()
        if not self._storage_path or not self._project_name:
            self._count_label.setText("No project loaded")
            return
        try:
            info = svc_get_project_info(self._storage_path, self._project_name)
            videos = info.get('videos', [])
            self._video_list.addItems(videos)
            self._count_label.setText(f"{len(videos)} videos")
        except Exception as e:
            self._count_label.setText(f"Error: {e}")

    def _on_video_selected(self, row):
        if row < 0 or not self._storage_path or not self._project_name:
            return
        video_name = self._video_list.item(row).text()
        video_path = os.path.join(
            self._storage_path, self._project_name, 'sources', video_name
        )
        if os.path.exists(video_path):
            self._player.load_video(video_path)

    def _browse_videos(self):
        if not self._project_name:
            QMessageBox.warning(
                self,
                "No Project Open",
                "No project open. Please open a project from the Project panel before adding videos.",
            )
            return

        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Video Files", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv);;All Files (*)",
        )
        if not paths:
            return

        results = svc_add_videos(self._storage_path, self._project_name, paths)
        log_lines = []
        for r in results:
            status = "✅" if r['success'] else "❌"
            log_lines.append(f"{status} {r['video_name']}: {r['message']}")

        self._log_text.setPlainText("\n".join(log_lines))
        self._refresh_video_list()
        self.videos_changed.emit()

    def _import_from_directory(self):
        if not self._project_name:
            QMessageBox.warning(
                self,
                "No Project Open",
                "No project open. Please open a project from the Project panel before adding videos.",
            )
            return

        directory = self._dir_input.text().strip()
        if not directory or not os.path.isdir(directory):
            QMessageBox.warning(
                self,
                "Invalid Directory",
                "Invalid directory path. Please enter an accessible server directory path.",
            )
            return

        result = svc_add_from_dir(self._storage_path, self._project_name, directory)
        log_lines = result.get('messages', [])
        log_lines.insert(0, f"Added: {result['success_count']} | Failed: {result['fail_count']}")

        self._log_text.setPlainText("\n".join(log_lines))
        self._refresh_video_list()
        self.videos_changed.emit()
