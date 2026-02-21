"""
CASTLE Desktop - Export Panel (Tab 7)

Select project data categories and package them into a ZIP archive.
Mirrors the Gradio export_ui tab, using shutil.copyfile for CIFS compatibility.
"""

import logging
import os
import shutil
import tempfile
import zipfile
from datetime import datetime

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QComboBox, QCheckBox, QProgressBar,
    QMessageBox, QFileDialog, QTextEdit,
)
from PyQt6.QtCore import Qt, pyqtSlot

from castle.desktop.services.worker_threads import ServiceWorker
from castle.service.export_service import (
    _collect_masks,
    _collect_latent,
    _collect_cluster_results,
    _collect_annotations,
    _collect_grid_videos,
    _collect_analysis,
    _collect_source_videos,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------

def _project_path(storage_path: str, project_name: str) -> str:
    return os.path.join(storage_path, project_name)


def _human_size(path: str) -> str:
    try:
        size = os.path.getsize(path)
    except OSError:
        return "?"
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def _run_export(
    storage_path, project_name,
    include_masks, include_latent, include_cluster,
    include_annotations, include_grid_videos, include_analysis,
    include_source_videos, session_id,
    progress_callback=None,
):
    """Collect files and create a ZIP. Returns path to the ZIP file."""
    pp = _project_path(storage_path, project_name)
    files = []

    if include_masks:
        files.extend(_collect_masks(pp))
    if include_latent:
        files.extend(_collect_latent(pp))
    if include_cluster:
        files.extend(_collect_cluster_results(pp))
    if include_annotations:
        files.extend(_collect_annotations(pp, session_id))
    if include_grid_videos:
        files.extend(_collect_grid_videos(pp))
    if include_analysis:
        files.extend(_collect_analysis(pp))
    if include_source_videos:
        files.extend(_collect_source_videos(pp))

    # Deduplicate
    seen: set = set()
    unique_files = []
    for src, arc in files:
        if arc not in seen:
            seen.add(arc)
            unique_files.append((src, arc))

    if not unique_files:
        return None, "Nothing to export — select at least one category."

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_name = f"{project_name}_export_{timestamp}.zip"
    tmp_dir = tempfile.mkdtemp(prefix="castle_export_")
    zip_path = os.path.join(tmp_dir, zip_name)

    total = len(unique_files)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for i, (src, arc_name) in enumerate(unique_files, 1):
            if not os.path.isfile(src):
                logger.warning("Export: missing file, skipping: %s", src)
                continue
            if progress_callback:
                progress_callback(int(i / total * 100), f"Packaging {arc_name}")
            staging = os.path.join(tmp_dir, f"_stage_{i}")
            shutil.copyfile(src, staging)
            zf.write(staging, arc_name)
            os.unlink(staging)

    size_str = _human_size(zip_path)
    return zip_path, f"Export complete! {zip_name} ({size_str}), {total} file(s)"


class ExportPanel(QWidget):
    """Export panel — package project data as a ZIP archive."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self._storage_path: str | None = None
        self._project_name: str | None = None
        self._worker: ServiceWorker | None = None

        self._setup_ui()

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Top: what to include
        layout.addWidget(self._create_include_group())

        # Session selector (for annotations)
        layout.addWidget(self._create_session_group())

        # Export button + status
        btn_row = QHBoxLayout()
        self._export_btn = QPushButton("📦 Export ZIP")
        self._export_btn.setObjectName("primaryButton")
        self._export_btn.clicked.connect(self._run_export)
        self._export_btn.setEnabled(False)
        btn_row.addWidget(self._export_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self._status_edit = QTextEdit()
        self._status_edit.setReadOnly(True)
        self._status_edit.setMaximumHeight(100)
        self._status_edit.setPlaceholderText("Status…")
        layout.addWidget(self._status_edit)

        self._result_label = QLabel("")
        self._result_label.setWordWrap(True)
        self._result_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self._result_label)

        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)
        layout.addWidget(self._progress_bar)

        layout.addStretch()

    def _create_include_group(self) -> QGroupBox:
        group = QGroupBox("Include in Export")
        layout = QVBoxLayout(group)

        self._chk_masks = QCheckBox("🎭 Masks (track/*/mask_list.h5) — may be very large!")
        self._chk_masks.setChecked(True)
        layout.addWidget(self._chk_masks)

        self._chk_latent = QCheckBox("🧠 Latent features (latent/)")
        self._chk_latent.setChecked(True)
        layout.addWidget(self._chk_latent)

        self._chk_cluster = QCheckBox(
            "📊 Cluster results (cluster/ — id.csv, cluster_*.npz, time_series_*.csv)"
        )
        self._chk_cluster.setChecked(True)
        layout.addWidget(self._chk_cluster)

        self._chk_annotations = QCheckBox(
            "🏷️ Annotations (cluster/sessions/{session}/annotations.csv)"
        )
        self._chk_annotations.setChecked(True)
        layout.addWidget(self._chk_annotations)

        self._chk_grid_videos = QCheckBox("🎬 Grid videos (cluster/grid_videos/*.mp4)")
        self._chk_grid_videos.setChecked(True)
        layout.addWidget(self._chk_grid_videos)

        self._chk_analysis = QCheckBox("📊 Analysis results (ethogram, metrics)")
        self._chk_analysis.setChecked(True)
        layout.addWidget(self._chk_analysis)

        self._chk_source = QCheckBox("📹 Source videos (sources/) — may be very large!")
        self._chk_source.setChecked(False)
        layout.addWidget(self._chk_source)

        return group

    def _create_session_group(self) -> QGroupBox:
        group = QGroupBox("Session (for Annotations)")
        layout = QHBoxLayout(group)

        layout.addWidget(QLabel("Session:"))
        self._session_combo = QComboBox()
        self._session_combo.setMinimumWidth(300)
        layout.addWidget(self._session_combo, stretch=1)

        self._refresh_btn = QPushButton("🔄 Refresh")
        self._refresh_btn.clicked.connect(self._refresh_sessions)
        layout.addWidget(self._refresh_btn)

        return group

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_project(self, storage_path: str, project_name: str):
        self._storage_path = storage_path
        self._project_name = project_name
        self._export_btn.setEnabled(True)
        self._result_label.setText("")
        self._status_edit.clear()
        self._refresh_sessions()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    @pyqtSlot()
    def _refresh_sessions(self):
        if not self._storage_path or not self._project_name:
            return
        try:
            from castle.service.session_manager import SessionManager
            mgr = SessionManager(self._storage_path, self._project_name)
            sessions = mgr.list_sessions()
            self._session_combo.clear()
            if not sessions:
                self._session_combo.addItem("(no sessions)", "")
            else:
                for s in sessions:
                    label = (
                        f"{s.name} — {s.n_clusters} clusters, "
                        f"bin_size={s.bin_size} ({s.updated_at[:16]})"
                    )
                    self._session_combo.addItem(label, s.session_id)
                active_id = mgr.get_active_session_id()
                if active_id:
                    for i in range(self._session_combo.count()):
                        if self._session_combo.itemData(i) == active_id:
                            self._session_combo.setCurrentIndex(i)
                            break
        except Exception as exc:
            logger.warning("Failed to refresh sessions: %s", exc)

    @pyqtSlot()
    def _run_export(self):
        if not self._storage_path or not self._project_name:
            QMessageBox.warning(
                self,
                "No Project Selected",
                "No project selected. Please open a project from the Project panel first.",
            )
            return

        # Ask where to save
        default_name = (
            f"{self._project_name}_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        )
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Export ZIP",
            os.path.join(os.path.expanduser("~"), default_name),
            "ZIP Archives (*.zip)",
        )
        if not save_path:
            return

        session_id = self._session_combo.currentData() or ""

        self._export_btn.setEnabled(False)
        self._progress_bar.setVisible(True)
        self._progress_bar.setValue(0)
        self._status_edit.clear()
        self._result_label.setText("")
        self._status_edit.append("Starting export…")

        self._worker = ServiceWorker(
            _run_export,
            self._storage_path,
            self._project_name,
            self._chk_masks.isChecked(),
            self._chk_latent.isChecked(),
            self._chk_cluster.isChecked(),
            self._chk_annotations.isChecked(),
            self._chk_grid_videos.isChecked(),
            self._chk_analysis.isChecked(),
            self._chk_source.isChecked(),
            session_id,
        )
        self._worker.finished.connect(
            lambda result: self._on_export_done(result, save_path)
        )
        self._worker.error.connect(self._on_export_error)
        self._worker.start()

    def _on_export_done(self, result, save_path: str):
        self._progress_bar.setVisible(False)
        self._export_btn.setEnabled(True)

        zip_path, message = result
        self._status_edit.append(message)

        if zip_path is None:
            QMessageBox.warning(self, "Export", message)
            return

        # Move to user-chosen location
        try:
            shutil.copyfile(zip_path, save_path)
            os.unlink(zip_path)
            self._result_label.setText(f"Saved: {save_path}")
            QMessageBox.information(self, "Export Complete", f"ZIP saved to:\n{save_path}")
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Export Error",
                f"Failed to save export ZIP to the selected location.\n\n{exc}\n\n"
                "Please check that the target directory is accessible and has enough disk space.",
            )

    @pyqtSlot(str)
    def _on_export_error(self, err: str):
        self._progress_bar.setVisible(False)
        self._export_btn.setEnabled(True)
        self._status_edit.append(f"Error: {err}")
        QMessageBox.critical(
            self,
            "Export Error",
            f"Export failed.\n\n{err}\n\nCheck the status log for details.",
        )
