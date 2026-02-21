"""
castle/desktop/widgets/wizard_panel.py
Quick Start Wizard panel for CASTLE Desktop (PyQt6).

Design goal: "An 80-year-old professor's first time using the app."
- Three-step flow via QStackedWidget
- Zero jargon
- Smart defaults (auto-configured by castle.service.auto_config)
- Plain-English error messages with fix suggestions
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QPixmap
from PyQt6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PIPELINE_STEPS = [
    ("preprocess", "Stabilise video"),
    ("track", "Track the animal"),
    ("extract", "Analyse movement features"),
    ("cluster", "Discover behaviour patterns"),
]

_STATUS_ICONS = {
    "pending": "⏳",
    "running": "🔄",
    "done": "✅",
    "error": "❌",
}


def _fmt_seconds(sec: float) -> str:
    sec = int(sec)
    if sec < 60:
        return f"{sec} sec"
    m, s = divmod(sec, 60)
    return f"{m} min {s:02d} sec"


# ---------------------------------------------------------------------------
# Worker thread for the pipeline
# ---------------------------------------------------------------------------

class PipelineWorker(QThread):
    """Runs the CASTLE pipeline in a background thread.

    Emits:
        step_update(step_key, status)   — "running" | "done" | "error"
        log_update(message)             — text to append to log
        finished_ok()                   — pipeline completed successfully
        finished_error(message)         — pipeline failed with plain-English msg
    """

    step_update = pyqtSignal(str, str)
    log_update = pyqtSignal(str)
    finished_ok = pyqtSignal()
    finished_error = pyqtSignal(str)

    def __init__(
        self,
        video_path: str,
        project_name: str,
        storage_path: str,
        body_roi: int,
        head_roi: int,
        config: dict,
        parent=None,
    ):
        super().__init__(parent)
        self.video_path = video_path
        self.project_name = project_name
        self.storage_path = storage_path
        self.body_roi = body_roi
        self.head_roi = head_roi
        self.config = config or {}

    # ------------------------------------------------------------------

    def run(self):  # noqa: PLR0912, PLR0915
        vpath = self.video_path
        pname = self.project_name
        spath = self.storage_path or "projects/"
        body = self.body_roi
        head = self.head_roi
        cfg = self.config

        # ---- Create project ----
        self.log_update.emit(f"📁 Creating project '{pname}'…")
        try:
            from castle.service.project_service import (  # noqa: PLC0415
                create_project,
            )

            create_project(spath, pname)
        except FileExistsError:
            self.log_update.emit(f"   Project '{pname}' already exists — reusing.")
        except Exception as exc:
            self.finished_error.emit(self._err("project", str(exc)))
            return

        # ---- Copy video ----
        self.log_update.emit("📂 Copying video into project…")
        try:
            src_dir = os.path.join(spath, pname, "sources")
            os.makedirs(src_dir, exist_ok=True)
            video_name = Path(vpath).name
            dest = os.path.join(src_dir, video_name)
            if not os.path.exists(dest):
                shutil.copyfile(vpath, dest)
        except Exception as exc:
            self.finished_error.emit(self._err("copy", str(exc)))
            return

        # ---- Preprocessing ----
        self.step_update.emit("preprocess", "running")
        self.log_update.emit("🎬 Stabilising video…")
        try:
            from castle.service.preprocessing_service import (  # noqa: PLC0415
                preprocess_stabilized_camera,
            )

            pre = cfg.get("preprocessing", {})
            result = preprocess_stabilized_camera(
                storage_path=spath,
                project_name=pname,
                video_name=video_name,
                body_roi_id=body,
                head_roi_id=head,
                fc=pre.get("fc", 0.25),
                margin=pre.get("margin", 75),
                min_crop=pre.get("min_crop", 300),
                output_size=pre.get("output_size", 518),
            )
            if result.get("status") != "ok":
                raise RuntimeError(result.get("message", "Unknown error"))
            self.step_update.emit("preprocess", "done")
            self.log_update.emit("✅ Video stabilised.")
        except Exception as exc:
            self.step_update.emit("preprocess", "error")
            self.finished_error.emit(self._err("preprocess", str(exc)))
            return

        # ---- Tracking ----
        self.step_update.emit("track", "running")
        self.log_update.emit("🐾 Tracking the animal — this may take a while…")
        try:
            from castle.service.tracking_service import run_tracking  # noqa: PLC0415

            ext = cfg.get("extraction", {})
            result = run_tracking(
                storage_path=spath,
                project_name=pname,
                video_name=video_name,
                batch_size=ext.get("batch_size", 8),
            )
            if result.get("status") != "ok":
                raise RuntimeError(result.get("message", "Unknown error"))
            self.step_update.emit("track", "done")
            self.log_update.emit("✅ Animal tracked.")
        except Exception as exc:
            self.step_update.emit("track", "error")
            self.finished_error.emit(self._err("track", str(exc)))
            return

        # ---- Feature extraction ----
        self.step_update.emit("extract", "running")
        self.log_update.emit("🔬 Analysing movement features…")
        try:
            from castle.service.extraction_service import (  # noqa: PLC0415
                extract_latent,
                make_preprocess_config,
            )

            ext = cfg.get("extraction", {})
            pre_cfg = make_preprocess_config(
                center_roi_switch=True,
                center_roi_id=body,
                center_roi_crop_width=ext.get("center_roi_crop_width", 300),
                center_roi_crop_height=ext.get("center_roi_crop_height", 300),
            )
            result = extract_latent(
                storage_path=spath,
                project_name=pname,
                video_name=video_name,
                preprocess=pre_cfg,
                batch_size=ext.get("batch_size", 8),
            )
            if result.get("status") != "ok":
                raise RuntimeError(result.get("message", "Unknown error"))
            self.step_update.emit("extract", "done")
            self.log_update.emit("✅ Features extracted.")
        except Exception as exc:
            self.step_update.emit("extract", "error")
            self.finished_error.emit(self._err("extract", str(exc)))
            return

        # ---- Clustering ----
        self.step_update.emit("cluster", "running")
        self.log_update.emit("🧩 Discovering behaviour patterns…")
        try:
            from castle.service.clustering_service import run_clustering  # noqa: PLC0415

            cl = cfg.get("clustering", {})
            result = run_clustering(
                storage_path=spath,
                project_name=pname,
                video_name=video_name,
                n_clusters=cl.get("n_clusters", 10),
            )
            if result.get("status") != "ok":
                raise RuntimeError(result.get("message", "Unknown error"))
            self.step_update.emit("cluster", "done")
            self.log_update.emit("✅ Behaviour clusters found!")
        except Exception as exc:
            self.step_update.emit("cluster", "error")
            self.finished_error.emit(self._err("cluster", str(exc)))
            return

        self.log_update.emit("🎉 Analysis complete!")
        self.finished_ok.emit()

    # ------------------------------------------------------------------

    @staticmethod
    def _err(step: str, exc_str: str) -> str:
        _hints = {
            "project": "Could not create the project folder. Check that the storage path exists and is writable.",
            "copy": "Could not copy the video file. Make sure there is enough disk space.",
            "preprocess": (
                "Video stabilisation failed.\n\n"
                "• Make sure the video plays normally in a media player.\n"
                "• Check there is enough disk space.\n"
                "• If the animal is very small, try a higher-resolution recording."
            ),
            "track": (
                "Animal tracking failed.\n\n"
                "• Make sure the animal is clearly visible throughout the video.\n"
                "• Check that the Region ID numbers match the tracking regions.\n"
                "• If you see a GPU memory error, try a shorter video clip."
            ),
            "extract": (
                "Feature extraction failed.\n\n"
                "• Make sure the tracking step finished without errors.\n"
                "• Try re-running tracking before retrying."
            ),
            "cluster": (
                "Behaviour discovery failed.\n\n"
                "• Make sure feature extraction finished successfully.\n"
                "• Your video may be too short — aim for at least 5 minutes."
            ),
        }
        hint = _hints.get(step, f"Unexpected error: {exc_str}")
        return f"{hint}\n\n(Technical detail: {exc_str})"


# ---------------------------------------------------------------------------
# Step pages
# ---------------------------------------------------------------------------

class _Step1Page(QWidget):
    """Step 1: Upload video + name the project."""

    video_selected = pyqtSignal(str)  # emits video path

    def __init__(self, parent=None):
        super().__init__(parent)
        self._video_path: Optional[str] = None
        self._setup_ui()
        self.setAcceptDrops(True)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        title = QLabel("<h2>📂 Step 1: Upload your video</h2>")
        title.setWordWrap(True)
        layout.addWidget(title)

        subtitle = QLabel(
            "Choose the video file you want to analyse.\n"
            "You can drag-and-drop it here, or click Browse."
        )
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        # Video selection
        video_group = QGroupBox("Video file")
        vg_layout = QHBoxLayout(video_group)
        self._video_label = QLabel("No video selected")
        self._video_label.setWordWrap(True)
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse_video)
        vg_layout.addWidget(self._video_label, stretch=4)
        vg_layout.addWidget(browse_btn, stretch=1)
        layout.addWidget(video_group)

        # Drop zone hint
        drop_hint = QLabel(
            "💡  You can also drag-and-drop a video file anywhere onto this window."
        )
        drop_hint.setWordWrap(True)
        layout.addWidget(drop_hint)

        # Video info
        self._info_label = QLabel("")
        self._info_label.setWordWrap(True)
        layout.addWidget(self._info_label)

        # Project name
        name_group = QGroupBox("Project name")
        ng_layout = QVBoxLayout(name_group)
        ng_hint = QLabel(
            "Give this analysis a name (letters, numbers, hyphens only).\n"
            "Example: mouse-open-field-trial1"
        )
        ng_hint.setWordWrap(True)
        self._name_input = QLineEdit()
        self._name_input.setPlaceholderText("e.g. mouse-experiment-2026")
        ng_layout.addWidget(ng_hint)
        ng_layout.addWidget(self._name_input)
        layout.addWidget(name_group)

        layout.addStretch()

    # ------------------------------------------------------------------

    def _browse_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select video file",
            "",
            "Video files (*.mp4 *.avi *.mov *.mkv);;All files (*)",
        )
        if path:
            self._set_video(path)

    def _set_video(self, path: str):
        self._video_path = path
        self._video_label.setText(path)
        self._info_label.setText(self._get_info(path))
        self.video_selected.emit(path)

    @staticmethod
    def _get_info(path: str) -> str:
        try:
            import cv2  # noqa: PLC0415

            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                return "⚠️ Could not read video info."
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            dur = frames / fps if fps else 0
            m, s = divmod(int(dur), 60)
            return f"📹 {w}×{h} pixels · {fps:.1f} fps · {m}:{s:02d} minutes"
        except Exception:
            return "ℹ️ Video info unavailable."

    # Drag-and-drop
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if Path(path).suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
                self._set_video(path)
            else:
                QMessageBox.warning(
                    self,
                    "Unsupported file",
                    "Please drop a video file (MP4, AVI, MOV, or MKV).",
                )

    # ------------------------------------------------------------------
    @property
    def video_path(self) -> Optional[str]:
        return self._video_path

    @property
    def project_name(self) -> str:
        return self._name_input.text().strip()


# ---------------------------------------------------------------------------

class _Step2Page(QWidget):
    """Step 2: Identify the animal region."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        title = QLabel("<h2>🐾 Step 2: Identify the animal</h2>")
        title.setWordWrap(True)
        layout.addWidget(title)

        instructions = QLabel(
            "Each animal or body region visible in your video has been assigned a number.\n\n"
            "• Enter the number for the main BODY of the animal (usually 1).\n"
            "• Enter the number for the HEAD region if your video has one (usually 2).\n\n"
            "Not sure? Leave the default values — they work for most recordings."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        # Frame preview
        self._frame_label = QLabel("(Video preview will appear here)")
        self._frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._frame_label.setMinimumHeight(240)
        self._frame_label.setStyleSheet("border: 1px solid #45475a; background: #181825;")
        layout.addWidget(self._frame_label)

        # ROI inputs
        roi_group = QGroupBox("Region numbers")
        rg_layout = QVBoxLayout(roi_group)

        body_row = QHBoxLayout()
        body_row.addWidget(QLabel("Body region number:"))
        self._body_spin = QSpinBox()
        self._body_spin.setRange(1, 99)
        self._body_spin.setValue(1)
        self._body_spin.setToolTip("Usually 1. The region covering the animal's torso.")
        body_row.addWidget(self._body_spin)
        rg_layout.addLayout(body_row)

        head_row = QHBoxLayout()
        head_row.addWidget(QLabel("Head region number (optional):"))
        self._head_spin = QSpinBox()
        self._head_spin.setRange(1, 99)
        self._head_spin.setValue(2)
        self._head_spin.setToolTip(
            "Usually 2. Leave at 2 if you don't have a separate head marker."
        )
        head_row.addWidget(self._head_spin)
        rg_layout.addLayout(head_row)

        layout.addWidget(roi_group)

        # Config preview
        self._config_label = QLabel("Settings will be shown after you upload a video.")
        self._config_label.setWordWrap(True)
        layout.addWidget(self._config_label)

        layout.addStretch()

    # ------------------------------------------------------------------

    def set_frame(self, frame_path: Optional[str]):
        if frame_path and os.path.exists(frame_path):
            pix = QPixmap(frame_path)
            self._frame_label.setPixmap(
                pix.scaled(
                    640,
                    360,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        else:
            self._frame_label.setText("(No preview available)")

    def set_config_summary(self, config: dict):
        pre = config.get("preprocessing", {})
        gpu = config.get("gpu_info", {})
        gpu_line = (
            f"GPU: {gpu.get('name', 'CPU only')} ({gpu.get('vram_mb', 0)} MB VRAM)"
            if gpu.get("available")
            else "Running on CPU (no GPU detected)"
        )
        self._config_label.setText(
            f"Auto-detected settings:\n"
            f"  • Smoothing level: {pre.get('fc', 0.25)}\n"
            f"  • Crop margin: {pre.get('margin', 75)} px\n"
            f"  • Min frame size: {pre.get('min_crop', 300)} px\n"
            f"  • Output frame: {pre.get('output_size', 518)} px\n"
            f"  • {gpu_line}\n\n"
            f"All settings are optimal for your video — nothing to change!"
        )

    @property
    def body_roi(self) -> int:
        return self._body_spin.value()

    @property
    def head_roi(self) -> int:
        return self._head_spin.value()


# ---------------------------------------------------------------------------

class _Step3Page(QWidget):
    """Step 3: Progress + status dashboard."""

    # Emitted when the user clicks "Open Behavior Explorer"
    open_explorer = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        title = QLabel("<h2>⚙️ Running analysis…</h2>")
        layout.addWidget(title)

        self._eta_label = QLabel("Calculating estimated time…")
        self._eta_label.setWordWrap(True)
        layout.addWidget(self._eta_label)

        # Overall progress bar
        self._progress = QProgressBar()
        self._progress.setRange(0, len(_PIPELINE_STEPS))
        self._progress.setValue(0)
        layout.addWidget(self._progress)

        # Pipeline step table
        self._table = QTableWidget(len(_PIPELINE_STEPS), 2)
        self._table.setHorizontalHeaderLabels(["Status", "Step"])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self._table.setMaximumHeight(200)
        for i, (_, label) in enumerate(_PIPELINE_STEPS):
            self._table.setItem(i, 0, QTableWidgetItem(_STATUS_ICONS["pending"]))
            self._table.setItem(i, 1, QTableWidgetItem(label))
        layout.addWidget(self._table)

        # Log
        log_label = QLabel("Progress log:")
        layout.addWidget(log_label)
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setPlaceholderText("Progress will appear here…")
        layout.addWidget(self._log)

        # Error message
        self._error_label = QLabel("")
        self._error_label.setWordWrap(True)
        self._error_label.setStyleSheet("color: #f38ba8;")
        self._error_label.setVisible(False)
        layout.addWidget(self._error_label)

        # Completion area
        self._done_widget = QWidget()
        done_layout = QVBoxLayout(self._done_widget)
        done_layout.setContentsMargins(0, 0, 0, 0)
        done_msg = QLabel(
            "<h3>🎉 Analysis complete!</h3>"
            "<p>Click below to explore the discovered behaviour patterns.</p>"
        )
        done_msg.setWordWrap(True)
        done_layout.addWidget(done_msg)
        self._open_btn = QPushButton("Open Behavior Explorer →")
        self._open_btn.setObjectName("primaryButton")
        self._open_btn.clicked.connect(self.open_explorer.emit)
        done_layout.addWidget(self._open_btn)
        self._done_widget.setVisible(False)
        layout.addWidget(self._done_widget)

        layout.addStretch()

    # ------------------------------------------------------------------

    def set_eta(self, eta_seconds: float):
        self._eta_label.setText(
            f"⏱️  Estimated time: {_fmt_seconds(eta_seconds)}"
        )

    def reset(self):
        for i in range(len(_PIPELINE_STEPS)):
            self._table.item(i, 0).setText(_STATUS_ICONS["pending"])
        self._progress.setValue(0)
        self._log.clear()
        self._error_label.setVisible(False)
        self._done_widget.setVisible(False)

    def update_step(self, step_key: str, status: str):
        idx = next((i for i, (k, _) in enumerate(_PIPELINE_STEPS) if k == step_key), None)
        if idx is not None:
            self._table.item(idx, 0).setText(_STATUS_ICONS.get(status, "?"))
        if status == "done":
            done_count = sum(
                1
                for i in range(len(_PIPELINE_STEPS))
                if self._table.item(i, 0).text() == _STATUS_ICONS["done"]
            )
            self._progress.setValue(done_count)

    def append_log(self, message: str):
        self._log.append(message)

    def show_error(self, message: str):
        self._error_label.setText(f"❌  {message}")
        self._error_label.setVisible(True)

    def show_done(self):
        self._progress.setValue(len(_PIPELINE_STEPS))
        self._done_widget.setVisible(True)


# ---------------------------------------------------------------------------
# Main WizardPanel widget
# ---------------------------------------------------------------------------

class WizardPanel(QWidget):
    """Three-step wizard panel — to be inserted as the first tab of MainWindow."""

    # Emitted when the pipeline finishes and the user wants to switch to
    # the Behavior Explorer (Microscope) tab.
    request_open_behavior_explorer = pyqtSignal()

    def __init__(self, parent=None, storage_path: str = "projects/"):
        super().__init__(parent)
        self._storage_path = storage_path
        self._video_path: Optional[str] = None
        self._config: dict = {}
        self._worker: Optional[PipelineWorker] = None
        self._frame_tmp: Optional[str] = None

        self._setup_ui()

    # ------------------------------------------------------------------

    def _setup_ui(self):
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(16, 16, 16, 16)
        root_layout.setSpacing(12)

        # Title bar
        header = QLabel(
            "<h1>🧭 Quick Start Wizard</h1>"
            "<p style='color: #a6adc8;'>New here? This wizard will guide you "
            "through your first analysis in 3 simple steps.</p>"
        )
        header.setWordWrap(True)
        root_layout.addWidget(header)

        # Step indicator
        self._step_indicator = QLabel("Step 1 of 3")
        self._step_indicator.setAlignment(Qt.AlignmentFlag.AlignRight)
        root_layout.addWidget(self._step_indicator)

        # Stacked pages
        self._stack = QStackedWidget()
        self._page1 = _Step1Page()
        self._page2 = _Step2Page()
        self._page3 = _Step3Page()

        self._stack.addWidget(self._page1)
        self._stack.addWidget(self._page2)
        self._stack.addWidget(self._page3)
        root_layout.addWidget(self._stack, stretch=1)

        # Navigation buttons
        nav_layout = QHBoxLayout()
        self._back_btn = QPushButton("← Back")
        self._back_btn.setVisible(False)
        self._back_btn.clicked.connect(self._go_back)
        self._next_btn = QPushButton("Continue →")
        self._next_btn.setObjectName("primaryButton")
        self._next_btn.clicked.connect(self._go_next)
        nav_layout.addWidget(self._back_btn)
        nav_layout.addStretch()
        nav_layout.addWidget(self._next_btn)
        root_layout.addLayout(nav_layout)

        # Connect signals
        self._page1.video_selected.connect(self._on_video_selected)
        self._page3.open_explorer.connect(self.request_open_behavior_explorer)

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _current_step(self) -> int:
        return self._stack.currentIndex()  # 0, 1, 2

    def _go_next(self):
        step = self._current_step()

        if step == 0:
            if not self._validate_step1():
                return
            self._prepare_step2()
            self._stack.setCurrentIndex(1)
            self._back_btn.setVisible(True)
            self._next_btn.setText("Start Analysis →")
            self._step_indicator.setText("Step 2 of 3")

        elif step == 1:
            self._prepare_step3()
            self._stack.setCurrentIndex(2)
            self._next_btn.setVisible(False)
            self._back_btn.setVisible(False)
            self._step_indicator.setText("Step 3 of 3")
            self._run_pipeline()

    def _go_back(self):
        step = self._current_step()
        if step == 1:
            self._stack.setCurrentIndex(0)
            self._back_btn.setVisible(False)
            self._next_btn.setVisible(True)
            self._next_btn.setText("Continue →")
            self._step_indicator.setText("Step 1 of 3")

    # ------------------------------------------------------------------
    # Validation & preparation
    # ------------------------------------------------------------------

    def _validate_step1(self) -> bool:
        import re  # noqa: PLC0415

        if not self._page1.video_path:
            QMessageBox.warning(self, "No video", "Please select a video file first.")
            return False
        name = self._page1.project_name
        if not name:
            QMessageBox.warning(self, "No project name", "Please enter a project name.")
            return False
        if not re.match(r"^[\w\-]+$", name):
            QMessageBox.warning(
                self,
                "Invalid project name",
                "Project name may only contain letters, numbers, and hyphens.",
            )
            return False
        return True

    def _on_video_selected(self, path: str):
        self._video_path = path
        # Auto-config in background to avoid blocking UI
        threading.Thread(target=self._run_auto_config, daemon=True).start()

    def _run_auto_config(self):
        try:
            from castle.service.auto_config import (  # noqa: PLC0415
                estimate_pipeline_time,
                get_gpu_info,
                recommend_config,
            )

            gpu = get_gpu_info()
            self._config = recommend_config(self._video_path, gpu_info=gpu)
            eta = estimate_pipeline_time(self._video_path, self._config)
            # Update ETA label from main thread
            self._page3.set_eta(eta)
        except Exception as exc:
            logger.warning("auto_config failed: %s", exc)
            self._config = {}

    def _prepare_step2(self):
        """Populate Step 2 page with frame preview and config summary."""
        # Extract first frame
        try:
            import cv2  # noqa: PLC0415
            import tempfile  # noqa: PLC0415

            cap = cv2.VideoCapture(self._video_path)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                if ret:
                    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                    cv2.imwrite(tmp.name, frame)
                    self._frame_tmp = tmp.name
                    self._page2.set_frame(tmp.name)
        except Exception:
            pass
        if self._config:
            self._page2.set_config_summary(self._config)

    def _prepare_step3(self):
        self._page3.reset()
        if self._config:
            try:
                from castle.service.auto_config import estimate_pipeline_time  # noqa: PLC0415

                eta = estimate_pipeline_time(self._video_path or "", self._config)
                self._page3.set_eta(eta)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Pipeline runner
    # ------------------------------------------------------------------

    def _run_pipeline(self):
        video_path = self._video_path or ""
        project_name = self._page1.project_name
        body_roi = self._page2.body_roi
        head_roi = self._page2.head_roi

        self._worker = PipelineWorker(
            video_path=video_path,
            project_name=project_name,
            storage_path=self._storage_path,
            body_roi=body_roi,
            head_roi=head_roi,
            config=self._config,
            parent=self,
        )
        self._worker.step_update.connect(self._page3.update_step)
        self._worker.log_update.connect(self._page3.append_log)
        self._worker.finished_ok.connect(self._on_pipeline_done)
        self._worker.finished_error.connect(self._on_pipeline_error)
        self._worker.start()

    def _on_pipeline_done(self):
        self._page3.show_done()
        self._next_btn.setVisible(False)

    def _on_pipeline_error(self, message: str):
        self._page3.show_error(message)
        # Allow user to go back and retry
        self._back_btn.setVisible(True)
        self._back_btn.setText("← Back (retry)")
        self._next_btn.setVisible(True)
        self._next_btn.setText("Retry Analysis")
        self._stack.setCurrentIndex(1)

    # ------------------------------------------------------------------

    def set_storage_path(self, path: str):
        self._storage_path = path
