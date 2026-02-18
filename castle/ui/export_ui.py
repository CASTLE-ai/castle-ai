"""
castle/ui/export_ui.py
Export Tab — Stage 5.

Allows users to select and download project data as a ZIP archive.
Supported exports: masks, latent features, cluster results, annotations,
grid videos, and source videos.
"""

import glob
import logging
import os
import shutil
import tempfile
import zipfile
from datetime import datetime

import gradio as gr

logger = logging.getLogger(__name__)


# ---------------------------
# Helpers
# ---------------------------


def _project_path(storage_path: str, project_name: str) -> str:
    return os.path.join(storage_path, project_name)


def _collect_masks(project_path: str) -> list[tuple[str, str]]:
    """Return list of (src_path, archive_name) for all mask_list.h5 files."""
    pattern = os.path.join(project_path, "track", "*", "mask_list.h5")
    results = []
    for src in glob.glob(pattern):
        video_name = os.path.basename(os.path.dirname(src))
        results.append((src, os.path.join("track", video_name, "mask_list.h5")))
    return results


def _collect_latent(project_path: str) -> list[tuple[str, str]]:
    """Return list of (src_path, archive_name) for all latent feature files."""
    latent_dir = os.path.join(project_path, "latent")
    results = []
    if not os.path.isdir(latent_dir):
        return results
    for root, _dirs, files in os.walk(latent_dir):
        for f in files:
            src = os.path.join(root, f)
            rel = os.path.relpath(src, project_path)
            results.append((src, rel))
    return results


def _collect_cluster_results(project_path: str) -> list[tuple[str, str]]:
    """Return (src, archive_name) for cluster id.csv, cluster_*.npz, time_series_*.csv."""
    cluster_dir = os.path.join(project_path, "cluster")
    results = []
    if not os.path.isdir(cluster_dir):
        return results
    for pattern in ("id.csv", "cluster_*.npz", "time_series_*.csv"):
        for src in glob.glob(os.path.join(cluster_dir, pattern)):
            rel = os.path.relpath(src, project_path)
            results.append((src, rel))
    return results


def _collect_annotations(project_path: str, session_id: str) -> list[tuple[str, str]]:
    """Return (src, archive_name) for the selected session's annotations.csv."""
    if not session_id:
        return []
    src = os.path.join(
        project_path, "cluster", "sessions", session_id, "annotations.csv"
    )
    if not os.path.isfile(src):
        return []
    rel = os.path.relpath(src, project_path)
    return [(src, rel)]


def _collect_grid_videos(project_path: str) -> list[tuple[str, str]]:
    """Return (src, archive_name) for all grid videos (.mp4)."""
    pattern = os.path.join(project_path, "cluster", "grid_videos", "*.mp4")
    results = []
    for src in glob.glob(pattern):
        rel = os.path.relpath(src, project_path)
        results.append((src, rel))
    return results


def _collect_analysis(project_path: str) -> list[tuple[str, str]]:
    """Return (src, archive_name) for analysis outputs (ethogram, metrics)."""
    results = []
    # Ethogram outputs (if saved to disk by the analysis tab)
    analysis_dir = os.path.join(project_path, "analysis")
    if os.path.isdir(analysis_dir):
        for root, _dirs, files in os.walk(analysis_dir):
            for f in files:
                src = os.path.join(root, f)
                rel = os.path.relpath(src, project_path)
                results.append((src, rel))
    # Also include any session-level analysis files
    sessions_dir = os.path.join(project_path, "cluster", "sessions")
    if os.path.isdir(sessions_dir):
        for sid in os.listdir(sessions_dir):
            sid_analysis = os.path.join(sessions_dir, sid, "analysis")
            if os.path.isdir(sid_analysis):
                for root, _dirs, files in os.walk(sid_analysis):
                    for f in files:
                        src = os.path.join(root, f)
                        rel = os.path.relpath(src, project_path)
                        results.append((src, rel))
    return results


def _collect_source_videos(project_path: str) -> list[tuple[str, str]]:
    """Return (src, archive_name) for all source video files."""
    sources_dir = os.path.join(project_path, "sources")
    results = []
    if not os.path.isdir(sources_dir):
        return results
    for root, _dirs, files in os.walk(sources_dir):
        for f in files:
            src = os.path.join(root, f)
            rel = os.path.relpath(src, project_path)
            results.append((src, rel))
    return results


def _human_size(path: str) -> str:
    """Return human-readable file size string."""
    try:
        size = os.path.getsize(path)
    except OSError:
        return "?"
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


# ---------------------------
# Event Handlers
# ---------------------------


def on_refresh_sessions(storage_path, project_name):
    """Refresh the session dropdown list (mirrors annotator_ui pattern)."""
    if not storage_path or not project_name:
        return gr.update(choices=[], value=None, visible=False)

    from castle.service.session_manager import SessionManager

    mgr = SessionManager(storage_path, project_name)
    sessions = mgr.list_sessions()

    if not sessions:
        return gr.update(
            choices=[("(no sessions)", "")], value="", visible=True
        )

    choices = [
        (
            f"{s.name} — {s.n_clusters} clusters, bin_size={s.bin_size} ({s.updated_at[:16]})",
            s.session_id,
        )
        for s in sessions
    ]
    active_id = mgr.get_active_session_id()
    default = active_id if active_id else sessions[0].session_id
    return gr.update(choices=choices, value=default, visible=True)


def on_export(
    storage_path,
    project_name,
    include_masks,
    include_latent,
    include_cluster,
    include_annotations,
    include_grid_videos,
    include_analysis,
    include_source_videos,
    session_id,
):
    """Collect selected files, copy to a temp dir, and zip for download.

    Uses ``shutil.copyfile`` throughout for CIFS compatibility.

    Args:
        storage_path: Root storage path (str).
        project_name: Project name (str).
        include_masks: bool — include mask h5 files.
        include_latent: bool — include latent features.
        include_cluster: bool — include cluster result files.
        include_annotations: bool — include session annotations.csv.
        include_grid_videos: bool — include grid videos.
        include_source_videos: bool — include source videos.
        session_id: Session ID for annotations.

    Yields:
        (status_markdown, file_path_or_None) tuples for streaming updates.
    """
    if not storage_path or not project_name:
        yield "**❌ Error:** No project selected.", None
        return

    pp = _project_path(storage_path, project_name)
    files: list[tuple[str, str]] = []

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

    # Deduplicate by archive name
    seen: set[str] = set()
    unique_files: list[tuple[str, str]] = []
    for src, arc in files:
        if arc not in seen:
            seen.add(arc)
            unique_files.append((src, arc))

    if not unique_files:
        yield "**⚠️ Nothing to export.** Select at least one category.", None
        return

    total = len(unique_files)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_name = f"{project_name}_export_{timestamp}.zip"

    yield f"**📦 Preparing export…** {total} file(s) to package.", None

    tmp_dir = tempfile.mkdtemp(prefix="castle_export_")
    zip_path = os.path.join(tmp_dir, zip_name)

    try:
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
            for i, (src, arc_name) in enumerate(unique_files, 1):
                if not os.path.isfile(src):
                    logger.warning("Export: source file missing, skipping: %s", src)
                    continue

                size_str = _human_size(src)
                yield (
                    f"**📦 Packaging…** ({i}/{total}) `{arc_name}` ({size_str})",
                    None,
                )

                # Copy via shutil.copyfile to a temp staging file, then add to zip
                staging = os.path.join(tmp_dir, f"_stage_{i}")
                try:
                    shutil.copyfile(src, staging)
                    zf.write(staging, arc_name)
                    os.unlink(staging)
                except Exception as exc:
                    logger.error("Export: failed to copy %s: %s", src, exc)
                    yield f"**⚠️ Warning:** Could not copy `{arc_name}`: {exc}", None

        zip_size = _human_size(zip_path)
        yield (
            f"**✅ Export complete!** `{zip_name}` ({zip_size}) — {total} file(s)",
            zip_path,
        )

    except Exception as exc:
        logger.exception("Export failed")
        yield f"**❌ Export failed:** {exc}", None


# ---------------------------
# UI Construction
# ---------------------------


def create_export_ui(storage_path, project_name):
    """Create the Export tab UI.

    Args:
        storage_path: gr.State / gr.Textbox holding the storage path.
        project_name: gr.State / gr.Textbox holding the project name.

    Returns:
        dict of Gradio UI components.
    """
    ui = {}

    with gr.Column():
        gr.Markdown("## 📦 Export Project Data")
        gr.Markdown(
            "Select the data categories to include in the ZIP archive. "
            "Large files (especially **masks**) may take a while to package."
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📂 Include")
                ui["include_masks"] = gr.Checkbox(
                    label="🎭 Masks (track/*/mask_list.h5) — may be very large!",
                    value=True,
                )
                ui["include_latent"] = gr.Checkbox(
                    label="🧠 Latent features (latent/)",
                    value=True,
                )
                ui["include_cluster"] = gr.Checkbox(
                    label="📊 Cluster results (cluster/ — id.csv, cluster_*.npz, time_series_*.csv)",
                    value=True,
                )
                ui["include_annotations"] = gr.Checkbox(
                    label="🏷️ Annotations (cluster/sessions/{session}/annotations.csv)",
                    value=True,
                )
                ui["include_grid_videos"] = gr.Checkbox(
                    label="🎬 Grid videos (cluster/grid_videos/*.mp4)",
                    value=True,
                )
                ui["include_analysis"] = gr.Checkbox(
                    label="📊 Analysis results (ethogram, metrics)",
                    value=True,
                )
                ui["include_source_videos"] = gr.Checkbox(
                    label="📹 Source videos (sources/)",
                    value=False,
                )

            with gr.Column(scale=1):
                gr.Markdown("### 🗂️ Session (for Annotations)")
                gr.Markdown(
                    "Annotations are session-scoped. Select the session whose "
                    "annotations to include."
                )
                with gr.Row():
                    ui["session_dropdown"] = gr.Dropdown(
                        label="Session",
                        choices=[],
                        interactive=True,
                        scale=3,
                    )
                    ui["refresh_btn"] = gr.Button("🔄 Refresh", scale=1)

        gr.Markdown("---")

        with gr.Row():
            ui["export_btn"] = gr.Button("📦 Export", variant="primary", scale=2)

        ui["status"] = gr.Markdown("**Status:** Ready")
        ui["download"] = gr.File(label="⬇️ Download ZIP", visible=False)

    # ---------------------------
    # Event Bindings
    # ---------------------------

    ui["refresh_btn"].click(
        fn=on_refresh_sessions,
        inputs=[storage_path, project_name],
        outputs=[ui["session_dropdown"]],
    )

    _export_inputs = [
        storage_path,
        project_name,
        ui["include_masks"],
        ui["include_latent"],
        ui["include_cluster"],
        ui["include_annotations"],
        ui["include_grid_videos"],
        ui["include_analysis"],
        ui["include_source_videos"],
        ui["session_dropdown"],
    ]

    def _run_export(*args):
        """Wrapper to consume the generator and return final status + file."""
        last_status = "**Status:** Ready"
        last_file = None
        for status, fpath in on_export(*args):
            last_status = status
            if fpath is not None:
                last_file = fpath
        return last_status, gr.update(value=last_file, visible=last_file is not None)

    ui["export_btn"].click(
        fn=_run_export,
        inputs=_export_inputs,
        outputs=[ui["status"], ui["download"]],
    )

    return ui
