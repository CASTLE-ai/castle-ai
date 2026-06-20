"""
castle/ui/export_ui.py
Export Tab — Stage 5.

Allows users to select and download project data as a ZIP archive.
Supported exports: masks, latent features, cluster results, annotations,
grid videos, and source videos.
"""

import json
import logging
import os
import shutil
import tempfile
import zipfile
from datetime import datetime

import gradio as gr

from castle.service.export_service import (
    _collect_masks,
    _collect_latent,
    _collect_cluster_results,
    _collect_annotations,
    _collect_grid_videos,
    _collect_analysis,
    _collect_source_videos,
    build_run_manifest,
)

logger = logging.getLogger(__name__)


# ---------------------------
# Helpers
# ---------------------------


def _project_path(storage_path: str, project_name: str) -> str:
    return os.path.join(storage_path, project_name)


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
        (status_markdown, gr.update for download component) tuples for streaming.
        The download component stays hidden until the final yield with the
        completed zip path.
    """
    _hide_download = gr.update(value=None, visible=False)

    if not storage_path or not project_name:
        yield "**❌ Error:** No project selected.", _hide_download
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
        yield "**⚠️ Nothing to export.** Select at least one category.", _hide_download
        return

    total = len(unique_files)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_name = f"{project_name}_export_{timestamp}.zip"

    yield f"**📦 Preparing export…** {total} file(s) to package.", _hide_download

    tmp_dir = tempfile.mkdtemp(prefix="castle_export_")
    zip_path = os.path.join(tmp_dir, zip_name)

    try:
        packaged_count = 0
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
            # Provenance manifest so the bundle is self-describing (CASTLE +
            # library/GPU versions, selected components, project + session info).
            try:
                selected = [
                    name for flag, name in (
                        (include_masks, "masks"),
                        (include_latent, "latent"),
                        (include_cluster, "cluster"),
                        (include_annotations, "annotations"),
                        (include_grid_videos, "grid_videos"),
                        (include_analysis, "analysis"),
                        (include_source_videos, "source_videos"),
                    ) if flag
                ]
                manifest = build_run_manifest(
                    pp, project_name=project_name, session_id=session_id,
                    components=selected, generated_at=timestamp,
                )
                zf.writestr(
                    "run_manifest.json",
                    json.dumps(manifest, indent=2, ensure_ascii=False),
                )
            except Exception as exc:  # never fail an export over the manifest
                logger.warning("Export: could not write run_manifest.json: %s", exc)

            for i, (src, arc_name) in enumerate(unique_files, 1):
                if not os.path.isfile(src):
                    logger.warning("Export: source file missing, skipping: %s", src)
                    continue

                size_str = _human_size(src)
                yield (
                    f"**📦 Packaging…** ({i}/{total}) `{arc_name}` ({size_str})",
                    _hide_download,
                )

                # Copy via shutil.copyfile to a temp staging file, then add to zip
                staging = os.path.join(tmp_dir, f"_stage_{i}")
                try:
                    shutil.copyfile(src, staging)
                    zf.write(staging, arc_name)
                    os.unlink(staging)
                    packaged_count += 1
                except Exception as exc:
                    logger.error("Export: failed to copy %s: %s", src, exc)
                    yield (
                        f"**⚠️ Warning:** Could not copy `{arc_name}`: {exc}",
                        _hide_download,
                    )

        zip_size = _human_size(zip_path)
        yield (
            f"**✅ Export complete!** `{zip_name}` ({zip_size}) — {packaged_count} file(s)",
            gr.update(value=zip_path, visible=True),
        )

    except Exception as exc:
        logger.exception("Export failed")
        yield f"**❌ Export failed:** {exc}", _hide_download


def on_export_nwb(
    storage_path: str,
    project_name: str,
    session_description: str,
    experimenter: str,
):
    """Export project cluster results to NWB format.

    Args:
        storage_path: Root storage path.
        project_name: Project name.
        session_description: NWB session description string.
        experimenter: Experimenter name.

    Returns:
        (status_markdown, gr.File update)
    """
    if not storage_path or not project_name:
        return "**❌ Error:** No project selected.", gr.update(value=None, visible=False)

    from castle.service.nwb_service import export_project_nwb

    project_path = os.path.join(storage_path, project_name)
    try:
        nwb_path = export_project_nwb(
            project_path=project_path,
            session_description=session_description or "CASTLE behavioral analysis",
            experimenter=experimenter or "",
        )
        size_str = _human_size(nwb_path)
        return (
            f"**✅ NWB export complete!** `{os.path.basename(nwb_path)}` ({size_str})",
            gr.update(value=nwb_path, visible=True),
        )
    except ImportError as exc:
        return (
            f"**❌ NWB export requires pynwb:** `pip install pynwb`. ({exc})",
            gr.update(value=None, visible=False),
        )
    except FileNotFoundError as exc:
        return (
            f"**❌ NWB export failed:** {exc}",
            gr.update(value=None, visible=False),
        )
    except Exception as exc:
        logger.exception("NWB export failed")
        return (
            f"**❌ NWB export failed:** {exc}",
            gr.update(value=None, visible=False),
        )


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
            ui["export_cancel_btn"] = gr.Button("Cancel", interactive=False, scale=1)

        ui["status"] = gr.Markdown("**Status:** Ready")
        ui["download"] = gr.File(label="⬇️ Download ZIP", visible=False)

        gr.Markdown("---")
        gr.Markdown("### 🧪 NWB Export")
        gr.Markdown(
            "Export project cluster results and ethogram data to "
            "[NWB (Neurodata Without Borders)](https://nwb.org/) format. "
            "Requires `pynwb` to be installed."
        )
        with gr.Row():
            ui["nwb_session_desc"] = gr.Textbox(
                label="Session Description",
                value="CASTLE behavioral analysis",
                scale=3,
            )
            ui["nwb_experimenter"] = gr.Textbox(
                label="Experimenter",
                value="",
                scale=1,
            )
        ui["nwb_export_btn"] = gr.Button("🧪 Export NWB", variant="secondary")
        ui["nwb_status"] = gr.Markdown("**NWB Status:** Ready")
        ui["nwb_download"] = gr.File(label="⬇️ Download NWB", visible=False)

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

    # Stream on_export's yields directly so users see per-file progress.
    # Requires app.queue() to be enabled at module scope (see app.py).
    def _before_export():
        return gr.update(interactive=False), gr.update(interactive=True)

    def _after_export():
        return gr.update(interactive=True), gr.update(interactive=False)

    _export_click = ui["export_btn"].click(
        fn=_before_export,
        outputs=[ui["export_btn"], ui["export_cancel_btn"]],
        queue=False,
    )
    # Save the generator event so the cancel button can cancel it.
    # Gradio 6.x requires the cancelled event to have queue=True;
    # the generator .then() has queue=True by default.
    _export_gen = _export_click.then(
        fn=on_export,
        inputs=_export_inputs,
        outputs=[ui["status"], ui["download"]],
    )
    _export_gen.then(
        fn=_after_export,
        outputs=[ui["export_btn"], ui["export_cancel_btn"]],
        queue=False,
    )

    ui["export_cancel_btn"].click(
        fn=_after_export,
        outputs=[ui["export_btn"], ui["export_cancel_btn"]],
        cancels=[_export_gen],
        queue=False,
    )

    ui["nwb_export_btn"].click(
        fn=on_export_nwb,
        inputs=[
            storage_path,
            project_name,
            ui["nwb_session_desc"],
            ui["nwb_experimenter"],
        ],
        outputs=[ui["nwb_status"], ui["nwb_download"]],
    )

    return ui
