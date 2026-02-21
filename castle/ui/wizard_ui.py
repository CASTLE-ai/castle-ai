"""
castle/ui/wizard_ui.py
Wizard Mode — Step-by-step guided analysis for first-time users.

Design goal: "An 80-year-old professor's first time using the app."
- Zero jargon
- One decision per screen
- Smart defaults for everything
- Plain-English error messages
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import gradio as gr

from castle.service.auto_config import get_gpu_info, recommend_config, estimate_pipeline_time
from castle.service.project_service import create_project

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_STEP_ICONS = {
    "pending": "⏳",
    "running": "🔄",
    "done": "✅",
    "error": "❌",
}

_PIPELINE_STEPS = [
    ("preprocess", "Stabilise video"),
    ("track", "Track the animal"),
    ("extract", "Analyse movement features"),
    ("cluster", "Discover behaviour patterns"),
]


def _fmt_seconds(sec: float) -> str:
    """Format seconds as e.g. '2 min 15 sec'."""
    sec = int(sec)
    if sec < 60:
        return f"{sec} sec"
    m, s = divmod(sec, 60)
    return f"{m} min {s} sec"


def _status_table(step_states: dict) -> str:
    """Build an HTML-ish markdown table for pipeline step status."""
    rows = []
    for key, label in _PIPELINE_STEPS:
        icon = _STEP_ICONS.get(step_states.get(key, "pending"), "⏳")
        rows.append(f"| {icon} | **{label}** |")
    header = "| Status | Step |\n|--------|------|\n"
    return header + "\n".join(rows)


def _first_frame_path(video_path: str) -> Optional[str]:
    """Extract the first frame of the video to a temp PNG."""
    try:
        import cv2  # noqa: PLC0415

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return None
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        cv2.imwrite(tmp.name, frame)
        return tmp.name
    except Exception:
        return None


def _video_info_text(video_path: str) -> str:
    """Human-readable video summary."""
    try:
        import cv2  # noqa: PLC0415

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return "⚠️ Could not read the video file."
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        duration = frames / fps if fps else 0
        m, s = divmod(int(duration), 60)
        return (
            f"📹 **{w} × {h}** pixels · **{fps:.1f}** frames/second · "
            f"**{m}:{s:02d}** minutes"
        )
    except Exception:
        return "ℹ️ Video info unavailable."


# ---------------------------------------------------------------------------
# WizardUI class
# ---------------------------------------------------------------------------

class WizardUI:
    """Builds the 'Quick Start Wizard' Gradio tab."""

    def __init__(self, storage_path_state: gr.State, project_name_state: gr.State):
        self._storage_state = storage_path_state
        self._project_state = project_name_state

    # ------------------------------------------------------------------
    # Public builder
    # ------------------------------------------------------------------

    def build(self) -> dict:  # noqa: PLR0914
        """Construct the wizard UI inside the currently active gr.Tab context.

        Returns a dict of Gradio components (for wiring if needed).
        """
        components: dict = {}

        gr.Markdown(
            """
# 🧭 Quick Start Wizard
**New here? This wizard will guide you through your first analysis in 3 simple steps.**
No technical knowledge required — we handle all the settings automatically.
""",
            elem_id="wizard-header",
        )

        # We use a top-level Tabs to simulate a multi-step flow.
        # Tab visibility is controlled by showing/hiding via gr.update.
        with gr.Tabs(elem_id="wizard-steps") as wizard_tabs:

            # ----------------------------------------------------------------
            # STEP 1 — Upload video
            # ----------------------------------------------------------------
            with gr.Tab(label="Step 1 — Upload Video", id="step1"):
                gr.Markdown(
                    """
## 📂 Step 1: Upload your video
Drag and drop your video file below (MP4, AVI, MOV are all fine).
We'll automatically detect the video settings — you don't need to change anything.
"""
                )
                with gr.Row():
                    with gr.Column(scale=2):
                        video_upload = gr.File(
                            label="🎬 Drop your video here",
                            file_types=["video", ".mp4", ".avi", ".mov", ".mkv"],
                            elem_id="wizard-video-upload",
                        )
                        project_name_input = gr.Textbox(
                            label="📁 Give this project a name",
                            placeholder="e.g. mouse-experiment-2026",
                            info=(
                                "Use letters, numbers, and hyphens. "
                                "A folder with this name will be created for your results."
                            ),
                        )
                    with gr.Column(scale=1):
                        video_info_md = gr.Markdown("_Upload a video to see its details._")
                        first_frame_img = gr.Image(
                            label="Preview (first frame)",
                            visible=False,
                            interactive=False,
                        )

                step1_status = gr.Markdown("")
                step1_next_btn = gr.Button(
                    "Continue →  (Step 2: Identify the Animal)",
                    variant="primary",
                    visible=False,
                )

            # ----------------------------------------------------------------
            # STEP 2 — Identify the animal
            # ----------------------------------------------------------------
            with gr.Tab(label="Step 2 — Identify Animal", id="step2"):
                gr.Markdown(
                    """
## 🐾 Step 2: Tell us which animal to track

Look at the image below (from your video). Each visible animal or body region
has a number. Enter the number that corresponds to the **main body** of the
animal you want to track.

> 💡 **Not sure?** Enter **1** — that's almost always the body.
> You can always change this later in the advanced settings.
"""
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        step2_frame_img = gr.Image(
                            label="Your video — first frame",
                            interactive=False,
                        )
                    with gr.Column(scale=1):
                        body_roi_id = gr.Number(
                            value=1,
                            label="Body region number",
                            minimum=1,
                            step=1,
                            precision=0,
                            info=(
                                "The number of the tracking region that covers "
                                "the animal's body. Usually 1."
                            ),
                        )
                        head_roi_id = gr.Number(
                            value=2,
                            label="Head region number (optional)",
                            minimum=1,
                            step=1,
                            precision=0,
                            info=(
                                "If your video has a separate head marker, "
                                "enter its number here. Otherwise leave as 2."
                            ),
                        )
                        config_preview_md = gr.Markdown(
                            "_We'll auto-configure everything else for you._"
                        )

                with gr.Row():
                    gr.Button("← Back")
                    step2_next_btn = gr.Button(
                        "Start Analysis →", variant="primary"
                    )

            # ----------------------------------------------------------------
            # STEP 3 — Progress / Run
            # ----------------------------------------------------------------
            with gr.Tab(label="Step 3 — Running Analysis", id="step3"):
                gr.Markdown(
                    """
## ⚙️ Step 3: Analysis in progress…

Sit back! We're now:
1. Stabilising your video
2. Tracking the animal
3. Analysing movement patterns
4. Discovering behaviour clusters

This may take several minutes depending on video length and your computer.
"""
                )
                run_status_md = gr.Markdown(_status_table({}))
                eta_md = gr.Markdown("_Calculating estimated time…_")
                run_btn = gr.Button(
                    "▶  Run Analysis",
                    variant="primary",
                    elem_id="wizard-run-btn",
                )
                run_log = gr.Textbox(
                    label="Progress log",
                    lines=10,
                    max_lines=20,
                    interactive=False,
                    placeholder="Progress will appear here…",
                )
                run_error_md = gr.Markdown("", visible=False)

            # ----------------------------------------------------------------
            # COMPLETION screen
            # ----------------------------------------------------------------
            with gr.Tab(label="✅ Done!", id="step_done"):
                gr.Markdown(
                    """
## 🎉 Analysis complete!

Your video has been analysed. Here's what to do next:

- Switch to the **4. Behavior Microscope** tab to explore the discovered behaviour clusters.
- Use the **Cluster Annotator** to label each cluster with a meaningful name.
- Visit **5. Analysis** to compare across sessions.

> 💡 All your results are saved in the project folder.
"""
                )
                go_to_annotator_btn = gr.Button(
                    "Open Behavior Explorer →", variant="primary"
                )
                gr.Markdown("")

        # ----------------------------------------------------------------
        # Hidden shared state
        # ----------------------------------------------------------------
        _video_path_state = gr.State(None)
        _config_state = gr.State(None)

        # ----------------------------------------------------------------
        # Event: video uploaded → show preview + info
        # ----------------------------------------------------------------
        def on_video_upload(file_obj):
            if file_obj is None:
                return (
                    gr.update(value="_Upload a video to see its details._"),
                    gr.update(visible=False, value=None),
                    "",
                    gr.update(visible=False),
                    None,
                )
            path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
            info = _video_info_text(path)
            frame = _first_frame_path(path)
            return (
                gr.update(value=info),
                gr.update(visible=frame is not None, value=frame),
                "",
                gr.update(visible=True),
                path,
            )

        video_upload.change(
            fn=on_video_upload,
            inputs=[video_upload],
            outputs=[
                video_info_md,
                first_frame_img,
                step1_status,
                step1_next_btn,
                _video_path_state,
            ],
        )

        # ----------------------------------------------------------------
        # Event: Step 1 → Step 2 (validate project name & show config)
        # ----------------------------------------------------------------
        def on_step1_next(video_path, project_name, storage_path):
            errors = []
            if not video_path:
                errors.append("Please upload a video first.")
            if not project_name or not project_name.strip():
                errors.append("Please enter a project name.")
            else:
                # Basic name sanity check
                import re  # noqa: PLC0415

                if not re.match(r"^[\w\-]+$", project_name.strip()):
                    errors.append(
                        "Project name may only contain letters, numbers, and hyphens."
                    )
            if errors:
                return (
                    gr.update(value="⚠️ " + " ".join(errors)),
                    gr.update(value=None),
                    gr.update(value="_Upload a video to see its details._"),
                    None,
                )

            # Auto-config
            try:
                gpu = get_gpu_info()
                cfg = recommend_config(video_path, gpu_info=gpu)
            except Exception as exc:
                logger.warning("auto_config failed: %s", exc)
                cfg = {}

            pre = cfg.get("preprocessing", {})
            gpu_info = cfg.get("gpu_info", {})
            gpu_line = (
                f"🖥️ GPU: **{gpu_info.get('name', 'CPU only')}** "
                f"({gpu_info.get('vram_mb', 0)} MB VRAM)"
                if gpu_info.get("available")
                else "🖥️ Running on **CPU** (no GPU detected)"
            )

            config_md = (
                f"**Auto-detected settings:**\n\n"
                f"- Smoothing level: `{pre.get('fc', 0.25)}`\n"
                f"- Crop margin: `{pre.get('margin', 75)} px`\n"
                f"- Min frame size: `{pre.get('min_crop', 300)} px`\n"
                f"- Output frame: `{pre.get('output_size', 518)} px`\n\n"
                f"{gpu_line}\n\n"
                f"*These are optimal for your video. Nothing to change!*"
            )

            frame = _first_frame_path(video_path)
            return (
                gr.update(value=""),  # clear step1 status
                gr.update(value=frame),  # step2 image
                gr.update(value=config_md),
                cfg,
            )

        step1_next_btn.click(
            fn=on_step1_next,
            inputs=[_video_path_state, project_name_input, self._storage_state],
            outputs=[step1_status, step2_frame_img, config_preview_md, _config_state],
        )

        # ----------------------------------------------------------------
        # Event: Step 2 → Step 3 (show ETA)
        # ----------------------------------------------------------------
        def on_step2_next(video_path, config):
            if not video_path:
                return gr.update(value=_status_table({})), gr.update(
                    value="⚠️ No video path — go back to Step 1."
                )
            try:
                eta = estimate_pipeline_time(video_path, config or {})
                eta_text = f"⏱️ **Estimated time:** {_fmt_seconds(eta)}"
            except Exception:
                eta_text = "⏱️ Time estimate unavailable."

            return (
                gr.update(value=_status_table({})),
                gr.update(value=eta_text),
            )

        step2_next_btn.click(
            fn=on_step2_next,
            inputs=[_video_path_state, _config_state],
            outputs=[run_status_md, eta_md],
        )

        # ----------------------------------------------------------------
        # Event: Run button — full pipeline
        # ----------------------------------------------------------------
        def on_run(video_path, project_name, storage_path, body_roi, head_roi, config):
            """Generator that yields (status_md, log, error_md) updates."""
            step_states: dict = {k: "pending" for k, _ in _PIPELINE_STEPS}

            def _emit(state_update: dict, log_line: str, error: str = ""):
                step_states.update(state_update)
                return (
                    _status_table(step_states),
                    log_line,
                    gr.update(value=error, visible=bool(error)),
                )

            if not video_path or not project_name:
                yield _emit({}, "⚠️ Missing video or project name.", "Please go back and fill in all fields.")
                return

            storage_path = storage_path or "projects/"
            body_roi = int(body_roi or 1)
            head_roi = int(head_roi or 2)

            # ---- Create project ----
            yield _emit({}, f"📁 Creating project '{project_name}'…")
            try:
                create_project(storage_path, project_name)
            except FileExistsError:
                yield _emit({}, f"📁 Project '{project_name}' already exists — reusing.")
            except Exception as exc:
                yield _emit({}, "", f"❌ Could not create project: {exc}")
                return

            # ---- Copy video into project sources ----
            yield _emit({}, "📂 Copying video into project…")
            try:
                src_path = os.path.join(storage_path, project_name, "sources")
                os.makedirs(src_path, exist_ok=True)
                video_name = Path(video_path).name
                dest = os.path.join(src_path, video_name)
                if not os.path.exists(dest):
                    shutil.copyfile(video_path, dest)
            except Exception as exc:
                yield _emit({}, "", f"❌ Could not copy video: {exc}")
                return

            # ---- Preprocessing ----
            yield _emit({"preprocess": "running"}, "🎬 Stabilising video…")
            try:
                from castle.service.preprocessing_service import (  # noqa: PLC0415
                    preprocess_stabilized_camera,
                )

                pre = (config or {}).get("preprocessing", {})
                result = preprocess_stabilized_camera(
                    storage_path=storage_path,
                    project_name=project_name,
                    video_name=video_name,
                    body_roi_id=body_roi,
                    head_roi_id=head_roi,
                    fc=pre.get("fc", 0.25),
                    margin=pre.get("margin", 75),
                    min_crop=pre.get("min_crop", 300),
                    output_size=pre.get("output_size", 518),
                )
                if result.get("status") != "ok":
                    raise RuntimeError(result.get("message", "Unknown error"))
                yield _emit({"preprocess": "done"}, "✅ Video stabilised.")
            except Exception as exc:
                friendly = _friendly_error("preprocess", str(exc))
                yield _emit({"preprocess": "error"}, "", friendly)
                return

            # ---- Tracking ----
            yield _emit({"track": "running"}, "🐾 Tracking the animal — this takes a while…")
            try:
                from castle.service.tracking_service import run_tracking  # noqa: PLC0415

                ext = (config or {}).get("extraction", {})
                result = run_tracking(
                    storage_path=storage_path,
                    project_name=project_name,
                    video_name=video_name,
                    batch_size=ext.get("batch_size", 8),
                )
                if result.get("status") != "ok":
                    raise RuntimeError(result.get("message", "Unknown error"))
                yield _emit({"track": "done"}, "✅ Animal tracked.")
            except Exception as exc:
                friendly = _friendly_error("track", str(exc))
                yield _emit({"track": "error"}, "", friendly)
                return

            # ---- Feature extraction ----
            yield _emit({"extract": "running"}, "🔬 Analysing movement features…")
            try:
                from castle.service.extraction_service import (  # noqa: PLC0415
                    extract_latent,
                    make_preprocess_config,
                )

                ext = (config or {}).get("extraction", {})
                pre_cfg = make_preprocess_config(
                    center_roi_switch=True,
                    center_roi_id=body_roi,
                    center_roi_crop_width=ext.get("center_roi_crop_width", 300),
                    center_roi_crop_height=ext.get("center_roi_crop_height", 300),
                )
                result = extract_latent(
                    storage_path=storage_path,
                    project_name=project_name,
                    video_name=video_name,
                    preprocess=pre_cfg,
                    batch_size=ext.get("batch_size", 8),
                )
                if result.get("status") != "ok":
                    raise RuntimeError(result.get("message", "Unknown error"))
                yield _emit({"extract": "done"}, "✅ Features extracted.")
            except Exception as exc:
                friendly = _friendly_error("extract", str(exc))
                yield _emit({"extract": "error"}, "", friendly)
                return

            # ---- Clustering ----
            yield _emit({"cluster": "running"}, "🧩 Discovering behaviour patterns…")
            try:
                from castle.service.clustering_service import run_clustering  # noqa: PLC0415

                cl = (config or {}).get("clustering", {})
                result = run_clustering(
                    storage_path=storage_path,
                    project_name=project_name,
                    video_name=video_name,
                    n_clusters=cl.get("n_clusters", 10),
                )
                if result.get("status") != "ok":
                    raise RuntimeError(result.get("message", "Unknown error"))
                yield _emit({"cluster": "done"}, "✅ Behaviour clusters found!")
            except Exception as exc:
                friendly = _friendly_error("cluster", str(exc))
                yield _emit({"cluster": "error"}, "", friendly)
                return

            yield _emit({}, "🎉 All done! Switch to the ✅ Done! tab.", "")

        run_btn.click(
            fn=on_run,
            inputs=[
                _video_path_state,
                project_name_input,
                self._storage_state,
                body_roi_id,
                head_roi_id,
                _config_state,
            ],
            outputs=[run_status_md, run_log, run_error_md],
        )

        components.update(
            {
                "wizard_tabs": wizard_tabs,
                "video_upload": video_upload,
                "project_name_input": project_name_input,
                "step1_next_btn": step1_next_btn,
                "step2_next_btn": step2_next_btn,
                "run_btn": run_btn,
                "go_to_annotator_btn": go_to_annotator_btn,
            }
        )
        return components


# ---------------------------------------------------------------------------
# Plain-English error explanations
# ---------------------------------------------------------------------------

_ERROR_HINTS: dict[str, dict[str, str]] = {
    "preprocess": {
        "No such file": "The video file was not found. Try re-uploading it.",
        "Permission": "The application does not have permission to write files. Check the storage folder.",
        "codec": "The video format is not supported. Try converting to MP4 first.",
        "default": (
            "The video stabilisation step failed.\n\n"
            "**What to try:**\n"
            "- Make sure the video plays normally in a media player.\n"
            "- Check that there is enough disk space.\n"
            "- If the animal is very small, try a higher-resolution recording."
        ),
    },
    "track": {
        "CUDA": "The GPU ran out of memory. Reduce video length or use a shorter clip.",
        "default": (
            "The animal tracking step failed.\n\n"
            "**What to try:**\n"
            "- Make sure the animal is clearly visible in the video.\n"
            "- Ensure the ROI IDs match the regions drawn in the Tracking ROIs tab.\n"
            "- Check that the video has not been corrupted."
        ),
    },
    "extract": {
        "default": (
            "Feature extraction failed.\n\n"
            "**What to try:**\n"
            "- Make sure the tracking step finished successfully.\n"
            "- Re-run tracking before retrying."
        ),
    },
    "cluster": {
        "default": (
            "Behaviour clustering failed.\n\n"
            "**What to try:**\n"
            "- Make sure the feature extraction step finished successfully.\n"
            "- Your video may be too short — try a recording of at least 5 minutes."
        ),
    },
}


def _friendly_error(step: str, exc_str: str) -> str:
    """Return a plain-English error message for a pipeline step failure."""
    hints = _ERROR_HINTS.get(step, {})
    for keyword, msg in hints.items():
        if keyword != "default" and keyword.lower() in exc_str.lower():
            return f"❌ **Something went wrong — {msg}**\n\n_(Technical detail: {exc_str})_"
    default = hints.get("default", f"An unexpected error occurred: {exc_str}")
    return f"❌ **{default}**\n\n_(Technical detail: {exc_str})_"
