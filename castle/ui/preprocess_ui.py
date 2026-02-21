"""
castle/ui/preprocess_ui.py
Gradio UI for stabilized camera preprocessing.

All service calls are delegated to castle.service.preprocessing_service.
"""

from __future__ import annotations

import logging
from typing import Any

import gradio as gr

from castle.utils.video_manager import get_project_videos

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper callbacks
# ---------------------------------------------------------------------------


def _list_videos(storage_path: str, project_name: str) -> gr.update:
    """Populate the video dropdown from the current project."""
    videos = get_project_videos(storage_path, project_name)
    return gr.update(choices=videos, value=videos[0] if videos else None)


def _run_preprocessing(
    storage_path: str,
    project_name: str,
    video_name: str,
    body_roi_id: int,
    head_roi_id: int,
    fc: float,
    order: int,
    margin: int,
    min_crop: int,
    output_size: int,
    preview_duration: float,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
) -> tuple[Any, str]:
    """Run stabilized camera preprocessing and return (preview_video, diag_text)."""
    if not storage_path or not project_name:
        gr.Warning("Please open a project first.")
        return None, "No project selected."

    if not video_name:
        gr.Warning("Please select a video.")
        return None, "No video selected."

    if body_roi_id == head_roi_id:
        gr.Warning("body_roi_id and head_roi_id must be different.")
        return None, "body_roi_id == head_roi_id — please fix ROI ids."

    from castle.service.preprocessing_service import preprocess_stabilized_camera

    def _cb(fraction: float, description: str = "") -> None:
        progress(fraction, desc=description or "Processing…")

    try:
        result = preprocess_stabilized_camera(
            storage_path=storage_path,
            project_name=project_name,
            video_name=video_name,
            body_roi_id=int(body_roi_id),
            head_roi_id=int(head_roi_id),
            fc=float(fc),
            order=int(order),
            margin=int(margin),
            min_crop=int(min_crop),
            output_size=int(output_size),
            preview_duration=float(preview_duration),
            progress_callback=_cb,
        )
    except Exception as exc:
        logger.exception("Stabilized camera preprocessing failed")
        gr.Warning(f"Preprocessing failed: {exc}")
        return None, f"Error: {exc}"

    diag = result["diagnostics"]
    diag_text = (
        f"✅ Preprocessing complete\n"
        f"  Video    : {result['preprocessed_video_path']}\n"
        f"  Preview  : {result['preview_path']}\n"
        f"  Frames   : {result['n_frames']}\n"
        f"\nDiagnostics:\n"
        f"  HP residual RMS  : {diag['hp_residual_rms']:.2f} px\n"
        f"  % frames at min_crop : {diag['pct_at_min_crop']:.1f}%\n"
        f"  Speed–crop correlation : {diag['speed_crop_correlation']:.3f}\n"
    )

    return result["preview_path"], diag_text


# ---------------------------------------------------------------------------
# UI factory
# ---------------------------------------------------------------------------


def create_preprocess_ui(
    storage_path: gr.State,
    project_name: gr.State,
    preprocess_tab: gr.Tab,
) -> dict[str, Any]:
    """Build the Preprocessing tab UI and wire up event handlers.

    Parameters
    ----------
    storage_path : gr.State
        Shared state holding the storage directory path.
    project_name : gr.State
        Shared state holding the current project name.
    preprocess_tab : gr.Tab
        The parent Tab component that triggers ``select`` events.

    Returns
    -------
    dict
        Dictionary of all created UI components (for visibility toggling).
    """
    ui: dict[str, Any] = {}

    with gr.Accordion("🎥 Stabilized Camera Preprocessing", open=True, visible=False) as ui["main_accordion"]:
        gr.Markdown(
            """
            Apply zero-phase Butterworth low-pass filtering to the tracked ROI trajectory
            and generate a stabilised, rotated, and cropped video optimised for DINOv2
            feature extraction.

            **Requires**: tracking must be completed for the selected video
            (`track/{video}/mask_list.h5` must exist).
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                ui["video_drop"] = gr.Dropdown(
                    label="Select Video",
                    choices=[],
                    interactive=True,
                    visible=False,
                )
                ui["body_roi_id"] = gr.Number(
                    label="Body ROI ID",
                    value=1,
                    precision=0,
                    minimum=1,
                    interactive=True,
                    visible=False,
                    info="ROI pixel value for the body (centroid + orientation reference).",
                )
                ui["head_roi_id"] = gr.Number(
                    label="Head ROI ID",
                    value=2,
                    precision=0,
                    minimum=1,
                    interactive=True,
                    visible=False,
                    info="ROI pixel value for the head (used to compute body→head angle).",
                )

                with gr.Accordion("⚙ Advanced Parameters", open=False, visible=False) as ui["adv_accordion"]:
                    ui["fc"] = gr.Number(
                        label="Low-pass cutoff (Hz)",
                        value=0.25,
                        minimum=0.001,
                        interactive=True,
                        info="Butterworth cutoff frequency. Lower = smoother camera.",
                    )
                    ui["order"] = gr.Number(
                        label="Filter order",
                        value=2,
                        precision=0,
                        minimum=1,
                        interactive=True,
                    )
                    ui["margin"] = gr.Number(
                        label="Crop margin (px)",
                        value=75,
                        precision=0,
                        minimum=0,
                        interactive=True,
                        info="Extra pixels added around the HP residual displacement.",
                    )
                    ui["min_crop"] = gr.Number(
                        label="Min crop size (px)",
                        value=300,
                        precision=0,
                        minimum=64,
                        interactive=True,
                    )
                    ui["output_size"] = gr.Number(
                        label="Output frame size (px)",
                        value=518,
                        precision=0,
                        minimum=64,
                        interactive=True,
                        info="Side length of square output frames. 518 = DINOv2 ViT-B/14 optimal.",
                    )
                    ui["preview_duration"] = gr.Number(
                        label="Preview duration (s)",
                        value=10.0,
                        minimum=1.0,
                        interactive=True,
                    )

                ui["run_btn"] = gr.Button(
                    "▶ Run Stabilized Camera",
                    variant="primary",
                    visible=False,
                )

            with gr.Column(scale=2):
                ui["preview_video"] = gr.Video(
                    label="Stabilised Preview",
                    visible=False,
                )
                ui["diag_text"] = gr.Textbox(
                    label="Diagnostics",
                    lines=10,
                    interactive=False,
                    visible=False,
                )

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    # Populate video list when the parent tab is selected
    preprocess_tab.select(
        fn=_list_videos,
        inputs=[storage_path, project_name],
        outputs=[ui["video_drop"]],
    )

    # Run preprocessing
    ui["run_btn"].click(
        fn=_run_preprocessing,
        inputs=[
            storage_path,
            project_name,
            ui["video_drop"],
            ui["body_roi_id"],
            ui["head_roi_id"],
            ui["fc"],
            ui["order"],
            ui["margin"],
            ui["min_crop"],
            ui["output_size"],
            ui["preview_duration"],
        ],
        outputs=[ui["preview_video"], ui["diag_text"]],
    )

    return ui
