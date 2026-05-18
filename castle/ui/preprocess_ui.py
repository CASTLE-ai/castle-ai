"""
castle/ui/preprocess_ui.py
Gradio UI for stabilized camera preprocessing (Kinematics Info Transfusion).

All service calls are delegated to castle.service.preprocessing_service and
castle.core.project.  No business logic lives in this module.
"""

from __future__ import annotations

import logging
from typing import Any, List

import gradio as gr

from castle.utils.video_manager import get_project_videos

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper callbacks
# ---------------------------------------------------------------------------


def _get_roi_ids(storage_path: str, project_name: str, video_name: str) -> List[int]:
    """Read unique non-zero ROI IDs from the tracking mask HDF5 file by sampling frames."""
    import h5py
    import numpy as np
    from pathlib import Path

    if not storage_path or not project_name or not video_name:
        return []
    try:
        from castle.core.project import get_project_config
        project_path, _ = get_project_config(storage_path, project_name)
        mask_path = Path(project_path) / "track" / video_name / "mask_list.h5"
        if not mask_path.exists():
            return []
        roi_ids: set[int] = set()
        with h5py.File(str(mask_path), "r") as f:
            sample_keys = sorted(f.keys(), key=lambda x: int(x) if x.isdigit() else 0)[:20]
            for key in sample_keys:
                roi_ids.update(int(v) for v in np.unique(f[key][:]) if v > 0)
        return sorted(roi_ids)
    except Exception:
        return []


def _list_videos(storage_path: str, project_name: str) -> gr.update:
    """Populate the video dropdown from the current project."""
    if not storage_path or not project_name:
        return gr.update(choices=[], value=None)
    videos = get_project_videos(storage_path, project_name)
    return gr.update(choices=videos, value=videos[0] if videos else None)


def _populate_roi_dropdowns(
    storage_path: str, project_name: str, video_name: str
) -> tuple:
    """Return updated choices for anterior/posterior ROI dropdowns based on mask data."""
    roi_ids = _get_roi_ids(storage_path, project_name, video_name)
    if not roi_ids:
        return (
            gr.update(choices=[], value=None,
                      info="No tracking data found. Complete tracking first."),
            gr.update(choices=[], value=None,
                      info="No tracking data found. Complete tracking first."),
        )
    default_ant = roi_ids[0] if len(roi_ids) >= 1 else None
    default_post = roi_ids[1] if len(roi_ids) >= 2 else (roi_ids[0] if roi_ids else None)
    return (
        gr.update(choices=roi_ids, value=default_ant,
                  info="ROI used as body centroid and orientation reference."),
        gr.update(choices=roi_ids, value=default_post,
                  info="ROI used to compute body→head angle."),
    )


def _check_min_crop_warning(min_crop: int, output_size: int) -> gr.update:
    """Show a warning when min_crop < 0.5 × output_size."""
    try:
        if int(min_crop) < 0.5 * int(output_size):
            return gr.update(
                visible=True,
                value=(
                    f"⚠️ **min_crop ({int(min_crop)} px) < output_size / 2 "
                    f"({int(output_size) // 2} px)**  — the crop may be upscaled "
                    "significantly, degrading feature quality."
                ),
            )
    except (TypeError, ValueError):
        pass
    return gr.update(visible=False, value="")


def _save_params(
    storage_path: str,
    project_name: str,
    anterior_roi_id: int,
    posterior_roi_id: int,
    fc: float,
    order: int,
    margin: int,
    min_crop: int,
    output_size: int,
) -> str:
    """Save current KIT parameters to the project config."""
    from castle.core.project import save_kit_params

    if not storage_path or not project_name:
        return "⚠️ No project open — cannot save."

    params = {
        "body_roi_id": int(anterior_roi_id),   # config key kept for backward compat
        "head_roi_id": int(posterior_roi_id),  # config key kept for backward compat
        "fc": float(fc),
        "order": int(order),
        "margin": int(margin),
        "min_crop": int(min_crop),
        "output_size": int(output_size),
    }
    try:
        save_kit_params(storage_path, project_name, params)
        return "✅ KIT parameters saved to project config."
    except Exception as exc:
        logger.exception("save_kit_params failed")
        return f"❌ Save failed: {exc}"


def _load_params(
    storage_path: str,
    project_name: str,
) -> tuple:
    """Load KIT parameters from the project config and return field updates."""
    from castle.core.project import load_kit_params

    if not storage_path or not project_name:
        return (
            gr.update(),  # anterior_roi_id
            gr.update(),  # posterior_roi_id
            gr.update(),  # fc
            gr.update(),  # order
            gr.update(),  # margin
            gr.update(),  # min_crop
            gr.update(),  # output_size
            "⚠️ No project open.",
        )

    params = load_kit_params(storage_path, project_name)
    if params is None:
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            "⚠️ No saved KIT params found. Run and save parameters first.",
        )

    return (
        gr.update(value=params.get("body_roi_id", 1)),   # → anterior_roi_id
        gr.update(value=params.get("head_roi_id", 2)),   # → posterior_roi_id
        gr.update(value=params.get("fc", 0.25)),
        gr.update(value=params.get("order", 2)),
        gr.update(value=params.get("margin", 75)),
        gr.update(value=params.get("min_crop", 300)),
        gr.update(value=params.get("output_size", 592)),
        "✅ Parameters loaded from project config.",
    )


def _run_preprocessing(
    storage_path: str,
    project_name: str,
    video_name: str,
    anterior_roi_id: int,
    posterior_roi_id: int,
    fc: float,
    order: int,
    margin: int,
    min_crop: int,
    output_size: int,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
) -> tuple[Any, str]:
    """Run stabilized camera preprocessing and return (full_video_path, diag_text)."""
    if not storage_path or not project_name:
        gr.Warning(
            "No project open. Please create or open a project in the 'Project' tab first."
        )
        return None, "No project selected."

    if not video_name:
        gr.Warning("No video selected. Please choose a video from the dropdown.")
        return None, "No video selected."

    if not anterior_roi_id or not posterior_roi_id:
        gr.Warning("Please select both Anterior and Posterior ROI IDs.")
        return None, "ROI IDs not selected."

    if anterior_roi_id == posterior_roi_id:
        gr.Warning(
            "Anterior ROI ID and Posterior ROI ID must be different. "
            "Please select a different ROI for each."
        )
        return None, "anterior_roi_id == posterior_roi_id — please fix ROI ids."

    from castle.service.preprocessing_service import preprocess_stabilized_camera
    from castle.core.project import save_kit_params

    def _cb(fraction: float, description: str = "") -> None:
        progress(fraction, desc=description or "Processing…")

    try:
        result = preprocess_stabilized_camera(
            storage_path=storage_path,
            project_name=project_name,
            video_name=video_name,
            body_roi_id=int(anterior_roi_id),
            head_roi_id=int(posterior_roi_id),
            fc=float(fc),
            order=int(order),
            margin=int(margin),
            min_crop=int(min_crop),
            output_size=int(output_size),
            preview_duration=10.0,  # still generated internally; not shown in UI
            progress_callback=_cb,
        )
    except Exception as exc:
        logger.exception("Stabilized camera preprocessing failed")
        gr.Warning(
            f"Stabilized camera preprocessing failed. Check that ROI tracking is "
            f"complete for this video and the ROI IDs are correct. Details: {exc}"
        )
        return None, f"Error: {exc}"

    # Auto-save params after a successful run
    try:
        save_kit_params(storage_path, project_name, {
            "body_roi_id": int(anterior_roi_id),
            "head_roi_id": int(posterior_roi_id),
            "fc": float(fc),
            "order": int(order),
            "margin": int(margin),
            "min_crop": int(min_crop),
            "output_size": int(output_size),
        })
    except Exception:
        logger.warning("Auto-save of KIT params failed (non-fatal)", exc_info=True)

    diag = result["diagnostics"]
    diag_text = (
        f"✅ Preprocessing complete\n"
        f"  Video    : {result['preprocessed_video_path']}\n"
        f"  Frames   : {result['n_frames']}\n"
        f"\nDiagnostics:\n"
        f"  HP residual RMS       : {diag['hp_residual_rms']:.2f} px\n"
        f"  % frames at min_crop  : {diag['pct_at_min_crop']:.1f}%\n"
        f"  Speed–crop correlation: {diag['speed_crop_correlation']:.3f}\n"
        f"\n💾 Parameters auto-saved to project config."
    )

    # Show the full stabilised video
    return result["preprocessed_video_path"], diag_text


# ---------------------------------------------------------------------------
# UI factory
# ---------------------------------------------------------------------------


def create_preprocess_ui(
    storage_path: gr.State,
    project_name: gr.State,
    preprocess_tab: gr.Tab,
) -> dict[str, Any]:
    """Build the Kinematics Info Transfusion tab UI and wire up event handlers.

    Returns a dict of all interactive components (no Accordion objects) so
    edit_ui.handle_edit_click can safely use list(ui.values()) as outputs.
    """
    ui: dict[str, Any] = {}

    # gr.Column is a valid output component for visibility control.
    # Accordion is NOT — so both accordions below are local vars only.
    with gr.Column(visible=False) as ui["wrapper"]:
        with gr.Accordion("🎥 Kinematics Info Transfusion", open=True):
            gr.Markdown(
                """
                Apply zero-phase Butterworth low-pass filtering to the tracked ROI trajectory
                and generate a stabilised, rotated, and cropped video aligned to the animal's
                body axis — optimised for DINOv2 / DINOv3 feature extraction.

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
                    )
                    ui["anterior_roi_id"] = gr.Dropdown(
                        label="Anterior ROI ID",
                        choices=[],
                        value=None,
                        interactive=True,
                        info="ROI used as body centroid and orientation reference (anterior end).",
                    )
                    ui["posterior_roi_id"] = gr.Dropdown(
                        label="Posterior ROI ID",
                        choices=[],
                        value=None,
                        interactive=True,
                        info="ROI used to compute body axis angle (posterior end).",
                    )

                    with gr.Accordion("⚙ Advanced Parameters", open=False):
                        ui["fc"] = gr.Number(
                            label="Low-pass cutoff (Hz)",
                            value=0.25,
                            minimum=0.001,
                            interactive=True,
                            info=(
                                "Butterworth low-pass filter cutoff frequency. Default: 0.25 Hz. "
                                "Lower values = smoother camera movement."
                            ),
                        )
                        ui["order"] = gr.Number(
                            label="Filter order",
                            value=2,
                            precision=0,
                            minimum=1,
                            interactive=True,
                            info="Butterworth filter order. Default: 2.",
                        )
                        ui["margin"] = gr.Number(
                            label="Crop margin (px)",
                            value=75,
                            precision=0,
                            minimum=0,
                            interactive=True,
                            info="Extra padding pixels added around the crop region. Default: 75 px.",
                        )
                        ui["min_crop"] = gr.Number(
                            label="Min crop size (px)",
                            value=300,
                            precision=0,
                            minimum=64,
                            interactive=True,
                            info="Minimum crop size in pixels. Default: 300 px.",
                        )
                        ui["output_size"] = gr.Radio(
                            label="Output frame size",
                            choices=[("518 px (DINOv2 ViT-B/14)", 518),
                                     ("592 px (DINOv3 ViT-B/16)", 592)],
                            value=592,
                            interactive=True,
                            info="Output resolution. Match the model you will use for extraction.",
                        )
                        ui["min_crop_warning"] = gr.Markdown(
                            value="",
                            visible=False,
                        )

                    with gr.Row():
                        ui["save_btn"] = gr.Button(
                            "💾 Save params",
                            variant="secondary",
                        )
                        ui["load_btn"] = gr.Button(
                            "📂 Load saved params",
                            variant="secondary",
                        )

                    ui["param_status"] = gr.Textbox(
                        label="",
                        value="",
                        interactive=False,
                        lines=1,
                    )

                    ui["run_btn"] = gr.Button(
                        "▶ Run Stabilized Camera",
                        variant="primary",
                    )

                with gr.Column(scale=2):
                    ui["preview_video"] = gr.Video(
                        label="Stabilised Video",
                    )
                    ui["diag_text"] = gr.Textbox(
                        label="Diagnostics",
                        lines=12,
                        interactive=False,
                    )

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    preprocess_tab.select(
        fn=_list_videos,
        inputs=[storage_path, project_name],
        outputs=[ui["video_drop"]],
    )

    # When video is selected, populate ROI dropdowns from mask file
    ui["video_drop"].change(
        fn=_populate_roi_dropdowns,
        inputs=[storage_path, project_name, ui["video_drop"]],
        outputs=[ui["anterior_roi_id"], ui["posterior_roi_id"]],
    )

    # min_crop warning: show when min_crop < 0.5 × output_size
    for trigger in (ui["min_crop"], ui["output_size"]):
        trigger.change(
            fn=_check_min_crop_warning,
            inputs=[ui["min_crop"], ui["output_size"]],
            outputs=[ui["min_crop_warning"]],
        )

    # Save params
    ui["save_btn"].click(
        fn=_save_params,
        inputs=[
            storage_path, project_name,
            ui["anterior_roi_id"], ui["posterior_roi_id"],
            ui["fc"], ui["order"], ui["margin"],
            ui["min_crop"], ui["output_size"],
        ],
        outputs=[ui["param_status"]],
    )

    # Load params
    ui["load_btn"].click(
        fn=_load_params,
        inputs=[storage_path, project_name],
        outputs=[
            ui["anterior_roi_id"], ui["posterior_roi_id"],
            ui["fc"], ui["order"], ui["margin"],
            ui["min_crop"], ui["output_size"],
            ui["param_status"],
        ],
    )

    # Run preprocessing (auto-saves on success, shows full stabilised video)
    ui["run_btn"].click(
        fn=_run_preprocessing,
        inputs=[
            storage_path,
            project_name,
            ui["video_drop"],
            ui["anterior_roi_id"],
            ui["posterior_roi_id"],
            ui["fc"],
            ui["order"],
            ui["margin"],
            ui["min_crop"],
            ui["output_size"],
        ],
        outputs=[ui["preview_video"], ui["diag_text"]],
    )

    return ui
