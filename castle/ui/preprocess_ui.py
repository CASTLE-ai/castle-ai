"""
castle/ui/preprocess_ui.py
Gradio UI for the Pre-process tab (KIT and Center ROI + Crop).

All service calls are delegated to castle.service.preprocessing_service and
castle.core.preprocess_session.  No business logic lives in this module.
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
    """Sample the first 20 frames of the mask H5 to find unique non-zero ROI IDs."""
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
    """Populate the video dropdown (single video + 'All' option)."""
    if not storage_path or not project_name:
        return gr.update(choices=[], value=None)
    videos = get_project_videos(storage_path, project_name)
    choices = videos + (["All"] if videos else [])
    return gr.update(choices=choices, value=videos[0] if videos else None)


def _populate_roi_dropdowns(
    storage_path: str, project_name: str, video_name: str
) -> tuple:
    """Return updated choices for Anterior/Posterior ROI dropdowns from mask data."""
    # When 'All' is selected, use the first available video for sampling
    actual_video = video_name
    if video_name == "All":
        videos = get_project_videos(storage_path, project_name)
        actual_video = videos[0] if videos else None

    roi_ids = _get_roi_ids(storage_path, project_name, actual_video) if actual_video else []

    if not roi_ids:
        empty = gr.update(choices=[], value=None, info="No tracking data found.")
        return empty, empty

    default_ant = roi_ids[0] if len(roi_ids) >= 1 else None
    default_post = roi_ids[1] if len(roi_ids) >= 2 else roi_ids[0]
    return (
        gr.update(choices=roi_ids, value=default_ant,
                  info="ROI used as body centroid and orientation reference (anterior end)."),
        gr.update(choices=roi_ids, value=default_post,
                  info="ROI used to compute body axis angle (posterior end)."),
    )


def _toggle_method_params(method: str) -> tuple:
    """Show KIT column or CenterROI column depending on selected method."""
    kit_visible = (method == "KIT")
    return (
        gr.update(visible=kit_visible),      # kit_params_col
        gr.update(visible=not kit_visible),  # center_roi_params_col
    )


def _compute_session_name(
    method: str,
    ant_roi: Any, post_roi: Any,
    fc: Any, order: Any, margin: Any, min_crop: Any, output_size: Any,
    center_roi_id: Any, crop_w: Any, crop_h: Any,
) -> str:
    """Compute and display the session name that would be created with current params."""
    try:
        from castle.core.preprocess_session import session_name_from_params, session_id_from_name
        if method == "KIT":
            params = {
                "anterior_roi_id": int(ant_roi or 1),
                "posterior_roi_id": int(post_roi or 2),
                "fc": float(fc or 0.25),
                "order": int(order or 2),
                "margin": int(margin or 75),
                "min_crop": int(min_crop or 300),
                "output_size": int(output_size or 592),
            }
        else:
            params = {
                "roi_id": int(center_roi_id or 1),
                "crop_width": int(crop_w or 300),
                "crop_height": int(crop_h or 300),
            }
        name = session_name_from_params(method, params)
        sid = session_id_from_name(name)
        return f"{name} [{sid}]"
    except Exception as exc:
        return f"(invalid params: {exc})"


def _list_sessions_dropdown(storage_path: str, project_name: str) -> gr.update:
    """Populate sessions dropdown, newest first."""
    if not storage_path or not project_name:
        return gr.update(choices=[], value=None)
    try:
        from castle.core.preprocess_session import list_sessions
        metas = list_sessions(storage_path, project_name)
        choices = [
            f"{m['session_name']} | {m['method']} | {len(m.get('videos', []))} videos"
            for m in metas
        ]
        return gr.update(choices=choices, value=choices[0] if choices else None)
    except Exception:
        return gr.update(choices=[], value=None)


def _delete_session_ui(
    storage_path: str, project_name: str, session_display: str
) -> tuple:
    """Delete the selected session and its latent entries, then refresh dropdown."""
    if not storage_path or not project_name or not session_display:
        return _list_sessions_dropdown(storage_path, project_name), "⚠️ No session selected."

    try:
        # session_display format: "KIT_a1_p2_fc0.25_sz592 | KIT | 2 videos"
        session_name = session_display.split(" | ")[0].strip()
        from castle.core.preprocess_session import session_id_from_name
        session_id = session_id_from_name(session_name)
        from castle.service.extraction_service import delete_session_with_latent_cleanup
        delete_session_with_latent_cleanup(storage_path, project_name, session_id)
        status = f"✅ Deleted session '{session_name}' ({session_id})."
    except Exception as exc:
        logger.exception("delete_session_ui failed")
        status = f"❌ Delete failed: {exc}"

    return _list_sessions_dropdown(storage_path, project_name), status


def _get_preview_frame(
    storage_path: str,
    project_name: str,
    video_name: str,
    method: str,
    ant_roi: Any,
    post_roi: Any,
    fc: Any,
    order: Any,
    margin: Any,
    min_crop: Any,
    output_size: Any,
    center_roi_id: Any,
    crop_w: Any,
    crop_h: Any,
) -> Any:
    """Return a single representative stabilised/cropped frame for preview."""
    import numpy as np

    if not storage_path or not project_name or not video_name or video_name == "All":
        return None

    try:
        from pathlib import Path
        import h5py
        from castle.core.project import get_project_config
        from castle.utils.video_io import ReadArray

        project_path, _ = get_project_config(storage_path, project_name)
        mask_h5_path = str(Path(project_path) / "track" / video_name / "mask_list.h5")
        source_path = str(Path(project_path) / "sources" / video_name)

        if method == "KIT":
            from castle.core.stabilized_camera import (
                StabilizedCamera,
                extract_centroids_from_masks,
                extract_orientations_from_masks,
            )
            body_roi_id = int(ant_roi or 1)
            head_roi_id = int(post_roi or 2)
            _fc = float(fc or 0.25)
            _order = int(order or 2)
            _margin = int(margin or 75)
            _min_crop = int(min_crop or 300)
            _output_size = int(output_size or 592)

            with ReadArray(source_path) as vr:
                n_frames = len(vr)
                _fps = vr.fps
                # Find first frame in the middle range
                target = n_frames // 4
                frame_bgr = vr[target]

            positions = extract_centroids_from_masks(mask_h5_path, body_roi_id, n_frames)
            angles = extract_orientations_from_masks(
                mask_h5_path, body_roi_id, head_roi_id, n_frames
            )
            cam = StabilizedCamera(
                positions=positions, angles=angles, fps=_fps,
                fc=_fc, order=_order, margin=_margin,
                min_crop=_min_crop, output_size=_output_size,
            )

            with h5py.File(mask_h5_path, "r") as f:
                orig_mask = f[str(target)][:] if str(target) in f else None
            if orig_mask is None:
                return None

            stabilised = cam.generate_frame(frame_bgr, target)
            trans_mask = cam.generate_mask(orig_mask, target)

            # Build a side-by-side (original | stabilised) BGR image
            combined = _make_preview_image(frame_bgr, orig_mask, stabilised, trans_mask)
            return combined[..., ::-1]  # BGR → RGB for Gradio

        else:  # CenterROI
            from castle.core.data import Preprocess

            _roi_id = int(center_roi_id or 1)
            _crop_w = int(crop_w or 300)
            _crop_h = int(crop_h or 300)
            preprocess = Preprocess(
                center_roi_switch=True,
                center_roi_id=_roi_id,
                center_roi_crop_width=_crop_w,
                center_roi_crop_height=_crop_h,
            )

            with ReadArray(source_path) as vr:
                n_frames = len(vr)
                target = n_frames // 4
                frame_bgr = vr[target]

            with h5py.File(mask_h5_path, "r") as f:
                orig_mask = f[str(target)][:] if str(target) in f else None
            if orig_mask is None:
                return None

            try:
                cropped_frame, cropped_mask = preprocess.transform(frame_bgr, orig_mask)
            except Exception:
                return None

            combined = _make_preview_image(frame_bgr, orig_mask, cropped_frame, cropped_mask)
            return combined[..., ::-1]

    except Exception as exc:
        logger.warning("Preview frame generation failed: %s", exc)
        return None


def _make_preview_image(orig_frame, orig_mask, proc_frame, proc_mask):
    """Build a side-by-side comparison: original+mask overlay | processed+mask overlay."""
    import cv2
    import numpy as np

    def _overlay(frame, mask):
        out = frame.copy()
        coloured = np.zeros_like(frame)
        coloured[mask > 0] = [0, 180, 0]  # green overlay for ROI
        out = cv2.addWeighted(out, 0.7, coloured, 0.3, 0)
        return out

    h1, w1 = orig_frame.shape[:2]
    h2, w2 = proc_frame.shape[:2]
    target_h = max(h1, h2)

    left = _overlay(orig_frame, orig_mask)
    right = _overlay(proc_frame, proc_mask)

    # Pad shorter side vertically
    if h1 < target_h:
        left = cv2.copyMakeBorder(left, 0, target_h - h1, 0, 0, cv2.BORDER_CONSTANT)
    if h2 < target_h:
        right = cv2.copyMakeBorder(right, 0, target_h - h2, 0, 0, cv2.BORDER_CONSTANT)

    return np.concatenate([left, right], axis=1)


def _run_preprocess(
    storage_path: str,
    project_name: str,
    video_name: str,
    method: str,
    ant_roi: Any,
    post_roi: Any,
    fc: Any,
    order: Any,
    margin: Any,
    min_crop: Any,
    output_size: Any,
    center_roi_id: Any,
    crop_w: Any,
    crop_h: Any,
    skip_existing: bool,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
) -> str:
    """Run pre-processing for one or all videos and return a progress log."""
    if not storage_path or not project_name:
        gr.Warning("No project open. Please create or open a project in the 'Project' tab first.")
        return "No project selected."

    if not video_name:
        gr.Warning("No video selected.")
        return "No video selected."

    from castle.core.project import get_project_config
    _, config = get_project_config(storage_path, project_name)
    video_list = sorted(config.get("source", [])) if video_name == "All" else [video_name]

    log_lines: list[str] = []

    if method == "KIT":
        if not ant_roi or not post_roi:
            gr.Warning("Please select both Anterior and Posterior ROI IDs.")
            return "ROI IDs not selected."
        if int(ant_roi) == int(post_roi):
            gr.Warning("Anterior and Posterior ROI IDs must be different.")
            return "anterior_roi_id == posterior_roi_id."

        from castle.service.preprocessing_service import preprocess_stabilized_camera
        kit_params = {
            "anterior_roi_id": int(ant_roi),
            "posterior_roi_id": int(post_roi),
            "fc": float(fc or 0.25),
            "order": int(order or 2),
            "margin": int(margin or 75),
            "min_crop": int(min_crop or 300),
            "output_size": int(output_size or 592),
        }

        for i, vname in enumerate(video_list):
            log_lines.append(f"[{i + 1}/{len(video_list)}] {vname}…")

            def _cb(frac: float, desc: str = "") -> None:
                progress(frac, desc=f"{vname}: {desc}" if desc else vname)

            try:
                result = preprocess_stabilized_camera(
                    storage_path=storage_path,
                    project_name=project_name,
                    video_name=vname,
                    kit_params=kit_params,
                    skip_existing=skip_existing,
                    progress_callback=_cb,
                )
                diag = result.get("diagnostics", {})
                sid = result.get("session_id", "")
                if diag:
                    log_lines.append(
                        f"  ✅ Done  session={sid}"
                        f"  frames={result.get('n_frames', '?')}"
                        f"  hp_rms={diag.get('hp_residual_rms', 0):.1f}px"
                        f"  @min_crop={diag.get('pct_at_min_crop', 0):.1f}%"
                    )
                else:
                    log_lines.append(f"  ✅ Skipped (already exists)  session={sid}")
            except Exception as exc:
                logger.exception("KIT preprocess failed for %s", vname)
                log_lines.append(f"  ❌ Error: {exc}")

    else:  # CenterROI
        if not center_roi_id:
            gr.Warning("Please select a ROI ID.")
            return "ROI ID not selected."

        from castle.service.preprocessing_service import preprocess_center_crop

        for i, vname in enumerate(video_list):
            log_lines.append(f"[{i + 1}/{len(video_list)}] {vname}…")

            def _cb(frac: float, desc: str = "") -> None:
                progress(frac, desc=f"{vname}: {desc}" if desc else vname)

            try:
                result = preprocess_center_crop(
                    storage_path=storage_path,
                    project_name=project_name,
                    video_name=vname,
                    roi_id=int(center_roi_id),
                    crop_width=int(crop_w or 300),
                    crop_height=int(crop_h or 300),
                    skip_existing=skip_existing,
                    progress_callback=_cb,
                )
                log_lines.append(
                    f"  ✅ Done  session={result.get('session_id', '')}  frames={result.get('n_frames', '?')}"
                )
            except Exception as exc:
                logger.exception("CenterROI preprocess failed for %s", vname)
                log_lines.append(f"  ❌ Error: {exc}")

    return "\n".join(log_lines)


# ---------------------------------------------------------------------------
# UI factory
# ---------------------------------------------------------------------------


def create_preprocess_ui(
    storage_path: gr.State,
    project_name: gr.State,
    preprocess_tab: gr.Tab,
) -> dict[str, Any]:
    """Build the Pre-process tab UI and wire up event handlers.

    Returns a dict of interactive components (no Accordion objects) so
    edit_ui.handle_edit_click can safely iterate over values if needed.
    """
    ui: dict[str, Any] = {}

    with gr.Column(visible=False) as ui["wrapper"]:
        gr.Markdown(
            """
            ## Pre-process (optional)
            Apply KIT (Kinematics Info Transfusion) or Center ROI + Crop to create a
            stabilised, aligned video and matching mask.  The output is saved as a
            **session** (parameter set) and can be selected in **Extract Latent**.

            **Requires**: tracking must be completed (`track/{video}/mask_list.h5` must exist).
            """
        )

        # -- Method selector -----------------------------------------------
        ui["method_radio"] = gr.Radio(
            label="Pre-processing method",
            choices=["KIT", "Center ROI + Crop"],
            value="KIT",
            interactive=True,
        )

        with gr.Row():
            # ---- KIT params column ----------------------------------------
            with gr.Column(visible=True) as ui["kit_params_col"]:
                gr.Markdown("### KIT Parameters")
                with gr.Row():
                    ui["anterior_roi_id"] = gr.Dropdown(
                        label="Anterior ROI ID",
                        choices=[],
                        value=None,
                        interactive=True,
                        info="Body centroid + orientation reference (anterior end).",
                    )
                    ui["posterior_roi_id"] = gr.Dropdown(
                        label="Posterior ROI ID",
                        choices=[],
                        value=None,
                        interactive=True,
                        info="Used to compute body axis angle (posterior end).",
                    )

                with gr.Accordion("⚙ Advanced KIT Parameters", open=False):
                    ui["fc"] = gr.Number(
                        label="Low-pass cutoff (Hz)",
                        value=0.25, minimum=0.001, interactive=True,
                        info="Butterworth LP filter cutoff. Default: 0.25 Hz.",
                    )
                    ui["order"] = gr.Number(
                        label="Filter order",
                        value=2, precision=0, minimum=1, interactive=True,
                        info="Butterworth filter order. Default: 2.",
                    )
                    ui["margin"] = gr.Number(
                        label="Crop margin (px)",
                        value=75, precision=0, minimum=0, interactive=True,
                        info="Extra padding pixels around the crop region. Default: 75 px.",
                    )
                    ui["min_crop"] = gr.Number(
                        label="Min crop size (px)",
                        value=300, precision=0, minimum=64, interactive=True,
                        info="Minimum crop size in pixels. Default: 300 px.",
                    )
                    ui["output_size"] = gr.Radio(
                        label="Output frame size",
                        choices=[("518 px (DINOv2 ViT-B/14)", 518),
                                 ("592 px (DINOv3 ViT-B/16)", 592)],
                        value=592, interactive=True,
                        info="Match the model you will use for extraction.",
                    )

            # ---- Center ROI + Crop params column -------------------------
            with gr.Column(visible=False) as ui["center_roi_params_col"]:
                gr.Markdown("### Center ROI + Crop Parameters")
                ui["center_roi_id"] = gr.Number(
                    label="ROI ID",
                    value=1, precision=0, minimum=1, interactive=True,
                    info="ROI to centre on.",
                )
                with gr.Row():
                    ui["crop_width"] = gr.Number(
                        label="Crop width (px)",
                        value=300, precision=0, minimum=32, interactive=True,
                    )
                    ui["crop_height"] = gr.Number(
                        label="Crop height (px)",
                        value=300, precision=0, minimum=32, interactive=True,
                    )

        # -- Session name preview ------------------------------------------
        ui["session_name_display"] = gr.Textbox(
            label="Session name (auto-computed)",
            value="",
            interactive=False,
            lines=1,
        )

        # -- Video + Preview -----------------------------------------------
        with gr.Row():
            with gr.Column(scale=1):
                ui["video_drop"] = gr.Dropdown(
                    label="Select Video",
                    choices=[],
                    interactive=True,
                    info="Select a single video or 'All' to process all videos.",
                )
                ui["preview_btn"] = gr.Button("🔍 Preview (single frame)", variant="secondary")

            with gr.Column(scale=2):
                ui["preview_image"] = gr.Image(
                    label="Preview: Original | Processed",
                    interactive=False,
                )

        # -- Session management -------------------------------------------
        gr.Markdown("---\n### Session Management")
        with gr.Row():
            ui["sessions_dropdown"] = gr.Dropdown(
                label="Existing sessions (newest first)",
                choices=[],
                value=None,
                interactive=True,
            )
            ui["delete_btn"] = gr.Button("🗑 Delete session", variant="stop")
        ui["session_status"] = gr.Textbox(
            label="",
            value="",
            interactive=False,
            lines=1,
        )

        # -- Run -----------------------------------------------------------
        gr.Markdown("---")
        ui["skip_existing"] = gr.Checkbox(label="Skip existing", value=True)
        ui["run_btn"] = gr.Button("▶ Run Pre-process", variant="primary")
        ui["log_text"] = gr.Textbox(
            label="Log",
            value="",
            interactive=False,
            lines=12,
        )

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    _kit_inputs = [
        storage_path, project_name, ui["video_drop"],
        ui["method_radio"],
        ui["anterior_roi_id"], ui["posterior_roi_id"],
        ui["fc"], ui["order"], ui["margin"], ui["min_crop"], ui["output_size"],
        ui["center_roi_id"], ui["crop_width"], ui["crop_height"],
    ]

    def _on_tab_select(sp, pn):
        vupd = _list_videos(sp, pn)
        supd = _list_sessions_dropdown(sp, pn)
        return vupd, supd

    preprocess_tab.select(
        fn=_on_tab_select,
        inputs=[storage_path, project_name],
        outputs=[ui["video_drop"], ui["sessions_dropdown"]],
    )

    # Populate ROI dropdowns when video changes
    ui["video_drop"].change(
        fn=_populate_roi_dropdowns,
        inputs=[storage_path, project_name, ui["video_drop"]],
        outputs=[ui["anterior_roi_id"], ui["posterior_roi_id"]],
    )

    # Toggle KIT / CenterROI params columns
    ui["method_radio"].change(
        fn=_toggle_method_params,
        inputs=[ui["method_radio"]],
        outputs=[ui["kit_params_col"], ui["center_roi_params_col"]],
    )

    # Auto-update session name display whenever any param changes
    _session_name_inputs = [
        ui["method_radio"],
        ui["anterior_roi_id"], ui["posterior_roi_id"],
        ui["fc"], ui["order"], ui["margin"], ui["min_crop"], ui["output_size"],
        ui["center_roi_id"], ui["crop_width"], ui["crop_height"],
    ]
    for trigger in _session_name_inputs:
        trigger.change(
            fn=_compute_session_name,
            inputs=_session_name_inputs,
            outputs=[ui["session_name_display"]],
        )

    # Preview button
    ui["preview_btn"].click(
        fn=_get_preview_frame,
        inputs=_kit_inputs,
        outputs=[ui["preview_image"]],
    )

    # Delete session
    ui["delete_btn"].click(
        fn=_delete_session_ui,
        inputs=[storage_path, project_name, ui["sessions_dropdown"]],
        outputs=[ui["sessions_dropdown"], ui["session_status"]],
    )

    # Run
    ui["run_btn"].click(
        fn=_run_preprocess,
        inputs=[
            storage_path, project_name, ui["video_drop"],
            ui["method_radio"],
            ui["anterior_roi_id"], ui["posterior_roi_id"],
            ui["fc"], ui["order"], ui["margin"], ui["min_crop"], ui["output_size"],
            ui["center_roi_id"], ui["crop_width"], ui["crop_height"],
            ui["skip_existing"],
        ],
        outputs=[ui["log_text"]],
    ).then(
        fn=_list_sessions_dropdown,
        inputs=[storage_path, project_name],
        outputs=[ui["sessions_dropdown"]],
    )

    return ui
