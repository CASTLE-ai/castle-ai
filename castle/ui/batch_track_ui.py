"""Batch tracking UI components for Gradio."""

import os
import json
import time
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
from tqdm import tqdm

from castle.core.logging_config import setup_logger

logger = setup_logger(__name__)

from ..utils.h5_io import H5IO  # noqa: E402
from ..utils.plot import generate_mix_image  # noqa: E402
from ..utils.video_io import ReadArray, WriteArray  # noqa: E402
from ..utils.tracking_manager import ROITracker, read_roi_labels  # noqa: E402
from ..utils.analysis_utils import compute_roi_info, save_kinematic_csv  # noqa: E402
from ..service.tracking_service import track_videos  # noqa: E402
from ..core.gpu_pool import available_cuda_devices  # noqa: E402


def update_video_count(storage_path_val, project_name_val):
    if not project_name_val:
        return gr.update(value="Please select a project first")
    videos = get_project_videos(storage_path_val, project_name_val)
    return gr.update(value=f"Found {len(videos)} videos")

def refresh_gallery(storage_path_val, project_name_val):
    if not project_name_val:
        return [], []
    return read_all_labels_to_gallery(storage_path_val, project_name_val)


def read_all_labels_without_video_filter(storage_path: str, project_name: str) -> List[Dict[str, Any]]:
    """
    Read all label data in the project without video filtering.
    
    Delegates to tracking_manager.read_roi_labels with include_metadata=True
    to get video_name, frame_index, and file_path for each label.
    """
    if not project_name:
        return []
    return read_roi_labels(storage_path, project_name, include_metadata=True)


def read_all_labels_to_gallery(storage_path: str, project_name: str) -> Tuple[List[Dict[str, Any]], List[Tuple[Any, str]]]:
    """
    Generate gallery list containing all labels.
    
    Args:
        storage_path: Storage path  
        project_name: Project name
        
    Returns:
        Tuple of label list and gallery list
    """
    # Use our wrapper function to read all labels
    label_list = read_all_labels_without_video_filter(storage_path, project_name)
    
    gallery_list = [
        (generate_mix_image(label["frame"], label["mask"]), label["index"])
        for label in label_list
    ]
    return label_list, gallery_list


def get_project_videos(storage_path: str, project_name: str) -> List[str]:
    """
    Get list of all videos in the project.
    
    Args:
        storage_path: Storage path
        project_name: Project name
        
    Returns:
        List of video names
    """
    if not project_name:
        return []
        
    project_path = os.path.join(storage_path, project_name)
    config_path = os.path.join(project_path, 'config.json')
    
    try:
        with open(config_path, 'r') as f:
            project_config = json.load(f)
        if 'source' in project_config:
            return sorted(project_config['source'])
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    
    return []


def generate_video_analysis(storage_path: str, project_name: str, video_name: str) -> Tuple[str, str]:
    """
    Generate CSV analysis and mix video for a single video.
    
    Args:
        storage_path: Storage path
        project_name: Project name  
        video_name: Video name
        
    Returns:
        Tuple of (csv_path, mix_video_path)
    """
    project_path = Path(storage_path) / project_name
    track_dir_path = project_path / "track" / video_name
    rois_results_path = track_dir_path / "mask_list.h5"
    
    csv_path = ""
    mix_video_path = ""
    
    if not rois_results_path.exists():
        return csv_path, mix_video_path
    
    try:
        # Load source video for mix video generation
        source_video_path = project_path / "sources" / video_name
        source_video = ReadArray(str(source_video_path))
        
        # Generate CSV analysis
        csv_path = generate_csv_analysis(storage_path, project_name, video_name, source_video)
        
        # Generate mix video
        mix_video_path = generate_mix_video_for_video(storage_path, project_name, video_name, source_video)
        
    except Exception as e:
        logger.error(f"Error generating analysis for {video_name}: {e}")
    
    return csv_path, mix_video_path


def generate_csv_analysis(storage_path: str, project_name: str, video_name: str, source_video) -> str:
    """Generate CSV kinematic analysis for a video.
    
    Uses shared compute_roi_info() and save_kinematic_csv() from analysis_utils.
    """
    project_path = Path(storage_path) / project_name
    track_dir_path = project_path / "track" / video_name
    rois_results_path = track_dir_path / "mask_list.h5"
    
    if not rois_results_path.exists():
        return ""
    
    try:
        rois_results = H5IO(str(rois_results_path), read_only=True)
        try:
            n_rois = rois_results.get_n_rois()
            total_frames = len(rois_results)

            roi_info_list = compute_roi_info(rois_results, n_rois, total_frames)
            csv_path = save_kinematic_csv(str(track_dir_path), video_name, roi_info_list)
        finally:
            rois_results.close()
        return csv_path

    except Exception as e:
        logger.error(f"Error generating CSV for {video_name}: {e}")
        return ""


def generate_mix_video_for_video(storage_path: str, project_name: str, video_name: str, source_video) -> str:
    """Generate mix video for a single video."""
    project_path = Path(storage_path) / project_name
    track_dir_path = project_path / "track" / video_name
    rois_results_path = track_dir_path / "mask_list.h5"
    
    if not rois_results_path.exists():
        return ""
    
    try:
        video_name_wo_extension = video_name.split('.')[0]
        output_path = track_dir_path / f'{video_name_wo_extension}-mix.mp4'

        output = WriteArray(str(output_path), fps=source_video.fps, crf=15)
        rois_results = H5IO(str(rois_results_path), read_only=True)
        try:
            n_frames = len(rois_results)

            for i in range(n_frames):
                rois = rois_results[i]
                frame = source_video[i]
                out_frame = generate_mix_image(frame, rois)
                output.append(out_frame)
        finally:
            rois_results.close()
            output.close()
        return str(output_path)

    except Exception as e:
        logger.error(f"Error generating mix video for {video_name}: {e}")
        return ""



_TRACK_BTN_IDLE = "Start Tracking All Videos"
_CANCEL_BTN_IDLE = "Cancel Tracking"


def _init_cancel_event() -> threading.Event:
    """Fresh per-run cancel flag, stored in gr.State before the work generator runs."""
    return threading.Event()


def _request_cancel(cancel_event):
    """Cancel handler: set the flag and immediately relabel the button.

    Immediate feedback (the generator keeps running its in-flight video, so the
    user would otherwise see no reaction). The work generator's final yield
    restores the button's idle label/state.
    """
    if cancel_event is not None:
        cancel_event.set()
    return gr.update(value="Canceling (finishing current video)…", interactive=False)


def track_all_videos(
    storage_path: str,
    project_name: str,
    model_aot: str = "r50_deaotl",
    skip_existing: bool = True,
    use_multi_gpu: bool = True,
    cancel_event=None,
    progress=gr.Progress(),
):
    """Execute tracking on all videos in the project (generator → crash-safe UI).

    Yields ``(progress_text, start_btn_update, cancel_btn_update)`` so the button
    states are owned by this function: the **first** yield flips to the running
    state (Start disabled, Cancel enabled) and the **final** yield always resets
    them — on success, error, *or* cancel. A `.then()` chain that set the buttons
    in a separate step would leave the UI stuck forever if the body raised; a
    generator with a catch-all + guaranteed final yield does not.

    Args:
        skip_existing: when True (default) only videos missing ``mask_list.h5`` are
            tracked; when False every video is (re-)tracked / overwritten.
        use_multi_gpu: when True and ≥2 GPUs are visible, spread whole videos
            across GPUs (passes explicit ``device_ids`` — independent of the
            ``CASTLE_MULTI_GPU`` env gate). Falls back to single-GPU otherwise.
        cancel_event: ``threading.Event``; once set, ``track_videos`` stops
            launching new videos (in-flight ones finish).
    """
    # First yield: running state. Start disabled, Cancel enabled.
    yield (
        "🚀 Batch tracking started… (live progress on the bar above; the full "
        "per-video log appears here when the run finishes).",
        gr.update(interactive=False),
        gr.update(value=_CANCEL_BTN_IDLE, interactive=True),
    )

    messages: List[str] = []
    try:
        if not project_name:
            messages.append("Error: No project selected")
        else:
            all_videos = get_project_videos(storage_path, project_name)
            if not all_videos:
                messages.append("Error: No videos found in project")
            else:
                # --- Pre-flight: decide which videos to process ---
                videos_to_process: List[str] = []
                messages.append(f"Starting pre-flight check for {len(all_videos)} videos...")
                for video_name in all_videos:
                    rois_results_path = Path(storage_path) / project_name / "track" / video_name / "mask_list.h5"
                    if skip_existing and rois_results_path.exists():
                        messages.append(f"  ⏩ Skipping existing video: {video_name}")
                    else:
                        videos_to_process.append(video_name)

                if not videos_to_process:
                    messages.append("All videos already tracked. Nothing to track.")
                    progress(1.0, desc="Batch tracking completed (no new videos)")
                else:
                    total_videos_to_process = len(videos_to_process)
                    messages.append(f"Found {total_videos_to_process} videos to process")

                    # Resolve the GPU set from the UI toggle (independent of the
                    # CASTLE_MULTI_GPU env gate). [] → sequential single-GPU path.
                    if use_multi_gpu:
                        device_ids = available_cuda_devices()
                        if not device_ids:
                            messages.append("ℹ️  Multi-GPU requested but <2 GPUs available; running single-GPU.")
                        else:
                            messages.append(f"🖥️  Multi-GPU: spreading videos across {len(device_ids)} GPUs.")
                    else:
                        device_ids = []

                    # --- Execution ---
                    # DeAOT is sequential within a video, so parallelism is
                    # video-level (one whole video per GPU). Per-video CSV +
                    # mix-video analysis runs in on_video_done; progress is
                    # reported per completed video (concurrent videos make a
                    # single frame-level bar meaningless).
                    success_count = {"n": 0}
                    failed_videos: List[str] = []

                    def _on_video_done(video_name: str, status: str) -> None:
                        if status == "Done":
                            success_count["n"] += 1
                            messages.append(f"✅ Completed tracking for video {video_name}")
                            logger.info("Completed tracking for %s", video_name)
                            try:
                                messages.append(f"Generating analysis files for {video_name}...")
                                csv_path, mix_video_path = generate_video_analysis(storage_path, project_name, video_name)
                                if csv_path:
                                    messages.append(f"  ✅ Generated CSV: {os.path.basename(csv_path)}")
                                if mix_video_path:
                                    messages.append(f"  ✅ Generated mix video: {os.path.basename(mix_video_path)}")
                            except Exception as e:  # noqa: BLE001
                                messages.append(f"  ⚠️  Warning: Failed to generate analysis files for {video_name}: {str(e)}")
                                logger.warning("Analysis generation failed for %s: %s", video_name, e)
                        elif status in ("Skipped", "Skip"):
                            messages.append(f"  Skipping existing video: {video_name}")
                        elif status == "Cancel":
                            messages.append(f"  🛑 Cancelled mid-track (partial output discarded): {video_name}")
                        else:
                            failed_videos.append(video_name)
                            messages.append(f"❌ Tracking failed for video {video_name}: {status}")
                            logger.error("Tracking failed for %s: %s", video_name, status)

                    # ETA: stamp t0 *after* pre-flight (skipped videos never enter
                    # the timing window), total = tracked-only count.
                    t0 = time.time()

                    def _progress_cb(frac: float, desc: str) -> None:
                        completed = round(frac * total_videos_to_process)
                        if completed == 0:  # guard div-by-zero on the first callback
                            progress(frac, desc=desc)
                            return
                        elapsed = time.time() - t0
                        eta = elapsed / completed * (total_videos_to_process - completed)
                        em, es = divmod(int(elapsed), 60)
                        rm, rs = divmod(int(eta), 60)
                        progress(
                            frac,
                            desc=(f"{completed}/{total_videos_to_process} · "
                                  f"elapsed {em}:{es:02d} · ETA ~{rm}:{rs:02d}"),
                        )

                    # Pre-flight already applied skip_existing, so skip_existing=False here.
                    track_videos(
                        storage_path, project_name, videos_to_process,
                        model=model_aot, skip_existing=False,
                        device_ids=device_ids,
                        progress_callback=_progress_cb,
                        on_video_done=_on_video_done,
                        cancel_event=cancel_event,
                    )

                    cancelled = cancel_event is not None and cancel_event.is_set()
                    progress(1.0, desc="Batch tracking cancelled" if cancelled else "Batch tracking completed")

                    head = "🛑 Batch tracking cancelled" if cancelled else "🎉 Batch tracking completed!"
                    result_msg = f"\n{head} Successfully processed {success_count['n']}/{total_videos_to_process} videos"
                    result_msg += "\n📊 CSV analysis files and 🎬 mix videos generated for successful tracks"
                    if failed_videos:
                        result_msg += f"\n⚠️  Failed videos: {', '.join(failed_videos)}"
                    messages.append(result_msg)
    except Exception as e:  # noqa: BLE001 - never re-raise; reset the UI below
        logger.exception("Batch tracking crashed")
        messages.append(f"\n❌ Batch tracking crashed: {e}")

    # Final yield: always reset the buttons (success / error / cancel).
    yield (
        "\n".join(messages),
        gr.update(value=_TRACK_BTN_IDLE, interactive=True),
        gr.update(value=_CANCEL_BTN_IDLE, interactive=False),
    )


def get_select_index(evt: gr.SelectData):
    """Get index of selected item"""
    return evt.index


_DELETE_LABEL_IDLE = "Delete Selected Label"


def _armed_label_text(filename: str) -> str:
    return f"⚠️ Delete '{filename}'? Click again to confirm"


def delete_selected_label(
    storage_path: str,
    project_name: str,
    label_list: List[Dict],
    selected_index: int,
    armed_index: Any,
) -> Tuple[List[Dict], List[Tuple], Any, Any]:
    """Two-click delete: first click arms (label shows filename), second click confirms.

    Args:
        storage_path: Storage path
        project_name: Project name
        label_list: Label list
        selected_index: Index of currently selected gallery item
        armed_index: Previously armed index (None if idle)

    Returns:
        (updated_label_list, gallery_list, delete_btn_update, new_armed_index)
    """
    # Nothing selected — reset to idle.
    if selected_index is None or not label_list or selected_index >= len(label_list):
        new_label_list, new_gallery = read_all_labels_to_gallery(storage_path, project_name)
        return (
            new_label_list,
            new_gallery,
            gr.update(value=_DELETE_LABEL_IDLE),
            None,
        )

    selected_label = label_list[selected_index]
    file_path = selected_label["file_path"]
    filename = os.path.basename(file_path)

    # Selection changed between arming and confirmation — re-arm on new target.
    if armed_index != selected_index:
        return (
            label_list,
            gr.update(),
            gr.update(value=_armed_label_text(filename)),
            selected_index,
        )

    # Same selection clicked twice — execute delete.
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            gr.Info(f"Deleted label file: {selected_label['index']}")
        else:
            gr.Info(f"File does not exist: {file_path}")
    except Exception as e:
        gr.Warning(f"Error occurred while deleting file: {str(e)}")

    new_label_list, new_gallery = read_all_labels_to_gallery(storage_path, project_name)
    return (
        new_label_list,
        new_gallery,
        gr.update(value=_DELETE_LABEL_IDLE),
        None,
    )


def _reset_delete_label_armed_state(*args):
    """Reset the armed state when the user changes selection."""
    return gr.update(value=_DELETE_LABEL_IDLE), None


def create_batch_track_ui(storage_path: str, project_name: str, batch_tracking_tab: gr.Tab) -> Tuple[Dict[str, Any], Dict[str, Any]]: # 修改簽名，新增 batch_tracking_tab 參數
    ui: Dict[str, Any] = {}
    states: Dict[str, Any] = {}
    
    # State variables
    states["label_list_state"] = gr.State(None)
    states["selected_index_state"] = gr.State(None)
    # Tracks which gallery index is currently armed for delete (None = idle).
    states["delete_armed_index_state"] = gr.State(None)
    # Per-run cancel flag (threading.Event), created fresh on each Start click.
    states["cancel_event"] = gr.State(None)
    
    with gr.Row():
        with gr.Column(scale=2):
            # ROI Knowledge area
            with gr.Group():
                
                ui["gallery"] = gr.Gallery(
                    label="ROI Prompts",
                    show_label=True,
                    allow_preview=False,
                    object_fit="contain",
                    interactive=False,
                    columns=3,
                    visible=False # 設置為預設不可見
                )
                
                ui["delete_selected_btn"] = gr.Button(
                    "Delete Selected Label",
                    variant="secondary",
                    visible=False # 設置為預設不可見
                )
        
        with gr.Column(scale=1):
            # Batch tracking control area
                ui["video_count_display"] = gr.Textbox(
                    label="Project Video Count",
                    interactive=False,
                    visible=False # 設置為預設不可見
                )
                
                ui["model_dropdown"] = gr.Dropdown(
                    choices=["r50_deaotl", "swinb_deaotl"],
                    label="Tracking Model",
                    info="ResNet-50 or Swin-transformer",
                    value="r50_deaotl",
                    interactive=True,
                    visible=False # 設置為預設不可見
                )

                # GPU status indicator + multi-GPU toggle (refreshed on tab-select
                # from the live device count). Default value reflects the hardware
                # at build time; the tab.select step corrects it on each visit.
                _avail_devices = available_cuda_devices()
                ui["gpu_indicator"] = gr.Textbox(
                    label="GPU Status",
                    interactive=False,
                    visible=False # 設置為預設不可見
                )
                ui["multi_gpu_toggle"] = gr.Checkbox(
                    label="Use multiple GPUs",
                    value=bool(_avail_devices),
                    interactive=bool(_avail_devices),
                    info="Spread whole videos across GPUs. Close other GPU/heavy apps before a big batch.",
                    visible=False # 設置為預設不可見
                )

                ui["skip_existing_checkbox"] = gr.Checkbox(
                    label="Skip already-tracked videos",
                    value=True,
                    info="On: only track videos missing mask_list.h5. Off: re-track everything (overwrites).",
                    visible=False # 設置為預設不可見
                )

                ui["track_all_btn"] = gr.Button(
                    "Start Tracking All Videos",
                    variant="primary",
                    size="lg",
                    visible=False # 設置為預設不可見
                )

                ui["cancel_tracking_btn"] = gr.Button(
                    "Cancel Tracking",
                    variant="stop",
                    interactive=False,
                    visible=False # 設置為預設不可見
                )

                ui["progress_text"] = gr.Textbox(
                    label="Progress & Results",
                    interactive=False,
                    visible=False, # 設置為預設不可見
                    lines=15,
                    max_lines=15
                )
    
    # Event bindings
    # Gallery selection event — also disarms any pending delete since the user
    # may have shifted focus to a different label.
    (
        ui["gallery"].select(
            fn=get_select_index,
            outputs=states["selected_index_state"],
        )
        .then(
            fn=_reset_delete_label_armed_state,
            inputs=None,
            outputs=[ui["delete_selected_btn"], states["delete_armed_index_state"]],
            queue=False,
        )
    )

    # Delete button event — two-click: first click arms (label shows filename),
    # second click confirms and executes.
    ui["delete_selected_btn"].click(
        fn=delete_selected_label,
        inputs=[
            storage_path,
            project_name,
            states["label_list_state"],
            states["selected_index_state"],
            states["delete_armed_index_state"],
        ],
        outputs=[
            states["label_list_state"],
            ui["gallery"],
            ui["delete_selected_btn"],
            states["delete_armed_index_state"],
        ],
        queue=False,
    )
    
    # Batch tracking button event. A generator that owns its own button states
    # (running → reset), preceded by a step that creates a fresh cancel flag.
    ui["track_all_btn"].click(
        fn=_init_cancel_event,
        outputs=states["cancel_event"],
        queue=False,
    ).then(
        fn=track_all_videos,
        inputs=[
            storage_path,
            project_name,
            ui["model_dropdown"],
            ui["skip_existing_checkbox"],
            ui["multi_gpu_toggle"],
            states["cancel_event"],
        ],
        outputs=[ui["progress_text"], ui["track_all_btn"], ui["cancel_tracking_btn"]],
        show_progress=True,
    )

    # Cancel: set the flag + immediate relabel (the generator's final yield
    # restores the idle label). queue=False so it runs even while the work
    # generator occupies the queue.
    ui["cancel_tracking_btn"].click(
        fn=_request_cancel,
        inputs=states["cancel_event"],
        outputs=ui["cancel_tracking_btn"],
        queue=False,
    )

    # 新增 batch_tracking_tab.select 事件綁定
    all_batch_ui_elements = [
        ui["gallery"],
        ui["delete_selected_btn"],
        ui["video_count_display"],
        ui["model_dropdown"],
        ui["gpu_indicator"],
        ui["multi_gpu_toggle"],
        ui["skip_existing_checkbox"],
        ui["track_all_btn"],
        ui["cancel_tracking_btn"],
        ui["progress_text"]
    ]

    def show_batch_track_ui(project_name_val):
        is_visible = project_name_val is not None
        return [gr.update(visible=is_visible)] * len(all_batch_ui_elements)

    def init_gpu_controls():
        """Set the GPU indicator text + toggle default/interactivity from the live
        device count. Returns an explicit component-keyed dict (robust to future
        component reordering, unlike positional tuples)."""
        try:
            import torch
            n = torch.cuda.device_count() if torch.cuda.is_available() else 0
        except Exception:  # noqa: BLE001
            n = 0
        if n >= 2:
            indicator = f"🖥️ {n} GPUs detected — multi-GPU available"
            toggle = gr.update(value=True, interactive=True)
        elif n == 1:
            indicator = "🖥️ 1 GPU (multi-GPU N/A)"
            toggle = gr.update(value=False, interactive=False)
        else:
            indicator = "⚠️ No CUDA GPU detected"
            toggle = gr.update(value=False, interactive=False)
        return {ui["gpu_indicator"]: gr.update(value=indicator), ui["multi_gpu_toggle"]: toggle}

    (
        batch_tracking_tab.select(
            fn=show_batch_track_ui,
            inputs=[project_name],
            outputs=all_batch_ui_elements
        )
        .then(
            fn=init_gpu_controls,
            inputs=None,
            outputs=[ui["gpu_indicator"], ui["multi_gpu_toggle"]],
        )
        .then( # 顯示 UI 後，再載入內容
            fn=update_video_count,
            inputs=[storage_path, project_name],
            outputs=ui["video_count_display"]
        )
        .then(
            fn=refresh_gallery,
            inputs=[storage_path, project_name],
            outputs=[states["label_list_state"], ui["gallery"]]
        )
    )

    return ui, states
