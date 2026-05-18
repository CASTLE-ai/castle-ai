"""
castle/ui/extract_ui.py
UI Layer for Extraction. 
Delegates all logic to castle.core.extractor.
"""

import logging
import os
import gradio as gr
from tqdm import tqdm  # 新增: 匯入 tqdm

from castle.core.data import Preprocess
from castle.core.extractor import extract_roi_latent_from_video, extract_roi_crop_video, extract_roi_rotation_latent_from_video
from castle.utils.video_manager import get_project_config
from castle.utils.video_io import VideoReader
from castle.utils.h5_io import H5IO
from castle.utils.plot import generate_mix_image

logger = logging.getLogger(__name__)

# ---------------------------
# Helpers
# ---------------------------
def handle_assertion_error(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (AssertionError, ValueError) as e:
            gr.Warning(f"Error: {e}")
            raise
        except Exception as e:
            gr.Warning(f"Unexpected Error: {e}")
            raise
    return wrapper

def init_select_video_list(storage_path, project_name):
    
    # 統一收集所有需要控制的 UI 元件的更新
    updates = []
    
    # 預設所有元件不可見，並清空數值。注意順序必須與 extract_tab.select 的 outputs 順序嚴格一致
    updates.extend([
        gr.update(visible=False), # select_model
        gr.update(visible=False), # select_roi_id
        gr.update(visible=False), # batch_size
        gr.update(choices=[], value=None, visible=False), # select_video
        gr.update(value=0, visible=False), # video_count
        gr.update(visible=False), # skip_existing
        gr.update(visible=False), # center_roi_switch
        gr.update(visible=False), # center_roi_id
        gr.update(visible=False), # center_roi_crop_width
        gr.update(visible=False), # center_roi_crop_height
        gr.update(visible=False), # remove_background_switch
        gr.update(visible=False), # apply_preprocess
        gr.update(visible=False), # display
        gr.update(visible=False), # adv_accordion
        gr.update(visible=False), # extract_btn
        gr.update(visible=False), # extract_crop_video_btn
        gr.update(visible=False), # latent_file_list
        gr.update(visible=False), # auto_batch_btn
        gr.update(value="", visible=False),  # mem_warning
    ])

    if not storage_path or not project_name:
        gr.Warning("No project selected. Please create or open a project in the 'Project' tab first.")
        return updates # 返回預設的隱藏狀態

    try:
        _, config = get_project_config(storage_path, project_name)
        
        # 獲取影片列表
        video_list_from_config = config.get('source', [])
        choices = sorted(video_list_from_config)
        
        # Project Video Count
        video_count_val = len(choices)
        
        # 如果有影片，則顯示相關 UI
        if video_count_val > 0:
            choices_with_all = list(choices) # 複製一份，避免修改原列表
            choices_with_all.append("All") # 確保 "All" 選項在有影片時才加入
            
            # 覆蓋預設的隱藏狀態，設定為可見
            updates[0] = gr.update(visible=True)  # select_model
            updates[1] = gr.update(visible=True)  # select_roi_id
            updates[2] = gr.update(visible=True)  # batch_size
            updates[3] = gr.update(choices=choices_with_all, value="All", visible=True)  # select_video
            updates[4] = gr.update(value=video_count_val, visible=True)  # video_count
            updates[5] = gr.update(visible=True)  # skip_existing
            updates[6] = gr.update(visible=True)  # center_roi_switch
            updates[7] = gr.update(visible=True)  # center_roi_id
            updates[8] = gr.update(visible=True)  # center_roi_crop_width
            updates[9] = gr.update(visible=True)  # center_roi_crop_height
            updates[10] = gr.update(visible=True) # remove_background_switch
            updates[11] = gr.update(visible=True) # apply_preprocess
            updates[12] = gr.update(visible=True) # display
            updates[13] = gr.update(visible=True) # adv_accordion
            updates[14] = gr.update(visible=True) # extract_btn
            updates[15] = gr.update(visible=True) # extract_crop_video_btn
            updates[16] = gr.update(visible=True) # latent_file_list
            updates[17] = gr.update(visible=True) # auto_batch_btn
            # mem_warning (updates[18]) stays hidden until reactive check triggers
        else:
            gr.Warning(
                "No videos found in this project. Please add videos in the "
                "'Source' tab before extracting features."
            )
            # updates 已經是預設的隱藏狀態，無需額外操作

    except Exception as e:
        gr.Warning(
            f"Failed to load video list. Please check that the project is correctly "
            f"configured. Details: {e}"
        )
        # updates 已經是預設的隱藏狀態，無需額外操作

    return updates

# ---------------------------
# Main Action Handlers
# ---------------------------

@handle_assertion_error
def ui_extract_roi_latent(
    storage_path: str, 
    project_name: str, 
    select_model: str, 
    select_roi: str, 
    select_video: str, 
    batch_size: str, 
    preprocess_args: Preprocess, 
    skip_existing: bool,
    rotate_roi_tail_switch: bool = True,
    pooling_method: str = 'weighted_average',
    pooling_scales_list: list = None,
    feature_layers_str: str = '',
    progress=gr.Progress()
) -> str:
    
    messages = []
    if not preprocess_args:
        raise ValueError("Please click Apply on Preprocess settings first.")

    # A-06: Parse advanced extraction options
    parsed_scales = [int(s) for s in pooling_scales_list] if pooling_scales_list else [1, 2, 4]
    parsed_layers = None
    if feature_layers_str and feature_layers_str.strip():
        try:
            parsed_layers = [int(x.strip()) for x in feature_layers_str.split(',') if x.strip()]
        except ValueError:
            raise ValueError(f"Invalid feature layers format: '{feature_layers_str}'. Use comma-separated integers.")
        
    _, config = get_project_config(storage_path, project_name)
    video_list = sorted(config['source']) if select_video == "All" else [select_video]
    
    # --- Pre-flight Check ---
    videos_to_process = []
    messages.append(f"Starting pre-flight check for {len(video_list)} videos...")
    if pooling_method == 'multiscale':
        messages.append(f"  Pooling: multiscale (scales={parsed_scales})")
    if parsed_layers:
        messages.append(f"  Feature layers: {parsed_layers}")
    for video_name in video_list:
        # Construct the expected output path
        # Replicate valid tag logic from extractor.py
        tags = []
        if preprocess_args.center_roi_switch:
            tags.append("ctr")
        if preprocess_args.remove_background_switch:
            tags.append("rmbg")
        # A-06: replicate tag logic
        if pooling_method == 'multiscale' and parsed_scales:
            scales_str = "x".join(str(s) for s in sorted(parsed_scales))
            tags.append(f"spp{scales_str}")
        if parsed_layers:
            layers_str = "x".join(str(lay) for lay in sorted(parsed_layers))
            tags.append(f"L{layers_str}")
        
        suffix = "_".join([select_model] + tags)
        
        latent_dir = os.path.join(storage_path, project_name, 'latent', select_model)
        latent_filename = f'{os.path.splitext(video_name)[0]}_ROI_{select_roi}_{suffix}.npz'
        output_path = os.path.join(latent_dir, latent_filename)
        
        if skip_existing and os.path.exists(output_path):
             messages.append(f"  ⏩ Skipping existing: {video_name}")
             continue
        videos_to_process.append(video_name)
    
    if not videos_to_process:
        messages.append("\n✅ All videos already have latent files. Nothing to extract.")
        return "\n".join(messages)
    
    messages.append(f"\nFound {len(videos_to_process)} new videos to process.")
    
    # --- Execution ---
    success_count = 0
    failed_videos = []
    
    def update_progress(p, desc=None):
        progress(p, desc=desc)

    for video_name in tqdm(videos_to_process, desc="Extracting Latents"):
        try:
            messages.append(f"\nProcessing {video_name}...")
            path = extract_roi_latent_from_video(
                storage_path=storage_path,
                project_name=project_name,
                video_name=video_name,
                roi_id=int(select_roi),
                model_name=select_model,
                batch_size=int(batch_size),
                preprocess_config=preprocess_args,
                skip_existing=skip_existing,
                progress_callback=update_progress,
                pooling_method=pooling_method,
                pooling_scales=parsed_scales if pooling_method == 'multiscale' else None,
                feature_layers=parsed_layers,
            )
            if path:
                messages.append(f"  ✅ Success: Latent file saved to {os.path.basename(path)}")
                success_count += 1
            else:
                messages.append(f"  ⚠️ Warning: Extraction returned no path for {video_name}, but no error was raised.")

        except Exception as e:
            failed_videos.append(video_name)
            messages.append(f"  ❌ Error processing {video_name}: {e}")

    # --- Final Summary ---
    summary_msg = f"\n\n🎉 Extraction Complete! \nSuccessfully processed {success_count}/{len(videos_to_process)} videos."
    if failed_videos:
        summary_msg += f"\n⚠️ Failed videos: {', '.join(failed_videos)}"
    messages.append(summary_msg)

    # --- Optional: Rotation Latent Extraction ---
    if rotate_roi_tail_switch:
        # Safety guard: re-validate tail ROI before GPU-heavy rotation extraction.
        # The checkbox may have been manually re-checked after Apply locked it out.
        import numpy as _np
        tail_roi_id = int(preprocess_args.rotate_roi_tail_id)
        _first_video = videos_to_process[0]
        _mask_path = os.path.join(storage_path, project_name, 'track', _first_video, 'mask_list.h5')
        _tail_roi_ok = True
        try:
            with H5IO(_mask_path) as _h5:
                _guard_mask = _h5.read_mask(0)
            if _guard_mask is not None:
                _roi_ids = set(_np.unique(_guard_mask[_guard_mask > 0]).tolist())
                if tail_roi_id not in _roi_ids or len(_roi_ids) < 2:
                    _tail_roi_ok = False
                    found_str = str(sorted(_roi_ids) or 'none')
                    messages.append(
                        f"\n⚠️ Rotation skipped: Tail ROI (ID {tail_roi_id}) not found or only 1 ROI "
                        f"in mask (detected: {found_str}). "
                        f"Click 'Apply' to re-validate preprocess settings."
                    )
        except Exception as _e:
            messages.append(f"\n⚠️ Could not validate tail ROI ({_e}); proceeding with rotation.")

        if _tail_roi_ok:
            messages.append("\n\n--- Extracting Rotation Latents (Rotate based on Tail is ON) ---")
            rot_success = 0
            rot_failed = []
            for video_name in tqdm(videos_to_process, desc="Extracting Rotation Latents"):
                try:
                    messages.append(f"\nRotation: Processing {video_name}...")
                    rpath = extract_roi_rotation_latent_from_video(
                        storage_path=storage_path,
                        project_name=project_name,
                        video_name=video_name,
                        roi_id=int(select_roi),
                        model_name=select_model,
                        batch_size=int(batch_size),
                        preprocess_config=preprocess_args,
                        skip_existing=skip_existing,
                        progress_callback=update_progress,
                    )
                    if rpath:
                        messages.append(f"  ✅ Rotation latent saved to {os.path.basename(rpath)}")
                        rot_success += 1
                    else:
                        messages.append(f"  ⚠️ Rotation extraction returned no path for {video_name}.")
                except Exception as e:
                    rot_failed.append(video_name)
                    messages.append(f"  ❌ Rotation error for {video_name}: {e}")

            rot_summary = f"\n🎉 Rotation Extraction Complete! Processed {rot_success}/{len(videos_to_process)} videos."
            if rot_failed:
                rot_summary += f"\n⚠️ Failed: {', '.join(rot_failed)}"
            messages.append(rot_summary)

    return "\n".join(messages)

@handle_assertion_error
def ui_extract_roi_crop_video(
    storage_path: str, 
    project_name: str, 
    select_roi: str, 
    select_video: str, 
    preprocess_args: Preprocess, 
    skip_existing: bool, 
    progress=gr.Progress()
) -> str:
    
    messages = []
    if not preprocess_args:
        raise ValueError("Please click Apply on Preprocess settings first.")

    _, config = get_project_config(storage_path, project_name)
    video_list = sorted(config['source']) if select_video == "All" else [select_video]
    
    # --- Pre-flight Check ---
    videos_to_process = []
    messages.append(f"Starting pre-flight check for {len(video_list)} cropped videos...")
    for video_name in video_list:
        # Construct the expected output path
        crop_dir = os.path.join(storage_path, project_name, 'crop', video_name)
        base_name, _ = os.path.splitext(video_name)
        crop_filename = f'{base_name}_ROI_{select_roi}_crop.mp4'
        output_path = os.path.join(crop_dir, crop_filename)

        if skip_existing and os.path.exists(output_path):
            messages.append(f"  ⏩ Skipping existing: {video_name}")
            continue
        videos_to_process.append(video_name)

    if not videos_to_process:
        messages.append("\n✅ All videos already have cropped versions. Nothing to extract.")
        return "\n".join(messages)

    messages.append(f"\nFound {len(videos_to_process)} new videos to process.")
    
    # --- Execution ---
    success_count = 0
    failed_videos = []
    
    def update_progress(p, desc=None):
        progress(p, desc=desc)

    for video_name in tqdm(videos_to_process, desc="Cropping Videos"):
        try:
            messages.append(f"\nProcessing {video_name}...")
            path = extract_roi_crop_video(
                storage_path=storage_path,
                project_name=project_name,
                video_name=video_name,
                roi_id=int(select_roi),
                preprocess_config=preprocess_args,
                skip_existing=skip_existing,
                progress_callback=update_progress
            )
            if path:
                messages.append(f"  ✅ Success: Cropped video saved to {os.path.basename(path)}")
                success_count += 1
            else:
                messages.append(f"  ⚠️ Warning: Cropping returned no path for {video_name}, but no error was raised.")

        except Exception as e:
            failed_videos.append(video_name)
            messages.append(f"  ❌ Error processing {video_name}: {e}")

    # --- Final Summary ---
    summary_msg = f"\n\n🎉 Cropping Complete! \nSuccessfully processed {success_count}/{len(videos_to_process)} videos."
    if failed_videos:
        summary_msg += f"\n⚠️ Failed videos: {', '.join(failed_videos)}"
    
    messages.append(summary_msg)
    return "\n".join(messages)


@handle_assertion_error
def ui_setting_preprocess(storage_path, project_name, select_video, center_roi_switch, center_roi_id,
                       center_roi_crop_width, center_roi_crop_height, rotate_roi_tail_switch, rotate_roi_tail_id, remove_background_switch):
    import numpy as np

    # Preview logic
    _, config = get_project_config(storage_path, project_name)
    video_list = sorted(config['source']) if select_video == "All" else [select_video]
    if not video_list:
        raise ValueError("No videos.")

    first_video = video_list[0]
    source_path = os.path.join(storage_path, project_name, 'sources', first_video)

    # Get mask — find first frame where mask area > 0
    track_dir = os.path.join(storage_path, project_name, 'track', first_video)
    with H5IO(os.path.join(track_dir, 'mask_list.h5')) as tracker:
        preview_idx = 0
        n_frames = len(tracker) if len(tracker) > 0 else 1
        for fi in range(min(n_frames, 500)):  # search up to 500 frames
            m = tracker.read_mask(fi)
            if m is not None and m.sum() > 0:
                preview_idx = fi
                break
        mask = tracker.read_mask(preview_idx)

    # Detect which ROI IDs actually exist in the mask before applying settings
    roi_ids_in_mask: set = set(np.unique(mask[mask > 0]).tolist()) if mask is not None else set()
    tail_roi = int(rotate_roi_tail_id)
    # Rotation requires the tail ROI to exist AND at least 2 distinct ROIs in the mask
    can_rotate = (tail_roi in roi_ids_in_mask) and (len(roi_ids_in_mask) >= 2)
    if not can_rotate:
        if bool(rotate_roi_tail_switch):
            found = sorted(roi_ids_in_mask)
            gr.Warning(
                f"Tail ROI (ID {tail_roi}) not found in tracking data or only 1 ROI detected "
                f"(detected ROI IDs: {found or 'none'}). "
                f"'Rotate based on Tail' has been disabled and locked — re-apply after fixing tracking."
            )
        rotate_roi_tail_switch = False

    # Boolean switches now arrive as bool from gr.Checkbox (no string conversion needed)
    preprocess = Preprocess(
        center_roi_switch=bool(center_roi_switch),
        center_roi_id=center_roi_id,
        center_roi_crop_width=center_roi_crop_width,
        center_roi_crop_height=center_roi_crop_height,
        rotate_roi_tail_switch=bool(rotate_roi_tail_switch),
        rotate_roi_tail_id=rotate_roi_tail_id,
        remove_background_switch=bool(remove_background_switch)
    )

    # Use VideoReader for preview
    with VideoReader(source_path) as vr:
        frame = vr.get_frame(preview_idx)

    pf, pm = preprocess.transform(frame, mask)
    mixed = generate_mix_image(pf, pm)

    # Third return value updates the checkbox.
    # If rotation is not possible, lock the checkbox (interactive=False) so the
    # user cannot re-enable it without clicking Apply again on valid tracking data.
    rotate_update = (
        gr.update(value=False, interactive=False)
        if not can_rotate
        else gr.update(value=bool(rotate_roi_tail_switch), interactive=True)
    )
    return preprocess, mixed, rotate_update


# ---------------------------
# KIT helper functions for extract UI
# ---------------------------

def _kit_load_params_for_display(storage_path_val: str, project_name_val: str) -> tuple:
    """Load KIT params and format them for display in the extract UI."""
    import json
    from castle.core.project import load_kit_params

    if not storage_path_val or not project_name_val:
        return gr.update(), "⚠️ No project open."

    params = load_kit_params(storage_path_val, project_name_val)
    if params is None:
        return (
            gr.update(),
            "⚠️ 尚未存有 KIT 參數，請先在 Tracking → Kinematics info transfusion 探索並 Save。",
        )
    return gr.update(value=json.dumps(params, indent=2)), "✅ KIT params loaded."


def _update_kit_conflict_warning(enable_kit: bool, center_roi: bool, rotate_tail: bool) -> gr.update:
    """Show a warning when KIT is enabled alongside Center ROI or Rotate tail."""
    if enable_kit and (center_roi or rotate_tail):
        which = []
        if center_roi:
            which.append("Center ROI")
        if rotate_tail:
            which.append("Rotate based on Tail")
        return gr.update(
            visible=True,
            value=(
                f"⚠️ **KIT 與 {' / '.join(which)} 語意重疊**，同時開啟可能產生衝突 "
                "（KIT 已對影像做穩定旋轉，Rotate Tail 會再旋轉一次）。"
                "確認繼續請直接點 Extract。"
            ),
        )
    return gr.update(visible=False, value="")


# ---------------------------
# UI Construction
# ---------------------------
def create_extract_ui(storage_path, project_name, extract_tab):
    ui = {}
    preprocess_state = gr.State(None)

    # ------------------------------------------------------------------
    # KIT accordion (above main settings)
    # ------------------------------------------------------------------
    with gr.Accordion("🎥 Kinematics info transfusion", open=False):
        gr.Markdown(
            "Enable KIT to transform both frame **and** mask with StabilizedCamera "
            "on-the-fly during extraction (no intermediate video written to disk). "
            "Parameters are loaded from the project config saved in "
            "**Tracking → Kinematics info transfusion**."
        )
        ui["enable_kit"] = gr.Checkbox(
            label="Enable KIT",
            value=False,
            info="When enabled, extraction uses the saved KIT parameters.",
        )
        with gr.Row():
            ui["load_kit_params_btn"] = gr.Button(
                "📂 Load saved params from project", variant="secondary"
            )
        ui["kit_params_display"] = gr.Textbox(
            label="Saved KIT params (read-only)",
            value="",
            interactive=False,
            lines=5,
        )
        ui["kit_param_status"] = gr.Textbox(
            label="",
            value="",
            interactive=False,
            lines=1,
        )
        ui["kit_conflict_warning"] = gr.Markdown(value="", visible=False)

    with gr.Row(visible=True):
        with gr.Column(scale=2):
            ui['select_model'] = gr.Dropdown(
                label="Visual Model",
                choices=["dinov2_vitb14_reg4_pretrain", "dinov3_vitb16", "dinov3_vitl16"],
                value="dinov2_vitb14_reg4_pretrain",
                visible=False,
                info="DINOv2/v3 backbone used for feature extraction.",
            )
            ui['select_roi_id'] = gr.Textbox(
                label="ROI ID",
                value="1",
                visible=False,
                info=(
                    "Region of Interest ID to extract features from. "
                    "Default: 1 (animal body). Must match the ROI tracked in Step 2."
                ),
            )
            ui['batch_size'] = gr.Textbox(
                label="Batch Size",
                value="32",
                visible=False,
                info="Frames processed per GPU/CPU batch. Use 'Auto Batch Size' to pick a safe value.",
            )
            ui['auto_batch_btn'] = gr.Button("Auto Batch Size", size="sm", visible=False)
            ui['select_video'] = gr.Dropdown(
                label="Target Video",
                value=None,
                visible=False,
                info="Select a specific video or 'All' to process the entire project.",
            )
            ui['video_count'] = gr.Number(
                label="Project Video Count",
                value=0,
                interactive=False,
                visible=False,
            )
            ui['skip_existing'] = gr.Checkbox(
                label="Skip existing files",
                value=True,
                visible=False,
                info="Skip videos that already have a latent file saved to disk.",
            )
            
        with gr.Column(scale=2):
            ui['center_roi_switch'] = gr.Checkbox(
                label="Center ROI",
                value=False,
                visible=False,
                info="Crop each frame centred on the chosen ROI before extracting features.",
            )
            ui['center_roi_id'] = gr.Number(
                label="Center ROI ID",
                value=1,
                visible=False,
                info="ROI ID used as the crop centre. Default: 1 (body centroid).",
            )
            ui['center_roi_crop_width'] = gr.Number(
                label="Crop Width",
                value=300,
                visible=False,
                info="Width of the crop region in pixels. Default: 300.",
            )
            ui['center_roi_crop_height'] = gr.Number(
                label="Crop Height",
                value=300,
                visible=False,
                info="Height of the crop region in pixels. Default: 300.",
            )
            ui['remove_background_switch'] = gr.Checkbox(
                label="Remove Background",
                value=False,
                visible=False,
                info="Zero out pixels outside the ROI mask before extracting features.",
            )
            ui['apply_preprocess'] = gr.Button("Apply", visible=False)
            
        with gr.Column(scale=4):
            ui['display'] = gr.Image(label='Display', interactive=False, visible=False)
            
            with gr.Accordion("Advanced Extraction Options", open=False, visible=False) as adv_accordion:
                ui['rotate_roi_tail_switch'] = gr.Checkbox(
                    label="Rotate based on Tail",
                    value=True,
                    info=(
                        "Automatically extract a rotation latent after the main extraction. "
                        "Aligns frames by body orientation using the tail ROI as reference."
                    ),
                )
                ui['rotate_roi_tail_id'] = gr.Number(
                    label="Tail ROI ID",
                    value=2,
                    info=(
                        "ROI ID for the tail used to compute body orientation. "
                        "Default: 2. Requires the tail to be tracked in Step 2."
                    ),
                )
                ui['pooling_method'] = gr.Radio(
                    choices=['weighted_average', 'multiscale'],
                    value='weighted_average',
                    label='Pooling Method',
                    info='weighted_average: single vector; multiscale: spatial pyramid pooling.',
                )
                ui['pooling_scales'] = gr.CheckboxGroup(
                    choices=['1', '2', '4', '8'],
                    value=['1', '2', '4'],
                    label='Multiscale Grid Sizes',
                    info='Only used when Pooling Method is multiscale. 1=global, 2=2×2, 4=4×4, 8=8×8.',
                )
                ui['feature_layers'] = gr.Textbox(
                    value='',
                    label='Feature Layers',
                    info='Comma-separated layer indices to concatenate (e.g. "3,7,11"). Empty = last layer only.',
                    placeholder='Leave empty for default (last layer)',
                )
            ui['adv_accordion'] = adv_accordion
            
            ui['mem_warning'] = gr.HTML(value="", visible=False)
            ui['extract_btn'] = gr.Button("Extract", visible=False)
            ui['extract_crop_video_btn'] = gr.Button("Extract Crop Video", visible=False)

            ui['latent_file_list'] = gr.Textbox(
                label="Log Output", 
                visible=False,
                lines=10,
                max_lines=20,
            )

    # 收集所有需要控制可見性的 UI 元件 (rotate/tail 元件在 accordion 內，不需要單獨控制)
    all_ui_elements_to_control = [
        ui['select_model'],       # 0
        ui['select_roi_id'],      # 1
        ui['batch_size'],         # 2
        ui['select_video'],       # 3
        ui['video_count'],        # 4
        ui['skip_existing'],      # 5
        ui['center_roi_switch'],  # 6
        ui['center_roi_id'],      # 7
        ui['center_roi_crop_width'],   # 8
        ui['center_roi_crop_height'],  # 9
        ui['remove_background_switch'], # 10
        ui['apply_preprocess'],   # 11
        ui['display'],            # 12
        ui['adv_accordion'],      # 13
        ui['extract_btn'],        # 14
        ui['extract_crop_video_btn'],  # 15
        ui['latent_file_list'],   # 16
        ui['auto_batch_btn'],     # 17
        ui['mem_warning'],        # 18
    ]

    # Event Binding
    extract_tab.select(init_select_video_list, inputs=[storage_path, project_name], outputs=all_ui_elements_to_control)
    
    ui['apply_preprocess'].click(
        ui_setting_preprocess,
        inputs=[storage_path, project_name, ui['select_video'], ui['center_roi_switch'],
                ui['center_roi_id'], ui['center_roi_crop_width'], ui['center_roi_crop_height'],
                ui['rotate_roi_tail_switch'], ui['rotate_roi_tail_id'], ui['remove_background_switch']],
        outputs=[preprocess_state, ui['display'], ui['rotate_roi_tail_switch']]
    )

    # KIT: load saved params into display
    ui["load_kit_params_btn"].click(
        fn=_kit_load_params_for_display,
        inputs=[storage_path, project_name],
        outputs=[ui["kit_params_display"], ui["kit_param_status"]],
    )

    # KIT conflict warning: update when relevant checkboxes change
    for _trigger in (ui["enable_kit"], ui["center_roi_switch"]):
        _trigger.change(
            fn=_update_kit_conflict_warning,
            inputs=[ui["enable_kit"], ui["center_roi_switch"], ui["rotate_roi_tail_switch"]],
            outputs=[ui["kit_conflict_warning"]],
        )

    def _extract_with_kit_routing(
        storage_path_val, project_name_val, select_model, select_roi_id,
        select_video, batch_size, preprocess_args, skip_existing,
        rotate_roi_tail_switch, pooling_method, pooling_scales_list, feature_layers_str,
        enable_kit, progress=gr.Progress(),
    ) -> str:
        """Route to KIT extraction or standard extraction depending on enable_kit."""
        if not enable_kit:
            return ui_extract_roi_latent(
                storage_path_val, project_name_val, select_model, select_roi_id,
                select_video, batch_size, preprocess_args, skip_existing,
                rotate_roi_tail_switch, pooling_method, pooling_scales_list, feature_layers_str,
                progress,
            )

        # KIT path
        from castle.core.project import load_kit_params, get_project_config
        from castle.service.extraction_service import extract_latent_with_kit

        msgs = []
        if enable_kit and (bool(preprocess_args.center_roi_switch if preprocess_args else False)
                           or bool(rotate_roi_tail_switch)):
            gr.Warning(
                "KIT 與 Center ROI / Rotate Tail 同時開啟。KIT 已完成空間穩定，"
                "再進行 Center ROI / Rotate Tail 可能產生衝突，請確認設定。"
            )

        kit_params = load_kit_params(storage_path_val, project_name_val)
        if kit_params is None:
            return (
                "❌ KIT enabled but no saved params found. "
                "Run KIT in Tracking → Kinematics info transfusion first, then Save."
            )

        _, config = get_project_config(storage_path_val, project_name_val)
        video_list = sorted(config.get("source", [])) if select_video == "All" else [select_video]

        def _cb(current: int, total: int, message: str) -> None:
            progress(current / total if total else 0, desc=message)

        success = 0
        for vname in video_list:
            try:
                msgs.append(f"KIT extracting {vname}…")
                path = extract_latent_with_kit(
                    storage_path=storage_path_val,
                    project_name=project_name_val,
                    video_name=vname,
                    roi_id=int(select_roi_id),
                    model_name=select_model,
                    batch_size=int(batch_size),
                    kit_params=kit_params,
                    skip_existing=bool(skip_existing),
                    progress_callback=_cb,
                )
                msgs.append(f"  ✅ {os.path.basename(path)}")
                success += 1
            except Exception as exc:
                msgs.append(f"  ❌ {vname}: {exc}")
                logger.exception("KIT extraction failed for %s", vname)

        msgs.append(f"\n🎉 KIT Extraction complete: {success}/{len(video_list)} succeeded.")
        return "\n".join(msgs)

    ui['extract_btn'].click(
        _extract_with_kit_routing,
        inputs=[storage_path, project_name, ui['select_model'], ui['select_roi_id'],
                ui['select_video'], ui['batch_size'], preprocess_state, ui['skip_existing'],
                ui['rotate_roi_tail_switch'],
                ui['pooling_method'], ui['pooling_scales'], ui['feature_layers'],
                ui['enable_kit']],
        outputs=ui['latent_file_list']
    )
    
    ui['extract_crop_video_btn'].click(
        ui_extract_roi_crop_video,
        inputs=[storage_path, project_name, ui['select_roi_id'],
                ui['select_video'], preprocess_state, ui['skip_existing']],
        outputs=ui['latent_file_list']
    )

    # Memory guard: reactive OOM check + auto batch size
    # Inputs: model, batch_size, rotate_tail, pooling_method, pooling_scales
    # rotate_roi_tail_switch is the dominant VRAM multiplier (7× per batch).
    # pooling_method/scales affect output dim only; model runs once regardless.
    _mem_inputs = [
        ui['select_model'],
        ui['batch_size'],
        ui['rotate_roi_tail_switch'],
        ui['pooling_method'],
        ui['pooling_scales'],
    ]

    def _mem_update(model_type, batch_size_str, rotate_tail, pooling_method, pooling_scales_list):
        import torch
        from castle.core.memory_guard import check as _check
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            batch_size = int(batch_size_str)
        except (ValueError, TypeError):
            return gr.update(value="", visible=False)
        risky, msg = _check(model_type, batch_size, device, rotate=bool(rotate_tail))
        if risky:
            return gr.update(
                value=f'<p style="color:#c05000;background:#fff4e6;padding:6px 10px;border-radius:4px;margin:4px 0">{msg}</p>',
                visible=True,
            )
        return gr.update(value="", visible=False)

    for _comp in _mem_inputs:
        _comp.change(_mem_update, inputs=_mem_inputs, outputs=ui['mem_warning'])

    def _auto_batch(model_type, rotate_tail, pooling_method, pooling_scales_list):
        import torch
        from castle.core.memory_guard import suggest_batch_size as _suggest
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return str(_suggest(model_type, device, rotate=bool(rotate_tail)))

    ui['auto_batch_btn'].click(
        _auto_batch,
        inputs=[ui['select_model'], ui['rotate_roi_tail_switch'], ui['pooling_method'], ui['pooling_scales']],
        outputs=ui['batch_size'],
    )

    return ui
