"""
castle/ui/extract_ui.py
UI Layer for Extraction. 
Delegates all logic to castle.core.extractor.
"""

import os
import gradio as gr
from typing import List
from tqdm import tqdm # 新增: 匯入 tqdm

from castle.core.data import Preprocess
from castle.core.extractor import extract_roi_latent_from_video, extract_roi_crop_video, extract_roi_rotation_latent_from_video

from castle.utils.video_manager import get_project_config
from castle.utils.video_io import VideoReader
from castle.utils.h5_io import H5IO
from castle.utils.plot import generate_mix_image

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
        gr.update(visible=False), # rotate_roi_tail_switch
        gr.update(visible=False), # rotate_roi_tail_id
        gr.update(visible=False), # remove_background_switch
        gr.update(visible=False), # apply_preprocess
        gr.update(visible=False), # display
        gr.update(visible=False), # extract_btn
        gr.update(visible=False), # extract_crop_video_btn
        gr.update(visible=False), # extract_rotation_latent_btn
        gr.update(visible=False)  # latent_file_list
    ])

    if not project_name:
        gr.Warning("No project selected.")
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
            updates[0] = gr.update(visible=True) # select_model
            updates[1] = gr.update(visible=True) # select_roi_id
            updates[2] = gr.update(visible=True) # batch_size
            updates[3] = gr.update(choices=choices_with_all, value="All", visible=True) # select_video
            updates[4] = gr.update(value=video_count_val, visible=True) # video_count
            updates[5] = gr.update(visible=True) # skip_existing
            updates[6] = gr.update(visible=True) # center_roi_switch
            updates[7] = gr.update(visible=True) # center_roi_id
            updates[8] = gr.update(visible=True) # center_roi_crop_width
            updates[9] = gr.update(visible=True) # center_roi_crop_height
            updates[10] = gr.update(visible=True) # rotate_roi_tail_switch
            updates[11] = gr.update(visible=True) # rotate_roi_tail_id
            updates[12] = gr.update(visible=True) # remove_background_switch
            updates[13] = gr.update(visible=True) # apply_preprocess
            updates[14] = gr.update(visible=True) # display
            updates[15] = gr.update(visible=True) # extract_btn
            updates[16] = gr.update(visible=True) # extract_crop_video_btn
            updates[17] = gr.update(visible=True) # extract_rotation_latent_btn
            updates[18] = gr.update(visible=True) # latent_file_list
        else:
            gr.Warning("No videos found in the selected project.")
            # updates 已經是預設的隱藏狀態，無需額外操作

    except Exception as e:
        gr.Warning(f"Error loading project videos: {e}")
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
    progress=gr.Progress()
) -> str:
    
    messages = []
    if not preprocess_args:
        raise ValueError("Please click Apply on Preprocess settings first.")
        
    _, config = get_project_config(storage_path, project_name)
    video_list = sorted(config['source']) if select_video == "All" else [select_video]
    
    # --- Pre-flight Check ---
    videos_to_process = []
    messages.append(f"Starting pre-flight check for {len(video_list)} videos...")
    for video_name in video_list:
        # Construct the expected output path
        # Replicate valid tag logic from extractor.py
        tags = []
        if preprocess_args.center_roi_switch: tags.append("ctr")
        if preprocess_args.remove_background_switch: tags.append("rmbg")
        
        suffix = "_".join([select_model] + tags)
        
        latent_dir = os.path.join(storage_path, project_name, 'latent', select_model)
        latent_filename = f'{os.path.splitext(video_name)[0]}_ROI_{select_roi}_{suffix}.npz'
        output_path = os.path.join(latent_dir, latent_filename)
        
        # A more robust check might be needed if the filename is more complex
        # For now, we assume a simple naming convention.
        # The core function `extract_roi_latent_from_video` should ideally return the path
        # for a more accurate check, but that's a bigger refactor.
        
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
                skip_existing=skip_existing, # The core function handles this, but we pre-check
                progress_callback=update_progress
            )
            if path:
                messages.append(f"  ✅ Success: Latent file saved to {os.path.basename(path)}")
                success_count += 1
            else:
                # This case might happen if the core function returns None without an error
                messages.append(f"  ⚠️ Warning: Extraction returned no path for {video_name}, but no error was raised.")

        except Exception as e:
            failed_videos.append(video_name)
            messages.append(f"  ❌ Error processing {video_name}: {e}")

    # --- Final Summary ---
    summary_msg = f"\n\n🎉 Extraction Complete! \nSuccessfully processed {success_count}/{len(videos_to_process)} videos."
    if failed_videos:
        summary_msg += f"\n⚠️ Failed videos: {', '.join(failed_videos)}"
    
    messages.append(summary_msg)
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
def ui_extract_roi_rotation_latent(
    storage_path: str, 
    project_name: str, 
    select_model: str, 
    select_roi: str, 
    select_video: str, 
    batch_size: str, 
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
    messages.append(f"Starting pre-flight check for {len(video_list)} videos for rotation latent extraction...")
    for video_name in video_list:
        # Construct the expected output path
        latent_dir = os.path.join(storage_path, project_name, 'latent', video_name)
        latent_filename = f'ROI_{select_roi}_rotation_latent.npz' # This might need adjustment
        output_path = os.path.join(latent_dir, latent_filename)
        
        if skip_existing and os.path.exists(output_path):
            messages.append(f"  ⏩ Skipping existing: {video_name}")
            continue
        videos_to_process.append(video_name)

    if not videos_to_process:
        messages.append("\n✅ All videos already have rotation latent files. Nothing to extract.")
        return "\n".join(messages)

    messages.append(f"\nFound {len(videos_to_process)} new videos to process.")
    
    # --- Execution ---
    success_count = 0
    failed_videos = []
    
    def update_progress(p, desc=None):
        progress(p, desc=desc)

    for video_name in tqdm(videos_to_process, desc="Extracting Rotation Latents"):
        try:
            messages.append(f"\nProcessing {video_name}...")
            path = extract_roi_rotation_latent_from_video(
                storage_path=storage_path,
                project_name=project_name,
                video_name=video_name,
                roi_id=int(select_roi),
                model_name=select_model,
                batch_size=int(batch_size),
                preprocess_config=preprocess_args,
                skip_existing=skip_existing,
                progress_callback=update_progress
            )
            if path:
                messages.append(f"  ✅ Success: Rotation latent file saved to {os.path.basename(path)}")
                success_count += 1
            else:
                messages.append(f"  ⚠️ Warning: Rotation extraction returned no path for {video_name}, but no error was raised.")

        except Exception as e:
            failed_videos.append(video_name)
            messages.append(f"  ❌ Error processing {video_name}: {e}")

    # --- Final Summary ---
    summary_msg = f"\n\n🎉 Rotation Extraction Complete! \nSuccessfully processed {success_count}/{len(videos_to_process)} videos."
    if failed_videos:
        summary_msg += f"\n⚠️ Failed videos: {', '.join(failed_videos)}"
    
    messages.append(summary_msg)
    return "\n".join(messages)


@handle_assertion_error
def ui_setting_preprocess(storage_path, project_name, select_video, center_roi_switch, center_roi_id,
                       center_roi_crop_width, center_roi_crop_height, rotate_roi_tail_switch, rotate_roi_tail_id, remove_background_switch):
    
    # M-01 Fix: Handle string-to-bool conversion in UI layer
    preprocess = Preprocess(
        center_roi_switch=(center_roi_switch == 'True'),
        center_roi_id=center_roi_id,
        center_roi_crop_width=center_roi_crop_width,
        center_roi_crop_height=center_roi_crop_height,
        rotate_roi_tail_switch=(rotate_roi_tail_switch == 'True'),
        rotate_roi_tail_id=rotate_roi_tail_id,
        remove_background_switch=(remove_background_switch == 'True')
    )
    
    # Preview logic
    _, config = get_project_config(storage_path, project_name)
    video_list = sorted(config['source']) if select_video == "All" else [select_video]
    if not video_list: raise ValueError("No videos.")
    
    first_video = video_list[0]
    source_path = os.path.join(storage_path, project_name, 'sources', first_video)
    
    # Use VideoReader for preview
    with VideoReader(source_path) as vr:
        frame = vr.get_frame(0)
    
    # Get mask
    track_dir = os.path.join(storage_path, project_name, 'track', first_video)
    tracker = H5IO(os.path.join(track_dir, 'mask_list.h5'))
    mask = tracker.read_mask(0)
    
    pf, pm = preprocess.transform(frame, mask)
    mixed = generate_mix_image(pf, pm)
    
    return preprocess, mixed


# ---------------------------
# UI Construction
# ---------------------------
from ..utils.video_manager import get_project_videos


# ---------------------------
# UI Construction
# ---------------------------
def create_extract_ui(storage_path, project_name, extract_tab):
    ui = {}
    preprocess_state = gr.State(None)
    
    with gr.Row(visible=True):
        with gr.Column(scale=2):
            ui['select_model'] = gr.Dropdown(
                label="Select Visual Model",
                choices=["dinov2_vitb14_reg4_pretrain", "dinov3_vitb16", "dinov3_vitl16"],
                value="dinov2_vitb14_reg4_pretrain",
                visible=False
            )
            ui['select_roi_id'] = gr.Textbox(label="Enter ROI ID", value="1", visible=False)
            ui['batch_size'] = gr.Textbox(label="Batch size", value="32", visible=False)
            ui['select_video'] = gr.Dropdown(label="Select Target Video", value=None, visible=False)
            ui['video_count'] = gr.Number(label="Project Video Count", value=0, interactive=False, visible=False) # 新增的 UI 元件
            ui['skip_existing'] = gr.Checkbox(label="Skip existing files", value=True, visible=False)
            
        with gr.Column(scale=2):
            ui['center_roi_switch'] = gr.Dropdown(label="Center ROI", value='False', choices=['True', 'False'], visible=False)
            ui['center_roi_id'] = gr.Number(label="Center ROI ID", value=1, visible=False)
            ui['center_roi_crop_width'] = gr.Number(label="width", value=300, visible=False)
            ui['center_roi_crop_height'] = gr.Number(label="height", value=300, visible=False)
            ui['rotate_roi_tail_switch'] = gr.Dropdown(label="Rotate based on Tail", value='False', choices=['True', 'False'], visible=False)
            ui['rotate_roi_tail_id'] = gr.Number(label="Tail ROI ID", value=2, visible=False)
            ui['remove_background_switch'] = gr.Dropdown(label="Remove Background", value='False', choices=['True', 'False'], visible=False)
            ui['apply_preprocess'] = gr.Button("Apply", visible=False)
            
        with gr.Column(scale=4):
            ui['display'] = gr.Image(label='Display', interactive=False, visible=False)
            ui['extract_btn'] = gr.Button("Extract", visible=False)
            ui['extract_crop_video_btn'] = gr.Button("Extract Crop Video", visible=False)
            ui['extract_rotation_latent_btn'] = gr.Button("Extract Rotation Latent", visible=False) 
 
            ui['latent_file_list'] = gr.Textbox(
                label="Log Output", 
                visible=False,
                lines=10,
                max_lines=20
            )

    # 收集所有需要控制可見性的 UI 元件
    all_ui_elements_to_control = [
        ui['select_model'],
        ui['select_roi_id'],
        ui['batch_size'],
        ui['select_video'],
        ui['video_count'], # 新增
        ui['skip_existing'],
        ui['center_roi_switch'],
        ui['center_roi_id'],
        ui['center_roi_crop_width'],
        ui['center_roi_crop_height'],
        ui['rotate_roi_tail_switch'],
        ui['rotate_roi_tail_id'],
        ui['remove_background_switch'],
        ui['apply_preprocess'],
        ui['display'],
        ui['extract_btn'],
        ui['extract_crop_video_btn'],
        ui['extract_rotation_latent_btn'],
        ui['latent_file_list']
    ]

    # Event Binding
    extract_tab.select(init_select_video_list, inputs=[storage_path, project_name], outputs=all_ui_elements_to_control)
    
    ui['apply_preprocess'].click(
        ui_setting_preprocess,
        inputs=[storage_path, project_name, ui['select_video'], ui['center_roi_switch'],
                ui['center_roi_id'], ui['center_roi_crop_width'], ui['center_roi_crop_height'],
                ui['rotate_roi_tail_switch'], ui['rotate_roi_tail_id'], ui['remove_background_switch']],
        outputs=[preprocess_state, ui['display']]
    )

    ui['extract_btn'].click(
        ui_extract_roi_latent,
        inputs=[storage_path, project_name, ui['select_model'], ui['select_roi_id'],
                ui['select_video'], ui['batch_size'], preprocess_state, ui['skip_existing']],
        outputs=ui['latent_file_list']
    )
    
    # Updated: Removed unnecessary inputs for crop
    ui['extract_crop_video_btn'].click(
        ui_extract_roi_crop_video,
        inputs=[storage_path, project_name, ui['select_roi_id'],
                ui['select_video'], preprocess_state, ui['skip_existing']],
        outputs=ui['latent_file_list']
    )

    ui['extract_rotation_latent_btn'].click(
        ui_extract_roi_rotation_latent,
        inputs=[storage_path, project_name, ui['select_model'], ui['select_roi_id'],
                ui['select_video'], ui['batch_size'], preprocess_state, ui['skip_existing']],
        outputs=ui['latent_file_list']
    )

    return ui
