import os
import json
import time
import cv2
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gradio as gr
from natsort import natsorted
from tqdm import tqdm # Import tqdm

from castle import generate_aot
from ..utils.h5_io import H5IO
from ..utils.plot import generate_mix_image, generate_mask_image
from ..utils.video_io import ReadArray, WriteArray
from ..utils.tracking_manager import ROITracker, read_roi_labels
from .plot_mask_info import Plotter


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
    
    This function uses the core logic from tracking_manager.read_roi_labels but without video filtering.
    """
    if not project_name:
        return []

    project_path = Path(storage_path) / project_name
    label_dir = project_path / "label"
    
    if not label_dir.exists():
        return []

    label_list = []
    # Iterate through all label folders (same logic as read_roi_labels)
    for label_folder in natsorted([p for p in label_dir.iterdir() if p.is_dir()]):
        video_basename = label_folder.name
        # Iterate through all .npz files in the folder
        for npz_file in natsorted(list(label_folder.glob("*.npz"))):
            index = npz_file.stem
            try:
                data = np.load(npz_file)
                # Check if 'frame' and 'mask' keys are present
                if "frame" not in data or "mask" not in data:
                    continue
                frame = data["frame"]
                mask = data["mask"]
                label_list.append({
                    "index": f"{index}, {video_basename}",  # Same format as track_ui
                    "frame": frame,
                    "mask": mask,
                    "video_name": video_basename,
                    "frame_index": index,
                    "file_path": str(npz_file)
                })
            except Exception as e:
                print(f"Error reading label file {npz_file}: {e}")
                continue
    
    return label_list


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
        print(f"Error generating analysis for {video_name}: {e}")
    
    return csv_path, mix_video_path


def generate_csv_analysis(storage_path: str, project_name: str, video_name: str, source_video) -> str:
    """Generate CSV kinematic analysis for a video."""
    project_path = Path(storage_path) / project_name
    track_dir_path = project_path / "track" / video_name
    rois_results_path = track_dir_path / "mask_list.h5"
    
    if not rois_results_path.exists():
        return ""
    
    try:
        rois_results = H5IO(str(rois_results_path))
        n_rois = rois_results.get_n_rois()
        total_frames = len(rois_results)
        
        roi_info_list = [{"x": [], "y": [], "area": []} for i in range(n_rois)]
        
        for i_frame in range(total_frames):
            for i in range(n_rois):
                if not rois_results.has_mask(i_frame):
                    roi_info_list[i]['x'].append(np.nan)
                    roi_info_list[i]['y'].append(np.nan)
                    roi_info_list[i]['area'].append(0)
                    continue

                mask = rois_results[i_frame][:]
                mask = cv2.inRange(mask, i+1, i+1)
                output = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_32S)
                num_labels, _, stats, centroids = output
                if num_labels <= 1:
                    roi_info_list[i]['x'].append(np.nan)
                    roi_info_list[i]['y'].append(np.nan)
                    roi_info_list[i]['area'].append(0)
                    continue

                areas = [stats[j, cv2.CC_STAT_AREA] for j in range(1, num_labels)]
                max_label = np.argmax(areas)
                roi_info_list[i]['x'].append((centroids[max_label + 1][0]))
                roi_info_list[i]['y'].append((centroids[max_label + 1][1]))
                roi_info_list[i]['area'].append(areas[max_label])
        
        # Convert to numpy arrays
        for i in range(n_rois):
            roi_info_list[i]['x'] = np.array(roi_info_list[i]['x'])
            roi_info_list[i]['y'] = np.array(roi_info_list[i]['y'])
            roi_info_list[i]['area'] = np.array(roi_info_list[i]['area']).astype(int)

        # Generate CSV
        video_name_wo_extension = video_name.split('.')[0]
        csv_path = track_dir_path / f'{video_name_wo_extension}-basic-information.csv'
        df = Plotter.create_pandas(roi_info_list)
        df.to_csv(str(csv_path), float_format="%.4f")
        
        del rois_results
        return str(csv_path)
        
    except Exception as e:
        print(f"Error generating CSV for {video_name}: {e}")
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
        rois_results = H5IO(str(rois_results_path))
        n_frames = len(rois_results)

        for i in range(n_frames):
            rois = rois_results[i]
            frame = source_video[i]
            out_frame = generate_mix_image(frame, rois)
            output.append(out_frame)

        del rois_results, output
        return str(output_path)
        
    except Exception as e:
        print(f"Error generating mix video for {video_name}: {e}")
        return ""



def track_all_videos(storage_path: str, project_name: str, model_aot: str = "r50_deaotl", progress=gr.Progress()) -> str:
    """
    Execute tracking on all videos in the project.
    
    Args:
        storage_path: Storage path
        project_name: Project name
        model_aot: Tracking model type
        progress: Gradio progress object
        
    Returns:
        Completion message with progress log
    """
    messages = []
    
    if not project_name:
        return "Error: No project selected"
    
    all_videos = get_project_videos(storage_path, project_name) # Rename to avoid conflict
    if not all_videos:
        return "Error: No videos found in project"
    
    # --- First Pass: Pre-flight Check (Identify videos to process) ---
    videos_to_process = []
    messages.append(f"Starting pre-flight check for {len(all_videos)} videos...")
    
    total_all_videos = len(all_videos)
    for i, video_name in enumerate(all_videos):
        progress((i + 1) / total_all_videos, desc=f"Pre-flight Check: {i+1}/{total_all_videos}")
        project_path = Path(storage_path) / project_name
        track_dir_path = project_path / "track" / video_name
        rois_results_path = track_dir_path / "mask_list.h5"
        
        if not rois_results_path.exists():
            videos_to_process.append(video_name)
        else:
            messages.append(f"  Skipping existing video: {video_name}")

    if not videos_to_process:
        messages.append("All videos already processed. Nothing to track.")
        progress(1.0, desc="Batch tracking completed (no new videos to track)")
        return "\n".join(messages)

    total_videos_to_process = len(videos_to_process)
    messages.append(f"Found {total_videos_to_process} new videos to process")
    
    # --- Second Pass: Execution (Process identified videos) ---
    success_count = 0
    failed_videos = []
    
    for i, video_name in enumerate(tqdm(videos_to_process, desc="Processing videos", unit="video")): # Use tqdm for console progress
        progress((i + 1) / total_videos_to_process, desc=f"Overall Progress: {i+1}/{total_videos_to_process}") # Gradio progress bar
        try:
            # Load video file
            project_path = Path(storage_path) / project_name
            video_path = project_path / "sources" / video_name
            
            if not video_path.exists():
                failed_videos.append(video_name)
                msg = f"Warning: Video file not found {video_name}"
                messages.append(msg)
                print(msg)
                continue
            
            # Create video reader
            source_video = ReadArray(str(video_path))
            total_frames = len(source_video)
            
            msg = f"Video {video_name}: {total_frames} frames total"
            messages.append(msg)
            print(msg)
            
            # Create tracker (track entire video)
            start_frame = 0
            stop_frame = total_frames - 1
            
            tracker = ROITracker(
                storage_path=storage_path,
                project_name=project_name, 
                video_source=source_video,
                start_frame=start_frame,
                stop_frame=stop_frame,
                model_type=model_aot
            )
            
            msg = f"Starting tracking for video {video_name} (frames {start_frame}-{stop_frame})"
            messages.append(msg)
            print(msg)
            
            # Execute tracking without progress bar (to avoid sub-progress complexity)
            result = tracker.track(progress=None)
            
            if result == "Done":
                success_count += 1
                msg = f"✅ Completed tracking for video {video_name}"
                messages.append(msg)
                print(msg)
                
                # Generate CSV and mix video after successful tracking
                try:
                    msg = f"Generating analysis files for {video_name}..."
                    messages.append(msg)
                    print(msg)
                    
                    csv_path, mix_video_path = generate_video_analysis(storage_path, project_name, video_name)
                    
                    if csv_path:
                        msg = f"  ✅ Generated CSV: {os.path.basename(csv_path)}"
                        messages.append(msg)
                        print(msg)
                    if mix_video_path:
                        msg = f"  ✅ Generated mix video: {os.path.basename(mix_video_path)}"
                        messages.append(msg)
                        print(msg)
                        
                except Exception as e:
                    msg = f"  ⚠️  Warning: Failed to generate analysis files for {video_name}: {str(e)}"
                    messages.append(msg)
                    print(msg)
                    
            elif result == "Cancel":
                msg = f"❌ Tracking for video {video_name} was cancelled"
                messages.append(msg)
                print(msg)
                break
            else:
                failed_videos.append(video_name)
                msg = f"❌ Tracking failed for video {video_name}: {result}"
                messages.append(msg)
                print(msg)
                
        except Exception as e:
            failed_videos.append(video_name)
            msg = f"❌ Error: Failed to process video {video_name}: {str(e)}"
            messages.append(msg)
            print(msg)
    
    progress(1.0, desc="Batch tracking completed")
    
    result_msg = f"\n🎉 Batch tracking completed! Successfully processed {success_count}/{total_videos_to_process} videos"
    result_msg += f"\n📊 CSV analysis files and 🎬 mix videos generated for successful tracks"
    if failed_videos:
        result_msg += f"\n⚠️  Failed videos: {', '.join(failed_videos)}"
    
    messages.append(result_msg)
    return "\n".join(messages)


def get_select_index(evt: gr.SelectData):
    """Get index of selected item"""
    return evt.index


def delete_selected_label(storage_path: str, project_name: str, label_list: List[Dict], selected_index: int) -> Tuple[List[Dict], List[Tuple]]:
    """
    Delete the selected label.
    
    Args:
        storage_path: Storage path
        project_name: Project name
        label_list: Label list
        selected_index: Selected index
        
    Returns:
        Updated label list and gallery list
    """
    if selected_index is None or not label_list or selected_index >= len(label_list):
        return read_all_labels_to_gallery(storage_path, project_name)
    
    try:
        selected_label = label_list[selected_index]
        file_path = selected_label["file_path"]
        
        if os.path.exists(file_path):
            os.remove(file_path)
            gr.Info(f"Deleted label file: {selected_label['index']}")
        else:
            gr.Info(f"File does not exist: {file_path}")
            
    except Exception as e:
        gr.Warning(f"Error occurred while deleting file: {str(e)}")
    
    return read_all_labels_to_gallery(storage_path, project_name)


def create_batch_track_ui(storage_path: str, project_name: str, batch_tracking_tab: gr.Tab) -> Tuple[Dict[str, Any], Dict[str, Any]]: # 修改簽名，新增 batch_tracking_tab 參數
    ui: Dict[str, Any] = {}
    states: Dict[str, Any] = {}
    
    # State variables
    states["label_list_state"] = gr.State(None)
    states["selected_index_state"] = gr.State(None)
    
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
                
                ui["track_all_btn"] = gr.Button(
                    "Start Tracking All Videos",
                    variant="primary",
                    size="lg",
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
    # 移除原有的 project_name.change 事件，這些將由 batch_tracking_tab.select 處理
    # project_name.change(
    #     fn=update_video_count,
    #     inputs=[storage_path, project_name],
    #     outputs=ui["video_count_display"]
    # )
    
    # project_name.change(
    #     fn=refresh_gallery,
    #     inputs=[storage_path, project_name],
    #     outputs=[states["label_list_state"], ui["gallery"]]
    # )
    
    # Gallery selection event
    ui["gallery"].select(
        fn=get_select_index,
        outputs=states["selected_index_state"]
    )
    
    # Delete button event
    ui["delete_selected_btn"].click(
        fn=delete_selected_label,
        inputs=[storage_path, project_name, states["label_list_state"], states["selected_index_state"]],
        outputs=[states["label_list_state"], ui["gallery"]]
    )
    
    # Batch tracking button event
    ui["track_all_btn"].click(
        fn=track_all_videos,
        inputs=[storage_path, project_name, ui["model_dropdown"]],
        outputs=ui["progress_text"],
        show_progress=True
    )
    
    # 新增 batch_tracking_tab.select 事件綁定
    all_batch_ui_elements = [
        ui["gallery"],
        ui["delete_selected_btn"],
        ui["video_count_display"],
        ui["model_dropdown"],
        ui["track_all_btn"],
        ui["progress_text"]
    ]

    def show_batch_track_ui(project_name_val):
        is_visible = project_name_val is not None
        return [gr.update(visible=is_visible)] * len(all_batch_ui_elements)
    
    (
        batch_tracking_tab.select(
            fn=show_batch_track_ui,
            inputs=[project_name],
            outputs=all_batch_ui_elements
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
