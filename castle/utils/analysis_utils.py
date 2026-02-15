"""
castle/utils/analysis_utils.py
Shared analysis utilities for post-track and batch-track UI modules.

Extracted from castle/ui/post_track_ui.py and castle/ui/batch_track_ui.py
to eliminate code duplication (C-07).
"""

import os
import cv2
import numpy as np
from typing import List, Dict

from castle.utils.h5_io import H5IO
from castle.utils.plot import generate_mix_image, generate_mask_image
from castle.utils.video_io import WriteArray


def compute_roi_info(rois_results: H5IO, n_rois: int, total_frames: int,
                     progress_fn=None) -> List[Dict[str, np.ndarray]]:
    """
    Compute per-ROI centroid (x, y) and area for every frame from tracked masks.
    
    This is the core analysis logic shared by both post_track_ui.plot_basic_mask_info()
    and batch_track_ui.generate_csv_analysis().
    
    Args:
        rois_results: H5IO object containing tracking masks
        n_rois: Number of ROIs
        total_frames: Total number of frames
        progress_fn: Optional callable for progress reporting. Called as progress_fn(frame_index).
                     Can be a tqdm iterator or gr.Progress.tqdm range.
    
    Returns:
        List of dicts, one per ROI, each with keys 'x', 'y', 'area' as numpy arrays.
    """
    roi_info_list = [{"x": [], "y": [], "area": []} for _ in range(n_rois)]

    frame_iter = range(total_frames)
    if progress_fn is not None:
        frame_iter = progress_fn(frame_iter)

    for i_frame in frame_iter:
        for i in range(n_rois):
            if not rois_results.has_mask(i_frame):
                roi_info_list[i]['x'].append(np.nan)
                roi_info_list[i]['y'].append(np.nan)
                roi_info_list[i]['area'].append(0)
                continue

            mask = rois_results[i_frame][:]
            mask = cv2.inRange(mask, i + 1, i + 1)
            output = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_32S)
            num_labels, _, stats, centroids = output
            if num_labels <= 1:
                roi_info_list[i]['x'].append(np.nan)
                roi_info_list[i]['y'].append(np.nan)
                roi_info_list[i]['area'].append(0)
                continue

            areas = [stats[j, cv2.CC_STAT_AREA] for j in range(1, num_labels)]
            max_label = np.argmax(areas)
            roi_info_list[i]['x'].append(centroids[max_label + 1][0])
            roi_info_list[i]['y'].append(centroids[max_label + 1][1])
            roi_info_list[i]['area'].append(areas[max_label])

    # Convert lists to numpy arrays
    for i in range(n_rois):
        roi_info_list[i]['x'] = np.array(roi_info_list[i]['x'])
        roi_info_list[i]['y'] = np.array(roi_info_list[i]['y'])
        roi_info_list[i]['area'] = np.array(roi_info_list[i]['area']).astype(int)

    return roi_info_list


def create_kinematic_dataframe(roi_info_list: List[Dict[str, np.ndarray]]) -> 'pd.DataFrame':
    """
    Create a pandas DataFrame with per-ROI kinematics (x, y, speed, area).
    
    This is the pure data logic extracted from castle.ui.plot_mask_info.Plotter.create_pandas()
    so that the utils layer doesn't depend on the UI layer.
    
    Args:
        roi_info_list: ROI info as returned by compute_roi_info()
    
    Returns:
        DataFrame with columns ROI{i}.x, ROI{i}.y, ROI{i}.speed, ROI{i}.area
    """
    import pandas as pd
    
    df_dict = {}
    for index, it in enumerate(roi_info_list):
        df_dict[f'ROI{index+1}.x'] = it['x']
        df_dict[f'ROI{index+1}.y'] = it['y']

        speed = np.zeros(len(it['x']))
        dx = np.array(it['x'][1:] - it['x'][:-1])
        dy = np.array(it['y'][1:] - it['y'][:-1])
        speed[1:] = np.sqrt(dx * dx + dy * dy)

        df_dict[f'ROI{index+1}.speed'] = speed
        df_dict[f'ROI{index+1}.area'] = it['area']

    return pd.DataFrame(df_dict)


def save_kinematic_csv(track_dir_path: str, video_name: str,
                       roi_info_list: List[Dict[str, np.ndarray]]) -> str:
    """
    Save ROI kinematic data (position, speed, area) to CSV.
    
    Args:
        track_dir_path: Directory path for the track output
        video_name: Video filename (with extension)
        roi_info_list: ROI info as returned by compute_roi_info()
    
    Returns:
        Path to the generated CSV file.
    """
    video_name_wo_extension = video_name.split('.')[0]
    csv_path = os.path.join(track_dir_path, f'{video_name_wo_extension}-basic-information.csv')
    df = create_kinematic_dataframe(roi_info_list)
    df.to_csv(csv_path, float_format="%.4f")
    return csv_path


def generate_mix_video(source_video, rois_results: H5IO, output_path: str,
                       fps: float, n_frames: int) -> str:
    """
    Generate a video overlaying mask contours on the source video.
    
    Args:
        source_video: Video reader object supporting indexing (source_video[i])
        rois_results: H5IO object containing tracking masks
        output_path: Output video file path
        fps: Frames per second for the output video
        n_frames: Number of frames to process
    
    Returns:
        Path to the generated mix video.
    """
    output = WriteArray(output_path, fps=fps, crf=15)

    for i in range(n_frames):
        rois = rois_results[i]
        frame = source_video[i]
        out_frame = generate_mix_image(frame, rois)
        output.append(out_frame)

    del output
    return output_path


def generate_mask_video(source_video_not_used, rois_results: H5IO, 
                        output_path: str, fps: float, n_frames: int) -> str:
    """
    Generate a video showing only the masks (no source frames).
    
    Args:
        source_video_not_used: Unused, kept for API symmetry
        rois_results: H5IO object containing tracking masks
        output_path: Output video file path
        fps: Frames per second for the output video
        n_frames: Number of frames to process
    
    Returns:
        Path to the generated mask video.
    """
    output = WriteArray(output_path, fps=fps, crf=15)

    for i in range(n_frames):
        rois = rois_results[i]
        out_frame = generate_mask_image(rois)
        output.append(out_frame)

    del output
    return output_path
