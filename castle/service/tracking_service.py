"""
castle/service/tracking_service.py
Service layer for ROI tracking operations.

All functions take simple types and return dicts.
No gradio imports.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Callable

from castle.core.project import get_project_config
from castle.utils.video_io import ReadArray
from castle.utils.h5_io import H5IO
from castle.utils.tracking_manager import ROITracker
from castle.utils.analysis_utils import compute_roi_info, save_kinematic_csv

logger = logging.getLogger(__name__)


def track_video(storage_path: str, project_name: str, video_name: str,
                model: str = 'r50_deaotl',
                start: int = 0, stop: int = -1,
                skip_existing: bool = True,
                progress_callback: Optional[Callable] = None) -> str:
    """
    Execute tracking on a single video.
    
    Args:
        storage_path: Root storage directory
        project_name: Project name
        video_name: Video filename
        model: Tracking model type ('r50_deaotl' or 'swinb_deaotl')
        start: Start frame (0-based)
        stop: Stop frame (-1 for end)
        skip_existing: Skip if mask_list.h5 already exists
        progress_callback: Optional progress callback(progress_fraction, description)
    
    Returns:
        Status string: 'Done', 'Skipped', 'Cancel', or error message
    """
    project_path = Path(storage_path) / project_name
    track_dir = project_path / 'track' / video_name
    mask_path = track_dir / 'mask_list.h5'
    
    if skip_existing and mask_path.exists():
        return 'Skipped'
    
    video_path = project_path / 'sources' / video_name
    if not video_path.exists():
        return f'Error: Video not found: {video_path}'
    
    try:
        source_video = ReadArray(str(video_path))
        total_frames = len(source_video)
        
        if stop < 0:
            stop = total_frames - 1
        
        tracker = ROITracker(
            storage_path=storage_path,
            project_name=project_name,
            video_source=source_video,
            start_frame=start,
            stop_frame=stop,
            model_type=model,
        )
        
        result = tracker.track(progress=None)
        return result
        
    except Exception as e:
        logger.error(f"Tracking failed for {video_name}: {e}", exc_info=True)
        return f'Error: {e}'


def get_tracking_status(storage_path: str, project_name: str, video_name: str) -> dict:
    """
    Check tracking status for a video.
    
    Args:
        storage_path: Root storage directory
        project_name: Project name
        video_name: Video filename
    
    Returns:
        dict with keys:
            'tracked': bool — whether mask_list.h5 exists
            'mask_path': str — path to mask file
            'n_rois': int — number of ROIs found (0 if not tracked)
            'n_frames': int — number of tracked frames (0 if not tracked)
            'csv_path': str — path to kinematic CSV if exists
            'mix_video_path': str — path to mix video if exists
    """
    project_path = Path(storage_path) / project_name
    track_dir = project_path / 'track' / video_name
    mask_path = track_dir / 'mask_list.h5'
    
    result = {
        'tracked': mask_path.exists(),
        'mask_path': str(mask_path),
        'n_rois': 0,
        'n_frames': 0,
        'csv_path': '',
        'mix_video_path': '',
    }
    
    if not mask_path.exists():
        return result
    
    try:
        h5 = H5IO(str(mask_path))
        result['n_rois'] = h5.get_n_rois()
        result['n_frames'] = len(h5)
        del h5
    except Exception as e:
        logger.warning(f"Could not read mask file {mask_path}: {e}")
    
    # Check for generated files
    video_basename = video_name.split('.')[0]
    csv_path = track_dir / f'{video_basename}-basic-information.csv'
    mix_path = track_dir / f'{video_basename}-mix.mp4'
    
    if csv_path.exists():
        result['csv_path'] = str(csv_path)
    if mix_path.exists():
        result['mix_video_path'] = str(mix_path)
    
    return result
