"""
castle/service/extraction_service.py
Service layer for latent extraction operations.

All functions take simple types and return strings/dicts.
No gradio imports.
"""

import logging
from typing import Optional, Callable

from castle.core.data import Preprocess
from castle.core.extractor import (
    extract_roi_latent_from_video,
    extract_roi_crop_video,
    extract_roi_rotation_latent_from_video,
)
from castle.core.project import get_project_config

logger = logging.getLogger(__name__)


def make_preprocess_config(
    center_roi_switch: bool = False,
    center_roi_id: int = 1,
    center_roi_crop_width: int = 300,
    center_roi_crop_height: int = 300,
    rotate_roi_tail_switch: bool = False,
    rotate_roi_tail_id: int = 2,
    remove_background_switch: bool = False,
) -> Preprocess:
    """
    Create a Preprocess config object from simple parameters.
    
    This wraps the Preprocess dataclass so callers don't need to import
    castle.core.data directly.
    
    Returns:
        Preprocess config object
    """
    return Preprocess(
        center_roi_switch=center_roi_switch,
        center_roi_id=center_roi_id,
        center_roi_crop_width=center_roi_crop_width,
        center_roi_crop_height=center_roi_crop_height,
        rotate_roi_tail_switch=rotate_roi_tail_switch,
        rotate_roi_tail_id=rotate_roi_tail_id,
        remove_background_switch=remove_background_switch,
    )


def extract_latent(
    storage_path: str,
    project_name: str,
    video_name: str,
    model: str,
    roi: int,
    batch_size: int = 32,
    preprocess_config: Optional[Preprocess] = None,
    skip_existing: bool = True,
    progress_callback: Optional[Callable] = None,
    pooling_method: str = 'weighted_average',
    pooling_scales: Optional[list] = None,
    feature_layers: Optional[list] = None,
) -> str:
    """
    Extract latent features from a tracked video ROI.
    
    Args:
        storage_path: Root storage directory
        project_name: Project name
        video_name: Video filename (or 'All' for all videos)
        model: Model name (e.g., 'dinov3_vitb16')
        roi: ROI ID
        batch_size: Batch size for extraction
        preprocess_config: Preprocess configuration. If None, uses defaults.
        skip_existing: Skip if latent file already exists
        progress_callback: Optional progress callback(fraction, description)
        pooling_method: 'weighted_average' (default) or 'multiscale'
        pooling_scales: Grid scales for multiscale pooling, e.g. [1, 2, 4]
        feature_layers: Layer indices for multi-layer extraction. None = last only.
    
    Returns:
        Path to saved latent file, or empty string on failure.
        If video_name is 'All', returns semicolon-separated paths.
    """
    if preprocess_config is None:
        preprocess_config = Preprocess()
    
    # Handle "All" videos
    _, config = get_project_config(storage_path, project_name)
    video_list = sorted(config.get('source', [])) if video_name == 'All' else [video_name]
    
    paths = []
    for vname in video_list:
        try:
            path = extract_roi_latent_from_video(
                storage_path=storage_path,
                project_name=project_name,
                video_name=vname,
                roi_id=roi,
                model_name=model,
                batch_size=batch_size,
                preprocess_config=preprocess_config,
                skip_existing=skip_existing,
                progress_callback=progress_callback,
                pooling_method=pooling_method,
                pooling_scales=pooling_scales,
                feature_layers=feature_layers,
            )
            if path:
                paths.append(path)
        except Exception as e:
            logger.error(f"Extraction failed for {vname}: {e}", exc_info=True)
    
    return ';'.join(paths)


# NOTE: Not yet exposed via CLI or UI
def extract_crop_video(
    storage_path: str,
    project_name: str,
    video_name: str,
    roi: int,
    preprocess_config: Optional[Preprocess] = None,
    skip_existing: bool = True,
    progress_callback: Optional[Callable] = None,
) -> str:
    """
    Extract cropped/preprocessed video for a tracked ROI.
    
    Args:
        storage_path: Root storage directory
        project_name: Project name
        video_name: Video filename (or 'All')
        roi: ROI ID
        preprocess_config: Preprocess configuration. If None, uses defaults.
        skip_existing: Skip if crop video already exists
        progress_callback: Optional progress callback(fraction, description)
    
    Returns:
        Path to saved crop video, or empty string on failure.
        If video_name is 'All', returns semicolon-separated paths.
    """
    if preprocess_config is None:
        preprocess_config = Preprocess()
    
    _, config = get_project_config(storage_path, project_name)
    video_list = sorted(config.get('source', [])) if video_name == 'All' else [video_name]
    
    paths = []
    for vname in video_list:
        try:
            path = extract_roi_crop_video(
                storage_path=storage_path,
                project_name=project_name,
                video_name=vname,
                roi_id=roi,
                preprocess_config=preprocess_config,
                skip_existing=skip_existing,
                progress_callback=progress_callback,
            )
            if path:
                paths.append(path)
        except Exception as e:
            logger.error(f"Crop extraction failed for {vname}: {e}", exc_info=True)
    
    return ';'.join(paths)


# NOTE: Not yet exposed via CLI or UI
def extract_rotation_latent(
    storage_path: str,
    project_name: str,
    video_name: str,
    model: str,
    roi: int,
    batch_size: int = 32,
    preprocess_config: Optional[Preprocess] = None,
    skip_existing: bool = True,
    progress_callback: Optional[Callable] = None,
) -> str:
    """
    Extract rotation-invariant latent features.
    
    Generates 7 rotated views and averages the latent representations.
    
    Args:
        storage_path: Root storage directory
        project_name: Project name
        video_name: Video filename (or 'All')
        model: Model name
        roi: ROI ID
        batch_size: Batch size for extraction
        preprocess_config: Preprocess configuration. If None, uses defaults.
        skip_existing: Skip if latent file already exists
        progress_callback: Optional progress callback(fraction, description)
    
    Returns:
        Path to saved latent file, or empty string on failure.
    """
    if preprocess_config is None:
        preprocess_config = Preprocess()
    
    _, config = get_project_config(storage_path, project_name)
    video_list = sorted(config.get('source', [])) if video_name == 'All' else [video_name]
    
    paths = []
    for vname in video_list:
        try:
            path = extract_roi_rotation_latent_from_video(
                storage_path=storage_path,
                project_name=project_name,
                video_name=vname,
                roi_id=roi,
                model_name=model,
                batch_size=batch_size,
                preprocess_config=preprocess_config,
                skip_existing=skip_existing,
                progress_callback=progress_callback,
            )
            if path:
                paths.append(path)
        except Exception as e:
            logger.error(f"Rotation extraction failed for {vname}: {e}", exc_info=True)
    
    return ';'.join(paths)
