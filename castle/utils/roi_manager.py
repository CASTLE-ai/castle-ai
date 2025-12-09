"""ROI (Region of Interest) management utilities for Castle AI."""

import os
import numpy as np

from .h5_io import H5IO
from .plot import generate_mask_image, generate_mix_image


def get_frame_display(storage_path, project_name, source_video, frame_index, display_mode='Image'):
    """Get frame display based on the selected mode.
    
    Args:
        storage_path: Path to the storage directory
        project_name: Name of the project
        source_video: Video array object
        frame_index: Index of the frame to display
        display_mode: Display mode ('Image', 'Image & Mask', 'Mask')
        
    Returns:
        numpy.ndarray: Frame image to display
    """
    # Return original frame for 'Image' mode
    if display_mode == 'Image':
        return source_video[frame_index]
    
    # Get paths for mask data
    project_path = os.path.join(storage_path, project_name)
    video_name = source_video.video_name
    track_dir_path = os.path.join(project_path, 'track', video_name)
    mask_list_path = os.path.join(track_dir_path, 'mask_list.h5')
    
    # Check if mask file exists
    if not os.path.exists(mask_list_path):
        print(f"Mask file not found: {mask_list_path}")
        return source_video[frame_index]

    if 0 > frame_index and frame_index >= len(source_video):
        print(f"Error frame_index: {frame_index}")
        return source_video[0]
    
    try:
        tracker = H5IO(mask_list_path)
        mask = tracker.read_mask(frame_index)
        
        if display_mode == 'Image & Mask':
            frame = source_video[frame_index]
            result = generate_mix_image(frame, mask)
        elif display_mode == 'Mask':
            result = generate_mask_image(mask)
        else:
            result = source_video[frame_index]
        
        del tracker
        return result
    
    except Exception as e:
        print(f"Error loading mask: {str(e)}")
        return source_video[frame_index]


def save_frame_to_knowledge(storage_path, project_name, source_video, frame_index):
    """Save a frame and its mask to the knowledge base.
    
    Args:
        storage_path: Path to the storage directory
        project_name: Name of the project
        source_video: Video array object
        frame_index: Index of the frame to save
        
    Returns:
        tuple: (success: bool, message: str)
    """
    try:
        # Setup paths
        project_path = os.path.join(storage_path, project_name)
        video_name = source_video.video_name
        
        # Create label directory
        label_dir_path = os.path.join(project_path, 'label', video_name)
        os.makedirs(label_dir_path, exist_ok=True)
        
        # Get mask path
        track_dir_path = os.path.join(project_path, 'track', video_name)
        mask_list_path = os.path.join(track_dir_path, 'mask_list.h5')
        
        # Check if mask file exists
        if not os.path.exists(mask_list_path):
            return False, f"Mask file not found. Please run tracking first."
        
        # Load frame and mask
        tracker = H5IO(mask_list_path)
        frame = source_video[frame_index]
        mask = tracker.read_mask(frame_index)
        del tracker
        
        # Save to knowledge base
        label_path = os.path.join(label_dir_path, f'{frame_index}')
        np.savez_compressed(label_path, frame=frame, mask=mask)
        
        return True, f"Saved ROI at frame {frame_index}"
    
    except Exception as e:
        return False, f"Failed to save ROI: {str(e)}"
