"""
castle/core/cluster.py
Core clustering logic and data aggregation.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional

from castle.core.interfaces import NotificationCallback
from castle.core.logging_config import setup_logger
from castle.utils.video_io import VideoReader, VideoIOError
from castle.utils.video_manager import get_project_config
from castle.utils.latent_explorer import Latent

logger = setup_logger(__name__)

# ---------------------------
# Helper Functions
# ---------------------------

def frame_to_timestamp(frame_number: int, fps: float) -> str:
    """Convert frame number to timestamp string (HH:MM:SS,mmm)."""
    seconds = frame_number / fps
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds_rem = seconds % 60
    milliseconds = (seconds_rem % 1) * 1000
    return f"{hours:02}:{minutes:02}:{int(seconds_rem):02},{int(milliseconds):03}"

def find_nearest_embedding(embedding_data: np.ndarray, x: float, y: float) -> Tuple[int, float]:
    """
    Find the nearest point in embedding space using KDTree.
    
    Args:
        embedding_data: 2D embedding array of shape (N, 2)
        x: Query x coordinate
        y: Query y coordinate
    
    Returns:
        Tuple of (index, distance) to nearest point
    """
    from scipy.spatial import KDTree
    tree = KDTree(embedding_data)
    distance, index = tree.query((x, y))
    return int(index), float(distance)

# ---------------------------
# Core Class: LatentAggregator
# ---------------------------

class LatentAggregator:
    """
    Aggregates latent features from multiple video files in a project.
    
    This class replaces the legacy MultiVideos class and provides:
    - Loading and concatenation of latent files across videos
    - Frame retrieval by global bin index
    - Subtitle generation for clustered behaviors
    
    Attributes:
        latents: Aggregated latent array of shape (N, feature_dim)
        videos_meta: List of (n_bins, video_name) tuples
        fps: Frames per second from first loaded video
        bin_size: Number of frames per bin
    """
    def __init__(self, storage_path: str, project_name: str, select_roi_id: int, bin_size: int, 
                 model_name: str, notify: Optional[NotificationCallback] = None) -> None:
        """
        Initialize the LatentAggregator.
        
        Args:
            storage_path: Root storage directory path
            project_name: Name of the project
            select_roi_id: ROI ID to filter latent files
            bin_size: Number of frames per temporal bin
            model_name: Name of the model to load latents for
            notify: Optional callback for progress/status notifications
        """
        self.storage_path = storage_path
        self.project_name = project_name
        self.source_path = os.path.join(storage_path, project_name, 'sources')
        self.project_path = os.path.join(storage_path, project_name)
        self.bin_size = bin_size
        self.model_name = model_name
        self.notify = notify or print  # Fallback to print
        
        # Load project configuration
        project_path, project_config = get_project_config(storage_path, project_name)
        self.project_path = project_path

        # Filter latents for the selected ROI
        roi_key = f'ROI_{select_roi_id}'
        
        # Latent files are stored in model-specific subdirectories
        latent_dir_path = os.path.join(storage_path, project_name, 'latent', model_name)
        
        self.latents: Optional[np.ndarray] = None
        self.videos_meta: List[Tuple[int, str]] = []
        self.fps: float = 30.0 # Default fallback
        
        latent_files = []
        if 'latent' in project_config:
            for filename, video_source_name in project_config['latent'].items():
                # Check 1: Must match ROI ID
                if roi_key not in filename: continue
                
                # Check 2: Must match Model Name (since new filenames contain it)
                # OR we just rely on file existence in the folder.
                # Let's rely on finding it in the folder + being in config.
                if model_name not in filename: continue 
                
                latent_files.append((filename, video_source_name))
        
        total_frames_loaded = 0
        latents_buffer = [] # Buffer for concatenating later
        
        # Load and aggregate latents
        for filename, video_source_name in latent_files:
            self.notify(f'Loading latent: {video_source_name}')
            try:
                latent_path = os.path.join(latent_dir_path, filename)
                if not os.path.exists(latent_path):
                     self.notify(f"Latent file missing: {latent_path}", "warning")
                     continue
                     
                loaded_data = np.load(latent_path)
                latent_chunk = loaded_data['latent']
                
                # Truncate to multiple of bin_size
                n_bins = len(latent_chunk) // bin_size
                n_frames_to_keep = n_bins * bin_size
                
                if n_frames_to_keep == 0:
                    continue
                
                # Setup fps from the first video found
                if not latents_buffer:
                    try:
                        video_path = os.path.join(self.source_path, video_source_name)
                        with VideoReader(video_path) as vr:
                            self.fps = vr.fps
                    except Exception as e:
                        self.notify(f"Warning: Could not read FPS from {video_source_name}, using default 30. Error: {e}", "warning")

                latents_buffer.append(latent_chunk[:n_frames_to_keep])
                self.videos_meta.append((n_bins, video_source_name))
                total_frames_loaded += n_frames_to_keep
                
            except Exception as e:
                self.notify(f"Error loading {filename}: {e}", "error")

        if latents_buffer:
             self.latents = np.concatenate(latents_buffer, axis=0)
             self.notify(f'Finished loading. Total aggregated latents: {len(self.latents)}')
        else:
             self.notify("Warning: No latents loaded.", "warning")

    def get_frame(self, index: int) -> Optional[np.ndarray]:
        """
        Retrieve the representative frame for a given global bin index.
        
        The frame is taken from the center of the bin (bin_size // 2).
        
        Args:
            index: Global bin index across all aggregated videos
            
        Returns:
            Frame as numpy array (H, W, 3) or None if retrieval fails
        """
        # index is the global bin index
        for n_bins_in_video, video_name in self.videos_meta:
            if index >= n_bins_in_video:
                index -= n_bins_in_video
                continue
            
            # Found the video
            video_path = os.path.join(self.source_path, video_name)
            # Calculate actual frame index (center of the bin)
            frame_idx = index * self.bin_size + self.bin_size // 2
            
            self.notify(f'Retrieving frame from {video_name} at index {frame_idx}')
            try:
                with VideoReader(video_path) as vr:
                    return vr.get_frame(frame_idx)
            except Exception as e:
                self.notify(f"Error reading frame: {e}", "error")
                return None
                
        self.notify('Error: Index out of bounds in Aggregator', "error")
        return None

    def get_latent_object(self) -> Latent:
        """Returns the high-level Latent explorer object."""
        if self.latents is None:
            # Return empty or handle error. Latent() might expect valid data.
            # Assuming Latent can handle empty or we shouldn't call this if init failed.
            pass
        return Latent(self.latents, self.bin_size)

    def generate_subtitles(self, syllables: np.ndarray, meta: Dict) -> List[str]:
        """
        Generates SRT subtitle files based on clustering results (syllables).
        """
        subtitle_output_dir = os.path.join(self.project_path, 'subtitles')
        os.makedirs(subtitle_output_dir, exist_ok=True)
        
        generated_files = []
        cum_bins = 0

        for n_bins_in_video, video_name in self.videos_meta:
            # Extract syllables corresponding to this video
            # Syllables are per-bin, so we repeat them to match frame-rate if we want per-frame arrays,
            # BUT the logic here seems to iterate changes in bins.
            
            this_video_syllables_bins = syllables[cum_bins : cum_bins + n_bins_in_video]
            
            # Expand bins to frames for precision? 
            # The original code repeated: data = np.repeat(this_video_syllabels, self.bin_size)
            data = np.repeat(this_video_syllables_bins, self.bin_size)
            
            srt_entries = []
            n_frames = len(data)
            
            # Find indices where behavior changes
            # Prepend -1 and append n-1 to handle start and end
            change_indices = np.arange(n_frames - 1)[data[:-1] != data[1:]]
            change_indices = np.concatenate([[-1], change_indices, [n_frames - 1]])
            
            for i in range(len(change_indices) - 1):
                start_frame = change_indices[i] + 1
                end_frame = change_indices[i+1] + 1
                
                start_time = frame_to_timestamp(start_frame, self.fps)
                end_time = frame_to_timestamp(end_frame, self.fps)
                
                behavior_id = data[start_frame]
                
                if behavior_id == -1:
                    behavior_name = "Unclustered"
                else:
                    # meta keys might be integers or strings, let's try both
                    if behavior_id in meta:
                         behavior_name = meta[behavior_id]['name']
                    elif str(behavior_id) in meta:
                         behavior_name = meta[str(behavior_id)]['name']
                    else:
                         behavior_name = f"Cluster {behavior_id}"

                srt_entries.append(f"{i + 1}\n{start_time} --> {end_time}\n{behavior_name}\n")
            
            srt_content = "\n".join(srt_entries)
            
            video_basename = os.path.splitext(os.path.basename(video_name))[0]
            output_path = os.path.join(subtitle_output_dir, video_basename + '.srt')
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(srt_content)
                
            generated_files.append(output_path)
            cum_bins += n_bins_in_video
            
        return generated_files
