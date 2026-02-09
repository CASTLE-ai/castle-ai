"""ROI tracking management utilities for Castle AI."""

import time
from pathlib import Path
from typing import Any, List, Dict, Tuple, Optional
import os

import numpy as np
import gradio as gr
from natsort import natsorted
import cv2  # Added for smart filtering
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm # 新增：匯入 tqdm

from .video_object_segment import generate_aot
from .h5_io import H5IO


def read_roi_labels(storage_path: str, project_name: str, video_name: Optional[str] = None) -> List[Dict[str, Any]]:
    """Read all ROI label files for the given project.
    
    Args:
        storage_path: Base storage directory
        project_name: Name of the project
        video_name: Optional specific video name to filter labels
        
    Returns:
        List of dictionaries containing label information with keys:
            - index: String identifier combining file index and video basename
            - frame: Frame data
            - mask: Corresponding mask
    """
    project_path = Path(storage_path) / project_name
    label_dir = project_path / "label"
    
    if not label_dir.exists():
        return []
    
    label_list = []
    
    # Iterate through all subdirectories in natural sorted order
    for label_folder in natsorted([p for p in label_dir.iterdir() if p.is_dir()]):
        video_basename = label_folder.name
        
        # Skip if filtering by video name and doesn't match
        if video_name and video_basename != video_name:
            continue
        
        # Iterate through all .npz files in the folder
        for npz_file in natsorted(list(label_folder.glob("*.npz"))):
            try:
                index = npz_file.stem
                data = np.load(npz_file)
                
                # Expect keys 'frame' and 'mask'
                if "frame" not in data or "mask" not in data:
                    print(f"Warning: Missing frame or mask in {npz_file}")
                    continue
                
                frame = data["frame"]
                mask = data["mask"]
                
                label_list.append({
                    "index": f"{index}, {video_basename}",
                    "frame": frame,
                    "mask": mask,
                })
            except Exception as e:
                print(f"Error loading label file {npz_file}: {str(e)}")
                continue
    
    return label_list


class TrackingDataset(Dataset):
    """Dataset for lazy loading of video frames for tracking."""
    def __init__(self, video_source: Any, frame_indices: List[int], transform: Any):
        """
        Initialize the dataset.

        Args:
            video_source: Video source object (e.g., ReadArray)
            frame_indices: List of frame indices to process
            transform: Preprocessing transform to apply to each frame
        """
        self.video_path = video_source.path  # Store path for worker
        self.frame_indices = frame_indices
        self.transform = transform
        self.reader = None # Initialize reader to None for lazy loading in worker

    def __len__(self) -> int:
        return len(self.frame_indices)

    def __getitem__(self, idx: int) -> Tuple[Any, int, Any]:
        if self.reader is None:
            # Each worker gets its own file handle to avoid conflicts
            from .video_io import ReadArray
            self.reader = ReadArray(self.video_path)

        frame_index = self.frame_indices[idx]
        frame = self.reader[frame_index]
        
        # Apply preprocessing transform
        sample = {'current_img': frame}
        processed_sample = self.transform(sample)
        frame_tensor = processed_sample[0]['current_img']
        
        # Return the original frame as well for display purposes
        return frame_tensor, frame_index, frame


class ROITracker:
    """ROI tracker for performing video object tracking using reference frames and masks."""
    
    def __init__(
        self,
        storage_path: str,
        project_name: str,
        video_source: Any,
        start_frame: int,
        stop_frame: int,
        model_type: str = "r50_deaotl",
    ) -> None:
        """Initialize the ROI tracker.
        
        Args:
            storage_path: Base storage directory
            project_name: Name of the project
            video_source: Video source object
            start_frame: Starting frame index
            stop_frame: Stopping frame index
            model_type: Tracking model type (e.g., 'r50_deaotl', 'swinb_deaotl')
        """
        self.cancel = False
        self.show_middle_result = False
        self.model_type = model_type
        
        # Setup paths
        project_path = Path(storage_path) / project_name
        # video_name = video_source.video_name
        self.track_dir = project_path / "track" / video_source.video_name
        self.track_dir.mkdir(parents=True, exist_ok=True)
        
        # Video parameters
        self.video_source = video_source
        self.start_frame = int(start_frame)
        self.stop_frame = int(stop_frame)
        self.max_memory_length = 30
        
        # Load reference knowledge from labels
        self.reference_frames = []
        label_list = read_roi_labels(storage_path, project_name) # 移除 video_source.video_name 參數
        self.n_rois = 0
        
        for label in label_list:
            frame, mask = label["frame"], label["mask"]
            self.reference_frames.append((frame, mask))
            # Update n_rois to be the maximum value found in masks
            self.n_rois = max(self.n_rois, int(np.max(mask)))
        
        # Current frame and mask for display
        self.current_frame = None
        self.current_mask = None

        # --- Smart Filtering Initialization ---
        self.smart_thresholds = {}
        for (_, mask) in self.reference_frames:
            obj_ids = np.unique(mask)
            for obj_id in obj_ids:
                if obj_id == 0: continue
                area = np.sum(mask == obj_id)
                # If multiple references, we take the one that gives us a reasonable baseline.
                # Here we just blindly set/overwrite, assuming references are consistent.
                # A safer bet is 10% of the reference area.
                self.smart_thresholds[obj_id] = area * 0.1
        print(f"Smart Filtering Thresholds: {self.smart_thresholds}")

    def _smart_filter(self, mask: np.ndarray) -> np.ndarray:
        """Apply automated smart filtering: Keep Largest Component (> threshold)."""
        if mask.max() == 0: return mask
        
        new_mask = np.zeros_like(mask)
        obj_ids = np.unique(mask)
        
        for obj_id in obj_ids:
            if obj_id == 0: continue
            
            # Binary mask for this object
            binary_mask = (mask == obj_id).astype(np.uint8)
            
            # Connectivity Analysis
            # num_labels includes background (0)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
            
            if num_labels <= 1: continue 
            
            threshold = self.smart_thresholds.get(obj_id, 50) # Default 50 pixels if untracked
            
            # Find largest component that meets threshold
            largest_area = -1
            largest_label = -1
            
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area > threshold and area > largest_area:
                    largest_area = area
                    largest_label = i
            
            # If we found a valid component, keep it
            if largest_label != -1:
                new_mask[labels == largest_label] = obj_id
                
        return new_mask
    
    def track(self, progress: Optional[gr.Progress] = None, skip_existing: bool = False) -> str:
        """Execute ROI tracking over specified frames using a parallelized DataLoader and batch inference."""
        time.sleep(0.5)

        # Initialize tracker model and HDF5 writer
        tracker = generate_aot(model_type=self.model_type)
        mask_list_path = self.track_dir / "mask_list.h5"
        
        # --- Start of new logic: Ensure a clean HDF5 file ---
        if os.path.exists(mask_list_path):
            if skip_existing:
                print(f"Skipping existing tracked file: {mask_list_path}")
                return "Skip"
                
            try:
                os.remove(mask_list_path)
                print(f"Removed existing HDF5 file: {mask_list_path}")
            except Exception as e:
                print(f"Warning: Could not remove existing HDF5 file {mask_list_path}. Error: {e}")
        # --- End of new logic ---

        mask_seq = H5IO(str(mask_list_path))

        # Write video and ROI configuration
        first_frame = self.video_source[0]
        mask_seq.write_config("n_rois", self.n_rois)
        mask_seq.write_config("total_frames", len(self.video_source))
        mask_seq.write_config("height", first_frame.shape[0])
        mask_seq.write_config("width", first_frame.shape[1])

        # Add all reference ROI frames to tracker's memory
        for frame, mask in self.reference_frames:
            tracker.add_reference_frame(frame, mask, self.n_rois, -1)

        # Determine tracking direction
        delta = 1 if self.start_frame < self.stop_frame else -1
        frame_range = list(range(self.start_frame, self.stop_frame + delta, delta))

        # Initialize DataLoader for batch processing
        # Initialize DataLoader for batch processing
        # Limit num_workers to prevent OOM (20% of CPU cores)
        num_workers = max(1, int(os.cpu_count() * 0.2))
        batch_size = 16  # Process frames in batches
        print(f"DEBUG: Tracking with {num_workers} workers and batch size {batch_size}")

        dataset = TrackingDataset(self.video_source, frame_range, tracker.transform)
        
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        
        # Iterator with progress bar, handling custom notification callback if present
        if progress is not None:
            iterator = progress.tqdm(loader, desc="Tracking frames")
        else:
            iterator = tqdm(loader, desc="Tracking frames")

        for frame_tensors, frame_indices, original_frames in iterator:
            # Check for cancellation flag
            if self.cancel:
                self.show_middle_result = False
                self.cancel = False
                del mask_seq
                return "Cancel"
            
            # Prepare batch of original sizes
            original_sizes = [frame.shape[:2] for frame in original_frames.numpy()]

            # Perform batch tracking
            mask_batch = tracker.track_batch(frame_tensors, original_sizes=original_sizes)

            # Process and save the batch of masks
            processed_masks = mask_batch.squeeze(1).detach().cpu().numpy().astype(np.uint8)
            
            for i in range(len(processed_masks)):
                frame_idx = frame_indices[i].item()
                mask_to_save = processed_masks[i]
                
                # Apply Smart Filtering (Keep Largest Component per class)
                mask_to_save = self._smart_filter(mask_to_save)
                
                # Update current state for display (with the last frame of the batch)
                self.current_frame = original_frames[i].numpy()
                self.current_mask = mask_to_save
                
                # Write mask to HDF5 file
                mask_seq.write_mask(frame_idx, mask_to_save)

        # Cleanup
        self.show_middle_result = False
        del mask_seq

        return "Done"
    
    def cancel_tracking(self) -> None:
        """Set flag to cancel tracking."""
        self.cancel = True
    
    def toggle_display_mode(self) -> None:
        """Toggle the display of intermediate results."""
        self.show_middle_result = not self.show_middle_result
    
    def get_current_result(self) -> Tuple[Optional[Any], Optional[Any]]:
        """Get current frame and mask.
        
        Returns:
            Tuple of (current_frame, current_mask)
        """
        return self.current_frame, self.current_mask

