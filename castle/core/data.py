"""
castle/core/data.py
Data structures and dataset classes.
"""

from typing import Optional, Tuple

import numpy as np
from torch.utils.data import Dataset
from castle.utils.video_io import VideoReader
from castle.utils.h5_io import H5IO
import cv2  # Added for interpolation flags
import logging
from castle.utils.video_align import (
    center_roi, rotate_based_on_roi_closest_center_point,
    rotate_based_on_point, crop, blank_page, rotate_based_on_deg
)

logger = logging.getLogger(__name__)

# ---------------------------
# 預處理類別 (Moved from castle/ui/extract_ui.py)
# ---------------------------
class Preprocess:
    """Frame preprocessing pipeline for ROI-centered extraction.

    Applies optional centering, rotation, cropping, and background removal
    to a video frame and its corresponding mask before feature extraction.

    Attributes:
        center_roi_switch: Whether to center the crop on a specific ROI.
        center_roi_id: ROI ID to center on.
        center_roi_crop_width: Crop width in pixels after centering.
        center_roi_crop_height: Crop height in pixels after centering.
        rotate_roi_tail_switch: Whether to normalize orientation using a tail ROI.
        rotate_roi_tail_id: ROI ID that defines the tail direction.
        remove_background_switch: Whether to zero out non-ROI pixels.
    """

    def __init__(self, center_roi_switch: bool = False, center_roi_id: int = 1,
                 center_roi_crop_width: int = 300, center_roi_crop_height: int = 300,
                 rotate_roi_tail_switch: bool = False, rotate_roi_tail_id: int = 2,
                 remove_background_switch: bool = False):
        # M-01 Fix: Core layer should not handle UI string conversion
        # Strict type checking instead
        if not isinstance(center_roi_switch, bool):
            raise TypeError(f"center_roi_switch must be bool, got {type(center_roi_switch)}")
        if not isinstance(rotate_roi_tail_switch, bool):
            raise TypeError(f"rotate_roi_tail_switch must be bool, got {type(rotate_roi_tail_switch)}")
        if not isinstance(remove_background_switch, bool):
            raise TypeError(f"remove_background_switch must be bool, got {type(remove_background_switch)}")
            
        self.center_roi_switch = center_roi_switch
        self.center_roi_id = int(center_roi_id)
        self.center_roi_crop_width = int(center_roi_crop_width)
        self.center_roi_crop_height = int(center_roi_crop_height)
        self.rotate_roi_tail_switch = rotate_roi_tail_switch
        self.rotate_roi_tail_id = int(rotate_roi_tail_id)
        self.remove_background_switch = remove_background_switch
        
        if center_roi_switch and (self.center_roi_crop_width <= 0 or self.center_roi_crop_height <= 0):
            raise ValueError(f"Crop dimensions must be positive, got width={self.center_roi_crop_width}, height={self.center_roi_crop_height}")

    def transform(self, frame: np.ndarray, mask: np.ndarray, deg: int = 0, 
                  precomputed_closest_point: Optional[Tuple[float, float]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply preprocessing transforms to a frame and its mask.
        
        Args:
            frame: Input video frame (H, W, 3)
            mask: Corresponding mask (H, W) or (H, W, 3)
            deg: Optional rotation degree for rotation-invariant extraction
            precomputed_closest_point: If provided, use this (x, y) point for tail rotation
                instead of computing from mask. Used with interpolate_missing_points
                to handle frames where tail ROI is not detected.
        """
        try:
            if self.center_roi_switch:
                f = center_roi(frame, mask, self.center_roi_id)
                m = center_roi(mask, mask, self.center_roi_id, flags=cv2.INTER_NEAREST)
                if self.rotate_roi_tail_switch:
                    if precomputed_closest_point is not None:
                        # Use precomputed (possibly interpolated) point
                        f = rotate_based_on_point(f, precomputed_closest_point)
                        m = rotate_based_on_point(m, precomputed_closest_point)
                    else:
                        # Original: compute from mask in real-time
                        f = rotate_based_on_roi_closest_center_point(f, m, self.rotate_roi_tail_id)
                        m = rotate_based_on_roi_closest_center_point(m, m, self.rotate_roi_tail_id, flags=cv2.INTER_NEAREST)
            else:
                f, m = frame, mask

            if deg > 0:
                f = rotate_based_on_deg(f, deg)
                m = rotate_based_on_deg(m, deg, flags=cv2.INTER_NEAREST)
            elif deg < 0:
                logger.warning(
                    "Negative rotation degree (%s) is not supported and will be skipped. "
                    "Use a positive degree value (0–360).",
                    deg,
                )

            if self.center_roi_switch:
                f = crop(f, self.center_roi_crop_height, self.center_roi_crop_width)
                m = crop(m, self.center_roi_crop_height, self.center_roi_crop_width)

            if self.remove_background_switch:
                f[m == 0] = 0
        except Exception as e:
            logger.error(f"Preprocessing transform failed for ROI ID {self.center_roi_id} (Center) and {self.rotate_roi_tail_id} (Tail). Error: {e}")
            f = blank_page(self.center_roi_crop_height, self.center_roi_crop_width)
            m = blank_page(self.center_roi_crop_height, self.center_roi_crop_width)
        return f, m

# ---------------------------
# 核心類別：支援多核心的 Dataset (Moved from castle/ui/extract_ui.py)
# ---------------------------
class VideoDataset(Dataset):
    """PyTorch Dataset that yields preprocessed (frame, mask) pairs from a video.

    Lazily opens the video file and HDF5 mask store per-worker to support
    multi-process DataLoader without file handle conflicts.

    Args:
        video_path: Path to the source video file.
        video_len: Total number of frames in the video.
        mask_path: Path to the HDF5 mask file (mask_list.h5).
        preprocess: Preprocess instance defining the transform pipeline.
        select_roi: ROI ID to extract.
        rotate_deg: Optional fixed rotation degree for rotation-invariant extraction.
        interpolated_points: Optional dict mapping frame index to (x, y) tail points.
    """

    def __init__(self, video_path: str, video_len: int, mask_path: str, preprocess: Preprocess, select_roi: int, 
                 rotate_deg: Optional[int] = None, interpolated_points: Optional[dict] = None):
        # 我們只存「路徑」，不存物件，避免多行程打架
        self.video_path = video_path
        self.video_len = video_len
        self.mask_path = mask_path
        self.preprocess = preprocess
        self.select_roi = select_roi
        self.rotate_deg = rotate_deg
        self.interpolated_points = interpolated_points  # {frame_idx: (x, y)} or None
        
        # 初始化設為 None，等 Worker 自己打開
        self.reader: Optional[VideoReader] = None 
        self.tracker: Optional[H5IO] = None

    def __len__(self) -> int:
        return self.video_len

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        # Worker 第一次工作時，才打開自己的檔案
        if self.reader is None:
            self.reader = VideoReader(self.video_path)
            
        if self.tracker is None:
            # 重新開啟 H5 檔案讀取 Mask
            self.tracker = H5IO(self.mask_path) 

        frame = self.reader[idx]
        mask = self.tracker.read_mask(idx)
        
        # Get precomputed closest point for this frame (if interpolation is active)
        closest_point = self.interpolated_points.get(idx) if self.interpolated_points else None
        
        if self.rotate_deg is not None:
             pf, pm = self.preprocess.transform(frame, mask, self.rotate_deg, precomputed_closest_point=closest_point)
        else:
             pf, pm = self.preprocess.transform(frame, mask, precomputed_closest_point=closest_point)
             
        return pf, pm
