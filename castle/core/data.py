"""
castle/core/data.py
Data structures and dataset classes.
"""

from typing import Optional, Tuple, Any

import numpy as np
import torch
from torch.utils.data import Dataset
from castle.utils.video_io import ReadArray
from castle.utils.h5_io import H5IO
from castle.utils.video_align import (
    center_roi, rotate_based_on_roi_closest_center_point,
    crop, blank_page, rotate_based_on_deg
)

# ---------------------------
# 預處理類別 (Moved from castle/ui/extract_ui.py)
# ---------------------------
class Preprocess:
    def __init__(self, center_roi_switch: bool, center_roi_id: int, center_roi_crop_width: int, center_roi_crop_height: int, 
                 rotate_roi_tail_switch: bool, rotate_roi_tail_id: int, remove_background_switch: bool = False):
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

    def transform(self, frame: np.ndarray, mask: np.ndarray, deg: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        try:
            if self.center_roi_switch:
                f = center_roi(frame, mask, self.center_roi_id)
                m = center_roi(mask, mask, self.center_roi_id)
                if self.rotate_roi_tail_switch:
                    f = rotate_based_on_roi_closest_center_point(f, m, self.rotate_roi_tail_id)
                    m = rotate_based_on_roi_closest_center_point(m, m, self.rotate_roi_tail_id)
                if deg > 0:
                    f = rotate_based_on_deg(f, deg)
                    m = rotate_based_on_deg(m, deg)
                f = crop(f, self.center_roi_crop_height, self.center_roi_crop_width)
                m = crop(m, self.center_roi_crop_height, self.center_roi_crop_width)
            else:
                f, m = frame, mask
            if self.remove_background_switch:
                f[m == 0] = 255
        except Exception as e:
            print(f"ERROR: Preprocessing transform failed for ROI ID {self.center_roi_id} (Center) and {self.rotate_roi_tail_id} (Tail). Error: {e}")
            f = blank_page(self.center_roi_crop_height, self.center_roi_crop_width)
            m = blank_page(self.center_roi_crop_height, self.center_roi_crop_width)
        return f, m

# ---------------------------
# 核心類別：支援多核心的 Dataset (Moved from castle/ui/extract_ui.py)
# ---------------------------
class VideoDataset(Dataset):
    def __init__(self, video_path: str, video_len: int, mask_path: str, preprocess: Preprocess, select_roi: int, rotate_deg: Optional[int] = None):
        # 我們只存「路徑」，不存物件，避免多行程打架
        self.video_path = video_path
        self.video_len = video_len
        self.mask_path = mask_path
        self.preprocess = preprocess
        self.select_roi = select_roi
        self.rotate_deg = rotate_deg 
        
        # 初始化設為 None，等 Worker 自己打開
        self.reader: Optional[ReadArray] = None 
        self.tracker: Optional[H5IO] = None

    def __len__(self) -> int:
        return self.video_len

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        # Worker 第一次工作時，才打開自己的檔案
        if self.reader is None:
            self.reader = ReadArray(self.video_path)
            
        if self.tracker is None:
            # 重新開啟 H5 檔案讀取 Mask
            self.tracker = H5IO(self.mask_path) 

        # Note: self.reader[idx] might raise index error if len is wrong, but we assume video_len is correct
        frame = self.reader[idx]
        mask = self.tracker.read_mask(idx)
        
        if self.rotate_deg is not None:
             pf, pm = self.preprocess.transform(frame, mask, self.rotate_deg)
        else:
             pf, pm = self.preprocess.transform(frame, mask)
             
        return pf, pm
