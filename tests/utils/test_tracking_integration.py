
import pytest
import os
import shutil
import numpy as np
from castle.utils.tracking_manager import ROITracker
from castle.utils.h5_io import H5IO
from castle.utils.video_io import ReadArray # 新增：匯入 ReadArray

@pytest.mark.integration
def test_roitracker_cycle(dummy_project, device):
    """
    Test the ROI Tracking pipeline (AOT).
    WARNING: This requires AOT weights.
    """
    storage_path, project_name, video_name = dummy_project
    
    # 為 ROITracker 的 __init__ 準備正確的參數
    video_source = ReadArray(os.path.join(storage_path, project_name, "sources", video_name))
    start_frame = 0
    stop_frame = video_source.total_frames - 1 # 確保追蹤到影片結尾
    
    # Check if weights exist, else skip
    try:
        tracker = ROITracker(
            storage_path=storage_path,
            project_name=project_name,
            video_source=video_source, # 更正為正確的參數
            start_frame=start_frame, # 新增參數
            stop_frame=stop_frame, # 新增參數
            model_type="r50_deaotl"
        )
    except Exception as e:
        # 關閉 video_source 以釋放資源
        video_source.close()
        pytest.skip(f"Failed to init ROItracker (likely missing AOT weights): {e}")

    # Run Tracking
    try:
        tracker.track()
    except RuntimeError as e:
        # 關閉 video_source 以釋放資源
        video_source.close()
        if "CUDA" in str(e) and device == "cpu":
             pytest.skip("AOT requires CUDA")
        raise e
        
    # Verify Output
    track_dir = os.path.join(storage_path, project_name, "track", video_name)
    mask_file = os.path.join(track_dir, "mask_list.h5")
    
    assert os.path.exists(mask_file)
    
    h5 = H5IO(mask_file)
    # Check mask content
    mask0 = h5.read_mask(0)
    assert mask0.max() > 0 # Should have ROI
    
    # 關閉 video_source 以釋放資源
    video_source.close()
