
import sys
import os
import shutil
import argparse

# Add project root to path
sys.path.append("/media/isonaei/Works/linux temp/castle-ai_castle-ai_rebuild")

from castle.utils.profiler import Profiler, SystemMonitor
from castle.utils.tracking_manager import ROITracker
from castle.core.extractor import extract_roi_latent_from_video
from castle.core.data import Preprocess
from castle.utils.video_io import VideoReader, ReadArray

# Mock objects if needed
class MockReader(ReadArray):
    pass

def profile_tracking(storage_path, project_name, video_name):
    print(f"--- Profiling Tracking: {video_name} ---")
    
    # 1. Setup minimal resources
    video_source = MockReader(os.path.join(storage_path, project_name, "sources", video_name))
    
    tracker = ROITracker(
        storage_path=storage_path, 
        project_name=project_name, 
        video_source=video_source,
        start_frame=0,
        stop_frame=200, # Only track 30 frames for speed
        model_type="r50_deaotl"
    )
    
    # Fake a reference frame if none exist (assuming ROI 1 exists from previous run or we create one)
    # Actually ROITracker reads labels from disk. We hope they exist.
    if not tracker.reference_frames:
        print("No reference frames found. Cannot profile tracking.")
        return

    tracker.track(skip_existing=False)

def profile_extraction(storage_path, project_name, video_name):
    print(f"--- Profiling Extraction: {video_name} ---")
    
    preprocess_config = Preprocess(
        center_roi_switch=True,
        center_roi_id=1,
        center_roi_crop_width=256,
        center_roi_crop_height=256,
        rotate_roi_tail_switch=False, # Simplify
        rotate_roi_tail_id=2,
        remove_background_switch=True
    )
    
    # Run only a few batches
    # We can't control frames easily in extract_roi_latent_from_video without modifying it or the dataloader.
    # But since we set batch_size, we can perhaps just run it. 
    # Or rely on the fact that we are only profiling.
    # Let's run it fully? If video is long it might take time.
    # The previous conversation implies using "current project stuff".
    
    # Actually, let's just run it. The user has "Video01.mp4".
    
    # Force batch size 4 for granular timing
    extract_roi_latent_from_video(
        storage_path=storage_path,
        project_name=project_name,
        video_name=video_name,
        roi_id=1,
        model_name="dinov2_vitb14", # Switch to DINOv2 Base
        batch_size=40, # Increase batch size
        preprocess_config=preprocess_config,
        skip_existing=False
    )

if __name__ == "__main__":
    storage_path = "."
    project_name = "Castle_Project_01"
    video_name = "Video01.mp4"
    
    # Ensure paths exist
    if not os.path.exists(os.path.join(storage_path, project_name)):
        # Fallback to absolute if dot fails
        storage_path = "/media/isonaei/Works/linux temp/castle-ai_castle-ai_rebuild/castle_storage" # Guessing from context, or use args
        
    # Check if storage exists there
    if not os.path.exists(storage_path):
         # Try creating a dummy if needed, but we need video files.
         # Assume running from root where castle_storage might be?
         # User said "use current project folder".
         # Let's check where the user's files are.
         pass

    # Actually lets use the paths from previous context if possible.
    # From conversation history: storage_path='/media/isonaei/Works/linux temp/castle-ai_castle-ai_rebuild' (Project root is here but where is data?)
    # "castle/ui/main_ui.py" usually sets defaults.
    # Let's assume the user calls this script from project root and data is in `projects/CASTLE-ai`.
    # Wait, the user has `castle-ai_castle-ai_rebuild/task.md`.
    # Let's assume project is "CASTLE-ai" and root is current dir.
    
    storage_path = "castle_storage" # Default graduao app storage
    # Updated based on check_paths
    storage_path = "/media/isonaei/Works/linux temp/castle-ai_castle-ai_rebuild/projects"
    project_name = "2025-12-18-10-01-00-Project"
    video_name = "SNI_D0_side_15030_1_male.mp4"
    
    # Find a video
    sources_dir = os.path.join(storage_path, project_name, "sources")
    if os.path.exists(sources_dir):
        videos = [f for f in os.listdir(sources_dir) if f.endswith(".mp4")]
        if videos:
            video_name = videos[0]
            print(f"Found video: {video_name}")
    
    monitor = Profiler()._system_monitor = SystemMonitor(interval=0.2)
    
    try:
        monitor.start()
        profile_tracking(storage_path, project_name, video_name)
    except Exception as e:
        print(f"Tracking failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        # profile_tracking might have stopped monitoring if it crashed? No, try/except handles it.
        profile_extraction(storage_path, project_name, video_name)
    except Exception as e:
        print(f"Extraction failed: {e}")
        import traceback
        traceback.print_exc()
    
    monitor.stop()
        
    Profiler().print_report()
    monitor.print_stats()
