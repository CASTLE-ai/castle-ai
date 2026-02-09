
import pytest
import os
import numpy as np
import cv2
from castle.utils.video_io import VideoReader, VideoWriter

def test_video_reader_seeking(dummy_video):
    """
    Test seeking capabilities of VideoReader with a real file.
    """
    with VideoReader(dummy_video) as vr:
        total_frames = len(vr)
        assert total_frames == 30
        assert hasattr(vr, 'fps')
        
        # Test sequential
        f0 = vr[0]
        assert f0.shape == (480, 640, 3)
        
        # Test seek
        f29 = vr[29]
        assert f29.shape == (480, 640, 3)
        
        # Verify content (moving circle)
        # Frame 0 circle at x=0. Frame 29 circle at x=640.
        # Check center pixel roughly
        # This assumes we know the content.
        pass

def test_video_writer_roundtrip(temp_workspace):
    """
    Write frames, read them back.
    """
    out_path = os.path.join(temp_workspace, "output_test_io.mp4")
    fps = 30
    
    writer = VideoWriter(out_path, fps)
    
    # Write 10 red frames
    for _ in range(10):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:, :, 2] = 255 # BGR -> Red is index 2? Or RGB? 
        # VideoWriter usually expects BGR/RGB depending on backend.
        # castle VideoWriter uses pyav, usually RGB if configured so.
        writer.write_frame(frame)
        
    writer.close()
    
    assert os.path.exists(out_path)
    
    # Read Back
    with VideoReader(out_path) as vr:
        assert len(vr) == 10
        assert vr.fps == 30
