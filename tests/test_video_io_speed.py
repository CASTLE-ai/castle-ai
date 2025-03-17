#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test the performance of ReadArray and WriteArray.

The test does the following:
1. Generate random images as video frames.
2. Write frames to a video file using WriteArray.
3. Read the video file using ReadArray.
4. Calculate and present the write and read speeds via warnings.
"""

import time
import numpy as np
import os
import warnings
import pytest
from castle.utils.video_io import ReadArray, WriteArray

def test_video_io_speed(tmp_path):
    """
    Test the speed of video I/O using WriteArray and ReadArray.
    The results are presented using warnings.
    """
    # Ensure warnings are always shown
    warnings.simplefilter("always", UserWarning)

    # Parameters
    num_frames = 1000
    frame_shape = (1024, 1024, 3)
    fps = 60
    crf = 15
    out_path = str(tmp_path / "test_output.mp4")
    
    # Generate random frames (simulate video frames)
    warnings.warn(f"Generating {num_frames} random frames with shape {frame_shape}...", UserWarning)
    frames = [np.random.randint(0, 256, frame_shape, dtype=np.uint8) for _ in range(num_frames)]
    
    # Test writing speed
    # warnings.warn(f"Testing WriteArray (crf={crf}) write speed...", UserWarning)
    writer = WriteArray(out_path, fps, crf)
    start_write = time.perf_counter()
    for frame in frames:
        writer.append(frame)
    writer.close()
    end_write = time.perf_counter()
    
    write_duration = end_write - start_write
    write_fps = num_frames / write_duration
    warnings.warn(f"Wrote {num_frames} frames in {write_duration:.3f} seconds (~{write_fps:.2f} fps)", UserWarning)
    
    # Test reading speed
    # warnings.warn("Testing ReadArray read speed...", UserWarning)
    start_read = time.perf_counter()
    video = ReadArray(out_path)
    total_frames = len(video)
    for i in range(total_frames):
        _ = video[i]
    end_read = time.perf_counter()
    
    read_duration = end_read - start_read
    read_fps = total_frames / read_duration
    warnings.warn(f"Read {total_frames} frames in {read_duration:.3f} seconds (~{read_fps:.2f} fps)", UserWarning)
    
    # Clean up: remove the generated file
    if os.path.exists(out_path):
        os.remove(out_path)
        warnings.warn(f"Removed temporary file: {out_path}", UserWarning)
    
    # Assert that the file has been removed
    assert not os.path.exists(out_path)

if __name__ == "__main__":
    pytest.main([__file__])
