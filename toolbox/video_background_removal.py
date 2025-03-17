#!/usr/bin/env python
# coding: utf-8
"""
This script processes video files by removing background and applying threshold filtering using GPU acceleration.
It supports command-line arguments for video path, output path, sample ratio for median calculation, and chunk size.
"""

import argparse
from castle.utils.video_io import ReadArray, WriteArray
from tqdm import tqdm
import numpy as np
import cupy as cp

def process_video(video_path, out_path, sample_ratio, chunk_size):
    """
    Process video by removing background and applying threshold filtering using GPU acceleration.
    
    Args:
        video_path (str): Path to input video file.
        out_path (str): Path for output video file.
        sample_ratio (float): Ratio of frames to sample for median frame calculation.
        chunk_size (int): Number of frames to process per chunk.
    """
    # Read the input video
    video = ReadArray(video_path)
    n_frame = len(video)
    fps = video.fps
    crf = 15  # Constant Rate Factor for video compression

    # Initialize output video writer
    out = WriteArray(out_path, fps, crf)

    # Calculate sample indices based on the sample_ratio (at least 1 frame)
    num_samples = max(1, int(n_frame * sample_ratio))
    sample_indices = np.linspace(0, n_frame - 1, num=num_samples, dtype=int)

    # Obtain sampled frames and convert to int16 for processing
    sampled_frames = np.array([video[i] for i in tqdm(sample_indices)]).astype(np.int16)
    # Transfer sampled frames to GPU
    sampled_frames_gpu = cp.asarray(sampled_frames)
    # Compute the median frame on GPU and invert it (255 - median)
    median_frame_gpu = 255 - cp.median(sampled_frames_gpu, axis=0).astype(cp.int16)
    
    del sampled_frames_gpu
    cp.get_default_memory_pool().free_all_blocks()

    # Process video in chunks to avoid memory overflow
    for i in tqdm(range(0, n_frame, chunk_size), desc="Processing chunks"):
        frames_list = [video[j] for j in range(i, min(i + chunk_size, n_frame))]
        frames_cpu = np.array(frames_list).astype(np.int16)
        # Transfer frames to GPU
        frames_gpu = cp.asarray(frames_cpu)
        # Invert the frames
        frames_gpu = 255 - frames_gpu
        # Compute difference with the median frame
        frames_gpu = frames_gpu - median_frame_gpu
        # Apply threshold: set pixel to 0 if difference is less than 30
        frames_gpu = cp.where(frames_gpu < 30, 0, frames_gpu)
        # Clip values between 0 and 255
        frames_gpu = cp.clip(frames_gpu, 0, 255)
        # Invert frames back
        frames_gpu = 255 - frames_gpu
        # Convert to 8-bit and transfer back to CPU
        frames_cpu_processed = cp.asnumpy(frames_gpu.astype(cp.uint8))
        # Append each processed frame to the output
        for frame in frames_cpu_processed:
            out.append(frame)
    out.close()

def main():
    parser = argparse.ArgumentParser(
        description="Process video by removing background and applying threshold filtering with GPU acceleration."
    )
    parser.add_argument("--video_path", type=str, required=True,
                        help="Path to input video file")
    parser.add_argument("--out_path", type=str, required=True,
                        help="Path for output video file")
    parser.add_argument("--sample_ratio", type=float, default=0.05,
                        help="Ratio of frames to sample for median frame calculation (default: 0.1)")
    parser.add_argument("--chunk_size", type=int, default=512,
                        help="Number of frames to process per chunk (default: 512)")

    args = parser.parse_args()
    process_video(args.video_path, args.out_path, args.sample_ratio, args.chunk_size)

if __name__ == "__main__":
    main()
