"""
tests/integration/conftest.py
Heavy fixtures for integration tests (GPU, video, models).
"""

import os
import json
import shutil
import tempfile

import cv2
import numpy as np
import pytest
import torch

os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

TEST_VIDEO_LEN = 30
TEST_VIDEO_FPS = 30
TEST_VIDEO_W = 640
TEST_VIDEO_H = 480


@pytest.fixture(scope="session")
def device():
    """Returns 'cuda' if available, else 'cpu'."""
    d = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n[Fixture] Testing on device: {d}")
    return d


@pytest.fixture(scope="session")
def temp_workspace():
    """Creates a temporary workspace directory that persists for the session."""
    temp_dir = tempfile.mkdtemp(prefix="castle_test_")
    print(f"\n[Fixture] Temporary workspace: {temp_dir}")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def dummy_video(temp_workspace):
    """Generates a real .mp4 video file with moving circles."""
    video_path = os.path.join(temp_workspace, "test_video.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, TEST_VIDEO_FPS, (TEST_VIDEO_W, TEST_VIDEO_H))

    for i in range(TEST_VIDEO_LEN):
        frame = np.zeros((TEST_VIDEO_H, TEST_VIDEO_W, 3), dtype=np.uint8)
        cx = int((i / TEST_VIDEO_LEN) * TEST_VIDEO_W)
        cy = TEST_VIDEO_H // 2
        cv2.circle(frame, (cx, cy), 50, (0, 255, 0), -1)
        cv2.putText(frame, f"Frame {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        out.write(frame)

    out.release()
    return video_path


@pytest.fixture(scope="session")
def dummy_project(temp_workspace, dummy_video):
    """Sets up a minimal castle-ai project structure."""
    project_name = "TestProject"
    project_path = os.path.join(temp_workspace, project_name)

    os.makedirs(os.path.join(project_path, "sources"), exist_ok=True)
    os.makedirs(os.path.join(project_path, "track"), exist_ok=True)
    os.makedirs(os.path.join(project_path, "latent"), exist_ok=True)

    video_name = os.path.basename(dummy_video)
    dest_video = os.path.join(project_path, "sources", video_name)
    shutil.copy(dummy_video, dest_video)

    config = {"source": [video_name], "latent": {}}
    with open(os.path.join(project_path, "config.json"), "w") as f:
        json.dump(config, f)

    try:
        import h5py
        track_dir = os.path.join(project_path, "track", video_name)
        os.makedirs(track_dir, exist_ok=True)
        mask_path = os.path.join(track_dir, "mask_list.h5")
        with h5py.File(mask_path, 'w') as f:
            for i in range(TEST_VIDEO_LEN):
                mask = np.zeros((TEST_VIDEO_H, TEST_VIDEO_W), dtype=np.uint8)
                mask[100:300, 100:300] = 1
                f.create_dataset(str(i), data=mask, compression="gzip")
    except ImportError:
        pass

    try:
        label_dir = os.path.join(project_path, "label", video_name)
        os.makedirs(label_dir, exist_ok=True)
        label_path = os.path.join(label_dir, "frame0-label.npz")
        dummy_frame = np.zeros((TEST_VIDEO_H, TEST_VIDEO_W, 3), dtype=np.uint8)
        dummy_mask = np.zeros((TEST_VIDEO_H, TEST_VIDEO_W), dtype=np.uint8)
        dummy_mask[100:300, 100:300] = 1
        np.savez_compressed(label_path, frame=dummy_frame, mask=dummy_mask)
    except Exception:
        pass

    return temp_workspace, project_name, video_name
