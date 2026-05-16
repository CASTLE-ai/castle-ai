"""
tests/conftest.py
Shared fixtures for the CASTLE test suite.

This file is kept lightweight — no torch, no gradio, no model loading.
Heavy fixtures for integration tests live in tests/integration/conftest.py.
"""

import os
import pytest
import shutil
import tempfile
from pathlib import Path

import numpy as np

os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'


# ---- Lightweight fixtures (safe for unit tests) ----

@pytest.fixture
def tmp_storage():
    """Create a temporary storage directory for project tests."""
    d = tempfile.mkdtemp(prefix="castle_unit_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture(scope="session")
def synthetic_video(tmp_path_factory) -> Path:
    """Generate a deterministic 10-frame 64×64 mp4 for utils tests.

    Each frame's red channel encodes the frame index (``frame[..., 0] = i * 25``),
    which lets unit tests assert "the reader gave me frame i" without parsing
    pixel content beyond a mean check.

    Session scope: encoding takes ~0.5s, and the file is read-only — sharing
    it across the suite saves time and reduces ffmpeg load on CI.

    Returns:
        Absolute path to the generated mp4.
    """
    from castle.utils.video_io import VideoWriter

    out_dir = tmp_path_factory.mktemp("synthetic_video")
    out_path = out_dir / "synth_10f_64x64.mp4"

    frames = np.zeros((10, 64, 64, 3), dtype=np.uint8)
    for i in range(10):
        frames[i, :, :, 0] = i * 25  # red channel encodes frame index

    with VideoWriter(out_path, fps=30.0, crf=18) as writer:
        for frame in frames:
            writer.write_frame(frame)

    return out_path
