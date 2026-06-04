"""Tests for the optimized mix-video path: the overlay function (plot.py) and the
encode_overlay_video pipeline (video_io.py)."""

import zlib

import av
import cv2
import h5py
import numpy as np
import pytest

from castle.utils.plot import generate_mix_image, colorize_mask
from castle.utils.video_io import encode_overlay_video, ReadArray


def test_generate_mix_image_invariants():
    frame = (np.random.rand(40, 50, 3) * 255).astype(np.uint8)
    mask = np.zeros((40, 50), np.uint8)
    mask[10:25, 12:30] = 1
    out = generate_mix_image(frame, mask, alpha=0.5)
    assert out.shape == frame.shape and out.dtype == np.uint8
    # Background (mask==0) is untouched.
    bg = mask == 0
    assert np.array_equal(out[bg], frame[bg])
    # Foreground equals the cv2 alpha blend with the palette colour.
    colorized = colorize_mask(mask)
    blended = cv2.addWeighted(frame, 0.5, colorized, 0.5, 0.0)
    fg = mask != 0
    assert np.array_equal(out[fg], blended[fg])


def test_colorize_mask_uses_palette_lut():
    mask = np.array([[0, 1], [2, 0]], np.uint8)
    col = colorize_mask(mask)
    assert col.shape == (2, 2, 3) and col.dtype == np.uint8
    assert np.array_equal(col[0, 0], [0, 0, 0])  # label 0 = background = black


def _make_source(path, n=12, w=64, h=64, fps=30):
    c = av.open(str(path), mode="w")
    s = c.add_stream("libx264", rate=fps)
    s.width = w; s.height = h; s.pix_fmt = "yuv420p"
    for i in range(n):
        img = np.full((h, w, 3), i * 5, np.uint8)
        for pkt in s.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
            c.mux(pkt)
    for pkt in s.encode():
        c.mux(pkt)
    c.close()


def _make_masks(path, n=12, w=64, h=64):
    with h5py.File(str(path), "w") as f:
        for i in range(n):
            m = np.zeros((h, w), np.uint8)
            m[20:40, 20:40] = 1
            f.create_dataset(str(i), data=m, dtype="uint8",
                             compression="gzip", compression_opts=3)
        f.create_dataset("total_frames", data=n)


def test_encode_overlay_video_produces_playable_output(tmp_path, monkeypatch):
    monkeypatch.setenv("CASTLE_VIDEO_ENCODER", "x264")  # deterministic, no GPU dependency
    src = tmp_path / "src.mp4"; mh5 = tmp_path / "mask_list.h5"; out = tmp_path / "out-mix.mp4"
    _make_source(src); _make_masks(mh5)
    res = encode_overlay_video(str(src), str(mh5), str(out), 30, generate_mix_image)
    assert res == str(out) and out.exists() and out.stat().st_size > 0
    with ReadArray(str(out)) as r:
        assert len(r) >= 10  # ~12 frames encoded


def test_encode_overlay_video_cancel_removes_partial(tmp_path, monkeypatch):
    import threading
    monkeypatch.setenv("CASTLE_VIDEO_ENCODER", "x264")
    src = tmp_path / "src.mp4"; mh5 = tmp_path / "mask_list.h5"; out = tmp_path / "out-mix.mp4"
    _make_source(src, n=12); _make_masks(mh5, n=12)
    ev = threading.Event(); ev.set()  # already cancelled
    from castle.core._centroid_worker import PreprocessCancelled
    with pytest.raises(PreprocessCancelled):
        encode_overlay_video(str(src), str(mh5), str(out), 30, generate_mix_image, cancel_event=ev)
    assert not out.exists()  # partial output cleaned up
