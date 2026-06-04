"""Tests for the shared video encoder (castle/core/video_encoder.py) — NVENC
selection + RUNTIME fallback — and the direct-chunk mask writer
(H5IO.write_mask_compressed)."""

import zlib

import h5py
import numpy as np

import castle.core.video_encoder as ve
from castle.utils.h5_io import H5IO


# --- open_encoder -----------------------------------------------------------

class _FakeStream:
    def __init__(self):
        self.codec_context = type("cc", (), {"thread_count": None})()


class _FakeContainer:
    """add_stream raises for h264_nvenc, succeeds for libx264 — simulates a host
    where NVENC compiles in but fails at runtime (driver/session limit)."""
    def __init__(self, nvenc_works):
        self.nvenc_works = nvenc_works

    def add_stream(self, codec, rate=None):
        if codec == "h264_nvenc" and not self.nvenc_works:
            raise RuntimeError("nvenc init failed (no usable session)")
        return _FakeStream()

    def close(self):
        pass


def _patch_av(monkeypatch, nvenc_works):
    import av
    monkeypatch.setattr(av, "open", lambda *a, **k: _FakeContainer(nvenc_works))


def test_open_encoder_forced_x264(monkeypatch):
    monkeypatch.setenv("CASTLE_VIDEO_ENCODER", "x264")
    _patch_av(monkeypatch, nvenc_works=True)
    _, _, codec = ve.open_encoder("/tmp/none.mp4", 30, 592, 592)
    assert codec == "libx264"


def test_open_encoder_runtime_fallback(monkeypatch):
    # auto, but NVENC can't encode at this size (validated via _nvenc_ok) → libx264.
    monkeypatch.setenv("CASTLE_VIDEO_ENCODER", "auto")
    monkeypatch.setattr(ve, "_nvenc_ok", lambda fps, w, h: False)
    _patch_av(monkeypatch, nvenc_works=True)
    _, _, codec = ve.open_encoder("/tmp/none.mp4", 30, 64, 64)
    assert codec == "libx264"


def test_open_encoder_forced_nvenc_falls_back_on_small_frame(monkeypatch):
    # Even with mode=nvenc, a size NVENC can't handle must degrade, not crash.
    monkeypatch.setenv("CASTLE_VIDEO_ENCODER", "nvenc")
    monkeypatch.setattr(ve, "_nvenc_ok", lambda fps, w, h: False)
    _patch_av(monkeypatch, nvenc_works=True)
    _, _, codec = ve.open_encoder("/tmp/none.mp4", 30, 48, 48)
    assert codec == "libx264"


def test_open_encoder_uses_nvenc_when_it_works(monkeypatch):
    monkeypatch.setenv("CASTLE_VIDEO_ENCODER", "auto")
    monkeypatch.setattr(ve, "_nvenc_ok", lambda fps, w, h: True)
    _patch_av(monkeypatch, nvenc_works=True)
    _, _, codec = ve.open_encoder("/tmp/none.mp4", 30, 592, 592)
    assert codec == "h264_nvenc"


def test_select_video_encoder_legacy_env_alias(monkeypatch):
    monkeypatch.delenv("CASTLE_VIDEO_ENCODER", raising=False)
    monkeypatch.setenv("CASTLE_PREPROCESS_ENCODER", "x264")  # legacy knob still honored
    assert ve.select_video_encoder(30, 592, 592) == "libx264"


# --- write_mask_compressed --------------------------------------------------

def test_write_mask_compressed_roundtrip(tmp_path):
    m = (np.random.rand(64, 80) * 3).astype(np.uint8)
    p = str(tmp_path / "out.h5")
    with H5IO(p) as h5:
        h5.write_mask_compressed(5, zlib.compress(m.tobytes(), 3), m.shape)
    with h5py.File(p, "r") as f:
        assert np.array_equal(f["5"][:], m)
    with H5IO(p, read_only=True) as h5:
        assert np.array_equal(h5.read_mask(5), m)


def test_write_mask_compressed_matches_normal_write(tmp_path):
    m = (np.random.rand(48, 48) * 2).astype(np.uint8)
    a, b = str(tmp_path / "a.h5"), str(tmp_path / "b.h5")
    with H5IO(a) as h5:
        h5.write_mask(0, m)
    with H5IO(b) as h5:
        h5.write_mask_compressed(0, zlib.compress(m.tobytes(), 3), m.shape)
    with h5py.File(a, "r") as fa, h5py.File(b, "r") as fb:
        assert np.array_equal(fa["0"][:], fb["0"][:])
