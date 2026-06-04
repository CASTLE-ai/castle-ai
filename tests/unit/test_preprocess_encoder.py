"""Tests for the pre-process video encoder selection + RUNTIME NVENC fallback
(castle/service/preprocessing_service._open_encoder) and the direct-chunk mask
writer (H5IO.write_mask_compressed)."""

import zlib

import h5py
import numpy as np

import castle.service.preprocessing_service as svc
from castle.utils.h5_io import H5IO


# --- _open_encoder ----------------------------------------------------------

class _FakeStream:
    def __init__(self):
        self.codec_context = type("cc", (), {"thread_count": None})()


class _FakeContainer:
    """add_stream raises for h264_nvenc, succeeds for libx264 — simulates a host
    where NVENC compiles in but fails at runtime (driver/session limit)."""
    def __init__(self, nvenc_works):
        self.nvenc_works = nvenc_works
        self.closed = False

    def add_stream(self, codec, rate=None):
        if codec == "h264_nvenc" and not self.nvenc_works:
            raise RuntimeError("nvenc init failed (no usable session)")
        return _FakeStream()

    def close(self):
        self.closed = True


def _patch_av(monkeypatch, nvenc_works):
    import av
    monkeypatch.setattr(av, "open", lambda *a, **k: _FakeContainer(nvenc_works))


def test_open_encoder_forced_x264(monkeypatch):
    monkeypatch.setenv("CASTLE_PREPROCESS_ENCODER", "x264")
    _patch_av(monkeypatch, nvenc_works=True)
    _, _, codec = svc._open_encoder("/tmp/none.mp4", 30, 592, 592)
    assert codec == "libx264"


def test_open_encoder_runtime_fallback(monkeypatch):
    # auto-select nvenc, but the real add_stream fails → must fall back to libx264.
    monkeypatch.setattr(svc, "_select_video_encoder", lambda fps: "h264_nvenc")
    _patch_av(monkeypatch, nvenc_works=False)
    _, _, codec = svc._open_encoder("/tmp/none.mp4", 30, 592, 592)
    assert codec == "libx264"


def test_open_encoder_uses_nvenc_when_it_works(monkeypatch):
    monkeypatch.setattr(svc, "_select_video_encoder", lambda fps: "h264_nvenc")
    _patch_av(monkeypatch, nvenc_works=True)
    _, _, codec = svc._open_encoder("/tmp/none.mp4", 30, 592, 592)
    assert codec == "h264_nvenc"


# --- write_mask_compressed --------------------------------------------------

def test_write_mask_compressed_roundtrip(tmp_path):
    m = (np.random.rand(64, 80) * 3).astype(np.uint8)
    p = str(tmp_path / "out.h5")
    with H5IO(p) as h5:
        h5.write_mask_compressed(5, zlib.compress(m.tobytes(), 3), m.shape)
    # read via plain h5py (what Extract Latent uses) AND via H5IO
    with h5py.File(p, "r") as f:
        assert np.array_equal(f["5"][:], m)
    with H5IO(p, read_only=True) as h5:
        assert np.array_equal(h5.read_mask(5), m)


def test_write_mask_compressed_matches_normal_write(tmp_path):
    # Pre-compressed direct-chunk write must read back identically to a normal
    # gzip create_dataset write.
    m = (np.random.rand(48, 48) * 2).astype(np.uint8)
    a, b = str(tmp_path / "a.h5"), str(tmp_path / "b.h5")
    with H5IO(a) as h5:
        h5.write_mask(0, m)
    with H5IO(b) as h5:
        h5.write_mask_compressed(0, zlib.compress(m.tobytes(), 3), m.shape)
    with h5py.File(a, "r") as fa, h5py.File(b, "r") as fb:
        assert np.array_equal(fa["0"][:], fb["0"][:])
