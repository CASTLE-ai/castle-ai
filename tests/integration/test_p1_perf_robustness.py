"""P1 performance + robustness tests.

Covers:

- PERF-01 ``_aggregate_latents``: pre-alloc path and memmap fall-back
  (threshold knob via ``CASTLE_MEMMAP_THRESHOLD_GB``).
- PERF-02 ``_load_prescan_cache`` / ``_save_prescan_cache`` round-trip and
  stale-key rejection.
- PERF-03 ``LatentAggregator`` frame cache: returns cached frame on second
  call and respects the LRU bound.
- PERF-07 helper ``_enable_cudnn_benchmark_if_not_strict`` honours
  ``cudnn.deterministic``.
- BUG-08 ``VideoReader`` raises ``VideoReadError`` for bogus paths.
- BUG-10 ``load_latent_safe`` raises ``LatentCorruptError`` for malformed
  ``.npz`` files.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List

import numpy as np
import pytest


# ---- PERF-01: aggregator pre-alloc + memmap --------------------------------

def test_aggregate_latents_in_memory_path() -> None:
    from castle.core.cluster import _aggregate_latents

    notes: List[str] = []
    chunks = [np.full((3, 4), float(i), dtype=np.float32) for i in range(2)]
    out = _aggregate_latents(
        chunks, cache_dir="/tmp/should_not_be_used",
        notify=lambda *args, **kwargs: notes.append("".join(map(str, args))),
    )
    assert out.shape == (6, 4)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out[0], np.zeros(4))
    np.testing.assert_array_equal(out[3], np.ones(4))
    # No notify message in the in-memory path
    assert notes == []


def test_aggregate_latents_memmap_fallback(tmp_path, monkeypatch) -> None:
    from castle.core.cluster import _aggregate_latents

    # Force a tiny threshold so any non-empty agg falls back to memmap
    monkeypatch.setenv("CASTLE_MEMMAP_THRESHOLD_GB", "0.0")

    notes: List[str] = []
    chunks = [np.ones((5, 3), dtype=np.float32)]
    cache_dir = tmp_path / "cache"
    out = _aggregate_latents(
        chunks, cache_dir=str(cache_dir),
        notify=lambda msg, level="info": notes.append(msg),
    )
    assert isinstance(out, np.memmap)
    assert out.shape == (5, 3)
    np.testing.assert_array_equal(out, np.ones((5, 3), dtype=np.float32))
    assert (cache_dir / "aggregated_latents.dat").exists()
    assert any("memmap" in m.lower() for m in notes), notes


def test_memmap_threshold_env_var_invalid_fallback(monkeypatch) -> None:
    from castle.core.cluster import _DEFAULT_MEMMAP_THRESHOLD_GB, _memmap_threshold_bytes

    monkeypatch.setenv("CASTLE_MEMMAP_THRESHOLD_GB", "notanumber")
    assert _memmap_threshold_bytes() == int(_DEFAULT_MEMMAP_THRESHOLD_GB * (1024 ** 3))


# ---- PERF-02: pre-scan sidecar cache ---------------------------------------

def test_prescan_cache_round_trip(tmp_path) -> None:
    from castle.core.extractor import _load_prescan_cache, _save_prescan_cache

    key = {"center_roi_id": 1, "rotate_roi_tail_id": 2, "video_len": 100}
    points = {0: (1.0, 2.0), 50: (3.0, 4.0), 99: (5.0, 6.0)}
    cache = tmp_path / "tail_roi_scan.json"

    _save_prescan_cache(str(cache), key, points)
    assert cache.exists()

    loaded = _load_prescan_cache(str(cache), key)
    assert loaded == {0: (1.0, 2.0), 50: (3.0, 4.0), 99: (5.0, 6.0)}


def test_prescan_cache_stale_key_returns_none(tmp_path) -> None:
    from castle.core.extractor import _load_prescan_cache, _save_prescan_cache

    cache = tmp_path / "tail_roi_scan.json"
    _save_prescan_cache(
        str(cache),
        {"center_roi_id": 1, "rotate_roi_tail_id": 2, "video_len": 100},
        {0: (1.0, 2.0)},
    )
    # Different rotate_roi_tail_id → no hit
    assert _load_prescan_cache(
        str(cache),
        {"center_roi_id": 1, "rotate_roi_tail_id": 3, "video_len": 100},
    ) is None


def test_prescan_cache_missing_file() -> None:
    from castle.core.extractor import _load_prescan_cache

    assert _load_prescan_cache("/tmp/does_not_exist_xyz.json", {"a": 1}) is None


# ---- PERF-03: frame cache --------------------------------------------------


class _RecordingReader:
    """Stub VideoReader that records get_frame calls."""

    def __init__(self, frames):
        self.frames = frames
        self.calls = 0

    def get_frame(self, idx):
        self.calls += 1
        return self.frames[idx]

    def close(self):
        pass


def _aggregator_with_stub_reader(reader, video_path="/stub/v.mp4"):
    """Construct a LatentAggregator without going through full __init__."""
    from castle.core.cluster import LatentAggregator
    from collections import OrderedDict
    import threading

    agg = LatentAggregator.__new__(LatentAggregator)
    agg.source_path = "/stub"
    agg.bin_size = 1
    agg.videos_meta = [(10, "v.mp4")]
    agg._video_reader_cache = OrderedDict()
    agg._video_reader_cache[video_path] = reader
    agg._cache_max_size = 8
    agg._frame_cache = OrderedDict()
    agg._frame_cache_max = 4   # Tiny for eviction test
    agg._cache_lock = threading.Lock()
    agg.notify = lambda *args, **kwargs: None
    return agg


def test_frame_cache_serves_repeated_reads_without_re_decoding() -> None:
    frames = [np.full((4, 4, 3), i, dtype=np.uint8) for i in range(10)]
    reader = _RecordingReader(frames)
    agg = _aggregator_with_stub_reader(reader)

    f1 = agg.get_frame(3)
    f2 = agg.get_frame(3)
    np.testing.assert_array_equal(f1, f2)
    assert reader.calls == 1, "Second hover should hit cache, not decode"


def test_frame_cache_evicts_lru_at_max() -> None:
    frames = [np.full((4, 4, 3), i, dtype=np.uint8) for i in range(10)]
    reader = _RecordingReader(frames)
    agg = _aggregator_with_stub_reader(reader)

    for i in range(5):  # frame_cache_max == 4 → first one evicted
        agg.get_frame(i)
    assert reader.calls == 5
    # Re-reading frame 0 should miss the cache (evicted) and increment calls
    agg.get_frame(0)
    assert reader.calls == 6
    # Re-reading frame 4 (most recent) hits cache
    agg.get_frame(4)
    assert reader.calls == 6


# ---- PERF-07: cudnn benchmark guard ---------------------------------------

def test_enable_cudnn_benchmark_respects_strict_mode() -> None:
    torch = pytest.importorskip("torch")
    from castle.core.extractor import _enable_cudnn_benchmark_if_not_strict

    if not torch.cuda.is_available():
        pytest.skip("No CUDA device — benchmark flag is irrelevant on CPU")

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    _enable_cudnn_benchmark_if_not_strict()
    assert torch.backends.cudnn.benchmark is False, (
        "strict mode (cudnn.deterministic=True) must NOT enable benchmark"
    )

    # Non-strict path flips it on
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    _enable_cudnn_benchmark_if_not_strict()
    assert torch.backends.cudnn.benchmark is True


# ---- BUG-08: VideoReader guard --------------------------------------------

def test_video_reader_raises_video_read_error_for_missing_file() -> None:
    """A non-existent path raises FileNotFoundError (not silently)."""
    from castle.utils.video_io import VideoReader

    with pytest.raises(FileNotFoundError):
        VideoReader("/tmp/this_video_does_not_exist_xyz.mp4")


# ---- BUG-10: safe latent loader -------------------------------------------

def test_load_latent_safe_happy_path(tmp_path) -> None:
    from castle.utils.safe_load import load_latent_safe

    p = tmp_path / "ok.npz"
    arr = np.random.default_rng(0).standard_normal((6, 4)).astype(np.float32)
    np.savez_compressed(p, latent=arr)
    out = load_latent_safe(p)
    np.testing.assert_array_equal(out, arr)


def test_load_latent_safe_missing_key_raises(tmp_path) -> None:
    from castle.core.types import LatentCorruptError
    from castle.utils.safe_load import load_latent_safe

    p = tmp_path / "wrong_key.npz"
    np.savez_compressed(p, other=np.zeros(3))
    with pytest.raises(LatentCorruptError) as ei:
        load_latent_safe(p)
    assert "'latent' key" in str(ei.value) or "missing the 'latent'" in str(ei.value)


def test_load_latent_safe_truncated_file_raises(tmp_path) -> None:
    from castle.core.types import LatentCorruptError
    from castle.utils.safe_load import load_latent_safe

    p = tmp_path / "trunc.npz"
    p.write_bytes(b"not really a zip")
    with pytest.raises(LatentCorruptError) as ei:
        load_latent_safe(p)
    assert "castle extract" in str(ei.value).lower()


def test_load_latent_safe_non_2d_raises(tmp_path) -> None:
    from castle.core.types import LatentCorruptError
    from castle.utils.safe_load import load_latent_safe

    p = tmp_path / "wrong_shape.npz"
    np.savez_compressed(p, latent=np.zeros((4, 4, 4)))
    with pytest.raises(LatentCorruptError):
        load_latent_safe(p)


def test_load_latent_safe_missing_file_raises() -> None:
    from castle.core.types import LatentCorruptError
    from castle.utils.safe_load import load_latent_safe

    with pytest.raises(LatentCorruptError):
        load_latent_safe("/tmp/never_existed_xyz.npz")
