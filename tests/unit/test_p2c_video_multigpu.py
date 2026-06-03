"""Routing tests for video-level multi-GPU (P2c): tracking + extraction.

No real GPUs / models — the per-video work functions are monkeypatched, so these
verify *routing* only (pool vs sequential, device pinning, reduced workers,
skip-existing, ordering).
"""

import os

import pytest

from castle.service import tracking_service as ts
from castle.service import extraction_service as es


# --------------------------------------------------------------------------
# tracking_service.track_videos
# --------------------------------------------------------------------------

def test_track_videos_pool_pins_distinct_devices(monkeypatch, tmp_path):
    import threading
    calls = []
    lock = threading.Lock()

    def fake_track_video(storage, project, video, *, model, start, stop, skip_existing, device=None, **k):
        with lock:
            calls.append((video, device))
        import time; time.sleep(0.01)  # hold the slot so both GPUs are exercised
        return "Done"

    monkeypatch.setattr(ts, "track_video", fake_track_video)
    out = ts.track_videos(str(tmp_path), "P", ["a.mp4", "b.mp4", "c.mp4", "d.mp4"],
                          device_ids=[0, 1], skip_existing=False)
    assert out == {"a.mp4": "Done", "b.mp4": "Done", "c.mp4": "Done", "d.mp4": "Done"}
    assert {dev for _, dev in calls} == {"cuda:0", "cuda:1"}  # both cards used
    assert all(str(dev).startswith("cuda:") for _, dev in calls)


def test_track_videos_sequential_when_single_device(monkeypatch, tmp_path):
    calls = []

    def fake_track_video(storage, project, video, *, model, start, stop, skip_existing, device=None, **k):
        calls.append((video, device))
        return "Done"

    monkeypatch.setattr(ts, "track_video", fake_track_video)
    out = ts.track_videos(str(tmp_path), "P", ["a.mp4", "b.mp4"], device_ids=[], skip_existing=False)
    assert out == {"a.mp4": "Done", "b.mp4": "Done"}
    assert all(dev is None for _, dev in calls)  # sequential never pins a device


def test_track_videos_skip_existing_preflight(monkeypatch, tmp_path):
    # b.mp4 already has a mask -> reported 'Skipped' without calling track_video
    mask = tmp_path / "P" / "track" / "b.mp4" / "mask_list.h5"
    mask.parent.mkdir(parents=True)
    mask.write_bytes(b"x")
    seen = []

    def fake_track_video(storage, project, video, **k):
        seen.append(video)
        return "Done"

    monkeypatch.setattr(ts, "track_video", fake_track_video)
    out = ts.track_videos(str(tmp_path), "P", ["a.mp4", "b.mp4"], device_ids=[], skip_existing=True)
    assert out["b.mp4"] == "Skipped"
    assert out["a.mp4"] == "Done"
    assert seen == ["a.mp4"]  # the existing one was never tracked


def test_track_videos_forwards_cancel_event_pool(monkeypatch, tmp_path):
    # The batch cancel_event must reach track_video (→ ROITracker.track) so an
    # in-flight video can abort mid-track, not just stop launching new ones.
    import threading
    ev = threading.Event()
    seen = []

    def fake_track_video(storage, project, video, *, model, start, stop, skip_existing,
                         device=None, num_workers=None, pin_memory=True, cancel_event=None):
        seen.append(cancel_event)
        return "Done"

    monkeypatch.setattr(ts, "track_video", fake_track_video)
    ts.track_videos(str(tmp_path), "P", ["a.mp4", "b.mp4"], device_ids=[0, 1],
                    skip_existing=False, cancel_event=ev)
    assert seen and all(c is ev for c in seen)  # same event threaded into every worker


def test_track_videos_forwards_cancel_event_sequential(monkeypatch, tmp_path):
    import threading
    ev = threading.Event()
    seen = []

    def fake_track_video(storage, project, video, *, model, start, stop, skip_existing,
                         cancel_event=None, **k):
        seen.append(cancel_event)
        return "Done"

    monkeypatch.setattr(ts, "track_video", fake_track_video)
    ts.track_videos(str(tmp_path), "P", ["a.mp4", "b.mp4"], device_ids=[],
                    skip_existing=False, cancel_event=ev)
    assert seen and all(c is ev for c in seen)


def test_track_videos_error_isolation(monkeypatch, tmp_path):
    def fake_track_video(storage, project, video, *, model, start, stop, skip_existing, device=None, **k):
        if video == "bad.mp4":
            raise RuntimeError("kaboom")
        return "Done"

    monkeypatch.setattr(ts, "track_video", fake_track_video)
    out = ts.track_videos(str(tmp_path), "P", ["ok.mp4", "bad.mp4"], device_ids=[0, 1], skip_existing=False)
    assert out["ok.mp4"] == "Done"
    assert out["bad.mp4"].startswith("Error:")


# --------------------------------------------------------------------------
# extraction_service.extract_latent  (batch 'All')
# --------------------------------------------------------------------------

def _patch_project(monkeypatch, videos):
    monkeypatch.setattr(es, "get_project_config", lambda sp, pn: ("/tmp/proj", {"source": videos}))


def test_extract_latent_parallel_pins_devices_and_reduces_workers(monkeypatch):
    import castle.core.gpu_pool as gp
    _patch_project(monkeypatch, ["v1.mp4", "v2.mp4"])
    monkeypatch.setattr(gp, "resolve_device_ids", lambda: [0, 1])

    import threading
    calls = []
    lock = threading.Lock()

    def fake_extract(**k):
        with lock:
            calls.append((k["video_name"], k.get("device"), k.get("num_workers")))
        return f"/p/{k['video_name']}.npz"

    monkeypatch.setattr(es, "extract_roi_latent_from_video", fake_extract)
    out = es.extract_latent("/tmp", "P", "All", model="dinov3_vitb16", roi=1)
    assert out == "/p/v1.mp4.npz;/p/v2.mp4.npz"  # ordered by video_list
    assert {v for v, _, _ in calls} == {"v1.mp4", "v2.mp4"}
    assert all(str(d).startswith("cuda:") for _, d, _ in calls)
    assert all(isinstance(w, int) and w >= 1 for _, _, w in calls)  # reduced num_workers


def test_extract_latent_sequential_when_flag_off(monkeypatch):
    import castle.core.gpu_pool as gp
    _patch_project(monkeypatch, ["v1.mp4", "v2.mp4"])
    monkeypatch.setattr(gp, "resolve_device_ids", lambda: [])

    auto_calls = []
    monkeypatch.setattr(es, "extract_roi_latent_from_video_auto",
                        lambda **k: (auto_calls.append(k["video_name"]) or f"/p/{k['video_name']}.npz"))
    out = es.extract_latent("/tmp", "P", "All", model="dinov3_vitb16", roi=1)
    assert auto_calls == ["v1.mp4", "v2.mp4"]  # sequential _auto path, in order
    assert out == "/p/v1.mp4.npz;/p/v2.mp4.npz"


def test_extract_latent_single_video_uses_auto_within_video_split(monkeypatch):
    # One video + multi-GPU on -> sequential branch -> _auto (within-video 2gpu split).
    import castle.core.gpu_pool as gp
    _patch_project(monkeypatch, ["only.mp4"])
    monkeypatch.setattr(gp, "resolve_device_ids", lambda: [0, 1])
    auto_calls = []
    monkeypatch.setattr(es, "extract_roi_latent_from_video_auto",
                        lambda **k: (auto_calls.append(k["video_name"]) or "/p/only.npz"))
    out = es.extract_latent("/tmp", "P", "All", model="dinov3_vitb16", roi=1)
    assert auto_calls == ["only.mp4"]
    assert out == "/p/only.npz"


# --------------------------------------------------------------------------
# Remediation: worker-division, pin_memory off, deterministic cuDNN
# --------------------------------------------------------------------------

def test_track_videos_pool_divides_workers_and_disables_pin_memory(monkeypatch, tmp_path):
    from castle.core.environment import get_num_workers
    captured = []

    def fake_track_video(storage, project, video, *, model, start, stop, skip_existing,
                         device=None, num_workers=None, pin_memory=True, cancel_event=None):
        captured.append((num_workers, pin_memory))
        return "Done"

    monkeypatch.setattr(ts, "track_video", fake_track_video)
    ts.track_videos(str(tmp_path), "P", ["a.mp4", "b.mp4"], device_ids=[0, 1], skip_existing=False)
    expected = max(1, get_num_workers("tracking") // 2)
    assert captured, "workers were never invoked"
    assert all(nw == expected for nw, _ in captured), f"expected {expected} workers/GPU, got {captured}"
    assert all(pm is False for _, pm in captured), "multi-GPU tracking must disable pin_memory"


def _recording_ctx(counter):
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        counter["n"] += 1
        yield
    return _ctx


def test_track_videos_default_speed_no_determinism(monkeypatch, tmp_path):
    # Default (speed): multi-GPU must NOT force cuDNN-deterministic.
    import castle.core.gpu_pool as gp
    counter = {"n": 0}
    monkeypatch.delenv("CASTLE_MULTI_GPU_DETERMINISTIC", raising=False)
    monkeypatch.setattr(gp, "cross_gpu_deterministic", _recording_ctx(counter))
    monkeypatch.setattr(ts, "track_video", lambda *a, **k: "Done")
    ts.track_videos(str(tmp_path), "P", ["a.mp4", "b.mp4"], device_ids=[0, 1], skip_existing=False)
    assert counter["n"] == 0, "default multi-GPU must not force determinism (speed)"


def test_track_videos_opt_in_determinism(monkeypatch, tmp_path):
    # CASTLE_MULTI_GPU_DETERMINISTIC=1 -> the pool runs under cross_gpu_deterministic.
    import castle.core.gpu_pool as gp
    counter = {"n": 0}
    monkeypatch.setenv("CASTLE_MULTI_GPU_DETERMINISTIC", "1")
    monkeypatch.setattr(gp, "cross_gpu_deterministic", _recording_ctx(counter))
    monkeypatch.setattr(ts, "track_video", lambda *a, **k: "Done")
    ts.track_videos(str(tmp_path), "P", ["a.mp4", "b.mp4"], device_ids=[0, 1], skip_existing=False)
    assert counter["n"] == 1, "CASTLE_MULTI_GPU_DETERMINISTIC=1 must enter deterministic ctx"


def test_extract_latent_default_speed_no_determinism(monkeypatch):
    import castle.core.gpu_pool as gp
    _patch_project(monkeypatch, ["v1.mp4", "v2.mp4"])
    monkeypatch.setattr(gp, "resolve_device_ids", lambda: [0, 1])
    monkeypatch.delenv("CASTLE_MULTI_GPU_DETERMINISTIC", raising=False)
    counter = {"n": 0}
    monkeypatch.setattr(gp, "cross_gpu_deterministic", _recording_ctx(counter))
    monkeypatch.setattr(es, "extract_roi_latent_from_video", lambda **k: f"/p/{k['video_name']}.npz")
    es.extract_latent("/tmp", "P", "All", model="dinov3_vitb16", roi=1)
    assert counter["n"] == 0, "default multi-GPU extraction must not force determinism (speed)"


def test_extract_latent_opt_in_determinism(monkeypatch):
    import castle.core.gpu_pool as gp
    _patch_project(monkeypatch, ["v1.mp4", "v2.mp4"])
    monkeypatch.setattr(gp, "resolve_device_ids", lambda: [0, 1])
    monkeypatch.setenv("CASTLE_MULTI_GPU_DETERMINISTIC", "1")
    counter = {"n": 0}
    monkeypatch.setattr(gp, "cross_gpu_deterministic", _recording_ctx(counter))
    monkeypatch.setattr(es, "extract_roi_latent_from_video", lambda **k: f"/p/{k['video_name']}.npz")
    es.extract_latent("/tmp", "P", "All", model="dinov3_vitb16", roi=1)
    assert counter["n"] == 1, "CASTLE_MULTI_GPU_DETERMINISTIC=1 must enter deterministic ctx"


# --------------------------------------------------------------------------
# Second-round hardening: device-encoder VRAM leak, cuDNN flag safety,
# split pin_memory, config-lock timeout
# --------------------------------------------------------------------------

import types  # noqa: E402


def _fake_device_encoder():
    return types.SimpleNamespace(model=object())


def test_clear_device_encoder_cache_empties_and_drops_model():
    # C1: cuda:1+ encoders must be evictable (they previously leaked VRAM forever).
    from castle.core import extractor as ex
    enc = _fake_device_encoder()
    ex._device_encoder_cache[("dinov3_vitb16", "cuda:1")] = enc
    ex.clear_device_encoder_cache()
    assert ex._device_encoder_cache == {}
    assert enc.model is None


def test_clear_model_cache_also_clears_device_encoders(monkeypatch):
    # C1: a model switch (models._evict_model_cache) must also free the
    # extractor's separate per-device cache, not just the model singleton.
    from castle.core import extractor as ex
    from castle.core import models as md
    ex._device_encoder_cache[("dinov2_vitb14_reg4_pretrain", "cuda:1")] = _fake_device_encoder()
    md.clear_model_cache()
    assert ex._device_encoder_cache == {}


def test_extract_latent_pool_tears_down_device_encoders(monkeypatch):
    # C1: the multi-GPU batch frees cuda:1 encoders when it finishes.
    import castle.core.gpu_pool as gp
    from castle.core import extractor as ex
    _patch_project(monkeypatch, ["v1.mp4", "v2.mp4"])
    monkeypatch.setattr(gp, "resolve_device_ids", lambda: [0, 1])
    monkeypatch.setattr(es, "extract_roi_latent_from_video", lambda **k: f"/p/{k['video_name']}.npz")
    calls = {"n": 0}
    monkeypatch.setattr(ex, "clear_device_encoder_cache", lambda: calls.__setitem__("n", calls["n"] + 1))
    es.extract_latent("/tmp", "P", "All", model="dinov3_vitb16", roi=1)
    assert calls["n"] == 1, "multi-GPU batch must tear down per-device encoders"


def test_cross_gpu_deterministic_restores_flags_on_exception():
    # C2: an exception inside the block must still restore the prior cuDNN flags
    # (a leaked deterministic=True would silently slow the long-lived process).
    import torch
    from castle.core.gpu_pool import cross_gpu_deterministic
    cudnn = torch.backends.cudnn
    saved = (cudnn.benchmark, cudnn.deterministic)
    try:
        cudnn.benchmark, cudnn.deterministic = True, False
        with pytest.raises(RuntimeError):
            with cross_gpu_deterministic():
                assert cudnn.deterministic is True and cudnn.benchmark is False
                raise RuntimeError("boom")
        assert cudnn.benchmark is True and cudnn.deterministic is False  # restored
    finally:
        cudnn.benchmark, cudnn.deterministic = saved


def test_build_loader_kwargs_pin_memory_param():
    # C3: pin_memory must be tunable (the within-video 2-GPU split passes False to
    # avoid the host-RAM OOM that the tracking path was already fixed for).
    from castle.core.extractor import _build_extractor_loader_kwargs
    assert _build_extractor_loader_kwargs(8, 0)["pin_memory"] is True       # default
    assert _build_extractor_loader_kwargs(8, 0, pin_memory=False)["pin_memory"] is False


def test_config_filelock_timeout_is_generous():
    # C4: a 5s timeout could spuriously fail a successful extraction on slow /
    # cross-process-contended storage; give it real headroom.
    from castle.core import project as pj
    assert pj._CONFIG_FILELOCK_TIMEOUT >= 30.0
