"""Routing tests for single-video multi-GPU tracking (track_video_2gpu).

track_videos must: send 1 video + ≥2 GPUs to the frame-split orchestrator, ≥2
videos to the video-level pool, and 1 GPU (or toggle off) to the sequential path.
The real GPU work is monkeypatched out.
"""

import castle.service.tracking_service as ts


def test_single_video_routes_to_2gpu(monkeypatch):
    seen = {}
    monkeypatch.setattr(ts, "track_video_2gpu",
                        lambda s, p, v, **k: (seen.__setitem__("2gpu", v) or "Done"))
    monkeypatch.setattr(ts, "track_video",
                        lambda s, p, v, **k: (seen.__setitem__("single", v) or "Done"))
    out = ts.track_videos("/s", "P", ["a.mp4"], device_ids=[0, 1], skip_existing=False)
    assert seen.get("2gpu") == "a.mp4" and "single" not in seen
    assert out["a.mp4"] == "Done"


def test_two_videos_use_pool_not_2gpu(monkeypatch):
    import castle.core.gpu_pool as gp
    seen = {}
    monkeypatch.setattr(ts, "track_video_2gpu",
                        lambda *a, **k: (seen.__setitem__("2gpu", True) or "Done"))
    monkeypatch.setattr(gp, "run_on_device_pool",
                        lambda items, worker, dev, **k: ["Done" for _ in items])
    out = ts.track_videos("/s", "P", ["a.mp4", "b.mp4"], device_ids=[0, 1], skip_existing=False)
    assert "2gpu" not in seen
    assert out == {"a.mp4": "Done", "b.mp4": "Done"}


def test_single_gpu_single_video_is_sequential(monkeypatch):
    seen = {}
    monkeypatch.setattr(ts, "track_video_2gpu",
                        lambda *a, **k: (seen.__setitem__("2gpu", True) or "Done"))
    monkeypatch.setattr(ts, "track_video",
                        lambda s, p, v, **k: (seen.__setitem__("single", v) or "Done"))
    out = ts.track_videos("/s", "P", ["a.mp4"], device_ids=[0], skip_existing=False)
    assert "2gpu" not in seen and seen.get("single") == "a.mp4"
    assert out["a.mp4"] == "Done"


def test_single_video_partial_range_not_split(monkeypatch):
    # A non-full range (start/stop) must NOT frame-split (warmup logic assumes [0,N)).
    seen = {}
    monkeypatch.setattr(ts, "track_video_2gpu",
                        lambda *a, **k: (seen.__setitem__("2gpu", True) or "Done"))
    monkeypatch.setattr(ts, "track_video",
                        lambda s, p, v, **k: (seen.__setitem__("single", v) or "Done"))
    ts.track_videos("/s", "P", ["a.mp4"], device_ids=[0, 1], skip_existing=False, start=10, stop=200)
    assert "2gpu" not in seen and seen.get("single") == "a.mp4"
