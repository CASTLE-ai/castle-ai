"""UI-routing tests for the Extract Latent tab (`ui_extract_roi_latent` generator).

The extractors + GPU pool are monkeypatched (no real GPUs/models), so these verify
the multi-GPU branching (video-level pool / single-video frame-split / sequential),
batch-granular cancel, and that the generator always yields a final button reset.
"""

import contextlib
import threading

import castle.ui.extract_ui as eu


class _FakeReader:
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False
    def __len__(self):
        return 100


def _patch_common(monkeypatch, sources):
    monkeypatch.setattr(eu, "get_project_config", lambda sp, pn: (None, {"source": list(sources)}))
    monkeypatch.setattr(eu, "ReadArray", lambda p: _FakeReader())
    monkeypatch.setattr(eu, "host_ram_available_bytes", lambda: 10 ** 12)
    monkeypatch.setattr(eu, "get_num_workers", lambda t: 8)
    monkeypatch.setattr(eu, "deterministic_ctx_if_enabled", lambda: contextlib.nullcontext())
    monkeypatch.setattr(eu, "clear_device_encoder_cache", lambda: None)


def _drive(**over):
    args = dict(
        storage_path="/store", project_name="P", select_model="dinov3_vitb16",
        select_roi=1, selected_videos=over.pop("videos", ["a.mp4", "b.mp4"]),
        batch_size="8", skip_existing=False, remove_background_switch=False,
        era_switch=False, era_roi_id=2, pooling_method="weighted_average",
        pooling_scales_list=["1"], feature_layers_str="", latent_dtype="float32",
        use_multi_gpu=over.pop("use_multi_gpu", False),
        session_display="(None — use raw source)",
        cancel_event=over.pop("cancel_event", None),
    )
    return list(eu.ui_extract_roi_latent(**args))


def test_sequential_single_gpu(monkeypatch):
    _patch_common(monkeypatch, ["a.mp4", "b.mp4"])
    monkeypatch.setattr(eu, "available_cuda_devices", lambda: [0, 1])
    seen = []
    monkeypatch.setattr(eu, "extract_roi_latent_from_video_auto",
                        lambda **kw: (seen.append(kw["video_name"]) or f"/lat/{kw['video_name']}.npz"))
    out = _drive(use_multi_gpu=False)
    assert seen == ["a.mp4", "b.mp4"]            # one-at-a-time, _auto path
    assert out[0][1]["interactive"] is False     # first yield: Extract disabled
    assert out[-1][1]["interactive"] is True     # final yield: Extract re-enabled
    assert out[-1][2]["interactive"] is False    # Cancel disabled


def test_multi_gpu_video_pool(monkeypatch):
    _patch_common(monkeypatch, ["a.mp4", "b.mp4", "c.mp4"])
    monkeypatch.setattr(eu, "available_cuda_devices", lambda: [0, 1])
    cap = {}

    def fake_pool(items, worker, device_ids, on_done=None, cancel_event=None):
        cap["device_ids"] = list(device_ids)
        out = []
        for i, it in enumerate(items):
            r = worker(it, f"cuda:{device_ids[i % len(device_ids)]}")
            out.append(r)
            if on_done:
                on_done(it, r)
        return out

    monkeypatch.setattr(eu, "run_on_device_pool", fake_pool)
    devs = []
    monkeypatch.setattr(eu, "extract_roi_latent_from_video",
                        lambda **kw: (devs.append(kw["device"]) or f"/lat/{kw['video_name']}.npz"))
    out = _drive(videos=["a.mp4", "b.mp4", "c.mp4"], use_multi_gpu=True)
    assert cap["device_ids"] == [0, 1]           # both GPUs used
    assert set(devs) == {"cuda:0", "cuda:1"}     # videos spread across devices
    assert out[-1][1]["interactive"] is True


def test_multi_gpu_single_video_framesplit(monkeypatch):
    _patch_common(monkeypatch, ["solo.mp4"])
    monkeypatch.setattr(eu, "available_cuda_devices", lambda: [0, 1])
    cap = {}
    monkeypatch.setattr(eu, "extract_roi_latent_from_video_2gpu",
                        lambda **kw: (cap.update(device_ids=list(kw["device_ids"])) or "/lat/solo.npz"))
    out = _drive(videos=["solo.mp4"], use_multi_gpu=True)
    assert cap["device_ids"] == [0, 1]           # frame-split across both GPUs
    assert out[-1][1]["interactive"] is True


def test_cancel_before_first_video(monkeypatch):
    _patch_common(monkeypatch, ["a.mp4", "b.mp4"])
    monkeypatch.setattr(eu, "available_cuda_devices", lambda: [])
    called = []
    monkeypatch.setattr(eu, "extract_roi_latent_from_video_auto",
                        lambda **kw: (called.append(kw["video_name"]) or "/x.npz"))
    ev = threading.Event(); ev.set()             # cancelled before launch
    out = _drive(use_multi_gpu=False, cancel_event=ev)
    assert called == []                          # no video extracted
    assert "🛑 Cancelled." in out[-1][3]
    assert out[-1][1]["interactive"] is True


def test_empty_selection_graceful(monkeypatch):
    _patch_common(monkeypatch, ["a.mp4"])
    monkeypatch.setattr(eu, "available_cuda_devices", lambda: [])
    out = _drive(videos=[], use_multi_gpu=False)
    assert out[-1][1]["interactive"] is True      # buttons reset even on no-op
