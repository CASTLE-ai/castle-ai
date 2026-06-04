"""UI-routing tests for the Batch Tracking tab (`track_all_videos` generator).

No real GPUs/models — `track_videos` and `get_project_videos` are monkeypatched,
so these verify that the UI threads the multi-GPU toggle, skip-existing checkbox
and cancel event into the service correctly, and that the generator always yields
a final button-reset state.
"""

import threading

import castle.ui.batch_track_ui as bt


def _drive(gen):
    """Exhaust the generator, returning the list of yielded output tuples."""
    return list(gen)


def _patch(monkeypatch, videos, capture):
    monkeypatch.setattr(bt, "get_project_videos", lambda sp, pn: list(videos))

    def fake_track_videos(storage, project, video_names, **kwargs):
        capture["video_names"] = list(video_names)
        capture["kwargs"] = kwargs
        return {v: "Done" for v in video_names}

    monkeypatch.setattr(bt, "track_videos", fake_track_videos)


def test_track_all_videos_multigpu_toggle_threads_device_ids(monkeypatch, tmp_path):
    cap = {}
    _patch(monkeypatch, ["a.mp4", "b.mp4"], cap)
    monkeypatch.setattr(bt, "available_cuda_devices", lambda: [0, 1])

    out = _drive(bt.track_all_videos(
        str(tmp_path), "P", "r50_deaotl",
        skip_existing=True, use_multi_gpu=True, cancel_event=threading.Event(),
        selected_videos=["a.mp4", "b.mp4"],
    ))

    assert cap["kwargs"]["device_ids"] == [0, 1]  # toggle ON -> both GPUs
    # First yield = running state, last yield = reset state (always present).
    assert len(out) >= 2
    assert out[-1][1]["interactive"] is True   # Start re-enabled
    assert out[-1][2]["interactive"] is False  # Cancel disabled


def test_track_all_videos_multigpu_off_passes_empty_device_ids(monkeypatch, tmp_path):
    cap = {}
    _patch(monkeypatch, ["a.mp4", "b.mp4"], cap)
    # available_cuda_devices would report 2, but the toggle is OFF -> single-GPU.
    monkeypatch.setattr(bt, "available_cuda_devices", lambda: [0, 1])

    _drive(bt.track_all_videos(
        str(tmp_path), "P", "r50_deaotl",
        skip_existing=True, use_multi_gpu=False, cancel_event=None,
        selected_videos=["a.mp4", "b.mp4"],
    ))
    assert not cap["kwargs"]["device_ids"]  # falsy -> sequential single-GPU


def test_track_all_videos_skip_existing_preflight(monkeypatch, tmp_path):
    # b.mp4 already tracked -> excluded from the list handed to track_videos.
    mask = tmp_path / "P" / "track" / "b.mp4" / "mask_list.h5"
    mask.parent.mkdir(parents=True)
    mask.write_bytes(b"x")
    cap = {}
    _patch(monkeypatch, ["a.mp4", "b.mp4"], cap)
    monkeypatch.setattr(bt, "available_cuda_devices", lambda: [])

    _drive(bt.track_all_videos(
        str(tmp_path), "P", "r50_deaotl",
        skip_existing=True, use_multi_gpu=False, cancel_event=None,
        selected_videos=["a.mp4", "b.mp4"],
    ))
    assert cap["video_names"] == ["a.mp4"]  # existing one filtered out
    # Pre-flight already applied the skip, so the service is told skip_existing=False.
    assert cap["kwargs"]["skip_existing"] is False


def test_track_all_videos_skip_existing_off_retracks_all(monkeypatch, tmp_path):
    mask = tmp_path / "P" / "track" / "b.mp4" / "mask_list.h5"
    mask.parent.mkdir(parents=True)
    mask.write_bytes(b"x")
    cap = {}
    _patch(monkeypatch, ["a.mp4", "b.mp4"], cap)
    monkeypatch.setattr(bt, "available_cuda_devices", lambda: [])

    _drive(bt.track_all_videos(
        str(tmp_path), "P", "r50_deaotl",
        skip_existing=False, use_multi_gpu=False, cancel_event=None,
        selected_videos=["a.mp4", "b.mp4"],
    ))
    assert cap["video_names"] == ["a.mp4", "b.mp4"]  # nothing skipped


def test_track_all_videos_threads_cancel_event(monkeypatch, tmp_path):
    cap = {}
    _patch(monkeypatch, ["a.mp4"], cap)
    monkeypatch.setattr(bt, "available_cuda_devices", lambda: [])
    ev = threading.Event()

    _drive(bt.track_all_videos(
        str(tmp_path), "P", "r50_deaotl",
        skip_existing=True, use_multi_gpu=False, cancel_event=ev,
        selected_videos=["a.mp4"],
    ))
    assert cap["kwargs"]["cancel_event"] is ev


def test_track_all_videos_crash_still_resets_buttons(monkeypatch, tmp_path):
    # A crash in track_videos must NOT leave the UI stuck: the generator's
    # catch-all + final yield must still reset Start/Cancel.
    monkeypatch.setattr(bt, "get_project_videos", lambda sp, pn: ["a.mp4"])
    monkeypatch.setattr(bt, "available_cuda_devices", lambda: [])

    def boom(*a, **k):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(bt, "track_videos", boom)
    out = _drive(bt.track_all_videos(
        str(tmp_path), "P", "r50_deaotl",
        skip_existing=True, use_multi_gpu=False, cancel_event=None,
        selected_videos=["a.mp4"],
    ))
    assert out[-1][1]["interactive"] is True    # Start re-enabled after crash
    assert out[-1][2]["interactive"] is False   # Cancel disabled
    assert "crashed" in out[-1][0]              # error surfaced in the log


def test_track_all_videos_mix_toggle_gates_analysis(monkeypatch, tmp_path):
    # When generate_mix=False / generate_csv=False, _on_video_done must call
    # generate_video_analysis with those flags (so mix/CSV are skipped).
    monkeypatch.setattr(bt, "available_cuda_devices", lambda: [])
    monkeypatch.setattr(bt, "get_project_videos", lambda sp, pn: ["a.mp4"])

    captured = {}

    def fake_analysis(storage, project, video, generate_csv=True, generate_mix=True, cancel_event=None):
        captured["csv"] = generate_csv
        captured["mix"] = generate_mix
        return ("", "")

    monkeypatch.setattr(bt, "generate_video_analysis", fake_analysis)

    def fake_track_videos(storage, project, video_names, **kwargs):
        # Drive the on_video_done callback so the analysis branch runs.
        cb = kwargs.get("on_video_done")
        for v in video_names:
            if cb:
                cb(v, "Done")
        return {v: "Done" for v in video_names}

    monkeypatch.setattr(bt, "track_videos", fake_track_videos)

    # CSV on, mix off → generate_video_analysis called with those exact flags.
    list(bt.track_all_videos(
        str(tmp_path), "P", "r50_deaotl",
        skip_existing=False, use_multi_gpu=False,
        generate_csv=True, generate_mix=False,
        cancel_event=None, selected_videos=["a.mp4"],
    ))
    assert captured == {"csv": True, "mix": False}

    # Both off → analysis is skipped entirely (generate_video_analysis NOT called).
    captured.clear()
    list(bt.track_all_videos(
        str(tmp_path), "P", "r50_deaotl",
        skip_existing=False, use_multi_gpu=False,
        generate_csv=False, generate_mix=False,
        cancel_event=None, selected_videos=["a.mp4"],
    ))
    assert captured == {}


def test_track_all_videos_empty_selection_is_graceful(monkeypatch, tmp_path):
    # No videos checked → don't call track_videos; surface a clear message.
    cap = {}
    _patch(monkeypatch, ["a.mp4", "b.mp4"], cap)
    monkeypatch.setattr(bt, "available_cuda_devices", lambda: [])
    out = _drive(bt.track_all_videos(
        str(tmp_path), "P", "r50_deaotl",
        skip_existing=True, use_multi_gpu=False, cancel_event=None,
        selected_videos=[],
    ))
    assert "kwargs" not in cap                  # track_videos never called
    assert "No videos selected" in out[-1][0]
    assert out[-1][1]["interactive"] is True    # buttons still reset


def test_request_cancel_sets_event_and_relabels():
    ev = threading.Event()
    upd = bt._request_cancel(ev)
    assert ev.is_set()
    assert upd["interactive"] is False
    assert "Canceling" in upd["value"]


def test_init_cancel_event_is_fresh_event():
    a = bt._init_cancel_event()
    b = bt._init_cancel_event()
    assert isinstance(a, threading.Event) and not a.is_set()
    assert a is not b


def test_available_cuda_devices_threshold(monkeypatch):
    import castle.core.gpu_pool as gp
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    assert gp.available_cuda_devices() == [0, 1]

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    assert gp.available_cuda_devices() == []  # single GPU -> no multi-GPU

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert gp.available_cuda_devices() == []
