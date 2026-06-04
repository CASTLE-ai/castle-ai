"""UI-routing tests for the Pre-process tab (`_run_preprocess` generator).

The service + project helpers are monkeypatched (no real videos/models), so these
verify the generator threads the selection + cancel event correctly and always
yields a final button-reset state — mirroring test_batch_track_ui.py.
"""

import threading

import castle.ui.preprocess_ui as pp


class _FakeReader:
    def __init__(self, n):
        self._n = n
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False
    def __len__(self):
        return self._n


def _drive(gen):
    return list(gen)


def _patch(monkeypatch, sources, capture):
    monkeypatch.setattr(pp, "ReadArray", lambda path: _FakeReader(100))
    monkeypatch.setattr(pp, "_check_crop_bounds", lambda *a, **k: None)

    import castle.core.project as proj
    monkeypatch.setattr(
        proj, "get_project_config",
        lambda sp, pn: ("/proj", {"source": list(sources)}),
    )

    import castle.service.preprocessing_service as svc

    def fake_kit(storage_path, project_name, video_name, kit_params,
                 skip_existing=True, progress_callback=None, cancel_event=None):
        capture.setdefault("videos", []).append(video_name)
        capture["cancel_event"] = cancel_event
        if progress_callback:
            progress_callback(1.0, "done")
        return {"session_id": "abc", "n_frames": 100,
                "diagnostics": {"hp_residual_rms": 1.0, "pct_at_min_crop": 2.0}}

    monkeypatch.setattr(svc, "preprocess_stabilized_camera", fake_kit)


def test_run_preprocess_kit_threads_selection_and_cancel(monkeypatch):
    cap = {}
    _patch(monkeypatch, ["a.mp4", "b.mp4"], cap)
    ev = threading.Event()
    out = _drive(pp._run_preprocess(
        "/store", "P", ["a.mp4", "b.mp4"], "KIT",
        1, 2, 0.25, 2, 75, 300, 592, None, None, None,
        True, ev,
    ))
    assert cap["videos"] == ["a.mp4", "b.mp4"]
    assert cap["cancel_event"] is ev
    # First yield = running (run disabled), last = reset (run enabled, cancel off).
    assert len(out) >= 2
    assert out[0][1]["interactive"] is False
    assert out[-1][1]["interactive"] is True
    assert out[-1][2]["interactive"] is False


def test_run_preprocess_empty_selection_is_graceful(monkeypatch):
    cap = {}
    _patch(monkeypatch, ["a.mp4"], cap)
    out = _drive(pp._run_preprocess(
        "/store", "P", [], "KIT",
        1, 2, 0.25, 2, 75, 300, 592, None, None, None,
        True, None,
    ))
    assert "videos" not in cap                    # service never called
    assert out[-1][1]["interactive"] is True      # buttons still reset


def test_run_preprocess_kit_missing_roi_is_graceful(monkeypatch):
    cap = {}
    _patch(monkeypatch, ["a.mp4"], cap)
    out = _drive(pp._run_preprocess(
        "/store", "P", ["a.mp4"], "KIT",
        None, None, 0.25, 2, 75, 300, 592, None, None, None,
        True, None,
    ))
    assert "videos" not in cap
    assert "ROI" in out[-1][0]
    assert out[-1][1]["interactive"] is True
