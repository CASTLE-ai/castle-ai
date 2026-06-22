"""Unit tests for castle.ui.annotator_ui event handlers.

These handlers are pure functions returning ``gr.update`` dicts, so they can be
exercised without a running Gradio app by monkeypatching the few side-effecting
collaborators (video generation, toasts, CSV writes).

Regression focus: selecting an already-annotated cluster must switch the
Behavior Label radio to the stored label (previously it was always reset to
None, so the user could not tell how a cluster had been labeled).
"""

import types

import numpy as np
import pytest

from castle.ui import annotator_ui


@pytest.fixture
def fake_data():
    return types.SimpleNamespace(
        cluster_meta={
            0: {"name": "init"},
            1: {"name": "cluster_1"},
            2: {"name": "cluster_2"},
        },
        cluster=np.array([1, 1, 2, 2, 2]),
        project_path="/tmp/proj",
        session_id="sess1",
        bin_size=1,
        fps=30.0,
    )


@pytest.fixture(autouse=True)
def _silence_toasts(monkeypatch):
    monkeypatch.setattr(annotator_ui.gr, "Info", lambda *a, **k: None)
    monkeypatch.setattr(annotator_ui.gr, "Warning", lambda *a, **k: None)


@pytest.fixture
def _stub_video(monkeypatch):
    monkeypatch.setattr(annotator_ui, "generate_grid_video", lambda **kw: "/tmp/grid.mp4")
    monkeypatch.setattr(annotator_ui, "find_bouts", lambda arr, cid: [(0, 1)])


def test_cluster_select_restores_saved_label(fake_data, _stub_video):
    """An annotated cluster restores its stored label + comment into the UI."""
    annotations = {
        "cluster_1": {"behavior_label": "Running", "scheme": "mice-10-class", "comment": "fast"}
    }
    name, _video, info, radio_upd, comment_upd = annotator_ui.on_cluster_select(
        "/sp", "proj", fake_data, annotations, "✅ cluster_1", 3
    )
    assert name == "cluster_1"
    assert radio_upd["value"] == "Running"
    assert comment_upd["value"] == "fast"
    assert "Running" in info  # label echoed in the info line too


def test_cluster_select_unlabeled_resets_to_none(fake_data, _stub_video):
    """An unlabeled cluster clears the radio (None) so auto-save still fires."""
    _name, _video, _info, radio_upd, comment_upd = annotator_ui.on_cluster_select(
        "/sp", "proj", fake_data, {}, "cluster_2", 3
    )
    assert radio_upd["value"] is None
    assert comment_upd["value"] == ""


def test_cluster_select_handles_nan_label(fake_data, _stub_video):
    """A NaN/empty stored label (hand-edited CSV) degrades to None, not 'nan'."""
    annotations = {"cluster_1": {"behavior_label": float("nan"), "comment": float("nan")}}
    _name, _video, _info, radio_upd, comment_upd = annotator_ui.on_cluster_select(
        "/sp", "proj", fake_data, annotations, "cluster_1", 3
    )
    assert radio_upd["value"] is None
    assert comment_upd["value"] == ""


def test_save_annotation_skips_redundant(fake_data, monkeypatch):
    """Restoring a label fires .change(); an unchanged annotation must not re-save."""
    called = {}
    monkeypatch.setattr(
        annotator_ui, "save_annotations", lambda *a, **k: called.setdefault("hit", True)
    )
    annotations = {
        "cluster_1": {
            "behavior_label": "Running",
            "scheme": "mice-10-class",
            "comment": "fast",
            "annotator": "user",
            "timestamp": "t",
        }
    }
    state, _radio_upd = annotator_ui.on_save_annotation(
        "/sp", "proj", fake_data, annotations, "cluster_1", "Running", "mice-10-class", "fast"
    )
    assert "hit" not in called
    assert state is annotations


def test_save_annotation_writes_on_real_change(fake_data, monkeypatch):
    """A genuine label change is persisted."""
    calls = {}
    monkeypatch.setattr(
        annotator_ui,
        "save_annotations",
        lambda sp, pn, ann, session_id=None: calls.update(ann=ann),
    )
    annotations = {
        "cluster_1": {"behavior_label": "Running", "scheme": "mice-10-class", "comment": "fast"}
    }
    _state, _radio_upd = annotator_ui.on_save_annotation(
        "/sp", "proj", fake_data, annotations, "cluster_1", "Walking", "mice-10-class", "fast"
    )
    assert calls["ann"]["cluster_1"]["behavior_label"] == "Walking"
