"""Tests for the shared progress helpers (castle/ui/progress_ui.py)."""

import threading
import time

from castle.ui import progress_ui as p


def test_fmt_dur_minutes_and_hours():
    assert p.fmt_dur(0) == "0:00"
    assert p.fmt_dur(65) == "1:05"
    assert p.fmt_dur(3725) == "1h02m"   # hours-aware
    assert p.fmt_dur(-5) == "0:00"      # clamps negatives


def test_status_md_frame_granular_bar():
    md = p.status_md(frames_done=50, total_frames=100, vids_done=1,
                     vids_total=4, t0=time.time(), cancelling=False)
    assert "50.0%" in md
    assert "50 / 100" in md
    assert "1/4** videos" in md
    assert "█" in md and "░" in md


def test_status_md_eta_warmup_gate():
    # Tiny fraction + just-started → ETA withheld (avoids absurd extrapolation).
    md = p.status_md(frames_done=1, total_frames=100000, vids_done=0,
                     vids_total=1, t0=time.time(), cancelling=False)
    assert "estimating…" in md
    # Past the elapsed gate → a real ETA appears.
    md2 = p.status_md(frames_done=1, total_frames=100000, vids_done=0,
                      vids_total=1, t0=time.time() - 30, cancelling=False)
    assert "estimating…" not in md2 and "ETA ~" in md2


def test_status_md_cancelling_prefix():
    md = p.status_md(10, 100, 0, 2, time.time(), cancelling=True)
    assert md.startswith("🛑 Cancelling…")


def test_request_cancel_sets_event_and_relabels():
    ev = threading.Event()
    upd = p.request_cancel(ev)
    assert ev.is_set()
    assert upd["interactive"] is False
    assert "Canceling" in upd["value"]


def test_init_cancel_event_is_fresh():
    a, b = p.init_cancel_event(), p.init_cancel_event()
    assert isinstance(a, threading.Event) and not a.is_set()
    assert a is not b
