"""Shared progress-bar / ETA / cancel helpers for the batch tabs.

Both Batch Tracking and Pre-process render their own unicode progress bar in a
dedicated ``gr.Markdown`` (so it never overlaps the log textbox), run work on a
background thread, and poll a ``threading.Event`` for cancellation. This module
holds the common pieces so the two tabs stay byte-identical in behaviour.
"""

import threading
import time

import gradio as gr

PROGRESS_BAR_WIDTH = 24
# ETA is meaningless when extrapolated from a tiny fraction (e.g. 0.03% → "55h"),
# so withhold it until there's a real sample: ≥2% done OR ≥20s elapsed.
ETA_MIN_FRAC = 0.02
ETA_MIN_ELAPSED = 20.0


def init_cancel_event() -> threading.Event:
    """Fresh per-run cancel flag, stored in gr.State before the work generator runs."""
    return threading.Event()


def request_cancel(cancel_event, label: str = "Canceling (stopping current video)…"):
    """Cancel handler: set the flag and immediately relabel the button.

    Immediate feedback. The in-flight video aborts within ~one frame-batch and
    no new videos launch; the work generator's final yield restores the idle
    label/state.
    """
    if cancel_event is not None:
        cancel_event.set()
    return gr.update(value=label, interactive=False)


def fmt_dur(seconds: float) -> str:
    """h/m/s duration, hours-aware (so a long run reads '2h05m', not '125:00')."""
    sec = int(max(0, seconds))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    return f"{m}:{s:02d}"


def status_md(frames_done: float, total_frames: int, vids_done: int,
              vids_total: int, t0: float, cancelling: bool) -> str:
    """Markdown for the dedicated status component: a unicode progress bar +
    ``frames / videos / elapsed / ETA``. Rendered in its own box (not the log
    textbox) so it never overlaps the log. The bar advances per frame-batch.

    ETA is withheld until enough progress accrues (see ``ETA_MIN_*``) — early
    extrapolation from ~0% produces absurd numbers — and is hours-aware.
    """
    elapsed = time.time() - t0
    if total_frames > 0:
        frac = min(1.0, frames_done / total_frames)
        lead = f"**{int(frames_done):,} / {total_frames:,}** frames · "
    else:  # frame counts unavailable → video-granular fallback
        frac = min(1.0, vids_done / vids_total) if vids_total else 0.0
        lead = ""

    if frac > 0 and (frac >= ETA_MIN_FRAC or elapsed >= ETA_MIN_ELAPSED):
        eta_str = "~" + fmt_dur(elapsed * (1 - frac) / frac)
    else:
        eta_str = "estimating…"

    filled = int(round(frac * PROGRESS_BAR_WIDTH))
    bar = "█" * filled + "░" * (PROGRESS_BAR_WIDTH - filled)
    prefix = "🛑 Cancelling… " if cancelling else ""
    return (f"{prefix}{lead}**{vids_done}/{vids_total}** videos · "
            f"elapsed {fmt_dur(elapsed)} · ETA {eta_str}\n\n"
            f"`{bar}` {frac * 100:.1f}%")
