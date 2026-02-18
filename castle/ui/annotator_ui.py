"""
castle/ui/annotator_ui.py
Cluster Annotator UI — Stage 4 extended tab (A-04).

Shows video clip previews of behavioral bouts per cluster and allows
behavior labeling with configurable classification schemes.

This tab is self-contained: it loads cluster data directly from disk via
:func:`castle.service.annotator_loader.load_annotator_data` and does NOT
require the Clustering workflow to have been run in the current session.
"""

import os
import datetime
import logging
import subprocess
import tempfile

import gradio as gr
import numpy as np

from castle.service.annotator_loader import (
    AnnotatorData,
    load_annotator_data,
    get_annotator_frame,
)
from castle.service.bout_service import find_bouts, generate_grid_video
from castle.service.annotation_service import (
    list_schemes,
    get_scheme_labels,
    save_scheme,
    load_annotations,
    save_annotations,
    DEFAULT_SCHEMES,
)

logger = logging.getLogger(__name__)


# ---------------------------
# Helpers
# ---------------------------


def _transcode_to_h264(video_path: str) -> None:
    """Re-encode *video_path* in-place to H.264 using ffmpeg libx264.

    The file is written to a temporary path first, then atomically
    replaces the original so that a partial failure leaves the mp4v
    file intact.

    Args:
        video_path: Path to an MP4 file written with the mp4v codec.
    """
    tmp_path = video_path + ".h264tmp.mp4"
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                video_path,
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "23",
                "-movflags",
                "+faststart",
                tmp_path,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            os.replace(tmp_path, video_path)
        else:
            logger.warning(
                "ffmpeg H.264 transcode failed for %s (keeping mp4v). stderr: %s",
                video_path,
                result.stderr[-300:],
            )
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    except FileNotFoundError:
        logger.warning("ffmpeg not found — keeping mp4v codec for %s", video_path)
    except Exception as exc:
        logger.warning("H.264 transcode error for %s: %s", video_path, exc)
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def _get_cluster_choices(annotator_data, annotations):
    """Build cluster list with ✅ for labeled ones."""
    if annotator_data is None or not hasattr(annotator_data, "cluster_meta"):
        return []
    choices = []
    for cid, meta in sorted(
        annotator_data.cluster_meta.items(), key=lambda x: x[1]["name"]
    ):
        name = meta["name"]
        if name == "init":
            continue
        prefix = "✅ " if name in annotations else ""
        choices.append(f"{prefix}{name}")
    return choices


def _strip_check(choice_str):
    """Remove ✅ prefix from a cluster choice string."""
    return choice_str.replace("✅ ", "").strip() if choice_str else ""


def _find_cluster_id_by_name(annotator_data, name):
    """Look up cluster ID from behavior name."""
    if annotator_data is None:
        return None
    for cid, meta in annotator_data.cluster_meta.items():
        if meta["name"] == name:
            return cid
    return None


def _extract_bouts_standalone(
    annotator_data: AnnotatorData,
    cluster_id: int,
    max_bouts: int = 9,
    max_frames: int = 60,
    output_dir: str = None,
    fps: float = 10.0,
):
    """Extract bout video clips from AnnotatorData without LatentAggregator.

    This mirrors the logic in :func:`castle.service.bout_service.extract_cluster_bouts`
    but uses :func:`get_annotator_frame` instead of an aggregator instance.

    Args:
        annotator_data: Loaded :class:`AnnotatorData`.
        cluster_id: Target cluster ID.
        max_bouts: Maximum number of bouts to extract.
        max_frames: Maximum frames per bout video.
        output_dir: Directory to save videos (tmpdir if None).
        fps: Video playback speed.

    Returns:
        List of MP4 file paths.
    """
    import cv2

    bouts = find_bouts(annotator_data.cluster, cluster_id)
    if not bouts:
        return []

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="castle_bouts_")
    os.makedirs(output_dir, exist_ok=True)

    cluster_name = "unknown"
    if cluster_id in annotator_data.cluster_meta:
        cluster_name = annotator_data.cluster_meta[cluster_id]["name"]

    video_paths = []
    for bout_idx, (start_bin, end_bin) in enumerate(bouts[:max_bouts]):
        bout_len = end_bin - start_bin
        if bout_len > max_frames:
            indices = np.linspace(start_bin, end_bin - 1, max_frames, dtype=int)
        else:
            indices = np.arange(start_bin, end_bin)

        frames = []
        for bin_idx in indices:
            frame = get_annotator_frame(annotator_data, int(bin_idx))
            if frame is not None:
                h, w = frame.shape[:2]
                max_side = 256
                if max(w, h) > max_side:
                    scale = max_side / max(w, h)
                    new_w, new_h = int(w * scale), int(h * scale)
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                frames.append(frame)

        if not frames:
            continue

        video_path = os.path.join(
            output_dir,
            f"bout_{cluster_name}_{bout_idx:02d}_bins{start_bin}-{end_bin}.mp4",
        )

        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
        for frame in frames:
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame
            out.write(frame_bgr)
        out.release()
        _transcode_to_h264(video_path)

        video_paths.append(video_path)
        logger.info("Saved bout video: %s (%d frames)", video_path, len(frames))

    return video_paths


# ---------------------------
# Event Handlers
# ---------------------------

def on_refresh_sessions(storage_path, project_name):
    """Refresh the session dropdown list."""
    if not storage_path or not project_name:
        return gr.update(choices=[], value=None, visible=False)

    from castle.service.session_manager import SessionManager
    mgr = SessionManager(storage_path, project_name)
    sessions = mgr.list_sessions()

    if not sessions:
        return gr.update(choices=[("(no sessions — load from disk)", "")], value="", visible=True)

    choices = [
        (f"{s.name} — {s.n_clusters} clusters, bin_size={s.bin_size} ({s.updated_at[:16]})", s.session_id)
        for s in sessions
    ]
    # Default to active session or latest
    active_id = mgr.get_active_session_id()
    default = active_id if active_id else (sessions[0].session_id if sessions else None)
    return gr.update(choices=choices, value=default, visible=True)


def on_load_cluster_data(storage_path, project_name, session_id):
    """Load cluster data from disk and populate the cluster radio."""
    if not storage_path or not project_name:
        gr.Warning("Select a project first.")
        return None, gr.update(choices=[], value=None), "**Status:** No project selected"

    sid = session_id if session_id else None
    logger.info("on_load_cluster_data: storage=%s project=%s session_id=%r", storage_path, project_name, sid)

    try:
        annotator_data = load_annotator_data(storage_path, project_name, session_id=sid)
    except FileNotFoundError as exc:
        gr.Warning(str(exc))
        return None, gr.update(choices=[], value=None), f"**Error:** {exc}"
    except Exception as exc:
        logger.exception("Failed to load cluster data")
        gr.Warning(f"Failed to load: {exc}")
        return None, gr.update(choices=[], value=None), f"**Error:** {exc}"

    annotations = load_annotations(storage_path, project_name)
    choices = _get_cluster_choices(annotator_data, annotations)
    n_clusters = len(choices)
    n_bins = len(annotator_data.cluster)
    status_msg = (
        f"**Loaded:** {n_clusters} clusters, {n_bins} bins "
        f"(bin_size={annotator_data.bin_size}, fps={annotator_data.fps:.1f})"
    )
    gr.Info(f"Loaded {n_clusters} clusters from {project_name}")
    return annotator_data, gr.update(choices=choices, value=None), status_msg


def on_cluster_select(storage_path, project_name, annotator_data, cluster_choice, grid_cols):
    """When user selects a cluster, generate a grid video and return its path."""
    cluster_name = _strip_check(cluster_choice)
    if not cluster_name or annotator_data is None:
        return None, "**Selected:** None"

    cluster_id = _find_cluster_id_by_name(annotator_data, cluster_name)
    if cluster_id is None:
        return None, f"**Error:** Cluster '{cluster_name}' not found"

    n_bins_in_cluster = int(np.sum(annotator_data.cluster == cluster_id))
    cols = int(grid_cols) if grid_cols else 3

    from castle.service.bout_service import find_bouts as _find_bouts

    all_bouts = _find_bouts(annotator_data.cluster, cluster_id)
    n_bouts = len(all_bouts)

    output_dir = os.path.join(
        annotator_data.project_path, "cluster", "grid_videos"
    )

    gr.Info(f"Generating {cols}×{cols} grid video for '{cluster_name}'…")
    video_path = generate_grid_video(
        annotator_data=annotator_data,
        cluster_id=cluster_id,
        grid_cols=cols,
        output_dir=output_dir,
    )

    info_text = (
        f"**{cluster_name}** — {n_bins_in_cluster} bins, {n_bouts} bouts"
    )
    return video_path, info_text


def on_scheme_change(storage_path, project_name, scheme_name):
    """When classification scheme changes, update the behavior label radio."""
    if not scheme_name:
        return gr.update(choices=[], value=None)
    labels = get_scheme_labels(storage_path, project_name, scheme_name)
    return gr.update(choices=labels, value=None)


def on_save_annotation(
    storage_path,
    project_name,
    annotator_data,
    annotations_state,
    cluster_choice,
    behavior_label,
    scheme_name,
):
    """Save a single cluster annotation."""
    cluster_name = _strip_check(cluster_choice)
    if not cluster_name or not behavior_label:
        gr.Info("Select a cluster and a behavior label first.")
        return annotations_state, _get_cluster_choices(annotator_data, annotations_state)

    annotations = dict(annotations_state) if annotations_state else {}
    annotations[cluster_name] = {
        "behavior_label": behavior_label,
        "scheme": scheme_name or "",
        "annotator": "user",
        "timestamp": datetime.datetime.now().isoformat(),
    }

    save_annotations(storage_path, project_name, annotations)
    gr.Info(f"Saved: {cluster_name} → {behavior_label}")

    return annotations, _get_cluster_choices(annotator_data, annotations)


def on_save_custom_scheme(storage_path, project_name, custom_name, custom_labels_text):
    """Save a custom classification scheme from user input."""
    if not custom_name or not custom_labels_text:
        gr.Info("Enter scheme name and labels (one per line).")
        return gr.update()

    labels = [line.strip() for line in custom_labels_text.strip().split("\n") if line.strip()]
    if not labels:
        gr.Info("No valid labels found.")
        return gr.update()

    save_scheme(storage_path, project_name, custom_name, labels)
    gr.Info(f"Saved scheme '{custom_name}' with {len(labels)} labels")

    schemes = list_schemes(storage_path, project_name)
    return gr.update(choices=list(schemes.keys()), value=custom_name)


# ---------------------------
# UI Construction
# ---------------------------

def create_annotator_ui(storage_path, project_name):
    """Create the Cluster Annotator tab UI.

    This tab is self-contained and does NOT require shared state from the
    Clustering tab.  Cluster data is loaded from disk via the
    "📂 Load Cluster Data" button.

    Args:
        storage_path: gr.State with storage path.
        project_name: gr.State with project name.

    Returns:
        dict of UI components.
    """
    ui = {}

    # Per-tab state
    annotator_data = gr.State(None)
    annotations_state = gr.State({})

    with gr.Column():
        # --- Load controls ---
        gr.Markdown("### 📋 Cluster Annotator")
        with gr.Row():
            ui["session_dropdown"] = gr.Dropdown(
                label="Select Session",
                choices=[],
                interactive=True,
                scale=3,
            )
            ui["refresh_btn"] = gr.Button("🔄", scale=0, min_width=50)
            ui["load_btn"] = gr.Button("📂 Load Cluster Data", variant="primary", scale=1)

        ui["load_status"] = gr.Markdown("**Status:** Not loaded")

        gr.Markdown("---")

        with gr.Row():
            # --- Left Column: Controls ---
            with gr.Column(scale=3):
                ui["cluster_radio"] = gr.Radio(
                    label="Select Cluster",
                    choices=[],
                    interactive=True,
                )

                ui["grid_cols"] = gr.Slider(
                    label="Grid size (N×N bouts)",
                    minimum=1,
                    maximum=5,
                    value=3,
                    step=1,
                    interactive=True,
                )

                ui["cluster_info"] = gr.Markdown("**Selected:** None")

                gr.Markdown("---")

                ui["scheme_dropdown"] = gr.Dropdown(
                    label="Classification Scheme",
                    choices=list(DEFAULT_SCHEMES.keys()),
                    value="10-class",
                    interactive=True,
                )

                ui["behavior_radio"] = gr.Radio(
                    label="🏷️ Behavior Label",
                    choices=DEFAULT_SCHEMES["10-class"],
                    interactive=True,
                )

                ui["save_annotation_btn"] = gr.Button("💾 Save Annotation", variant="primary")

                gr.Markdown("---")

                with gr.Accordion("Custom Scheme", open=False):
                    ui["custom_scheme_name"] = gr.Textbox(
                        label="Scheme name",
                        placeholder="my-custom-scheme",
                    )
                    ui["custom_scheme_labels"] = gr.Textbox(
                        label="Labels (one per line)",
                        lines=5,
                        placeholder="Running\nWalking\nImmobile\n...",
                    )
                    ui["save_scheme_btn"] = gr.Button("Save Scheme")

            # --- Right Column: Grid Video ---
            with gr.Column(scale=7):
                ui["grid_video"] = gr.Video(
                    label="Grid Video — Most Representative Bouts",
                    autoplay=True,
                    loop=True,
                    interactive=False,
                )

    # ---------------------------
    # Event Bindings
    # ---------------------------

    # Refresh session list
    ui["refresh_btn"].click(
        fn=on_refresh_sessions,
        inputs=[storage_path, project_name],
        outputs=[ui["session_dropdown"]],
    )

    # Auto-refresh sessions on Load: refresh dropdown first, then load data
    ui["load_btn"].click(
        fn=on_refresh_sessions,
        inputs=[storage_path, project_name],
        outputs=[ui["session_dropdown"]],
    ).then(
        fn=on_load_cluster_data,
        inputs=[storage_path, project_name, ui["session_dropdown"]],
        outputs=[annotator_data, ui["cluster_radio"], ui["load_status"]],
    )

    # Select cluster → generate grid video
    ui["cluster_radio"].change(
        fn=on_cluster_select,
        inputs=[
            storage_path,
            project_name,
            annotator_data,
            ui["cluster_radio"],
            ui["grid_cols"],
        ],
        outputs=[ui["grid_video"], ui["cluster_info"]],
    )

    # Change classification scheme → update labels
    ui["scheme_dropdown"].change(
        fn=on_scheme_change,
        inputs=[storage_path, project_name, ui["scheme_dropdown"]],
        outputs=ui["behavior_radio"],
    )

    # Save annotation
    ui["save_annotation_btn"].click(
        fn=on_save_annotation,
        inputs=[
            storage_path,
            project_name,
            annotator_data,
            annotations_state,
            ui["cluster_radio"],
            ui["behavior_radio"],
            ui["scheme_dropdown"],
        ],
        outputs=[annotations_state, ui["cluster_radio"]],
    )

    # Save custom scheme
    ui["save_scheme_btn"].click(
        fn=on_save_custom_scheme,
        inputs=[storage_path, project_name, ui["custom_scheme_name"], ui["custom_scheme_labels"]],
        outputs=ui["scheme_dropdown"],
    )

    return ui
