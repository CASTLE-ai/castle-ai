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
import gradio as gr
import numpy as np

from castle.service.annotator_loader import (
    AnnotatorData,
    load_annotator_data,
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


def _resolve_mask_h5_path(annotator_data: AnnotatorData) -> str | None:
    """Attempt to find the mask_list.h5 for the first video in the project.

    Looks for ``{project_path}/track/{video_name}/mask_list.h5``.

    Args:
        annotator_data: Loaded :class:`AnnotatorData`.

    Returns:
        Absolute path to the H5 file if it exists, else *None*.
    """
    if not annotator_data.videos_meta:
        return None
    _, first_video = annotator_data.videos_meta[0]
    video_name = os.path.basename(first_video)
    mask_path = os.path.join(annotator_data.project_path, "track", video_name, "mask_list.h5")
    if os.path.exists(mask_path):
        return mask_path
    return None


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
        gr.Warning(
            "No project selected. Please create or open a project in the "
            "'Project' tab first."
        )
        return None, gr.update(choices=[], value=None), "**Status:** No project selected"

    sid = session_id if session_id else None
    logger.info("on_load_cluster_data: storage=%s project=%s session_id=%r", storage_path, project_name, sid)

    try:
        annotator_data = load_annotator_data(storage_path, project_name, session_id=sid)
    except FileNotFoundError as exc:
        gr.Warning(
            "Cluster data not found. Please complete the clustering step (Step 3) "
            "and generate at least one session before annotating."
        )
        return None, gr.update(choices=[], value=None), f"**Error:** {exc}"
    except Exception as exc:
        logger.exception("Failed to load cluster data")
        gr.Warning(
            "Failed to load cluster data. Check that the clustering session exists "
            "and try refreshing the session dropdown."
        )
        return None, gr.update(choices=[], value=None), f"**Error:** {exc}"

    # Load annotations scoped to this session
    annotations = load_annotations(storage_path, project_name, session_id=sid)
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

    all_bouts = find_bouts(annotator_data.cluster, cluster_id)
    n_bouts = len(all_bouts)

    output_dir = os.path.join(
        annotator_data.project_path, "cluster", "grid_videos"
    )

    # Resolve mask path for ROI overlay
    mask_h5_path = _resolve_mask_h5_path(annotator_data)

    gr.Info(f"Generating {cols}×{cols} grid video for '{cluster_name}'…")
    video_path = generate_grid_video(
        annotator_data=annotator_data,
        cluster_id=cluster_id,
        grid_cols=cols,
        output_dir=output_dir,
        mask_h5_path=mask_h5_path,
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
    comment,
):
    """Save a single cluster annotation scoped to the loaded session."""
    cluster_name = _strip_check(cluster_choice)
    if not cluster_name or not behavior_label:
        return annotations_state, gr.update()

    # Resolve session_id from AnnotatorData
    session_id = annotator_data.session_id if annotator_data is not None else None

    annotations = dict(annotations_state) if annotations_state else {}
    annotations[cluster_name] = {
        "behavior_label": behavior_label,
        "scheme": scheme_name or "",
        "comment": comment or "",
        "annotator": "user",
        "timestamp": datetime.datetime.now().isoformat(),
    }

    save_annotations(storage_path, project_name, annotations, session_id=session_id)
    gr.Info(f"Saved: {cluster_name} → {behavior_label}")

    choices = _get_cluster_choices(annotator_data, annotations)
    # Keep current selection valid — find the matching choice with potential ✅ prefix
    current_value = None
    for c in choices:
        if _strip_check(c) == cluster_name:
            current_value = c
            break
    return annotations, gr.update(choices=choices, value=current_value)


def on_save_custom_scheme(storage_path, project_name, custom_name, custom_labels_text):
    """Save a custom classification scheme from user input."""
    if not custom_name or not custom_labels_text:
        gr.Info("Please enter a scheme name and at least one label (one per line).")
        return gr.update()

    labels = [line.strip() for line in custom_labels_text.strip().split("\n") if line.strip()]
    if not labels:
        gr.Info("No valid labels found. Add at least one label (one per line) in the Labels field.")
        return gr.update()

    save_scheme(storage_path, project_name, custom_name, labels)
    gr.Info(f"Saved scheme '{custom_name}' with {len(labels)} labels")

    schemes = list_schemes(storage_path, project_name)
    return gr.update(choices=list(schemes.keys()), value=custom_name)


# ---------------------------
# UI Construction
# ---------------------------

def create_annotator_ui(storage_path, project_name, annotator_tab=None):
    """Create the Cluster Annotator tab UI.

    This tab is self-contained and does NOT require shared state from the
    Clustering tab.  Cluster data is loaded from disk via the
    "📂 Load Cluster Data" button.

    Args:
        storage_path: gr.State with storage path.
        project_name: gr.State with project name.
        annotator_tab: gr.Tab reference for auto-loading sessions on tab enter.

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
                    info=(
                        "Number of behavior clips per row/column in the preview grid. "
                        "Default: 3 (3×3 = 9 clips). Larger grids show more variety "
                        "but take longer to generate."
                    ),
                )

                ui["speed_slider"] = gr.Slider(
                    label="Playback Speed",
                    minimum=0.1,
                    maximum=2.0,
                    value=1.0,
                    step=0.1,
                    interactive=True,
                    info=(
                        "Video playback speed multiplier. Default: 1.0× (normal speed). "
                        "Increase to quickly scan through fast behaviors."
                    ),
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

                ui["comment_box"] = gr.Textbox(
                    label="💬 Comment (optional)",
                    placeholder="e.g. mostly grooming with some head movement",
                    lines=2,
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

    # Auto-load session list when entering the tab
    if annotator_tab is not None:
        annotator_tab.select(
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

    # Inputs for cluster-select (no speed — speed is JS-only)
    _video_inputs = [
        storage_path,
        project_name,
        annotator_data,
        ui["cluster_radio"],
        ui["grid_cols"],
    ]
    _video_outputs = [ui["grid_video"], ui["cluster_info"]]

    # Select cluster → generate grid video
    ui["cluster_radio"].change(
        fn=on_cluster_select,
        inputs=_video_inputs,
        outputs=_video_outputs,
    )

    # Grid size change → regenerate
    ui["grid_cols"].change(
        fn=on_cluster_select,
        inputs=_video_inputs,
        outputs=_video_outputs,
    )

    # Speed change → JS-only: set playbackRate on the video element
    ui["speed_slider"].change(
        fn=None,
        inputs=[ui["speed_slider"]],
        outputs=[],
        js="(speed) => { document.querySelectorAll('video').forEach(v => v.playbackRate = speed); }",
    )

    # Change classification scheme → update labels
    ui["scheme_dropdown"].change(
        fn=on_scheme_change,
        inputs=[storage_path, project_name, ui["scheme_dropdown"]],
        outputs=ui["behavior_radio"],
    )

    _save_inputs = [
        storage_path,
        project_name,
        annotator_data,
        annotations_state,
        ui["cluster_radio"],
        ui["behavior_radio"],
        ui["scheme_dropdown"],
        ui["comment_box"],
    ]
    _save_outputs = [annotations_state, ui["cluster_radio"]]

    # Save annotation (button + auto-save on label change / comment blur)
    ui["save_annotation_btn"].click(
        fn=on_save_annotation,
        inputs=_save_inputs,
        outputs=_save_outputs,
    )
    ui["behavior_radio"].change(
        fn=on_save_annotation,
        inputs=_save_inputs,
        outputs=_save_outputs,
    )
    ui["comment_box"].blur(
        fn=on_save_annotation,
        inputs=_save_inputs,
        outputs=_save_outputs,
    )

    # Save custom scheme
    ui["save_scheme_btn"].click(
        fn=on_save_custom_scheme,
        inputs=[storage_path, project_name, ui["custom_scheme_name"], ui["custom_scheme_labels"]],
        outputs=ui["scheme_dropdown"],
    )

    return ui
