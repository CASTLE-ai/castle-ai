"""
castle/ui/annotator_ui.py
Cluster Annotator UI — Stage 4 extended tab (A-04).

Shows GIF previews of behavioral bouts per cluster and allows
behavior labeling with configurable classification schemes.
"""

import os
import datetime
import gradio as gr
import numpy as np

from castle.service.bout_service import extract_cluster_bouts
from castle.service.annotation_service import (
    list_schemes,
    get_scheme_labels,
    save_scheme,
    load_annotations,
    save_annotations,
    DEFAULT_SCHEMES,
)


# ---------------------------
# Helpers
# ---------------------------

def _get_cluster_choices(latents, annotations):
    """Build cluster list with ✅ for labeled ones."""
    if latents is None or not hasattr(latents, 'cluster_meta'):
        return []
    choices = []
    for cid, meta in sorted(latents.cluster_meta.items(), key=lambda x: x[1]['name']):
        name = meta['name']
        if name == 'init':
            continue
        prefix = "✅ " if name in annotations else ""
        choices.append(f"{prefix}{name}")
    return choices


def _strip_check(choice_str):
    """Remove ✅ prefix from a cluster choice string."""
    return choice_str.replace("✅ ", "").strip() if choice_str else ""


def _find_cluster_id_by_name(latents, name):
    """Look up cluster ID from behavior name."""
    if latents is None:
        return None
    return latents.behavior_name2cluster_id.get(name)


# ---------------------------
# Event Handlers
# ---------------------------

def on_cluster_select(storage_path, project_name, latents, aggregator, cluster_choice, grid_cols):
    """When user selects a cluster, generate bout GIFs and return gallery."""
    cluster_name = _strip_check(cluster_choice)
    if not cluster_name or latents is None or aggregator is None:
        return [], f"**Selected:** None"

    cluster_id = _find_cluster_id_by_name(latents, cluster_name)
    if cluster_id is None:
        return [], f"**Error:** Cluster '{cluster_name}' not found"

    # Count bins
    n_bins = int(np.sum(latents.cluster == cluster_id))
    max_bouts = int(grid_cols * grid_cols) if grid_cols else 9

    # Output dir: temp within cluster path
    output_dir = os.path.join(storage_path, project_name, 'cluster', 'bout_gifs', cluster_name)

    gif_paths = extract_cluster_bouts(
        aggregator=aggregator,
        latents=latents,
        cluster_id=cluster_id,
        max_bouts=max_bouts,
        max_frames=60,
        output_dir=output_dir,
        fps=10.0,
    )

    info_text = f"**{cluster_name}** — {n_bins} bins, {len(gif_paths)} bouts"
    return gif_paths, info_text


def on_scheme_change(storage_path, project_name, scheme_name):
    """When classification scheme changes, update the behavior label radio."""
    if not scheme_name:
        return gr.update(choices=[], value=None)
    labels = get_scheme_labels(storage_path, project_name, scheme_name)
    return gr.update(choices=labels, value=None)


def on_save_annotation(
    storage_path, project_name, latents, annotations_state,
    cluster_choice, behavior_label, scheme_name
):
    """Save a single cluster annotation."""
    cluster_name = _strip_check(cluster_choice)
    if not cluster_name or not behavior_label:
        gr.Info("Select a cluster and a behavior label first.")
        return annotations_state, _get_cluster_choices(latents, annotations_state)

    annotations = dict(annotations_state) if annotations_state else {}
    annotations[cluster_name] = {
        'behavior_label': behavior_label,
        'scheme': scheme_name or '',
        'annotator': 'user',
        'timestamp': datetime.datetime.now().isoformat(),
    }

    # Persist to CSV
    save_annotations(storage_path, project_name, annotations)
    gr.Info(f"Saved: {cluster_name} → {behavior_label}")

    return annotations, _get_cluster_choices(latents, annotations)


def on_save_custom_scheme(storage_path, project_name, custom_name, custom_labels_text):
    """Save a custom classification scheme from user input."""
    if not custom_name or not custom_labels_text:
        gr.Info("Enter scheme name and labels (one per line).")
        return gr.update()

    labels = [l.strip() for l in custom_labels_text.strip().split('\n') if l.strip()]
    if not labels:
        gr.Info("No valid labels found.")
        return gr.update()

    save_scheme(storage_path, project_name, custom_name, labels)
    gr.Info(f"Saved scheme '{custom_name}' with {len(labels)} labels")

    # Update the scheme dropdown
    schemes = list_schemes(storage_path, project_name)
    return gr.update(choices=list(schemes.keys()), value=custom_name)


# ---------------------------
# UI Construction
# ---------------------------

def create_annotator_ui(storage_path, project_name, latents_state, mulvideo_state):
    """Create the Cluster Annotator tab UI.

    Args:
        storage_path: gr.State with storage path
        project_name: gr.State with project name
        latents_state: gr.State holding the Latent object
        mulvideo_state: gr.State holding the LatentAggregator

    Returns:
        dict of UI components
    """
    ui = {}

    # Annotations state (dict: cluster_name → annotation info)
    annotations_state = gr.State({})

    with gr.Row():
        # --- Left Column: Controls ---
        with gr.Column(scale=3):
            gr.Markdown("### 📋 Cluster Annotator")

            ui['cluster_radio'] = gr.Radio(
                label="Select Cluster",
                choices=[],
                interactive=True,
            )

            ui['grid_cols'] = gr.Slider(
                label="Grid size (N×N bouts)",
                minimum=1, maximum=5, value=3, step=1,
                interactive=True,
            )

            ui['cluster_info'] = gr.Markdown("**Selected:** None")

            gr.Markdown("---")

            ui['scheme_dropdown'] = gr.Dropdown(
                label="Classification Scheme",
                choices=list(DEFAULT_SCHEMES.keys()),
                value="10-class",
                interactive=True,
            )

            ui['behavior_radio'] = gr.Radio(
                label="🏷️ Behavior Label",
                choices=DEFAULT_SCHEMES["10-class"],
                interactive=True,
            )

            ui['save_annotation_btn'] = gr.Button("💾 Save Annotation", variant="primary")

            gr.Markdown("---")

            with gr.Accordion("Custom Scheme", open=False):
                ui['custom_scheme_name'] = gr.Textbox(
                    label="Scheme name",
                    placeholder="my-custom-scheme",
                )
                ui['custom_scheme_labels'] = gr.Textbox(
                    label="Labels (one per line)",
                    lines=5,
                    placeholder="Running\nWalking\nImmobile\n...",
                )
                ui['save_scheme_btn'] = gr.Button("Save Scheme")

        # --- Right Column: GIF Gallery ---
        with gr.Column(scale=7):
            ui['gallery'] = gr.Gallery(
                label="Bout Previews",
                columns=3,
                rows=3,
                height="auto",
                object_fit="contain",
                interactive=False,
            )

    # ---------------------------
    # Event Bindings
    # ---------------------------

    # Load annotations when tab opens (via latents change)
    def _load_existing_annotations(storage_val, project_val, latents_val):
        if not storage_val or not project_val or latents_val is None:
            return {}, []
        annotations = load_annotations(storage_val, project_val)
        choices = _get_cluster_choices(latents_val, annotations)
        return annotations, gr.update(choices=choices, value=None)

    # Refresh cluster list on scheme dropdown focus (proxy for tab activation)
    ui['scheme_dropdown'].focus(
        fn=_load_existing_annotations,
        inputs=[storage_path, project_name, latents_state],
        outputs=[annotations_state, ui['cluster_radio']],
    )

    # Also refresh when cluster_radio gets focus (if supported)
    if hasattr(ui['cluster_radio'], 'focus'):
        ui['cluster_radio'].focus(
            fn=_load_existing_annotations,
            inputs=[storage_path, project_name, latents_state],
            outputs=[annotations_state, ui['cluster_radio']],
        )

    # Select cluster → generate GIFs
    ui['cluster_radio'].change(
        fn=on_cluster_select,
        inputs=[storage_path, project_name, latents_state, mulvideo_state,
                ui['cluster_radio'], ui['grid_cols']],
        outputs=[ui['gallery'], ui['cluster_info']],
    )

    # Change classification scheme → update labels
    ui['scheme_dropdown'].change(
        fn=on_scheme_change,
        inputs=[storage_path, project_name, ui['scheme_dropdown']],
        outputs=ui['behavior_radio'],
    )

    # Save annotation
    ui['save_annotation_btn'].click(
        fn=on_save_annotation,
        inputs=[storage_path, project_name, latents_state, annotations_state,
                ui['cluster_radio'], ui['behavior_radio'], ui['scheme_dropdown']],
        outputs=[annotations_state, ui['cluster_radio']],
    )

    # Save custom scheme
    ui['save_scheme_btn'].click(
        fn=on_save_custom_scheme,
        inputs=[storage_path, project_name, ui['custom_scheme_name'], ui['custom_scheme_labels']],
        outputs=ui['scheme_dropdown'],
    )

    return ui
