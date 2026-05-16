"""
castle/ui/cluster_page_ui.py
UI Layer for Clustering — layout construction and event binding only.

Heavy logic has been extracted to:
  - castle.ui.embedding_scatter   (EmbeddingScatterPlot)
  - castle.ui.cluster_handlers    (event handler functions)
  - castle.ui.cluster_tree        (build_cluster_tree_html, build_cluster_tree_choices)
"""

import logging
import re
import json

import gradio as gr

# Import from split modules
# Cluster tree functions imported by cluster_handlers
from castle.ui.cluster_handlers import (
    embedding_plot_click,
    collapse_accordion,
    update_select_cluster_list,
    generate_embedding,
    generate_local_cluster,
    label_all_and_submit,
    on_tree_node_select,
    restore_session,
    init_mulvideo,
    handle_undo,
    handle_redo,
    update_history_buttons,
    check_session_exists,
    save_cluster_model,
    apply_cluster_model,
    export_representatives,
)

logger = logging.getLogger(__name__)

# ---------------------------
# Templates & Presets (UI Config)
# ---------------------------

umap_config_template = '''[
    {
        "n_neighbors": 100,
        "min_dist": 0.0,
        "n_components": 2,
        "n_epochs": 5000
    }
]'''

umap_config_low_magnification_template = '''[
    {
        "n_neighbors": 30,
        "min_dist": 0.0,
        "n_components": 2,
        "n_epochs": 5000
    }
]'''

umap_config_intermediate_magnification_template = '''[
    {
        "n_neighbors": 30,
        "min_dist": 0.0,
        "n_components": 5,
        "n_epochs": 5000
    },
    {
        "n_neighbors": 30,
        "min_dist": 0.0,
        "n_components": 2,
        "n_epochs": 5000
    }
]'''

umap_config_high_magnification_template = '''[
    {
        "n_neighbors": 30,
        "min_dist": 0.0,
        "n_components": 10,
        "n_epochs": 5000
    },
    {
        "n_neighbors": 30,
        "min_dist": 0.0,
        "n_components": 2,
        "n_epochs": 5000
    }
]'''

preset_dropdown_list = ['Low-magnification objective 1000', 'Low-magnification objective 500', 'Low-magnification objective 300', 'Low-magnification objective 100', 'Low-magnification objective 50', 'Low-magnification objective 25']
preset_dropdown_list += ['Intermediate-magnification objective (1000, 500)', 'Intermediate-magnification objective (500, 300)', 'Intermediate-magnification objective (300, 100)', 'Intermediate-magnification objective (100, 50)', 'Intermediate-magnification objective (50, 25)']
preset_dropdown_list += ['High-magnification objective (1000, 500)', 'High-magnification objective (500, 300)', 'High-magnification objective (300, 100)', 'High-magnification objective (100, 50)', 'High-magnification objective (50, 25)']
preset_dropdown_list += ['Super-high-magnification objective (500, 300, 100)', 'Super-high-magnification objective (300, 100, 50)', 'Super-high-magnification objective (100, 50, 25)']


# ---------------------------
# Helpers
# ---------------------------

def _format_session_status(info):
    """Returns (restore_btn_update, session_status_update, session_dropdown_update)."""
    if info is None or not isinstance(info, dict):
        return (
            gr.update(interactive=False),
            gr.update(value="No previous sessions found. Use **⚙️ New Session** to start.", visible=True),
            gr.update(choices=[], visible=False),
        )
    
    try:
        sessions = info.get('sessions', [])
        if not sessions:
            return (
                gr.update(interactive=False),
                gr.update(value="No previous sessions found. Use **⚙️ New Session** to start.", visible=True),
                gr.update(choices=[], visible=False),
            )
        
        choices = [(f"{s.name} — {s.n_clusters} clusters ({s.updated_at[:16]})", s.session_id) for s in sessions[:10]]
        latest_id = sessions[0].session_id
        
        return (
            gr.update(interactive=True),
            gr.update(value=f"**{len(sessions)} session(s) found.** Select one to restore.", visible=True),
            gr.update(choices=choices, value=latest_id, visible=True),
        )
    except Exception:
        logger.exception("Error formatting session status")
        return (
            gr.update(interactive=False),
            gr.update(value="Error loading sessions.", visible=True),
            gr.update(choices=[], visible=False),
        )

def update_umap_config_text_with_preset(preset_dropdown):
    """根據使用者選擇的預設來生成對應的 UMAP 配置字串並調整 n_neighbors 數值"""
    if preset_dropdown is None:
        return umap_config_template
    
    if 'Low-magnification objective' in preset_dropdown:
        numbers = re.findall(r'\d+', preset_dropdown)
        if numbers:
            n_neighbors = int(numbers[0])
            config = [{
                "n_neighbors": n_neighbors,
                "min_dist": 0.0,
                "n_components": 2,
                "n_epochs": 5000
            }]
            return json.dumps(config, indent=4)
    
    elif 'Intermediate-magnification objective' in preset_dropdown:
        numbers = re.findall(r'\d+', preset_dropdown)
        if len(numbers) >= 2:
            n_neighbors_1 = int(numbers[0])
            n_neighbors_2 = int(numbers[1])
            config = [
                {"n_neighbors": n_neighbors_1, "min_dist": 0.0, "n_components": 5, "n_epochs": 5000},
                {"n_neighbors": n_neighbors_2, "min_dist": 0.0, "n_components": 2, "n_epochs": 5000}
            ]
            return json.dumps(config, indent=4)
    
    elif 'Super-high-magnification objective' in preset_dropdown:
        numbers = re.findall(r'\d+', preset_dropdown)
        if len(numbers) >= 3:
            n_neighbors_1 = int(numbers[0])
            n_neighbors_2 = int(numbers[1])
            n_neighbors_3 = int(numbers[2])
            config = [
                {"n_neighbors": n_neighbors_1, "min_dist": 0.0, "n_components": 15, "n_epochs": 5000},
                {"n_neighbors": n_neighbors_2, "min_dist": 0.0, "n_components": 5, "n_epochs": 5000},
                {"n_neighbors": n_neighbors_3, "min_dist": 0.0, "n_components": 2, "n_epochs": 5000}
            ]
            return json.dumps(config, indent=4)

    elif 'High-magnification objective' in preset_dropdown:
        numbers = re.findall(r'\d+', preset_dropdown)
        if len(numbers) >= 2:
            n_neighbors_1 = int(numbers[0])
            n_neighbors_2 = int(numbers[1])
            config = [
                {"n_neighbors": n_neighbors_1, "min_dist": 0.0, "n_components": 10, "n_epochs": 5000},
                {"n_neighbors": n_neighbors_2, "min_dist": 0.0, "n_components": 2, "n_epochs": 5000}
            ]
            return json.dumps(config, indent=4)
    
    return umap_config_template


# ---------------------------
# UI Construction
# ---------------------------

def create_cluster_page_ui(storage_path, project_name, cluster_page_tab):
    ui = dict()
    
    # Section 1: Previous Sessions
    with gr.Accordion("📂 Previous Sessions", open=True) as ui['previous_sessions_accordion']:
        ui['session_status'] = gr.Markdown("*Checking for previous sessions...*")
        ui['session_dropdown'] = gr.Dropdown(label="Select Session", choices=[], interactive=True, visible=False)
        ui['restore_btn'] = gr.Button("Restore Previous Session", interactive=False, visible=True)
    
    # Section 2: New Session
    with gr.Accordion("⚙️ New Session", open=False) as ui['cluster_input_accordion']:
        ui['select_model'] = gr.Dropdown(
            label="Select Visual Model",
            choices=["dinov2_vitb14_reg4_pretrain", "dinov3_vitb16", "dinov3_vitl16"],
            value="dinov3_vitb16",
            interactive=True,
            visible=True
        )
        ui['select_roi_id'] = gr.Textbox(
            label="Enter ROI ID",
            value="1",
            info=(
                "ROI (Region of Interest) ID(s) for clustering. "
                "Use the same ROI as used during feature extraction (Step 3). "
                "Example: 1, or 1,2,3 for multiple ROIs."
            ),
            visible=True,
        )
        ui['bin_size'] = gr.Number(
            label='Time window (frame)',
            value=1,
            interactive=True,
            visible=True,
            info=(
                "Number of frames aggregated into one behavior bin. Default: 1. "
                "Larger values create smoother but less temporally precise analysis. "
                "Use the same value for all sessions in a project."
            ),
        )
        ui['reset'] = gr.Button("Initialize", interactive=True, visible=True)
        
    # State Holders
    latents = gr.State(None)
    local_latents = gr.State(None)
    local_embedding_plot = gr.State(None)
    mulvideo = gr.State(None)  # Holds LatentAggregator instance
    session_info = gr.State(None)
    history_state = gr.State(None)  # HistoryManager for undo/redo
    overwrite_state = gr.State(False)  # Submit-overwrite confirmation gate

    with gr.Row(visible=True) as ui['cluster_row_main']:
        with gr.Column(scale=2):
            ui['cluster_tree_html'] = gr.HTML(
                value="<em style='color:#888;font-size:12px'>No clusters yet.</em>",
                label="Cluster Tree",
            )
            # Hidden textbox — JS onclick on tree nodes writes here via native
            # value setter + input/change event dispatch (castleTreeClick in
            # main_ui.py).  We use CSS to hide rather than visible=False
            # because hidden Gradio components occasionally swallow
            # synthetic .change() / .input() events, which breaks the
            # auto-restore handler bound below.
            ui['cluster_tree_select'] = gr.Textbox(
                value="",
                interactive=True,
                elem_id="castle-tree-select",
                elem_classes=["castle-tree-select-hidden"],
                show_label=False,
                container=False,
            )
            gr.HTML(
                "<style>.castle-tree-select-hidden{position:absolute!important;"
                "left:-9999px!important;width:1px!important;height:1px!important;"
                "overflow:hidden!important;}</style>"
            )
            ui['preset_dropdown'] = gr.Dropdown(preset_dropdown_list, value='Low-magnification objective 100', label="UMAP preset", visible=True, interactive=True)
            ui['umap_config_text'] = gr.Textbox(label='UMAP configs', value=umap_config_template, lines=8, max_lines=8, interactive=True, visible=True)
            with gr.Row():
                ui['umap_seed'] = gr.Textbox(
                    label='UMAP seed',
                    value='',
                    placeholder='Empty = re-roll',
                    interactive=True,
                    scale=4,
                    info=(
                        "Leave blank to draw a fresh seed each run. Paste a seed "
                        "from a previous status line to lock the layout."
                    ),
                )
                ui['umap_reroll'] = gr.Button("🎲 Re-roll", scale=1, variant="secondary")
            ui['umap_run'] = gr.Button("Generate Embedding", interactive=True, visible=True)
            ui['umap_seed_status'] = gr.Markdown(value="", visible=True)
            ui['eps'] = gr.Number(
                label='epsilon-neighborhood radius',
                interactive=True,
                visible=True,
                value=1,
                step=0.1,
                minimum=0.1,
                maximum=10,
                info=(
                    "DBSCAN neighborhood radius. Larger values = fewer, bigger clusters; "
                    "smaller values = more, finer-grained clusters. Default: 1.0. "
                    "Adjust based on the density of the embedding scatter plot."
                ),
            )
            ui['cluster_run'] = gr.Button("Generate Cluster", interactive=True, visible=True)
            ui['enter_submit_all_btn'] = gr.Button("Submit", interactive=True, visible=True, variant="primary")
            ui['submit_status'] = gr.Markdown("", visible=True)
            with gr.Row():
                ui['undo_btn'] = gr.Button("↩️ Undo", interactive=False)
                ui['redo_btn'] = gr.Button("↪️ Redo", interactive=False)
            ui['history_info'] = gr.Textbox(label="History", interactive=False, max_lines=1)
        with gr.Column(scale=8):
            ui['embedding_plot'] = gr.Image(label='Embedding', interactive=False, visible=True)
            ui['display'] = gr.Video(label='Display', interactive=False, visible=True, autoplay=True, loop=True)  
            ui['display_eps'] = gr.File(label="Display EPS", interactive=False, visible=True)
            
    ui['syllables_plot'] = gr.Plot(label='Syllable', visible=True)

    # ---- HITL warning banner ----
    gr.Markdown(
        "⚠️ **Cluster labels are only meaningful after human validation.** "
        "Before exporting or training downstream models: "
        "(1) inspect representative frames for each cluster, "
        "(2) verify cluster boundaries by adjusting `eps`, and "
        "(3) assign behaviorally meaningful labels. "
        "CASTLE intentionally provides no \"one-click cluster\" entry point."
    )

    # ---- Cluster representatives export (UX-02) ----
    with gr.Accordion("🖼️ Export Cluster Representatives", open=False):
        gr.Markdown(
            "Save N representative frames per labelled cluster — useful as "
            "paper figures or as the 'face validity' check before submitting."
        )
        with gr.Row():
            ui['representatives_n'] = gr.Number(
                label="Frames per cluster", value=9, minimum=1, maximum=64, step=1,
            )
            ui['representatives_selection'] = gr.Dropdown(
                label="Selection",
                choices=["medoid", "random"],
                value="medoid",
                info="medoid = closest to cluster centroid (most representative); random = uniform sample.",
            )
        ui['representatives_btn'] = gr.Button(
            "🖼️ Export Representatives", variant="secondary",
        )
        ui['representatives_file'] = gr.File(
            label="⬇️ Download (.zip)", visible=False, interactive=False,
        )
        ui['representatives_status'] = gr.Markdown("")

    # ---- Save / Apply Cluster Model Section ----
    with gr.Accordion("💾 Save / Apply Cluster Model", open=False) as ui['model_transfer_accordion']:
        gr.Markdown(
            "Transfer a trained cluster model to another project. "
            "Equivalent to `castle cluster save-model` / `castle cluster apply-model`."
        )
        with gr.Row():
            ui['save_model_btn'] = gr.Button("💾 Save Model", variant="secondary")
            ui['save_model_file'] = gr.File(
                label="⬇️ Download Model", visible=False, interactive=False
            )
        ui['save_model_status'] = gr.Markdown("")
        gr.Markdown("---")
        ui['apply_model_file'] = gr.File(
            label="📂 Upload Cluster Model (.npz)", file_types=[".npz"], interactive=True
        )
        ui['apply_model_btn'] = gr.Button("📂 Apply Model", variant="secondary")
        ui['apply_model_status'] = gr.Markdown("")

    with gr.Row(visible=True) as ui['cluster_row_files']:
        with gr.Column(scale=2):
            ui['behavior_id_csv'] = gr.File(label="Behavior ID", interactive=False, visible=True)
        with gr.Column(scale=2):
            ui['behavior_time_series_csv'] = gr.File(label="Behavior time series", interactive=False, visible=True, file_count="multiple")
        with gr.Column(scale=2):
            ui['behavior_time_series_srt'] = gr.File(label="Behavior time series (SRT)", interactive=False, visible=True)

    # --- Event Bindings ---

    # Auto-detect previous session when entering the tab
    cluster_page_tab.select(
        fn=lambda sp, pn: check_session_exists(sp, pn),
        inputs=[storage_path, project_name],
        outputs=[session_info]
    ).then(
        fn=_format_session_status,
        inputs=[session_info],
        outputs=[ui['restore_btn'], ui['session_status'], ui['session_dropdown']]
    )

    # Initialize: create aggregator + check for previous session
    ui['reset'].click(
        fn=init_mulvideo,
        inputs=[storage_path, project_name, ui['select_roi_id'], ui['bin_size'], ui['select_model']],
        outputs=[mulvideo, latents, session_info]
    ).then(
        fn=_format_session_status,
        inputs=[session_info],
        outputs=[ui['restore_btn'], ui['session_status'], ui['session_dropdown']]
    ).then(
        fn=update_select_cluster_list,
        inputs=latents,
        outputs=[ui['cluster_tree_html'], ui['cluster_tree_select']],
    ).then(
        fn=collapse_accordion,
        outputs=ui['cluster_input_accordion']
    )

    # Restore previous session (B-03: also restores UMAP embedding)
    ui['restore_btn'].click(
        fn=restore_session,
        inputs=[storage_path, project_name, ui['select_roi_id'], ui['bin_size'], ui['select_model'], ui['session_dropdown']],
        outputs=[mulvideo, latents, ui['syllables_plot'],
                 ui['cluster_tree_html'], ui['cluster_tree_select'],
                 ui['behavior_id_csv'], ui['behavior_time_series_csv'],
                 local_embedding_plot, ui['embedding_plot']],
    ).then(
        fn=lambda: (gr.update(visible=False), gr.update(value="Session restored successfully."), gr.update(visible=False)),
        outputs=[ui['restore_btn'], ui['session_status'], ui['session_dropdown']]
    )

    ui['preset_dropdown'].select(
        fn=update_umap_config_text_with_preset,
        inputs=ui['preset_dropdown'],
        outputs=ui['umap_config_text']
    )
    ui['umap_run'].click(
        fn=generate_embedding,
        inputs=[
            latents,
            ui['cluster_tree_select'],
            ui['umap_config_text'],
            ui['umap_seed'],
            storage_path,
            project_name,
        ],
        outputs=[
            local_latents, local_embedding_plot,
            ui['embedding_plot'], ui['umap_seed_status'],
        ],
    )

    # 🎲 Re-roll: clear the seed textbox so the next run draws a fresh seed.
    ui['umap_reroll'].click(
        fn=lambda: ("", ""),
        outputs=[ui['umap_seed'], ui['umap_seed_status']],
    )

    ui['embedding_plot'].select(
        fn=embedding_plot_click,
        inputs=[mulvideo, local_embedding_plot],
        outputs=[ui['embedding_plot'], ui['display']]
    )
    ui['cluster_run'].click(
        fn=generate_local_cluster,
        inputs=[local_latents, ui['eps'], history_state],
        outputs=[local_embedding_plot, ui['embedding_plot'], history_state],
    ).then(
        fn=update_history_buttons,
        inputs=[history_state],
        outputs=[ui['undo_btn'], ui['redo_btn'], ui['history_info']],
    )
    # Submit: auto-label all clusters and persist (with overwrite confirmation)
    ui['enter_submit_all_btn'].click(
        fn=label_all_and_submit,
        inputs=[storage_path, project_name, latents, local_latents, mulvideo,
                ui['cluster_tree_select'], history_state,
                ui['umap_config_text'], ui['eps'], overwrite_state],
        outputs=[ui['syllables_plot'],
                 ui['cluster_tree_html'], ui['cluster_tree_select'],
                 ui['behavior_id_csv'], ui['behavior_time_series_csv'],
                 ui['behavior_time_series_srt'], local_embedding_plot,
                 ui['embedding_plot'], ui['display_eps'],
                 history_state, overwrite_state, ui['submit_status']],
    ).then(
        fn=update_history_buttons,
        inputs=[history_state],
        outputs=[ui['undo_btn'], ui['redo_btn'], ui['history_info']],
    )

    # Auto-restore prior UMAP/eps when a tree node is clicked.
    # NB: use .change() not .input() — .input() requires a "real" user-typed
    # event and does not fire reliably for hidden textboxes whose value is
    # mutated via JS native setter + dispatchEvent.
    ui['cluster_tree_select'].change(
        fn=on_tree_node_select,
        inputs=[ui['cluster_tree_select'], latents, storage_path, project_name],
        outputs=[ui['umap_config_text'], ui['eps'], ui['embedding_plot'],
                 local_latents, overwrite_state, ui['submit_status']],
    )

    # Undo / Redo
    ui['undo_btn'].click(
        fn=handle_undo,
        inputs=[local_latents, latents, history_state],
        outputs=[local_embedding_plot, ui['embedding_plot'], history_state, ui['history_info'],
                 ui['cluster_tree_html'], ui['cluster_tree_select']],
    ).then(
        fn=update_history_buttons,
        inputs=[history_state],
        outputs=[ui['undo_btn'], ui['redo_btn'], ui['history_info']],
    )

    ui['redo_btn'].click(
        fn=handle_redo,
        inputs=[local_latents, latents, history_state],
        outputs=[local_embedding_plot, ui['embedding_plot'], history_state, ui['history_info'],
                 ui['cluster_tree_html'], ui['cluster_tree_select']],
    ).then(
        fn=update_history_buttons,
        inputs=[history_state],
        outputs=[ui['undo_btn'], ui['redo_btn'], ui['history_info']],
    )

    def _auto_update_tree(sp, pn, lat):
        if check_session_exists(sp, pn) is not None:
            return update_select_cluster_list(lat)
        return gr.update(), gr.update()

    cluster_page_tab.select(
        fn=_auto_update_tree,
        inputs=[storage_path, project_name, latents],
        outputs=[ui['cluster_tree_html'], ui['cluster_tree_select']],
    )

    # Save Cluster Model
    ui['save_model_btn'].click(
        fn=save_cluster_model,
        inputs=[storage_path, project_name],
        outputs=[ui['save_model_file'], ui['save_model_status']],
    )

    # Apply Cluster Model
    ui['apply_model_btn'].click(
        fn=apply_cluster_model,
        inputs=[storage_path, project_name, ui['apply_model_file']],
        outputs=[ui['apply_model_status']],
    )

    # UX-02: Export Cluster Representatives
    ui['representatives_btn'].click(
        fn=export_representatives,
        inputs=[
            storage_path, project_name, latents, mulvideo,
            ui['representatives_n'], ui['representatives_selection'],
        ],
        outputs=[ui['representatives_file'], ui['representatives_status']],
    )

    # Expose shared state for Annotator tab (A-04)
    shared_states = {
        'latents': latents,
        'mulvideo': mulvideo,
    }

    visibility_components = {
        'previous_sessions_accordion': ui['previous_sessions_accordion'],
        'cluster_input_accordion': ui['cluster_input_accordion'],
        'cluster_row_main': ui['cluster_row_main'],
        'syllables_plot': ui['syllables_plot'],
        'cluster_row_files': ui['cluster_row_files'],
    }

    return visibility_components, shared_states
