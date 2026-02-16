"""
castle/ui/cluster_page_ui.py
UI Layer for Clustering — layout construction and event binding only.

Heavy logic has been extracted to:
  - castle.ui.embedding_scatter   (EmbeddingScatterPlot)
  - castle.ui.cluster_handlers    (event handler functions)
  - castle.ui.cluster_tree        (build_cluster_tree_markdown)
"""

import re
import json

import gradio as gr

from castle.core.cluster import auto_generate_cluster_name

# Import from split modules
# Cluster tree functions imported by cluster_handlers
from castle.ui.cluster_handlers import (
    embedding_plot_click,
    collapse_accordion,
    update_select_cluster_list,
    generate_embedding,
    generate_local_cluster,
    label_local_cluster,
    label_all_and_submit,
    restore_session,
    import_info_from_local_latent,
    init_mulvideo,
    handle_undo,
    handle_redo,
    update_history_buttons,
    check_session_exists,
)


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
    
    with gr.Accordion('Input setting', visible=False) as ui['cluster_input_accordion']:
            ui['select_model'] = gr.Dropdown(
                label="Select Visual Model",
                choices=["dinov2_vitb14_reg4_pretrain", "dinov3_vitb16", "dinov3_vitl16"],
                value="dinov3_vitb16",
                interactive=True,
                visible=True
            )
            ui['select_roi_id'] = gr.Textbox(label="Enter ROI ID", value="1", info="ex: 1,2,3.", visible=True)
            ui['bin_size'] = gr.Number(label='Time window (frame)', value=1, interactive=True, visible=True)
            ui['reset'] = gr.Button("Initialize", interactive=True, visible=True)
            ui['restore_btn'] = gr.Button("Restore Previous Session", interactive=False, visible=True)
            ui['session_status'] = gr.Markdown("", visible=False)
        
    # State Holders
    latents = gr.State(None)
    local_latents = gr.State(None)
    local_embedding_plot = gr.State(None)
    mulvideo = gr.State(None)  # Holds LatentAggregator instance
    session_info = gr.State(None)
    history_state = gr.State(None)  # HistoryManager for undo/redo

    with gr.Row(visible=True) as ui['cluster_row_main']:
        with gr.Column(scale=2):
            ui['cluster_tree_radio'] = gr.Radio(label="Cluster Tree", choices=[], interactive=True, visible=True)
            ui['preset_dropdown'] = gr.Dropdown(preset_dropdown_list, value='Low-magnification objective 100', label="UMAP preset", visible=True, interactive=True)
            ui['umap_config_text'] = gr.Textbox(label='UMAP configs', value=umap_config_template, lines=8, max_lines=8, interactive=True, visible=True)
            ui['umap_run'] = gr.Button("Generate Embedding", interactive=True, visible=True)
            ui['eps'] = gr.Number(label='epsilon-neighborhood radius', interactive=True, visible=True, value=1, step=0.1, minimum=0.1, maximum=10)
            ui['cluster_run'] = gr.Button("Generate Cluster", interactive=True, visible=True)
            ui['label_cluster_id'] = gr.Number(label='Cluster id', interactive=True, visible=True)
            ui['label_cluster_name'] = gr.Textbox(label='Cluster name', interactive=True, visible=True)
            ui['label_cluster_btn'] = gr.Button("Enter", interactive=True, visible=True)
            ui['label_cluster_submit_btn'] = gr.Button("Submit", interactive=True, visible=True)
            ui['enter_submit_all_btn'] = gr.Button("Enter & Submit all", interactive=True, visible=True)
            with gr.Row():
                ui['undo_btn'] = gr.Button("↩️ Undo", interactive=False)
                ui['redo_btn'] = gr.Button("↪️ Redo", interactive=False)
            ui['history_info'] = gr.Textbox(label="History", interactive=False, max_lines=1)
        with gr.Column(scale=8):
            ui['embedding_plot'] = gr.Image(label='Embedding', interactive=False, visible=True)
            ui['display'] = gr.Video(label='Display', interactive=False, visible=True, autoplay=True, loop=True)  
            ui['display_eps'] = gr.File(label="Display EPS", interactive=False, visible=True)
            
    ui['syllables_plot'] = gr.Plot(label='Syllable', visible=True)
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
        fn=lambda info: (
            gr.update(interactive=info is not None),
            gr.update(value=f"**Previous session found:** {info['cluster_count']} clusters", visible=info is not None) if info else (gr.update(interactive=False), gr.update(visible=False))
        ),
        inputs=[session_info],
        outputs=[ui['restore_btn'], ui['session_status']]
    )

    # Initialize: create aggregator + check for previous session
    ui['reset'].click(
        fn=init_mulvideo,
        inputs=[storage_path, project_name, ui['select_roi_id'], ui['bin_size'], ui['select_model']],
        outputs=[mulvideo, latents, session_info]
    ).then(
        fn=lambda info: (
            gr.update(interactive=info is not None),
            gr.update(value=f"**Previous session found:** {info['cluster_count']} clusters", visible=info is not None) if info else gr.update(visible=False)
        ),
        inputs=[session_info],
        outputs=[ui['restore_btn'], ui['session_status']]
    ).then(
        fn=update_select_cluster_list,
        inputs=latents,
        outputs=ui['cluster_tree_radio']
    )

    # Restore previous session (B-03: also restores UMAP embedding)
    ui['restore_btn'].click(
        fn=restore_session,
        inputs=[storage_path, project_name, ui['select_roi_id'], ui['bin_size'], ui['select_model']],
        outputs=[mulvideo, latents, ui['syllables_plot'], ui['cluster_tree_radio'],
                 ui['behavior_id_csv'], ui['behavior_time_series_csv'],
                 local_embedding_plot, ui['embedding_plot']]
    ).then(
        fn=lambda: (gr.update(visible=False), gr.update(visible=False)),
        outputs=[ui['restore_btn'], ui['session_status']]
    )

    ui['cluster_tree_radio'].select(
        fn=update_select_cluster_list,
        inputs=latents,
        outputs=ui['cluster_tree_radio']
    )

    ui['preset_dropdown'].select(
        fn=update_umap_config_text_with_preset,
        inputs=ui['preset_dropdown'],
        outputs=ui['umap_config_text']
    )
    ui['umap_run'].click(
        fn=generate_embedding,
        inputs=[latents, ui['cluster_tree_radio'], ui['umap_config_text']],
        outputs=[local_latents, local_embedding_plot, ui['embedding_plot']]
    )

    ui['reset'].click(
        fn=collapse_accordion,
        outputs=ui['cluster_input_accordion']
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
    ui['label_cluster_btn'].click(
        fn=label_local_cluster,
        inputs=[local_latents, ui['label_cluster_id'], ui['label_cluster_name'], history_state],
        outputs=[history_state],
    ).then(
        fn=update_history_buttons,
        inputs=[history_state],
        outputs=[ui['undo_btn'], ui['redo_btn'], ui['history_info']],
    )
    
    # Auto-generate cluster name when ID changes
    ui['label_cluster_id'].change(
        fn=auto_generate_cluster_name,
        inputs=[ui['cluster_tree_radio'], ui['label_cluster_id']],
        outputs=ui['label_cluster_name']
    )

    ui['label_cluster_submit_btn'].click(
        fn=import_info_from_local_latent,
        inputs=[storage_path, project_name, latents, local_latents, mulvideo],
        outputs=[ui['syllables_plot'], ui['cluster_tree_radio'], ui['behavior_id_csv'], ui['behavior_time_series_csv'], ui['behavior_time_series_srt'], local_embedding_plot, ui['embedding_plot'], ui['display_eps']],
    )

    # Enter & Submit all: auto-label all clusters and submit
    ui['enter_submit_all_btn'].click(
        fn=label_all_and_submit,
        inputs=[storage_path, project_name, latents, local_latents, mulvideo, ui['cluster_tree_radio'], history_state],
        outputs=[ui['syllables_plot'], ui['cluster_tree_radio'], ui['behavior_id_csv'], ui['behavior_time_series_csv'], ui['behavior_time_series_srt'], local_embedding_plot, ui['embedding_plot'], ui['display_eps'], history_state],
    ).then(
        fn=update_history_buttons,
        inputs=[history_state],
        outputs=[ui['undo_btn'], ui['redo_btn'], ui['history_info']],
    )

    # Undo / Redo
    ui['undo_btn'].click(
        fn=handle_undo,
        inputs=[local_latents, latents, history_state],
        outputs=[local_embedding_plot, ui['embedding_plot'], history_state, ui['history_info'], ui['cluster_tree_radio']],
    ).then(
        fn=update_history_buttons,
        inputs=[history_state],
        outputs=[ui['undo_btn'], ui['redo_btn'], ui['history_info']],
    )

    ui['redo_btn'].click(
        fn=handle_redo,
        inputs=[local_latents, latents, history_state],
        outputs=[local_embedding_plot, ui['embedding_plot'], history_state, ui['history_info'], ui['cluster_tree_radio']],
    ).then(
        fn=update_history_buttons,
        inputs=[history_state],
        outputs=[ui['undo_btn'], ui['redo_btn'], ui['history_info']],
    )

    # Auto-update cluster list when tab is selected
    cluster_page_tab.select(
        fn=update_select_cluster_list,
        inputs=latents,
        outputs=ui['cluster_tree_radio']
    )

    # Expose shared state for Annotator tab (A-04)
    shared_states = {
        'latents': latents,
        'mulvideo': mulvideo,
    }

    visibility_components = {
        'cluster_input_accordion': ui['cluster_input_accordion'],
        'cluster_row_main': ui['cluster_row_main'],
        'syllables_plot': ui['syllables_plot'],
        'cluster_row_files': ui['cluster_row_files'],
    }

    return visibility_components, shared_states
