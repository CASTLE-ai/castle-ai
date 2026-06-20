"""
castle/ui/cluster_page_ui.py
UI Layer for Clustering — layout construction and event binding only.

Heavy logic has been extracted to:
  - castle.visualization.embedding_scatter (EmbeddingScatterPlot)
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
    check_session_exists,
    show_delete_confirmation,
    confirm_delete_session,
    show_delete_cache_confirmation,
    confirm_delete_cache,
    save_cluster_model,
    apply_cluster_model,
    export_representatives,
    list_latent_choices,
    build_prepare_handler,
    list_prepare_choices,
)
from castle.core.config import get_color_mode, set_color_mode
from castle.ui.progress_ui import init_cancel_event, request_cancel
from castle.ui.video_select import build_video_selector, wire_video_selector

logger = logging.getLogger(__name__)

# Sentinel dropdown label for the legacy (no prepared cache) raw-latent path.
LEGACY_SOURCE_LABEL = "(legacy raw — no cache)"

# ---------------------------
# Templates & Presets (UI Config)
# ---------------------------

umap_config_template = '''[
    {
        "n_neighbors": 100,
        "min_dist": 0.0,
        "n_components": 2,
        "n_epochs": 500
    }
]'''

umap_config_low_magnification_template = '''[
    {
        "n_neighbors": 30,
        "min_dist": 0.0,
        "n_components": 2,
        "n_epochs": 500
    }
]'''

umap_config_intermediate_magnification_template = '''[
    {
        "n_neighbors": 30,
        "min_dist": 0.0,
        "n_components": 5,
        "n_epochs": 500
    },
    {
        "n_neighbors": 30,
        "min_dist": 0.0,
        "n_components": 2,
        "n_epochs": 500
    }
]'''

umap_config_high_magnification_template = '''[
    {
        "n_neighbors": 30,
        "min_dist": 0.0,
        "n_components": 10,
        "n_epochs": 500
    },
    {
        "n_neighbors": 30,
        "min_dist": 0.0,
        "n_components": 2,
        "n_epochs": 500
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
                "n_epochs": 500
            }]
            return json.dumps(config, indent=4)
    
    elif 'Intermediate-magnification objective' in preset_dropdown:
        numbers = re.findall(r'\d+', preset_dropdown)
        if len(numbers) >= 2:
            n_neighbors_1 = int(numbers[0])
            n_neighbors_2 = int(numbers[1])
            config = [
                {"n_neighbors": n_neighbors_1, "min_dist": 0.0, "n_components": 5, "n_epochs": 500},
                {"n_neighbors": n_neighbors_2, "min_dist": 0.0, "n_components": 2, "n_epochs": 500}
            ]
            return json.dumps(config, indent=4)
    
    elif 'Super-high-magnification objective' in preset_dropdown:
        numbers = re.findall(r'\d+', preset_dropdown)
        if len(numbers) >= 3:
            n_neighbors_1 = int(numbers[0])
            n_neighbors_2 = int(numbers[1])
            n_neighbors_3 = int(numbers[2])
            config = [
                {"n_neighbors": n_neighbors_1, "min_dist": 0.0, "n_components": 15, "n_epochs": 500},
                {"n_neighbors": n_neighbors_2, "min_dist": 0.0, "n_components": 5, "n_epochs": 500},
                {"n_neighbors": n_neighbors_3, "min_dist": 0.0, "n_components": 2, "n_epochs": 500}
            ]
            return json.dumps(config, indent=4)

    elif 'High-magnification objective' in preset_dropdown:
        numbers = re.findall(r'\d+', preset_dropdown)
        if len(numbers) >= 2:
            n_neighbors_1 = int(numbers[0])
            n_neighbors_2 = int(numbers[1])
            config = [
                {"n_neighbors": n_neighbors_1, "min_dist": 0.0, "n_components": 10, "n_epochs": 500},
                {"n_neighbors": n_neighbors_2, "min_dist": 0.0, "n_components": 2, "n_epochs": 500}
            ]
            return json.dumps(config, indent=4)
    
    return umap_config_template


# ---------------------------
# UI Construction
# ---------------------------

def _on_init_click():
    """Leading handler for the Initialize button — emits a top-right toast.

    Fires a ``gr.Info`` toast the instant the button is clicked (so the feedback
    is unmissable, top-right) AND returns the inline session_status loading line.
    A plain lambda can only return an update; a named handler can do both. The
    inline line + gr.Progress bridge the gap while latents load.
    """
    gr.Info(
        "⏳ Initializing Explore session — loading latents… "
        "(this can take a while for a large prepared cache)"
    )
    return gr.update(value="⏳ Initializing…")


def create_cluster_page_ui(storage_path, project_name, cluster_page_tab):
    ui = dict()
    
    with gr.Tabs():
        with gr.Tab("Prepare"):
            gr.Markdown(
                "**Prepare** — build a reduced latent cache once (downsample → "
                "per-sample L2 → PCA), then explore it many times. Each settings "
                "combo is a separate cache; changing settings makes a new one."
            )
            ui['prep_model'] = gr.Dropdown(
                label="Model",
                choices=["dinov2_vitb14_reg4_pretrain", "dinov3_vitb16", "dinov3_vitl16"],
                value="dinov3_vitb16", interactive=True,
            )
            ui['prep_files'] = build_video_selector(
                label="Latent files (mice) to include", visible=True,
            )
            with gr.Row():
                ui['prep_downsample'] = gr.Checkbox(value=True, label="Downsample")
                ui['prep_target_fps'] = gr.Number(value=60, label="Target fps cap")
                ui['prep_normalize'] = gr.Radio(
                    choices=["l2", "none"], value="l2", label="Per-sample normalize",
                )
            with gr.Row():
                ui['prep_pca'] = gr.Checkbox(value=True, label="PCA (center-only)")
                ui['prep_K'] = gr.Number(value=1024, label="PCA K (wide basis)")
                ui['prep_fit_fraction'] = gr.Number(
                    value=1.0, label="PCA fit fraction", minimum=0.01, maximum=1.0, step=0.01,
                )
            ui['prep_spp_scales'] = gr.CheckboxGroup(
                label="SPP scales to combine (multiscale)",
                choices=["1", "2", "4"],
                value=["1", "2", "4"],
                interactive=True,
                info=(
                    "For multiscale latents: which spatial-pyramid scales to combine "
                    "INTO this cache — 1=1×1 (global), 2=2×2, 4=4×4. The chosen blocks "
                    "are sliced from each combined latent file and concatenated before "
                    "PCA, so each combination is its own cache (build several to "
                    "compare). Selecting every available scale = the full latent "
                    "(no slicing). Ignored for weighted_average."
                ),
            )
            with gr.Row():
                ui['prep_refresh_btn'] = gr.Button("↻ Refresh file list")
                ui['prep_build_btn'] = gr.Button("⚙️ Build cache", variant="primary", scale=4)
                ui['prep_cancel_btn'] = gr.Button(
                    "Cancel", variant="secondary", interactive=False, scale=1,
                )
            # Frame-granular progress bar in its own component (never overlaps the
            # log), rendered by us → Gradio's overlay disabled on the click. Same
            # look as the Extract / Pre-process tabs (shared progress_ui.status_md).
            ui['prep_status'] = gr.Markdown(value="")
            # Per-run cancel flag, created fresh before the build generator runs.
            ui['prep_cancel_event'] = gr.State(None)
            ui['prep_log'] = gr.Textbox(label="Log", value="", interactive=False, lines=12)
        with gr.Tab("Explore (UMAP/DBSCAN)"):
            # Section 1: Previous Sessions
            with gr.Accordion("📂 Previous Sessions", open=True) as ui['previous_sessions_accordion']:
                ui['session_status'] = gr.Markdown("*Checking for previous sessions...*")
                ui['session_dropdown'] = gr.Dropdown(label="Select Session", choices=[], interactive=True, visible=False)
                with gr.Row():
                    ui['restore_btn'] = gr.Button("Restore Previous Session", interactive=False, visible=True)
                    ui['delete_session_btn'] = gr.Button("🗑 Delete Session", variant="secondary", interactive=True, visible=True)
                ui['delete_warning_md'] = gr.Markdown("", visible=False)
                ui['delete_confirm_btn'] = gr.Button("⚠️ Confirm Delete", variant="stop", visible=False)
    
            # Section 2: New Session
            with gr.Accordion("⚙️ New Session", open=False) as ui['cluster_input_accordion']:
                ui['select_model'] = gr.Dropdown(
                    label="Select Visual Model",
                    choices=["dinov2_vitb14_reg4_pretrain", "dinov3_vitb16", "dinov3_vitl16"],
                    value="dinov3_vitb16",
                    interactive=True,
                    visible=True
                )
                ui['pooling'] = gr.Dropdown(
                    label="Latent pooling",
                    choices=["auto", "weighted_average", "multiscale"],
                    value="auto",
                    interactive=True,
                    info=(
                        "Which extracted-latent variant to cluster when a project has "
                        "more than one for the same model. auto = use the majority and "
                        "warn if mixed. weighted_average ≈ 768-d; multiscale/SPP is wider "
                        "(e.g. 16128-d). Legacy raw only (a prepared cache is already one "
                        "variant)."
                    ),
                )
                # SPP scale selection now lives in the Prepare tab (scales are
                # combined BEFORE PCA, baked into each cache). The legacy raw path
                # still combines all available scales by default.
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
                        "Use the same value for all sessions in a project. "
                        "For a prepared cache the window is counted in DOWNSAMPLED frames "
                        "(e.g. W=2 on a 60 fps cache ≈ 33 ms), not raw frames."
                    ),
                )
                ui['data_source'] = gr.Dropdown(
                    label="Data source",
                    choices=[LEGACY_SOURCE_LABEL],
                    value=LEGACY_SOURCE_LABEL,
                    interactive=True,
                    info="Pick a prepared cache (built in the Prepare tab), or legacy raw (no cache).",
                )
                # Delete the prepared cache currently selected above (no-op for
                # legacy raw). Two-step confirm, mirroring session deletion.
                ui['prep_delete_btn'] = gr.Button(
                    "🗑 Delete selected cache", variant="secondary", interactive=True,
                )
                ui['prep_delete_warning_md'] = gr.Markdown("", visible=False)
                ui['prep_delete_confirm_btn'] = gr.Button(
                    "⚠️ Confirm delete cache", variant="stop", visible=False,
                )
                ui['variance_pct'] = gr.Number(
                    label="Explained variance % (PCA → UMAP)",
                    value=95,
                    precision=0,
                    interactive=True,
                    info=("Prepared cache only. % of variance to keep → sets k' "
                          "(PCA dims fed to UMAP). Blank/0 = 95%. Lower % = fewer "
                          "dims = less memory. Ignored for legacy."),
                )
                ui['reset'] = gr.Button("Initialize", interactive=True, visible=True)

            # State Holders
            latents = gr.State(None)
            local_latents = gr.State(None)
            local_embedding_plot = gr.State(None)
            mulvideo = gr.State(None)  # Holds LatentAggregator instance
            session_info = gr.State(None)
            overwrite_state = gr.State(False)  # Submit-overwrite confirmation gate
            # Last temp clip path from this Gradio session — replaces the
            # module-level _last_clip_path global so two concurrent users do not
            # race to delete each other's clip (3-A).
            last_clip_path_state = gr.State(None)

            # Compact CSS for the control bar: Gradio's default form/block padding
            # + inter-block gaps make this bar very tall, pushing the panels below
            # off-screen. Scope tight padding/gaps/heights to #castle-controls-row
            # only (panels below keep normal sizing).
            gr.HTML(
                "<style>"
                "#castle-controls-row{gap:3px!important;}"
                "#castle-controls-row .gap{gap:2px!important;}"
                "#castle-controls-row .form{padding:2px!important;gap:2px!important;}"
                "#castle-controls-row .block{padding:1px 6px!important;}"
                "#castle-controls-row label{margin-bottom:0!important;}"
                "#castle-controls-row textarea,#castle-controls-row input{padding:3px 6px!important;}"
                "#castle-controls-row button{min-height:28px!important;padding:3px 8px!important;}"
                "#castle-controls-row .wrap{padding:1px!important;}"
                # pull the panels up tight under the control bar
                "#castle-panels-row{margin-top:-6px!important;}"
                "</style>"
            )
            # Horizontal control bar ABOVE the panels. UMAP big column = two
            # sub-columns: (preset on top, seed/backend/Subsample% inline below) |
            # (UMAP configs with Generate Embedding stacked under it). DBSCAN is
            # the narrow column on the right.
            with gr.Row(visible=True, equal_height=False, elem_id="castle-controls-row") as ui['cluster_controls_row']:
                # --- UMAP group (one cohesive panel, two sub-columns) ---
                with gr.Column(scale=6, min_width=460):
                    with gr.Group():
                        with gr.Row():
                            # sub-column: preset + the inline param row
                            with gr.Column(scale=3, min_width=250):
                                ui['preset_dropdown'] = gr.Dropdown(preset_dropdown_list, value='Low-magnification objective 100', label="UMAP preset", visible=True, interactive=True)
                                with gr.Row():
                                    ui['umap_seed'] = gr.Textbox(
                                        label='seed', value='', placeholder='random',
                                        interactive=True, min_width=70, scale=1,
                                    )
                                    ui['umap_device'] = gr.Radio(
                                        choices=["GPU", "CPU"], value="GPU", label="Backend",
                                        min_width=130, scale=2,
                                    )
                                    # Subsample folded into one % field: 100 = all
                                    # points (off); lower it to subsample.
                                    ui['umap_subsample_pct'] = gr.Number(
                                        label="Subsample %", value=100, precision=0,
                                        minimum=1, maximum=100, interactive=True,
                                        min_width=90, scale=1,
                                    )
                            # sub-column: configs + Generate Embedding (stacked)
                            with gr.Column(scale=2, min_width=190):
                                ui['umap_config_text'] = gr.Textbox(label='UMAP configs', value=umap_config_template, lines=6, max_lines=8, interactive=True, visible=True, elem_id="castle-umap-config")
                                ui['umap_run'] = gr.Button("Generate Embedding", interactive=True, visible=True, size="sm")
                    ui['umap_seed_status'] = gr.Markdown(value="", visible=True)
                # --- DBSCAN + Submit group (narrower) ---
                with gr.Column(scale=2, min_width=160):
                    with gr.Row():
                        ui['eps'] = gr.Number(
                            label='eps (radius)', interactive=True, visible=True,
                            value=1, step=0.1, minimum=0.1, maximum=10, min_width=85,
                        )
                        ui['min_samples'] = gr.Number(
                            label='min points', interactive=True, visible=True,
                            value=5, precision=0, minimum=1, min_width=85,
                        )
                    ui['cluster_run'] = gr.Button("Generate Cluster", interactive=True, visible=True, size="sm")
                    ui['enter_submit_all_btn'] = gr.Button("Submit", interactive=True, visible=True, variant="primary", size="sm")
                    ui['overwrite_confirm_btn'] = gr.Button(
                        "⚠️ Confirm Overwrite", variant="stop", visible=False, size="sm",
                    )
                    ui['overwrite_warning_md'] = gr.Markdown("", visible=False)
                    ui['submit_status'] = gr.Markdown("", visible=True)

            # Three panels side by side: cluster tree | embedding | clip preview.
            # Embedding height is FIXED so generating a plot never reflows.
            with gr.Row(visible=True, equal_height=False, elem_id="castle-panels-row") as ui['cluster_row_main']:
                # --- cluster tree (scrolls internally) ---
                with gr.Column(scale=2, min_width=190):
                    ui['cluster_tree_html'] = gr.HTML(
                        value="<em style='color:#888;font-size:12px'>No clusters yet.</em>",
                        label="Cluster Tree",
                    )
                    # Colour-vision toggle: flips the unified palette mode for the
                    # whole tool. The tree recolours immediately; figures/scatter
                    # recolour on their next render.
                    ui['color_mode_radio'] = gr.Radio(
                        choices=[("Colorblind-safe", "colorblind"), ("Vibrant", "normal")],
                        value=get_color_mode(),
                        label="Colour vision",
                        container=True,
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
                # --- embedding scatter ---
                with gr.Column(scale=5, min_width=320):
                    ui['embedding_plot'] = gr.Image(label='Embedding', interactive=False, visible=True, height=460)
                # --- clip preview ---
                with gr.Column(scale=4, min_width=280):
                    ui['display'] = gr.Video(label='Display', height=330, interactive=False, visible=True, autoplay=True, loop=True)
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

    # When a prepared cache is the data source, model / pooling / ROID are baked
    # into the cache and ignored downstream — hide them to avoid the impression
    # they still apply. Legacy raw uses all three, so show them for the sentinel.
    def _toggle_source_fields(source):
        is_legacy = (not source) or str(source).startswith("(legacy")
        return (
            gr.update(visible=is_legacy),  # select_model
            gr.update(visible=is_legacy),  # pooling
            gr.update(visible=is_legacy),  # select_roi_id
        )

    ui['data_source'].change(
        fn=_toggle_source_fields,
        inputs=[ui['data_source']],
        outputs=[ui['select_model'], ui['pooling'], ui['select_roi_id']],
    )

    # Prepared-cache deletion (two-step confirm; operates on the data_source pick).
    ui['prep_delete_btn'].click(
        fn=show_delete_cache_confirmation,
        inputs=[storage_path, project_name, ui['data_source']],
        outputs=[ui['prep_delete_warning_md'], ui['prep_delete_confirm_btn']],
    )
    ui['prep_delete_confirm_btn'].click(
        fn=confirm_delete_cache,
        inputs=[storage_path, project_name, ui['data_source']],
        outputs=[ui['data_source'], ui['prep_delete_warning_md'], ui['prep_delete_confirm_btn']],
    )

    # Initialize: create aggregator + check for previous session.
    # First flip a VISIBLE status to "loading" the instant the button is clicked
    # (the init outputs are all gr.State, so without this there's no on-screen
    # feedback while latents load); _format_session_status overwrites it after.
    ui['reset'].click(
        fn=_on_init_click,
        inputs=None,
        outputs=[ui['session_status']],
    ).then(
        fn=init_mulvideo,
        inputs=[storage_path, project_name, ui['select_roi_id'], ui['bin_size'], ui['select_model'],
                ui['data_source'], ui['variance_pct'], ui['pooling']],
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
        fn=lambda: "init",
        outputs=[ui['cluster_tree_select']],
    ).then(
        fn=collapse_accordion,
        outputs=ui['cluster_input_accordion']
    )

    # Colour-vision toggle → set the unified palette mode + re-render the tree.
    def _on_color_mode_change(mode, lat):
        try:
            set_color_mode(mode)
        except ValueError:
            pass
        return update_select_cluster_list(lat)

    ui['color_mode_radio'].change(
        fn=_on_color_mode_change,
        inputs=[ui['color_mode_radio'], latents],
        outputs=[ui['cluster_tree_html'], ui['cluster_tree_select']],
    )

    # Restore previous session (B-03: also restores UMAP embedding)
    ui['restore_btn'].click(
        fn=restore_session,
        inputs=[storage_path, project_name, ui['select_roi_id'], ui['bin_size'], ui['select_model'], ui['session_dropdown']],
        outputs=[mulvideo, latents, ui['syllables_plot'],
                 ui['cluster_tree_html'], ui['cluster_tree_select'],
                 ui['behavior_id_csv'], ui['behavior_time_series_csv'],
                 local_embedding_plot, ui['embedding_plot'], local_latents],
    ).then(
        # Only claim success when restore_session actually produced latents.
        # On failure it returns Nones (and already showed a gr.Warning); keep the
        # restore/select controls available instead of a misleading success line.
        fn=lambda restored_latents: (
            (
                gr.update(visible=False),
                gr.update(value="Session restored successfully."),
                gr.update(visible=False),
                gr.update(visible=False),
            )
            if restored_latents is not None
            else (
                gr.update(visible=True),
                gr.update(value="⚠️ Session restore failed — see the warning above. "
                                "Pick another session or initialize a new one."),
                gr.update(visible=True),
                gr.update(visible=False),
            )
        ),
        inputs=[latents],
        outputs=[ui['restore_btn'], ui['session_status'], ui['session_dropdown'], ui['delete_session_btn']]
    )

    # Delete session — double-confirm flow
    # Step 1: show warning + confirm button
    ui['delete_session_btn'].click(
        fn=show_delete_confirmation,
        inputs=[ui['session_dropdown']],
        outputs=[ui['delete_warning_md'], ui['delete_confirm_btn']],
    )
    # Step 2: actually delete + refresh session list
    ui['delete_confirm_btn'].click(
        fn=confirm_delete_session,
        inputs=[storage_path, project_name, ui['session_dropdown']],
        outputs=[session_info, ui['delete_warning_md'], ui['delete_confirm_btn'],
                 latents, mulvideo, local_latents, local_embedding_plot,
                 ui['cluster_tree_html'], ui['cluster_tree_select']],
    ).then(
        fn=_format_session_status,
        inputs=[session_info],
        outputs=[ui['restore_btn'], ui['session_status'], ui['session_dropdown']],
    )
    # Reset warning when user picks a different session in the dropdown
    ui['session_dropdown'].change(
        fn=lambda: (gr.update(visible=False, value=""), gr.update(visible=False)),
        outputs=[ui['delete_warning_md'], ui['delete_confirm_btn']],
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
            ui['umap_device'],
            ui['umap_subsample_pct'],
        ],
        outputs=[
            local_latents, local_embedding_plot,
            ui['embedding_plot'], ui['umap_seed_status'],
        ],
    )

    ui['embedding_plot'].select(
        fn=embedding_plot_click,
        inputs=[mulvideo, local_embedding_plot, last_clip_path_state],
        outputs=[ui['embedding_plot'], ui['display'], last_clip_path_state],
    )
    ui['cluster_run'].click(
        fn=generate_local_cluster,
        inputs=[local_latents, ui['eps'], ui['min_samples']],
        outputs=[local_embedding_plot, ui['embedding_plot']],
    )

    _submit_inputs = [
        storage_path, project_name, latents, local_latents, mulvideo,
        ui['cluster_tree_select'],
        ui['umap_config_text'], ui['eps'], overwrite_state,
        ui['preset_dropdown'], ui['umap_seed'], ui['min_samples'],
    ]
    _submit_outputs = [
        ui['syllables_plot'],
        ui['cluster_tree_html'], ui['cluster_tree_select'],
        ui['behavior_id_csv'], ui['behavior_time_series_csv'],
        ui['behavior_time_series_srt'], local_embedding_plot,
        ui['embedding_plot'], ui['display_eps'],
        overwrite_state, ui['overwrite_confirm_btn'],
        ui['overwrite_warning_md'], ui['submit_status'],
    ]

    # Submit: auto-label all clusters and persist (with overwrite confirmation).
    # overwrite_confirm_btn calls the same fn — by that point overwrite_state is True.
    ui['enter_submit_all_btn'].click(
        fn=label_all_and_submit,
        inputs=_submit_inputs,
        outputs=_submit_outputs,
    )
    ui['overwrite_confirm_btn'].click(
        fn=label_all_and_submit,
        inputs=_submit_inputs,
        outputs=_submit_outputs,
    )

    # Auto-restore prior UMAP/eps/preset/seed when a tree node is clicked.
    # NB: use .change() not .input() — .input() requires a "real" user-typed
    # event and does not fire reliably for hidden textboxes whose value is
    # mutated via JS native setter + dispatchEvent.
    ui['cluster_tree_select'].change(
        fn=on_tree_node_select,
        inputs=[ui['cluster_tree_select'], latents, storage_path, project_name],
        outputs=[ui['umap_config_text'], ui['eps'], ui['embedding_plot'],
                 local_latents, local_embedding_plot, overwrite_state,
                 ui['submit_status'], ui['preset_dropdown'], ui['umap_seed'],
                 ui['min_samples']],
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

    # --- Prepare sub-tab bindings ---
    wire_video_selector(ui['prep_files'])
    _prep_files_io = dict(
        inputs=[storage_path, project_name, ui['prep_model']],
        outputs=[ui['prep_files']['group'], ui['prep_files']['all_state']],
    )
    ui['prep_refresh_btn'].click(fn=list_latent_choices, **_prep_files_io)
    ui['prep_model'].change(fn=list_latent_choices, **_prep_files_io)
    # Build: fresh cancel flag → generator (live log + status bar). Same chain as
    # the Extract / Pre-process tabs. show_progress="hidden" because we render our
    # own bar into prep_status.
    ui['prep_build_btn'].click(
        fn=init_cancel_event,
        outputs=ui['prep_cancel_event'],
        queue=False,
    ).then(
        fn=build_prepare_handler,
        inputs=[storage_path, project_name, ui['prep_model'], ui['prep_files']['group'],
                ui['prep_downsample'], ui['prep_target_fps'], ui['prep_normalize'],
                ui['prep_pca'], ui['prep_K'], ui['prep_fit_fraction'],
                ui['prep_spp_scales'], ui['prep_cancel_event']],
        outputs=[ui['prep_log'], ui['prep_build_btn'], ui['prep_cancel_btn'],
                 ui['prep_status'], ui['data_source']],
        show_progress="hidden",
    )
    # Cancel: set the flag + relabel; the generator's final yield restores the
    # idle label and aborts the build at its next checkpoint (partial cache
    # removed). queue=False so it runs while the generator owns the queue. NO
    # cancels=[…] — an abrupt cancel would skip the reset yield.
    ui['prep_cancel_btn'].click(
        fn=lambda ce: request_cancel(ce, "Canceling (stopping current step)…"),
        inputs=ui['prep_cancel_event'],
        outputs=ui['prep_cancel_btn'],
        queue=False,
    )
    # Populate the file list + refresh the data-source dropdown on tab entry.
    cluster_page_tab.select(fn=list_latent_choices, **_prep_files_io)
    cluster_page_tab.select(
        fn=lambda sp, pn: gr.update(choices=list_prepare_choices(sp, pn)),
        inputs=[storage_path, project_name],
        outputs=[ui['data_source']],
    )

    # Expose shared state for Annotator tab (A-04) and main_ui wiring.
    shared_states = {
        'latents': latents,
        'mulvideo': mulvideo,
        # data_source dropdown so main_ui can re-list prepared caches on
        # project-open (the in-tab tab.select trigger above doesn't fire when a
        # project is opened while already on / before entering this tab).
        'data_source': ui['data_source'],
    }

    visibility_components = {
        'previous_sessions_accordion': ui['previous_sessions_accordion'],
        'cluster_input_accordion': ui['cluster_input_accordion'],
        'cluster_controls_row': ui['cluster_controls_row'],
        'cluster_row_main': ui['cluster_row_main'],
        'syllables_plot': ui['syllables_plot'],
        'cluster_row_files': ui['cluster_row_files'],
    }

    return visibility_components, shared_states
