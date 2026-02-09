"""
castle/ui/cluster_page_ui.py
UI Layer for Clustering.
"""

import os
import json
import io
import re  # Added missing import for regex
import gradio as gr
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from scipy.spatial import KDTree

from castle.core.cluster import LatentAggregator
from castle.utils.latent_explorer import Latent  # Still needed for type hinting or direct access if any

# Configure matplotlib to reduce warnings about open figures
matplotlib.rcParams['figure.max_open_warning'] = 50

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

dbscan_config_template='''{
    "eps": 1.0
}'''

preset_dropdown_list = ['Low-magnification objective 1000', 'Low-magnification objective 500', 'Low-magnification objective 300', 'Low-magnification objective 100', 'Low-magnification objective 50', 'Low-magnification objective 25']
preset_dropdown_list += ['Intermediate-magnification objective (1000, 500)', 'Intermediate-magnification objective (500, 300)', 'Intermediate-magnification objective (300, 100)', 'Intermediate-magnification objective (100, 50)', 'Intermediate-magnification objective (50, 25)']
preset_dropdown_list += ['High-magnification objective (1000, 500)', 'High-magnification objective (500, 300)', 'High-magnification objective (300, 100)', 'High-magnification objective (100, 50)', 'High-magnification objective (50, 25)']

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

def padding(mi, mx, scale=1.05):
    mid = (mi + mx) / 2
    d = mx - mi
    return (mid - (d / 2) * scale), (mid + (d / 2) * scale)

# ---------------------------
# UI Components
# ---------------------------

class EmbeddingScatterPlot:
    """
    Handles plotting of embedding data and interaction (click to find nearest point).
    Ideally, the logic parts (KDTree) should move to Core, but kept here for Phase 2 as per plan.
    """
    def __init__(self, local_latents):
        data = local_latents.embedding
        self.local_latents = local_latents
        self.data = data
        
        # Calculate bounds once
        self.xlim = padding(data[:,0].min(), data[:,0].max())
        self.ylim = padding(data[:,1].min(), data[:,1].max())
        
        self.selected_point = (np.nan, np.nan)
        self.selected_index = -1
        # M-03: KDTree removed - now computed in Core layer via find_nearest_embedding
    
    def pixel_2_embedding(self, px, py):
        px, py = float(px), float(py)
        # Warning: self.width/height depend on the LAST GENERATED image size.
        # This is a bit fragile if image size changes, but standard Gradio Image usually fixed or consistent.
        if not hasattr(self, 'width') or not hasattr(self, 'height'):
             return 0, 0 # Fallback
             
        ex = (px / self.width) * (self.xlim[1] - self.xlim[0]) + self.xlim[0]
        ey = (py / self.height) * (self.ylim[1] - self.ylim[0]) + self.ylim[0]
        return ex, ey

    def plot(self):
        fig = plt.figure()
        self.local_latents.plot_embedding()
        plt.scatter(self.selected_point[0], self.selected_point[1], color='red')
        plt.axis('off')
        plt.xlim(self.xlim)
        plt.ylim(self.ylim[1], self.ylim[0])

        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf)

        self.width, self.height = img.size
        return img
    
    def plot_named_embedding(self):
        fig = plt.figure()
        self.local_latents.plot_name_embedding()
        plt.scatter(self.selected_point[0], self.selected_point[1], color='red')
        plt.axis('off')
        plt.xlim(self.xlim)
        plt.ylim(self.ylim[1], self.ylim[0])

        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf)

        self.width, self.height = img.size
        return img

    def save_named_embedding(self, save_path):
        index_mask = self.local_latents.index_mask
        masked_emb = self.local_latents.embedding
        masked_cls = self.local_latents.cluster
        config = self.local_latents.configs
        n_samples = len(index_mask)
        n_features = masked_emb.shape[-1]

        emb = np.zeros((n_samples, n_features)) + np.nan
        emb[index_mask] = masked_emb

        cls = np.zeros(n_samples, dtype=np.int16) - 1
        cls[index_mask] = masked_cls

        np.savez_compressed(save_path, emb=emb, cls=cls, config=config)
    
    def click(self, x, y):
        x, y = self.pixel_2_embedding(x, y)
        index = self.near_point(x, y)
        
        self.selected_point = self.data[index]
        self.selected_index = index # Local index in 'data'
        
        # Map back to global index
        self.selected_index = np.arange(len(self.local_latents.index_mask))[self.local_latents.index_mask][index]
        return self.plot()
        
    def near_point(self, x, y):
        # M-03: Delegate to Core layer
        from castle.core.cluster import find_nearest_embedding
        index, _ = find_nearest_embedding(self.data, x, y)
        return index

# ---------------------------
# Event Handlers
# ---------------------------

def embedding_plot_click(aggregator, Z_plt, evt: gr.SelectData):
    """
    Handle click on embedding plot.
    aggregator: LatentAggregator instance
    Z_plt: EmbeddingScatterPlot instance
    """
    if hasattr(evt, 'index'):
        emb_plot = Z_plt.click(evt.index[0], evt.index[1])
    else:
        gr.Info('click event error')
        return None, None
        
    index = Z_plt.selected_index
    # Use Aggregator to get frame
    frame = aggregator.get_frame(index)
    
    if frame is None:
        # Return fallback blank image if frame fetch fails
        return emb_plot, None 
        
    return emb_plot, frame

def collapse_accordion():
    return gr.update(open=False)

def update_select_cluster_list(latents):
    if hasattr(latents, 'behavior_name2cluster_id'):
        li = [k for k,v in latents.behavior_name2cluster_id.items()]
    else:
        li = []
        gr.Info('Latent init error, please wait.')
    return gr.update(choices=li)

def generate_embedding(latents, cluster_name, cfg_str):
    try:
        cfg = json.loads(cfg_str)
    except:
        gr.Info('UMAP config JSON format error')
        return None, None, None
    
    local_latents = latents.select(selected_cluster=cluster_name)
    if len(local_latents.data) == 0:
        gr.Info('This Cluster is empty.')
        return None, None, None
    
    local_latents.build_embedding(cfg)
    Z_plt = EmbeddingScatterPlot(local_latents)
    return local_latents, Z_plt, Z_plt.plot()

def generate_local_cluster(local_latents, eps):
    try:
        cfg = json.loads(dbscan_config_template)
    except:
        gr.Info('Cluster JSON format error')
        return None, None
    
    cfg['eps'] = eps
    local_latents.build_cluster(method='dbscan', configs=cfg)
    Z_plt = EmbeddingScatterPlot(local_latents)
    return Z_plt, Z_plt.plot()

def label_local_cluster(local_latents, cluster_id, cluster_name):
    if not cluster_name:
        gr.Info('Name is empty')
        return 
    local_latents.label_cluster(cluster_id, cluster_name)  
    gr.Info(f'Named {cluster_id} as {cluster_name}')

def convert_latent_cluster_to_subtitle(storage_path, project_name, latents, aggregator):
    # Delegate to Aggregator
    return aggregator.generate_subtitles(latents.cluster, latents.cluster_meta)

def import_info_from_local_latent(storage_path, project_name, latents, local_latents, aggregator):
    try:
        # start_cluster_id = latents.num_cluster
        latents.import_local_latent(local_latents)
    except Exception as e:
        gr.Info(f'Import error: {e}')
        # Return mostly None to indicate failure safely
        return (None,) * 8

    # Plot Syllables
    fig = plt.figure(figsize=(12, 2))
    latents.plot_syllables()
    plt.tight_layout()

    # Save CSVs
    df1 = pd.DataFrame({
        'Id': [k for k, v in latents.cluster_meta.items()],
        'Name': [v['name'] for k, v in latents.cluster_meta.items()],
    })
    df2 = pd.DataFrame({
        'behavior': np.repeat(latents.cluster, latents.time_window)
    })

    cluster_path = os.path.join(storage_path, project_name, 'cluster')
    os.makedirs(cluster_path, exist_ok=True)

    df1_path = os.path.join(cluster_path, 'id.csv')
    df2_path = os.path.join(cluster_path, 'time_series.csv')
    df1.to_csv(df1_path, index=False)  
    df2.to_csv(df2_path)

    # Generate Subtitles
    subtitle_paths = convert_latent_cluster_to_subtitle(storage_path, project_name, latents, aggregator)
    # Note: subtitle_paths is a list. Gradio File component can accept list of paths.
    
    # Save Embedding
    Z_plt = EmbeddingScatterPlot(local_latents)
    cluster_name = ""
    for _, it in local_latents.export.items():
        cluster_name += it['name'] + '_'

    local_embedding_path = os.path.join(cluster_path, f'cluster_{cluster_name}.npz')
    Z_plt.save_named_embedding(save_path=local_embedding_path)

    return (fig, update_select_cluster_list(latents), df1_path, df2_path, subtitle_paths, 
            Z_plt, Z_plt.plot_named_embedding(), local_embedding_path)

def init_mulvideo(storage_path, project_name, select_roi_id, bin_size, select_model):
    """
    Initializes LatentAggregator (formerly MultiVideos)
    """
    if not project_name:
        return None, None
    
    # Create Gradio-compatible notification callback
    def notify_callback(msg: str, level: str = "info"):
        if level == "error":
            gr.Warning(msg)
        else:
            gr.Info(msg)
    
    try:
        aggregator = LatentAggregator(
            storage_path, project_name, select_roi_id, bin_size,
            model_name=select_model,
            notify=notify_callback
        )
        return aggregator, aggregator.get_latent_object()
    except Exception as e:
        gr.Warning(f"Initialization Failed: {e}")
        return None, None

# ---------------------------
# UI Construction
# ---------------------------

def create_cluster_page_ui(storage_path, project_name, cluster_page_tab):
    ui = dict()
    # Create a main container to control visibility of the entire page
    # Create a main container to control visibility of the entire page
    
    with gr.Accordion('Input setting', visible=False) as ui['cluster_input_accordion']:
            ui['select_model'] = gr.Dropdown(
                label="Select Visual Model",
                choices=["dinov2_vitb14_reg4_pretrain", "dinov3_vitb16", "dinov3_vitl16"],
                value="dinov3_vitb16",
                interactive=True,
                visible=True # Initially visible (controlled by parent accordion)
            )
            ui['select_roi_id'] = gr.Textbox(label="Enter ROI ID", value="1", info="ex: 1,2,3.", visible=True)
            ui['bin_size'] = gr.Number(label='Time window (frame)', value=1, interactive=True, visible=True)
            ui['reset'] = gr.Button("Initialize", interactive=True, visible=True)
        
    # State Holders
    latents = gr.State(None)
    local_latents = gr.State(None)
    local_embedding_plot = gr.State(None) # Holds EmbeddingScatterPlot instance
    mulvideo = gr.State(None) # Holds LatentAggregator instance

    # Manually track the visibility of this entire row
    with gr.Row(visible=True) as ui['cluster_row_main']:
        with gr.Column(scale=2):
            ui['select_cluster'] = gr.Dropdown(label="Select Cluster", visible=True, interactive=True)
            ui['preset_dropdown'] = gr.Dropdown(preset_dropdown_list, value='Low-magnification objective 100', label="UMAP preset", visible=True, interactive=True)
            ui['umap_config_text'] = gr.Textbox(label='UMAP configs', value=umap_config_template, lines=8, max_lines=8, interactive=True, visible=True)
            ui['umap_run'] = gr.Button("Generate Embedding", interactive=True, visible=True)
            ui['eps'] = gr.Number(label='epsilon-neighborhood radius', interactive=True, visible=True, value=1, step=0.1, minimum=0.1, maximum=10)
            ui['cluster_run'] = gr.Button("Generate Cluster", interactive=True, visible=True)
            ui['label_cluster_id'] = gr.Number(label='Cluster id', interactive=True, visible=True)
            ui['label_cluster_name'] = gr.Textbox(label='Cluster name', interactive=True, visible=True)
            ui['label_cluster_btn'] = gr.Button("Enter", interactive=True, visible=True)
            ui['label_cluster_submit_btn'] = gr.Button("Submit", interactive=True, visible=True)
        with gr.Column(scale=8):
            ui['embedding_plot'] = gr.Image(label='Embedding', interactive=False, visible=True)
            ui['display'] = gr.Image(label='Display', interactive=False, visible=True)  
            ui['display_eps'] = gr.File(label="Display EPS", interactive=False, visible=True)
            
    ui['syllables_plot'] = gr.Plot(label='Syllable', visible=True)
    with gr.Row(visible=True) as ui['cluster_row_files']:
        with gr.Column(scale=2):
            ui['behavior_id_csv'] = gr.File(label="Behavior ID", interactive=False, visible=True)
        with gr.Column(scale=2):
            ui['behavior_time_series_csv'] = gr.File(label="Behavior time series", interactive=False, visible=True)
        with gr.Column(scale=2):
            ui['behavior_time_series_srt'] = gr.File(label="Behavior time series (SRT)", interactive=False, visible=True)

    # Event Bindings
    ui['reset'].click(
        fn=init_mulvideo,
        inputs=[storage_path, project_name, ui['select_roi_id'], ui['bin_size'], ui['select_model']],
        outputs=[mulvideo, latents]
    )

    ui['select_cluster'].focus(
        fn=update_select_cluster_list,
        inputs=latents,
        outputs=ui['select_cluster']
    )

    ui['preset_dropdown'].select(
        fn=update_umap_config_text_with_preset,
        inputs=ui['preset_dropdown'],
        outputs=ui['umap_config_text']
    )
    ui['umap_run'].click(
        fn=generate_embedding,
        inputs=[latents, ui['select_cluster'], ui['umap_config_text']],
        outputs=[local_latents, local_embedding_plot, ui['embedding_plot']]
    )

    ui['reset'].click(
        fn=collapse_accordion,
        outputs=ui['cluster_input_accordion']
    )

    ui['embedding_plot'].select(
        fn=embedding_plot_click,
        inputs=[mulvideo, local_embedding_plot],
        outputs= [ui['embedding_plot'], ui['display']]
    )
    ui['cluster_run'].click(
        fn=generate_local_cluster,
        inputs=[local_latents, ui['eps']],
        outputs=[local_embedding_plot, ui['embedding_plot'] ],
    )
    ui['label_cluster_btn'].click(
        fn=label_local_cluster,
        inputs=[local_latents, ui['label_cluster_id'], ui['label_cluster_name']],
    )
    ui['label_cluster_submit_btn'].click(
        fn=import_info_from_local_latent,
        inputs=[storage_path, project_name, latents, local_latents, mulvideo],
        outputs=[ui['syllables_plot'], ui['select_cluster'], ui['behavior_id_csv'], ui['behavior_time_series_csv'], ui['behavior_time_series_srt'], local_embedding_plot, ui['embedding_plot'], ui['display_eps']],
    )

    # Auto-update cluster list when tab is selected
    cluster_page_tab.select(
        fn=update_select_cluster_list,
        inputs=latents,
        outputs=ui['select_cluster']
    )

    # Return only the top-level containers for visibility toggling
    # This avoids updating gr.Column directly and avoids recursive updates on children
    return {
        'cluster_input_accordion': ui['cluster_input_accordion'],
        'cluster_row_main': ui['cluster_row_main'],
        'syllables_plot': ui['syllables_plot'],
        'cluster_row_files': ui['cluster_row_files']
    }