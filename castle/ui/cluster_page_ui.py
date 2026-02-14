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

from castle.core.cluster import LatentAggregator, auto_generate_cluster_name
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
        
        # F-01: Build KDTree once and cache for reuse
        from scipy.spatial import KDTree
        self._kdtree = KDTree(data)
    
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
        # F-01: Use cached KDTree for O(log n) lookup
        from castle.core.cluster import find_nearest_embedding
        index, _ = find_nearest_embedding(self.data, x, y, tree=self._kdtree)
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

def plot_syllables_per_video(latents, aggregator):
    """Plot syllables with one video per row, x-axis in seconds."""
    from matplotlib.patches import Patch

    cluster = latents.cluster
    cluster_meta = latents.cluster_meta
    time_window = latents.time_window
    videos_meta = aggregator.videos_meta
    fps = aggregator.fps
    bin_size = aggregator.bin_size

    n_videos = len(videos_meta)

    fig, axes = plt.subplots(n_videos, 1, figsize=(14, 0.8 * n_videos), squeeze=False)
    axes = axes.flatten()

    def palette(c):
        if c in cluster_meta:
            return cluster_meta[c]['color']
        else:
            return 'grey'
    
    cum = 0
    for video_idx, (vn, video_name) in enumerate(videos_meta):
        ax = axes[video_idx]
        video_cluster = cluster[cum:cum + vn]
        
        n = len(video_cluster)
        key_frames = [0] + [i + 1 for i in range(n - 1) if video_cluster[i] != video_cluster[i + 1]] + [n]
        
        widths = [(key_frames[j+1] - key_frames[j]) * bin_size / fps for j in range(len(key_frames)-1)]
        colors = [palette(video_cluster[key_frames[j]]) for j in range(len(key_frames)-1)]
        lefts = [key_frames[j] * bin_size / fps for j in range(len(key_frames)-1)]
        
        total_seconds = n * bin_size / fps
        
        ax.bar(lefts, height=[1]*len(widths), width=widths, color=colors, align='edge', edgecolor='none')
        ax.set_xlim(0, total_seconds)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        
        video_basename = os.path.basename(video_name).split('.')[0]
        ax.set_title(video_basename, fontsize=9, loc='left')
        
        cum += vn
    
    unique_clusters = sorted(set(cluster))
    if -1 in unique_clusters:
        unique_clusters.remove(-1)
    
    legend_handles = [Patch(color=palette(cat), label=cluster_meta[cat]['name']) for cat in unique_clusters if cat in cluster_meta]
    
    if legend_handles:
        axes[-1].legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, -0.3), 
                       ncol=min(len(legend_handles), 6), fontsize=8)
    
    plt.tight_layout()
    return fig


def label_all_and_submit(storage_path, project_name, latents, local_latents, aggregator, parent_name):
    """Auto-label all clusters and submit."""
    unique_clusters = np.unique(local_latents.cluster)
    
    count = 0
    for cluster_id in unique_clusters:
        if cluster_id == -1:
            continue
        
        cluster_name = auto_generate_cluster_name(parent_name, cluster_id)
        local_latents.label_cluster(cluster_id, cluster_name)
        count += 1
        
    gr.Info(f'Auto-labeled {count} clusters.')
    
    return import_info_from_local_latent(storage_path, project_name, latents, local_latents, aggregator)


def check_session_exists(storage_path, project_name):
    """Check if previous session files exist."""
    if project_name is None:
        return None
    cluster_path = os.path.join(storage_path, project_name, 'cluster')
    id_csv = os.path.join(cluster_path, 'id.csv')
    if not os.path.exists(id_csv):
        return None
    id_df = pd.read_csv(id_csv)
    cluster_count = len(id_df) - 1  # Exclude root
    return {'cluster_count': cluster_count, 'id_csv': id_csv}


def restore_session(storage_path, project_name, select_roi_id, bin_size, select_model):
    """Restore latents from saved CSV files."""
    if project_name is None:
        return None, None, None, None, None, None

    def notify_callback(msg, level="info"):
        if level == "error":
            gr.Warning(msg)
        else:
            gr.Info(msg)

    aggregator = LatentAggregator(
        storage_path, project_name, select_roi_id, bin_size,
        model_name=select_model,
        notify=notify_callback
    )
    latents = aggregator.get_latent_object()

    cluster_path = os.path.join(storage_path, project_name, 'cluster')

    # Restore cluster_meta from id.csv
    id_csv_path = os.path.join(cluster_path, 'id.csv')
    id_df = pd.read_csv(id_csv_path)
    for _, row in id_df.iterrows():
        cluster_id = int(row['Id'])
        color = row.get('Color', 'grey')
        latents.cluster_meta[cluster_id] = {'name': row['Name'], 'color': color}
        latents.behavior_name2cluster_id[row['Name']] = cluster_id
        if color != 'grey':
            latents.used_palette.add(color)
    latents.num_cluster = len(id_df)

    # Restore cluster assignments from time_series CSVs
    cum = 0
    df2_paths = []
    for vn, v in aggregator.videos_meta:
        video_basename = os.path.basename(v).split('.')[0]
        ts_path = os.path.join(cluster_path, f'time_series_{video_basename}.csv')
        if os.path.exists(ts_path):
            ts_df = pd.read_csv(ts_path)
            bin_clusters = ts_df['behavior'].values[::aggregator.bin_size][:vn]
            latents.cluster[cum:cum+len(bin_clusters)] = bin_clusters
            df2_paths.append(ts_path)
        cum += vn

    gr.Info(f'Restored session with {latents.num_cluster - 1} clusters')

    fig = plot_syllables_per_video(latents, aggregator)

    return aggregator, latents, fig, update_select_cluster_list(latents), id_csv_path, df2_paths


def convert_latent_cluster_to_subtitle(storage_path, project_name, latents, aggregator):
    # Delegate to Aggregator
    return aggregator.generate_subtitles(latents.cluster, latents.cluster_meta)

def import_info_from_local_latent(storage_path, project_name, latents, local_latents, aggregator):
    try:
        latents.import_local_latent(local_latents)
    except Exception as e:
        gr.Info(f'Import error: {e}')
        return (None,) * 8

    # Plot syllables with one video per row
    fig = plot_syllables_per_video(latents, aggregator)

    # Save ID CSV (with Color column for session restore)
    df1 = pd.DataFrame({
        'Id': [k for k, v in latents.cluster_meta.items()],
        'Name': [v['name'] for k, v in latents.cluster_meta.items()],
        'Color': [v['color'] for k, v in latents.cluster_meta.items()],
    })

    cluster_path = os.path.join(storage_path, project_name, 'cluster')
    os.makedirs(cluster_path, exist_ok=True)

    df1_path = os.path.join(cluster_path, 'id.csv')
    df1.to_csv(df1_path, index=False)

    # Generate per-video time_series CSV files
    df2_paths = []
    cum = 0
    for vn, v in aggregator.videos_meta:
        video_cluster = latents.cluster[cum:cum + vn]
        video_frames = np.repeat(video_cluster, latents.time_window)
        df2 = pd.DataFrame({'behavior': video_frames})

        video_basename = os.path.basename(v).split('.')[0]
        df2_path = os.path.join(cluster_path, f'time_series_{video_basename}.csv')
        df2.to_csv(df2_path)
        df2_paths.append(df2_path)
        cum += vn

    # Generate Subtitles
    subtitle_paths = convert_latent_cluster_to_subtitle(storage_path, project_name, latents, aggregator)
    
    # Save Embedding
    Z_plt = EmbeddingScatterPlot(local_latents)
    cluster_name = ""
    for _, it in local_latents.export.items():
        cluster_name += it['name'] + '_'

    local_embedding_path = os.path.join(cluster_path, f'cluster_{cluster_name}.npz')
    Z_plt.save_named_embedding(save_path=local_embedding_path)

    return (fig, update_select_cluster_list(latents), df1_path, df2_paths, subtitle_paths, 
            Z_plt, Z_plt.plot_named_embedding(), local_embedding_path)

def init_mulvideo(storage_path, project_name, select_roi_id, bin_size, select_model):
    """
    Initializes LatentAggregator (formerly MultiVideos)
    """
    if not project_name:
        return None, None, None
    
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
        session_info = check_session_exists(storage_path, project_name)
        return aggregator, aggregator.get_latent_object(), session_info
    except Exception as e:
        gr.Warning(f"Initialization Failed: {e}")
        return None, None, None

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
            ui['restore_btn'] = gr.Button("Restore Previous Session", interactive=True, visible=False)
            ui['session_status'] = gr.Markdown("", visible=False)
        
    # State Holders
    latents = gr.State(None)
    local_latents = gr.State(None)
    local_embedding_plot = gr.State(None)
    mulvideo = gr.State(None)  # Holds LatentAggregator instance
    session_info = gr.State(None)

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
            ui['enter_submit_all_btn'] = gr.Button("Enter & Submit all", interactive=True, visible=True)
        with gr.Column(scale=8):
            ui['embedding_plot'] = gr.Image(label='Embedding', interactive=False, visible=True)
            ui['display'] = gr.Image(label='Display', interactive=False, visible=True)  
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

    # Initialize: create aggregator + check for previous session
    ui['reset'].click(
        fn=init_mulvideo,
        inputs=[storage_path, project_name, ui['select_roi_id'], ui['bin_size'], ui['select_model']],
        outputs=[mulvideo, latents, session_info]
    ).then(
        fn=lambda info: (
            gr.update(visible=info is not None),
            gr.update(value=f"**Previous session found:** {info['cluster_count']} clusters", visible=info is not None) if info else gr.update(visible=False)
        ),
        inputs=[session_info],
        outputs=[ui['restore_btn'], ui['session_status']]
    )

    # Restore previous session
    ui['restore_btn'].click(
        fn=restore_session,
        inputs=[storage_path, project_name, ui['select_roi_id'], ui['bin_size'], ui['select_model']],
        outputs=[mulvideo, latents, ui['syllables_plot'], ui['select_cluster'], ui['behavior_id_csv'], ui['behavior_time_series_csv']]
    ).then(
        fn=lambda: (gr.update(visible=False), gr.update(visible=False)),
        outputs=[ui['restore_btn'], ui['session_status']]
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
        outputs=[ui['embedding_plot'], ui['display']]
    )
    ui['cluster_run'].click(
        fn=generate_local_cluster,
        inputs=[local_latents, ui['eps']],
        outputs=[local_embedding_plot, ui['embedding_plot']],
    )
    ui['label_cluster_btn'].click(
        fn=label_local_cluster,
        inputs=[local_latents, ui['label_cluster_id'], ui['label_cluster_name']],
    )
    
    # Auto-generate cluster name when ID changes
    ui['label_cluster_id'].change(
        fn=auto_generate_cluster_name,
        inputs=[ui['select_cluster'], ui['label_cluster_id']],
        outputs=ui['label_cluster_name']
    )

    ui['label_cluster_submit_btn'].click(
        fn=import_info_from_local_latent,
        inputs=[storage_path, project_name, latents, local_latents, mulvideo],
        outputs=[ui['syllables_plot'], ui['select_cluster'], ui['behavior_id_csv'], ui['behavior_time_series_csv'], ui['behavior_time_series_srt'], local_embedding_plot, ui['embedding_plot'], ui['display_eps']],
    )

    # Enter & Submit all: auto-label all clusters and submit
    ui['enter_submit_all_btn'].click(
        fn=label_all_and_submit,
        inputs=[storage_path, project_name, latents, local_latents, mulvideo, ui['select_cluster']],
        outputs=[ui['syllables_plot'], ui['select_cluster'], ui['behavior_id_csv'], ui['behavior_time_series_csv'], ui['behavior_time_series_srt'], local_embedding_plot, ui['embedding_plot'], ui['display_eps']],
    )

    # Auto-update cluster list when tab is selected
    cluster_page_tab.select(
        fn=update_select_cluster_list,
        inputs=latents,
        outputs=ui['select_cluster']
    )

    return {
        'cluster_input_accordion': ui['cluster_input_accordion'],
        'cluster_row_main': ui['cluster_row_main'],
        'syllables_plot': ui['syllables_plot'],
        'cluster_row_files': ui['cluster_row_files']
    }