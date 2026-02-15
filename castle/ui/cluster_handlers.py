"""
castle/ui/cluster_handlers.py
Event handler functions for the clustering page.

These functions are bound to Gradio button clicks.
They take Gradio state values as input and return updated values.
"""

import os
import json
import glob

import gradio as gr
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from castle.core.cluster import LatentAggregator, auto_generate_cluster_name
from castle.ui.embedding_scatter import EmbeddingScatterPlot
from castle.ui.cluster_tree import build_cluster_tree_markdown


# ---------------------------
# Templates & Presets (UI Config)
# ---------------------------

dbscan_config_template = '''{
    "eps": 1.0
}'''


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
        li = [k for k, v in latents.behavior_name2cluster_id.items()]
    else:
        li = []
        gr.Info('Latent init error, please wait.')
    return gr.update(choices=li)


def generate_embedding(latents, cluster_name, cfg_str, progress=gr.Progress()):
    try:
        cfg = json.loads(cfg_str)
    except:
        gr.Info('UMAP config JSON format error')
        return None, None, None
    
    local_latents = latents.select(selected_cluster=cluster_name)
    if len(local_latents.data) == 0:
        gr.Info('This Cluster is empty.')
        return None, None, None
    
    # C-05: Report progress between UMAP stages
    def umap_progress(stage, total):
        progress(stage / total, desc=f"UMAP Stage {stage + 1}/{total}...")

    local_latents.build_embedding(cfg, progress_callback=umap_progress)

    progress(1.0, desc="Building plot...")
    Z_plt = EmbeddingScatterPlot(local_latents)
    return local_latents, Z_plt, Z_plt.plot()


def generate_local_cluster(local_latents, eps, progress=gr.Progress()):
    try:
        cfg = json.loads(dbscan_config_template)
    except:
        gr.Info('Cluster JSON format error')
        return None, None
    
    cfg['eps'] = eps
    progress(0, desc="Running DBSCAN...")
    local_latents.build_cluster(method='dbscan', configs=cfg)
    progress(1.0, desc="Building plot...")
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


def _find_latest_npz(cluster_path):
    """Find the most recently modified cluster_*.npz file in cluster_path."""
    npz_files = glob.glob(os.path.join(cluster_path, 'cluster_*.npz'))
    if not npz_files:
        return None
    # Sort by modification time, newest first
    npz_files.sort(key=os.path.getmtime, reverse=True)
    return npz_files[0]


def _restore_embedding_from_npz(npz_path, latents):
    """Restore a LocalLatent and EmbeddingScatterPlot from a saved .npz file.

    The npz contains:
      - emb: (N, 2) array with NaN for non-selected indices
      - cls: (N,) int array with -1 for non-selected indices
      - config: UMAP config used

    Returns (local_latents, Z_plt) or (None, None) on failure.
    """
    from castle.utils.latent_explorer import LocalLatent
    from collections import Counter
    try:
        data = np.load(npz_path, allow_pickle=True)
        emb_full = data['emb']   # (N, 2) with NaN
        cls_full = data['cls']   # (N,) with -1
        config = data['config']

        # Determine which indices were selected (non-NaN in embedding)
        valid_mask = ~np.isnan(emb_full[:, 0])

        masked_emb = emb_full[valid_mask]
        masked_cls = cls_full[valid_mask]

        # Reconstruct LocalLatent
        local_data = latents.data[valid_mask] if hasattr(latents, 'data') else masked_emb
        local_latents = LocalLatent(
            data=local_data,
            index_mask=valid_mask,
            color_avoid=latents.used_palette,
            device=latents.device,
        )
        local_latents.embedding = masked_emb
        local_latents.cluster = masked_cls
        local_latents.configs = config.tolist() if hasattr(config, 'tolist') else config

        # Reconstruct export dict from the cluster assignments + global meta
        for cid_local in np.unique(masked_cls):
            if cid_local == -1:
                continue
            global_indices = np.where(valid_mask)[0]
            global_cluster_vals = latents.cluster[global_indices]
            local_mask = masked_cls == cid_local
            if not np.any(local_mask):
                continue
            global_ids = global_cluster_vals[local_mask]
            global_id = Counter(global_ids.tolist()).most_common(1)[0][0]

            if global_id in latents.cluster_meta:
                meta = latents.cluster_meta[global_id]
                local_latents.export[cid_local] = {
                    'name': meta['name'],
                    'color': meta['color'],
                }

        Z_plt = EmbeddingScatterPlot(local_latents)
        return local_latents, Z_plt

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, None


def restore_session(storage_path, project_name, select_roi_id, bin_size, select_model):
    """Restore latents from saved CSV files and optionally restore UMAP embedding."""
    if project_name is None:
        return None, None, None, None, None, None, None, None

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

    # Restore UMAP embedding from saved .npz (B-03)
    restored_local_latents = None
    restored_Z_plt = None
    restored_emb_img = None

    npz_path = _find_latest_npz(cluster_path)
    if npz_path:
        restored_local_latents, restored_Z_plt = _restore_embedding_from_npz(npz_path, latents)
        if restored_Z_plt is not None:
            restored_emb_img = restored_Z_plt.plot_named_embedding()
            gr.Info(f'Restored UMAP embedding from {os.path.basename(npz_path)}')

    gr.Info(f'Restored session with {latents.num_cluster - 1} clusters')

    fig = plot_syllables_per_video(latents, aggregator)

    return (aggregator, latents, fig, update_select_cluster_list(latents),
            id_csv_path, df2_paths, restored_Z_plt, restored_emb_img)


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
