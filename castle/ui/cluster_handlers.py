"""
castle/ui/cluster_handlers.py
Event handler functions for the clustering page.

These functions are bound to Gradio button clicks.
They take Gradio state values as input and return updated values.
"""

import logging
import os
import json
import glob
import subprocess
import cv2
import tempfile

import gradio as gr
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from castle.core.cluster import LatentAggregator, auto_generate_cluster_name
from castle.ui.embedding_scatter import EmbeddingScatterPlot
from castle.service.history_service import HistoryManager
from castle.service.session_manager import SessionManager

logger = logging.getLogger(__name__)


# ---------------------------
# Templates & Presets (UI Config)
# ---------------------------

dbscan_config_template = '''{
    "eps": 1.0
}'''


def _transcode_to_h264(video_path: str) -> None:
    """Re-encode *video_path* in-place to H.264 using ffmpeg libx264.

    The file is written to a temporary path first, then atomically
    replaces the original so that a partial failure leaves the mp4v
    file intact.

    Args:
        video_path: Path to an MP4 file written with the mp4v codec.
    """
    import logging

    _log = logging.getLogger(__name__)
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
            _log.warning(
                "ffmpeg H.264 transcode failed for %s (keeping mp4v). stderr: %s",
                video_path,
                result.stderr[-300:],
            )
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    except FileNotFoundError:
        _log.warning("ffmpeg not found — keeping mp4v codec for %s", video_path)
    except Exception as exc:
        _log.warning("H.264 transcode error for %s: %s", video_path, exc)
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ---------------------------
# Event Handlers
# ---------------------------

def _generate_clip(aggregator, center_bin, n_frames=30, fps=15.0):
    """Generate a short H.264 MP4 clip around center_bin."""
    half = n_frames // 2
    start = max(0, center_bin - half)
    end = start + n_frames

    frames = []
    for i in range(start, end):
        frame = aggregator.get_frame(int(i))
        if frame is not None:
            frames.append(frame)

    if not frames:
        return None

    h, w = frames[0].shape[:2]
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(tmp.name, fourcc, fps, (w, h))
    for f in frames:
        bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR) if len(f.shape) == 3 else f
        out.write(bgr)
    out.release()
    _transcode_to_h264(tmp.name)
    return tmp.name


def embedding_plot_click(aggregator, Z_plt, evt: gr.SelectData):
    """
    Handle click on embedding plot.
    aggregator: LatentAggregator instance
    Z_plt: EmbeddingScatterPlot instance
    """
    if hasattr(evt, 'index'):
        emb_plot = Z_plt.click(evt.index[0], evt.index[1])
    else:
        gr.Info('Embedding click failed. Please try clicking on a data point again.')
        return None, None
        
    index = Z_plt.selected_index
    # Generate a short video clip around the selected bin
    clip_path = _generate_clip(aggregator, index)
    
    if clip_path is None:
        # Return fallback blank video if clip generation fails
        return emb_plot, None 
        
    return emb_plot, clip_path


def collapse_accordion():
    return gr.update(open=False)


def update_select_cluster_list(latents):
    from castle.ui.cluster_tree import build_cluster_tree_choices
    
    if latents is None:
        return gr.update(choices=[], value=None)
    
    if not hasattr(latents, 'cluster_meta') or not hasattr(latents, 'cluster'):
        gr.Info('Session not ready yet. Please wait for initialization to complete, then try again.')
        return gr.update(choices=[], value=None)
    
    choices = build_cluster_tree_choices(latents.cluster_meta, latents.cluster)
    return gr.update(choices=choices)


def generate_embedding(latents, cluster_name, cfg_str, progress=gr.Progress()):
    if latents is None:
        gr.Info(
            "Session not initialized. Please click '⚙️ New Session' to initialize "
            "before generating an embedding."
        )
        return None, None, None

    try:
        cfg = json.loads(cfg_str)
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        gr.Info(
            f"Invalid UMAP configuration. Please check the JSON format and try again. "
            f"Details: {e}"
        )
        return None, None, None

    local_latents = latents.select(selected_cluster=cluster_name)
    if len(local_latents.data) == 0:
        gr.Info(
            "This cluster has no data points. Select a different cluster or run "
            "clustering again with adjusted parameters."
        )
        return None, None, None
    
    # C-05: Report progress between UMAP stages
    def umap_progress(stage, total):
        progress(stage / total, desc=f"UMAP Stage {stage + 1}/{total}...")

    local_latents.build_embedding(cfg, progress_callback=umap_progress)

    progress(1.0, desc="Building plot...")
    Z_plt = EmbeddingScatterPlot(local_latents)
    return local_latents, Z_plt, Z_plt.plot()


def generate_local_cluster(local_latents, eps, history, progress=gr.Progress()):
    if local_latents is None:
        gr.Info(
            "No embedding available. Please click 'Generate Embedding' to run UMAP first."
        )
        return None, None, history

    try:
        cfg = json.loads(dbscan_config_template)
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        gr.Info(
            f"Invalid cluster configuration JSON. Please check the template format. "
            f"Details: {e}"
        )
        return None, None, history

    if history is None:
        history = HistoryManager()

    cfg['eps'] = eps
    # Save pre-clustering state (only if we have something to restore to)
    if hasattr(local_latents, 'embedding') and hasattr(local_latents, 'cluster'):
        history.save_state(local_latents, f"DBSCAN clustering (eps={eps})")
    
    progress(0, desc="Running DBSCAN...")
    local_latents.build_cluster(method='dbscan', configs=cfg)
    progress(1.0, desc="Building plot...")
    Z_plt = EmbeddingScatterPlot(local_latents)
    return Z_plt, Z_plt.plot(), history


def label_local_cluster(local_latents, cluster_id, cluster_name, history):
    if not cluster_name:
        gr.Info('Please enter a name for the cluster before clicking Enter.')
        return history

    if history is None:
        history = HistoryManager()

    # Only save state if cluster exists
    if hasattr(local_latents, 'cluster') and hasattr(local_latents, 'embedding'):
        history.save_state(local_latents, f"Label cluster {cluster_id} as {cluster_name}")
    
    local_latents.label_cluster(cluster_id, cluster_name)
    gr.Info(f'Named {cluster_id} as {cluster_name}')
    return history


def plot_syllables_per_video(latents, aggregator):
    """Plot syllables with one video per row, x-axis in seconds."""
    from matplotlib.patches import Patch

    cluster = latents.cluster
    cluster_meta = latents.cluster_meta
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


def label_all_and_submit(storage_path, project_name, latents, local_latents, aggregator, parent_name, history):
    """Auto-label all clusters and submit."""
    if history is None:
        history = HistoryManager()

    if not hasattr(local_latents, 'cluster'):
        gr.Warning(
            "No clusters available to submit. Please click 'Generate Cluster' to "
            "create clusters before submitting."
        )
        return (None, None, None, None, None, None, None, None, history)

    # Save state including parent (for undo of submit)
    if hasattr(local_latents, 'embedding'):
        history.save_state(local_latents, "Submit all clusters to parent", parent=latents)

    unique_clusters = np.unique(local_latents.cluster)

    count = 0
    for cluster_id in unique_clusters:
        if cluster_id == -1:
            continue

        cluster_name = auto_generate_cluster_name(parent_name, cluster_id)
        local_latents.label_cluster(cluster_id, cluster_name)
        count += 1

    gr.Info(f'Auto-labeled {count} clusters.')

    result = import_info_from_local_latent(storage_path, project_name, latents, local_latents, aggregator)
    
    # At the end of label_all_and_submit, after successful submit:
    mgr = SessionManager(storage_path, project_name)
    active_id = mgr.get_active_session_id()
    if active_id:
        mgr.snapshot_to_session(active_id)
        n_clusters = len([k for k in latents.cluster_meta if latents.cluster_meta[k]['name'] != 'init'])
        mgr.save_session_state(active_id, n_clusters)
    
    return result + (history,)


def check_session_exists(storage_path, project_name):
    """Check for existing sessions using SessionManager."""
    if project_name is None:
        return None
    mgr = SessionManager(storage_path, project_name)
    
    # Try legacy migration first
    mgr.migrate_legacy()
    
    sessions = mgr.list_sessions()
    if not sessions:
        return None
    
    return {
        'sessions': sessions,
        'count': len(sessions),
        'latest': sessions[0],  # sorted by updated_at desc
    }


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

    except Exception:
        logger.exception("Failed to restore embedding from %s", npz_path)
        return None, None


def restore_session(storage_path, project_name, select_roi_id, bin_size, select_model, session_id=None):
    """Restore latents from saved CSV files and optionally restore UMAP embedding."""
    _empty = (None, None, None, None, None, None, None, None)
    if project_name is None:
        return _empty

    def notify_callback(msg, level="info"):
        if level == "error":
            gr.Warning(msg)
        else:
            gr.Info(msg)

    try:
        return _do_restore_session(storage_path, project_name, select_roi_id, bin_size, select_model, session_id, notify_callback)
    except Exception as e:
        gr.Warning(
            f"Failed to restore session. Your saved session files may be missing or "
            f"corrupted. Try initializing a new session instead. Details: {e}"
        )
        return _empty


def _do_restore_session(storage_path, project_name, select_roi_id, bin_size, select_model, session_id, notify_callback):
    """Inner restore logic."""
    mgr = SessionManager(storage_path, project_name)
    session_info = None
    if session_id:
        session_info = mgr.get_session(session_id)
        mgr.activate_session(session_id)
    else:
        sessions = mgr.list_sessions()
        if sessions:
            session_info = sessions[0]
            mgr.activate_session(sessions[0].session_id)

    # Use session's saved parameters if available, fall back to UI values
    if session_info:
        select_model = session_info.model or select_model
        select_roi_id = str(session_info.roi_id) if session_info.roi_id else select_roi_id
        bin_size = session_info.bin_size if session_info.bin_size else bin_size

    aggregator = LatentAggregator(
        storage_path, project_name, select_roi_id, int(bin_size),
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
        gr.Info(f'Failed to import cluster results into the session. Details: {e}')
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
        latents = aggregator.get_latent_object()
        
        # After creating aggregator successfully:
        mgr = SessionManager(storage_path, project_name)
        mgr.create_session(
            model=select_model,
            roi_id=int(select_roi_id) if select_roi_id else 1,
            bin_size=int(bin_size),
            total_frames=len(aggregator.latents) if aggregator.latents is not None else 0,
        )
        
        session_info = check_session_exists(storage_path, project_name)
        return aggregator, latents, session_info
    except Exception as e:
        gr.Warning(
            f"Session initialization failed. Please ensure latent features have been "
            f"extracted (Step 3) and the ROI ID is correct. Details: {e}"
        )
        return None, None, None


# ---------------------------
# Undo / Redo Handlers
# ---------------------------

def handle_undo(local_latents, latents, history):
    """Undo the last clustering operation and redraw the plot."""
    if history is None or not history.can_undo:
        gr.Info("Nothing to undo — no recorded actions yet.")
        return gr.update(), gr.update(), history, _history_status(history), gr.update()

    desc = history.undo(local_latents, parent=latents)

    # Check if restored state is valid before plotting
    if not hasattr(local_latents, 'cluster') or not hasattr(local_latents, 'embedding'):
        gr.Info("Cannot undo: no valid previous state found.")
        return gr.update(), gr.update(), history, _history_status(history), gr.update()
    
    gr.Info(f"Undone: {desc}")

    Z_plt = EmbeddingScatterPlot(local_latents)
    
    # Refresh cluster tree from parent latents (may have been restored too)
    from castle.ui.cluster_tree import build_cluster_tree_choices
    tree_update = gr.update()
    if latents is not None and hasattr(latents, 'cluster_meta') and hasattr(latents, 'cluster'):
        choices = build_cluster_tree_choices(latents.cluster_meta, latents.cluster)
        tree_update = gr.update(choices=choices)
    
    return Z_plt, Z_plt.plot(), history, _history_status(history), tree_update


def handle_redo(local_latents, latents, history):
    """Redo the last undone clustering operation and redraw the plot."""
    if history is None or not history.can_redo:
        gr.Info("Nothing to redo — no undone actions available.")
        return gr.update(), gr.update(), history, _history_status(history), gr.update()

    desc = history.redo(local_latents, parent=latents)

    # Check if restored state is valid before plotting
    if not hasattr(local_latents, 'cluster') or not hasattr(local_latents, 'embedding'):
        gr.Info("Cannot redo: no valid next state found.")
        return gr.update(), gr.update(), history, _history_status(history), gr.update()
    
    gr.Info(f"Redone: {desc}")

    Z_plt = EmbeddingScatterPlot(local_latents)
    
    # Refresh cluster tree from parent latents
    from castle.ui.cluster_tree import build_cluster_tree_choices
    tree_update = gr.update()
    if latents is not None and hasattr(latents, 'cluster_meta') and hasattr(latents, 'cluster'):
        choices = build_cluster_tree_choices(latents.cluster_meta, latents.cluster)
        tree_update = gr.update(choices=choices)
    
    return Z_plt, Z_plt.plot(), history, _history_status(history), tree_update


def _history_status(history):
    """Return a human-readable status string for the history UI."""
    if history is None:
        return ""
    parts = []
    if history.can_undo:
        parts.append(f"Undo: {history.undo_description}")
    if history.can_redo:
        parts.append("Redo available")
    return " | ".join(parts) if parts else "No history"


def update_history_buttons(history):
    """Return interactive states for undo/redo buttons + status text."""
    if history is None:
        return gr.update(interactive=False), gr.update(interactive=False), ""
    return (
        gr.update(interactive=history.can_undo),
        gr.update(interactive=history.can_redo),
        _history_status(history),
    )


# ---------------------------
# Auto-Cluster Handler
# ---------------------------

def run_auto_cluster(storage_path, project_name, latents, mulvideo, max_depth, min_frames, progress=gr.Progress()):
    """Run recursive hierarchical auto-clustering.

    Args:
        storage_path: Root storage path.
        project_name: Project name.
        latents: LatentAggregator instance (holds global latents).
        mulvideo: LatentAggregator instance (aggregator).
        max_depth: Maximum recursion depth.
        min_frames: Minimum frames per cluster to continue splitting.
        progress: Gradio progress tracker.

    Returns:
        (syllables_plot, cluster_tree_radio, behavior_id_csv,
         behavior_time_series_csv, behavior_time_series_srt,
         local_embedding_plot, embedding_plot, display_eps, status_str)
    """
    _empty = (None, None, None, None, None, None, None, None, "**❌ Auto-cluster failed.**")

    if latents is None or mulvideo is None:
        gr.Warning(
            "Session not initialized. Please click '⚙️ New Session' and initialize "
            "before running Auto-Cluster."
        )
        return _empty

    from castle.service.clustering_service import ClusteringSession

    progress(0.0, desc="Initialising Auto-Cluster …")

    def _progress_cb(msg: str):
        gr.Info(msg)

    try:
        session = ClusteringSession(
            storage_path,
            project_name,
            roi=getattr(mulvideo, 'roi_id', 1),
            bin_size=getattr(mulvideo, 'bin_size', 1),
            model=getattr(mulvideo, 'model_name', 'dinov3_vitb16'),
        )
        # Copy existing cluster state into the service session so it starts
        # from whatever labels are already assigned.
        session.latents = latents
        session.aggregator = mulvideo

        progress(0.1, desc="Running recursive Auto-Cluster …")
        result = session.auto_cluster(
            cluster_name='init',
            max_depth=int(max_depth),
            min_frames=int(min_frames),
            progress_callback=_progress_cb,
        )
    except Exception as exc:
        logger.exception("Auto-cluster failed")
        gr.Warning(
            f"Auto-clustering failed. Try reducing 'Max Depth' or increasing "
            f"'Min Frames' and try again. Details: {exc}"
        )
        return _empty

    # After clustering, commit results back
    progress(0.85, desc="Committing results …")
    try:
        # Build a dummy local_latents from root selection so we can call
        # import_info_from_local_latent to generate files.
        local_latents_root = latents.select(selected_cluster='init')
        commit_result = import_info_from_local_latent(
            storage_path, project_name, latents, local_latents_root, mulvideo
        )
    except Exception as exc:
        logger.exception("Auto-cluster commit failed")
        gr.Warning(
            f"Auto-clustering succeeded but saving results failed. Your cluster "
            f"assignments may not be saved — try clicking 'Submit' manually. Details: {exc}"
        )
        commit_result = (None, None, None, None, None, None, None, None)

    progress(1.0, desc="Done!")

    n_leaves = result.get('leaf_count', '?')
    status = (
        f"**✅ Auto-Cluster complete!** "
        f"{n_leaves} leaf clusters, max_depth={max_depth}, min_frames={min_frames}."
    )

    gr.Info(f"Auto-cluster finished: {n_leaves} leaf clusters.")
    return commit_result + (status,)


# ---------------------------
# Save / Apply Cluster Model Handlers
# ---------------------------

def save_cluster_model(storage_path, project_name):
    """Save the current project's cluster model to a .npz file.

    Returns:
        (gr.File update, status_markdown)
    """
    if not storage_path or not project_name:
        return gr.update(value=None, visible=False), "**❌ No project selected.**"

    import os
    from castle.service.clustering_service import save_project_cluster_model

    project_path = os.path.join(storage_path, project_name)
    try:
        model_path = save_project_cluster_model(
            project_path=project_path,
            model_name=project_name,
        )
        gr.Info(f"Cluster model saved: {os.path.basename(model_path)}")
        return gr.update(value=model_path, visible=True), f"**✅ Model saved:** `{model_path}`"
    except FileNotFoundError as exc:
        gr.Warning(
            "Cluster model files not found. Please complete the clustering step and "
            "submit clusters before saving a model."
        )
        return gr.update(value=None, visible=False), f"**❌ Save failed:** {exc}"
    except Exception as exc:
        logger.exception("save_cluster_model failed")
        gr.Warning(
            f"Failed to save cluster model. Check that the project folder is "
            f"accessible and clustering has been completed. Details: {exc}"
        )
        return gr.update(value=None, visible=False), f"**❌ Save failed:** {exc}"


def apply_cluster_model(storage_path, project_name, model_file):
    """Apply a saved cluster model (.npz) to the current project.

    Args:
        storage_path: Root storage path.
        project_name: Target project name.
        model_file: Path to the saved model .npz (from gr.File).

    Returns:
        status_markdown
    """
    if not storage_path or not project_name:
        return "**❌ No project selected.**"
    if not model_file:
        return "**❌ No model file provided.** Upload a cluster_model.npz."

    import os
    from castle.service.clustering_service import apply_cluster_model_to_project

    project_path = os.path.join(storage_path, project_name)
    # model_file from gr.File is the temp path string
    model_path = model_file if isinstance(model_file, str) else model_file.name
    try:
        result = apply_cluster_model_to_project(
            model_path=model_path,
            project_path=project_path,
        )
        n = result.get('n_frames', '?')
        gr.Info(f"Cluster model applied: {n} frames classified.")
        return f"**✅ Model applied!** {n} frames classified. Output written to `{result.get('output_csv', '')}`"
    except FileNotFoundError as exc:
        gr.Warning(
            "Cluster model file not found. Please upload a valid cluster_model.npz file."
        )
        return f"**❌ Apply failed:** {exc}"
    except Exception as exc:
        logger.exception("apply_cluster_model failed")
        gr.Warning(
            f"Failed to apply cluster model. Ensure the model file matches the "
            f"project's feature type and dimensions. Details: {exc}"
        )
        return f"**❌ Apply failed:** {exc}"
