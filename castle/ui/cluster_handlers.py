"""
castle/ui/cluster_handlers.py
Event handler functions for the clustering page.

These functions are bound to Gradio button clicks.
They take Gradio state values as input and return updated values.
"""

import logging
import os
import json
from pathlib import Path

import gradio as gr
import numpy as np

from castle.service.history_service import HistoryManager
from castle.service.session_manager import SessionManager

logger = logging.getLogger(__name__)

# Tracks the last temp clip path so it can be cleaned up on the next call.
_last_clip_path: str | None = None


# ---------------------------
# Event Handlers
# ---------------------------

def _generate_clip(aggregator, center_bin, n_frames=30, fps=15.0):
    """Wrapper kept for back-compat — delegates to clip_service."""
    from castle.service.clip_service import generate_clip_with_roi_overlay
    return generate_clip_with_roi_overlay(
        aggregator, center_bin, n_frames=n_frames, fps=fps,
    )


def embedding_plot_click(aggregator, Z_plt, evt: gr.SelectData):
    """
    Handle click on embedding plot.
    aggregator: LatentAggregator instance
    Z_plt: EmbeddingScatterPlot instance
    """
    global _last_clip_path

    # Clean up the previous temp clip now that Gradio has already served it.
    if _last_clip_path is not None:
        try:
            os.unlink(_last_clip_path)
        except OSError:
            pass
        _last_clip_path = None

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

    _last_clip_path = clip_path
    return emb_plot, clip_path


def collapse_accordion():
    return gr.update(open=False)


def update_select_cluster_list(latents):
    """Return ``(html_update, textbox_reset)`` for the tree HTML and selection state.

    The 2-tuple maps to ``(ui['cluster_tree_html'], ui['cluster_tree_select'])``
    in every Gradio output binding.  The textbox reset clears any prior
    selection whenever the tree is rebuilt.
    """
    from castle.ui.cluster_tree import build_cluster_tree_html

    if latents is None:
        return (
            gr.update(value="<em style='color:#888'>No session.</em>"),
            gr.update(value=""),
        )

    if not hasattr(latents, 'cluster_meta') or not hasattr(latents, 'cluster'):
        gr.Info('Session not ready yet. Please wait for initialization to complete, then try again.')
        return (
            gr.update(value=""),
            gr.update(value=""),
        )

    html = build_cluster_tree_html(latents.cluster_meta, latents.cluster)
    return gr.update(value=html), gr.update(value="")


def generate_embedding(
    latents,
    cluster_name,
    cfg_str,
    umap_seed_str: str = "",
    storage_path: str | None = None,
    project_name: str | None = None,
    progress=gr.Progress(),
):
    """Thin Gradio wrapper around
    :func:`castle.service.clustering_service.run_umap_on_cluster`.

    Returns ``(local_latents, scatter_plot, plot_image, status_md)``.
    """
    from castle.core.types import InsufficientDataError
    from castle.service.clustering_service import run_umap_on_cluster
    from castle.service.plotting_service import build_scatter_plot

    if latents is None:
        gr.Info(
            "Session not initialized. Please click '⚙️ New Session' to initialize "
            "before generating an embedding."
        )
        return None, None, None, ""

    try:
        cfg = json.loads(cfg_str)
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        gr.Info(
            f"Invalid UMAP configuration. Please check the JSON format and try again. "
            f"Details: {e}"
        )
        return None, None, None, ""

    base_seed: int | None = None
    if isinstance(umap_seed_str, str) and umap_seed_str.strip():
        try:
            base_seed = int(umap_seed_str.strip())
        except ValueError:
            gr.Info(
                f"UMAP seed must be an integer; got {umap_seed_str!r}. "
                f"Leave blank to re-roll, or enter an integer to lock."
            )
            return None, None, None, ""

    # When locking to a specific seed, strip any pre-existing random_state
    # from the dicts so the service's seed-injection path takes priority.
    if base_seed is not None:
        if isinstance(cfg, list):
            cfg = [(dict(c) if isinstance(c, dict) else c) for c in cfg]
            for c in cfg:
                if isinstance(c, dict):
                    c.pop('random_state', None)
        elif isinstance(cfg, dict):
            cfg = dict(cfg)
            cfg.pop('random_state', None)

    log_path = _resolve_umap_log_path(storage_path, project_name)

    def umap_progress(stage, total):
        progress(stage / total, desc=f"UMAP Stage {stage + 1}/{total}...")

    try:
        result = run_umap_on_cluster(
            latents, cluster_name, cfg,
            base_seed=base_seed,
            progress_callback=umap_progress,
            log_path=log_path,
        )
    except InsufficientDataError as e:
        gr.Info(str(e))
        return None, None, None, ""
    except Exception as e:
        # UMAP itself failed (e.g. n_neighbors >= n_samples at the cuML
        # boundary). Surface a friendly hint based on the first cfg.
        first_cfg = cfg[0] if isinstance(cfg, list) else cfg
        n_neighbors = (
            first_cfg.get('n_neighbors', '?')
            if isinstance(first_cfg, dict) else '?'
        )
        err_str = str(e).lower()
        if 'n_neighbors' in err_str or ('larger' in err_str and 'sample' in err_str) or not err_str:
            gr.Warning(
                f"UMAP failed: n_neighbors ({n_neighbors}) may be too large. "
                f"Try reducing it. Details: {e or type(e).__name__}"
            )
        else:
            gr.Warning(f"UMAP failed: {e or type(e).__name__}")
        return None, None, None, ""

    gr.Info(f"UMAP done. seed={result.resolved_seeds[0]}. Re-run to reproduce.")
    progress(1.0, desc="Building plot...")
    Z_plt, img = build_scatter_plot(result.local_latents)
    return result.local_latents, Z_plt, img, f"✅ **{result.status_text}**"


def _resolve_umap_log_path(storage_path, project_name) -> str | None:
    """Build the umap_log.jsonl path for the active session, if any.

    Returns None when either input is empty or no active session exists.
    """
    if not storage_path or not project_name:
        return None
    try:
        mgr = SessionManager(storage_path, project_name)
        session_id = mgr.get_active_session_id()
        if not session_id:
            return None
        return os.path.join(
            mgr.sessions_path, session_id, "umap_log.jsonl"
        )
    except Exception as exc:
        logger.debug("Could not build umap_log.jsonl path: %s", exc)
        return None


def generate_local_cluster(local_latents, eps, history, progress=gr.Progress()):
    """Thin Gradio wrapper around
    :func:`castle.service.clustering_service.run_dbscan_on_local`.
    """
    from castle.core.types import InsufficientDataError
    from castle.service.clustering_service import run_dbscan_on_local
    from castle.service.plotting_service import build_scatter_plot

    if history is None:
        history = HistoryManager()

    if hasattr(local_latents, 'embedding') and hasattr(local_latents, 'cluster'):
        history.save_state(local_latents, f"DBSCAN clustering (eps={eps})")

    progress(0, desc="Running DBSCAN...")
    try:
        run_dbscan_on_local(local_latents, float(eps))
    except InsufficientDataError as e:
        gr.Info(str(e))
        return None, None, history
    progress(1.0, desc="Building plot...")
    Z_plt, img = build_scatter_plot(local_latents)
    return Z_plt, img, history


def label_local_cluster(local_latents, cluster_id, cluster_name, history):
    if local_latents is None:
        gr.Warning('No embedding available. Please generate an embedding and cluster before labelling.')
        return history

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
    """Wrapper kept for back-compat — delegates to plotting_service."""
    from castle.service.plotting_service import plot_syllables_per_video as _impl
    return _impl(latents, aggregator)


def on_tree_node_select(node_name, latents, storage_path, project_name):
    """Restore prior UMAP/eps/preset/seed state when a cluster tree node
    is selected.

    Resolution order:
        1. ``cluster/node_{name}_meta.json`` sidecar (new submissions).
        2. Legacy fallback: pick the ``cluster_*.npz`` whose export names
           are immediate children of ``node_name`` and rebuild from there.

    Outputs (in order, matching the .change binding):
        umap_config_text, eps, embedding_plot_image, local_latents_state,
        overwrite_state (always reset to False), submit_status_md,
        preset_dropdown, umap_seed_textbox.
    """
    from castle.service.clustering_service import (
        load_node_meta, restore_local_latent_from_npz,
        find_cluster_npz_for_parent,
    )
    from castle.service.plotting_service import build_named_scatter_plot

    no_update = (
        gr.update(), gr.update(), gr.update(), gr.update(),
        False, gr.update(), gr.update(), gr.update(),
    )
    if not node_name or not storage_path or not project_name:
        return no_update

    cluster_path = os.path.join(storage_path, project_name, 'cluster')

    meta = load_node_meta(cluster_path, node_name)
    npz_path: str | None = None
    if meta is not None and meta.get('embedding_npz'):
        candidate = os.path.join(cluster_path, meta['embedding_npz'])
        if os.path.exists(candidate):
            npz_path = candidate

    # Legacy fallback: no sidecar (or its npz is missing) → infer from
    # cluster_*.npz files whose export names match this parent.
    if npz_path is None and latents is not None:
        npz_path = find_cluster_npz_for_parent(cluster_path, node_name, latents)

    if meta is None and npz_path is None:
        # Fresh / un-submitted node — leave settings alone, clear status.
        return (
            gr.update(), gr.update(), gr.update(), gr.update(),
            False,
            gr.update(value=""),
            gr.update(), gr.update(),
        )

    umap_cfg_update = gr.update()
    eps_update = gr.update()
    preset_update = gr.update()
    seed_update = gr.update()

    if meta is not None:
        if meta.get('umap_config'):
            umap_cfg_update = gr.update(value=meta['umap_config'])
        if meta.get('eps') is not None:
            eps_update = gr.update(value=meta['eps'])
        if meta.get('preset'):
            preset_update = gr.update(value=meta['preset'])
        if meta.get('umap_seed') is not None:
            seed_update = gr.update(value=str(meta['umap_seed']))

    emb_img_update = gr.update()
    local_latents_update = gr.update()
    if npz_path and latents is not None:
        local_ll, _ = restore_local_latent_from_npz(npz_path, latents, parent_cluster_name=node_name)
        if local_ll is not None:
            _, named_img = build_named_scatter_plot(local_ll)
            emb_img_update = gr.update(value=named_img)
            local_latents_update = local_ll
            # If we used the legacy fallback (no sidecar), fish the seed
            # out of the rebuilt configs so it still reaches the UI.
            if meta is None:
                cfgs = getattr(local_ll, 'configs', None)
                if cfgs:
                    first = cfgs[0] if isinstance(cfgs, list) else cfgs
                    if isinstance(first, dict) and first.get('random_state') is not None:
                        seed_update = gr.update(value=str(int(first['random_state'])))

    return (
        umap_cfg_update, eps_update, emb_img_update, local_latents_update,
        False,
        gr.update(value=f"📂 Loaded saved state for **{node_name}**."),
        preset_update, seed_update,
    )


def label_all_and_submit(
    storage_path, project_name, latents, local_latents, aggregator,
    parent_name, history,
    umap_config_str="", eps_value=None, overwrite_confirmed=False,
    preset_value=None, umap_seed_str="",
):
    """Thin Gradio wrapper: auto-label every DBSCAN cluster then submit.

    Delegates the auto-label loop to
    :func:`castle.service.clustering_service.auto_label_local_clusters` and
    the submit + persist step to ``import_info_from_local_latent`` (which
    is itself a thin wrapper around ``submit_local_to_global``).

    Returns an 11-tuple appending ``(history, overwrite_confirmed_next,
    submit_status_markdown)`` to the 9 outputs from
    ``import_info_from_local_latent``. When the parent node already has a
    submitted result, the first click returns early with a warning and
    flips ``overwrite_confirmed_next`` to True; a second click goes through.
    """
    from castle.core.types import InsufficientDataError
    from castle.service.clustering_service import (
        auto_label_local_clusters, load_node_meta,
    )

    if history is None:
        history = HistoryManager()

    # Overwrite-confirmation gate: only kicks in when sidecar meta exists.
    cluster_path = os.path.join(storage_path or '', project_name or '', 'cluster')
    existing_meta = load_node_meta(cluster_path, parent_name) if parent_name else None
    if existing_meta is not None and not overwrite_confirmed:
        gr.Warning(
            f"'{parent_name}' already has a submitted clustering result. "
            f"Click Submit again to overwrite it."
        )
        warn_md = (
            f"⚠️ **'{parent_name}' already submitted.** "
            f"Click **Submit** again to overwrite."
        )
        # 9 Nones (outputs of import_info_from_local_latent) + history + True + status
        return (None,) * 9 + (history, True, warn_md)

    if hasattr(local_latents, 'embedding'):
        history.save_state(local_latents, "Submit all clusters to parent", parent=latents)

    try:
        count = auto_label_local_clusters(local_latents, parent_name)
    except InsufficientDataError as e:
        gr.Warning(str(e))
        return (None,) * 9 + (history, False, "")

    gr.Info(f'Auto-labeled {count} clusters.')

    seed_arg: int | None = None
    if isinstance(umap_seed_str, str) and umap_seed_str.strip():
        try:
            seed_arg = int(umap_seed_str.strip())
        except ValueError:
            seed_arg = None

    result = import_info_from_local_latent(
        storage_path, project_name, latents, local_latents, aggregator,
        parent_cluster_name=parent_name,
        umap_config_str=umap_config_str,
        eps_value=eps_value,
        preset_value=preset_value,
        umap_seed=seed_arg,
    )
    if result[0] is None:
        return result + (history, False, "")

    mgr = SessionManager(storage_path, project_name)
    active_id = mgr.get_active_session_id()
    if active_id:
        mgr.snapshot_to_session(active_id)
        n_clusters = len([
            k for k in latents.cluster_meta
            if latents.cluster_meta[k]['name'] != 'init'
        ])
        mgr.save_session_state(active_id, n_clusters)

    return result + (history, False, f"✅ Submitted '{parent_name}'.")


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
    """Wrapper kept for back-compat with existing handler call sites."""
    from castle.service.clustering_service import find_latest_cluster_npz
    return find_latest_cluster_npz(cluster_path)


def _restore_embedding_from_npz(npz_path, latents):
    """Wrapper kept for back-compat — delegates to service + plotting helpers."""
    from castle.service.clustering_service import restore_local_latent_from_npz
    from castle.service.plotting_service import build_scatter_plot

    local_latents, _ = restore_local_latent_from_npz(npz_path, latents)
    if local_latents is None:
        return None, None
    plot, _img = build_scatter_plot(local_latents)
    return local_latents, plot


def restore_session(storage_path, project_name, select_roi_id, bin_size, select_model, session_id=None):
    """Restore latents from saved CSV files and optionally restore UMAP embedding."""
    _empty = (None, None, None, None, None, None, None, None, None)
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
    """Thin Gradio wrapper around
    :func:`castle.service.clustering_service.restore_session_from_disk`.

    Adds the Gradio-side embedding plot rendering (which the service
    layer deliberately doesn't import) plus a couple of ``gr.Info``
    notifications.
    """
    from castle.service.clustering_service import restore_session_from_disk
    from castle.service.plotting_service import build_named_scatter_plot

    artifacts = restore_session_from_disk(
        storage_path, project_name,
        select_roi_id=select_roi_id, bin_size=bin_size, select_model=select_model,
        session_id=session_id, notify=notify_callback,
    )

    restored_Z_plt = None
    restored_emb_img = None
    if artifacts.local_latents is not None:
        restored_Z_plt, restored_emb_img = build_named_scatter_plot(
            artifacts.local_latents,
        )
        gr.Info('Restored UMAP embedding from saved npz.')

    gr.Info(f'Restored session with {artifacts.latents.num_cluster - 1} clusters')

    tree_html_upd, tree_dd_upd = artifacts.cluster_choices
    return (artifacts.aggregator, artifacts.latents,
            artifacts.syllables_fig, tree_html_upd, tree_dd_upd,
            artifacts.id_csv_path, artifacts.time_series_paths,
            restored_Z_plt, restored_emb_img)


def convert_latent_cluster_to_subtitle(storage_path, project_name, latents, aggregator):
    # Delegate to Aggregator
    return aggregator.generate_subtitles(latents.cluster, latents.cluster_meta)


def import_info_from_local_latent(
    storage_path, project_name, latents, local_latents, aggregator,
    parent_cluster_name=None, umap_config_str=None, eps_value=None,
    preset_value=None, umap_seed=None,
):
    """Thin Gradio wrapper around
    :func:`castle.service.clustering_service.submit_local_to_global`.

    Preserves the 9-tuple return shape:
    ``(syllables_fig, cluster_html, cluster_select_reset, id_csv,
       time_series_csvs, srt_paths, local_latents, named_embedding_image,
       embedding_npz_path)``.
    """
    from castle.service.clustering_service import submit_local_to_global
    from castle.service.plotting_service import build_named_scatter_plot

    try:
        artifacts = submit_local_to_global(
            latents, local_latents, aggregator,
            storage_path=storage_path, project_name=project_name,
            parent_cluster_name=parent_cluster_name,
            umap_config_str=umap_config_str,
            eps_value=eps_value,
            preset_value=preset_value,
            umap_seed=umap_seed,
        )
    except Exception as e:
        gr.Info(f'Failed to import cluster results into the session. Details: {e}')
        return (None,) * 9

    tree_html_upd, tree_dd_upd = artifacts.cluster_choices
    if artifacts.embedding_path is None:
        gr.Warning(
            "Embedding not available on local_latents — skipping scatter plot. "
            "Run 'Generate Cluster' with UMAP/t-SNE enabled before submitting."
        )
        return (artifacts.syllables_fig, tree_html_upd, tree_dd_upd,
                artifacts.id_csv_path, artifacts.time_series_paths,
                artifacts.subtitle_paths, None, None, None)

    Z_plt, named_img = build_named_scatter_plot(artifacts.local_latents)
    return (artifacts.syllables_fig, tree_html_upd, tree_dd_upd,
            artifacts.id_csv_path, artifacts.time_series_paths,
            artifacts.subtitle_paths, Z_plt, named_img,
            artifacts.embedding_path)


def init_mulvideo(storage_path, project_name, select_roi_id, bin_size, select_model):
    """Thin Gradio wrapper around
    :func:`castle.service.clustering_service.init_clustering_aggregator`.
    """
    from castle.service.clustering_service import init_clustering_aggregator

    if not project_name:
        return None, None, None

    def notify_callback(msg: str, level: str = "info"):
        if level == "error":
            gr.Warning(msg)
        else:
            gr.Info(msg)

    try:
        artifacts = init_clustering_aggregator(
            storage_path, project_name,
            select_roi_id=select_roi_id, bin_size=bin_size,
            select_model=select_model, notify=notify_callback,
        )
        session_info = check_session_exists(storage_path, project_name)
        return artifacts.aggregator, artifacts.latents, session_info
    except Exception as e:
        gr.Warning(
            f"Session initialization failed. Please ensure latent features have been "
            f"extracted (Step 3) and the ROI ID is correct. Details: {e}"
        )
        return None, None, None


# ---------------------------
# Undo / Redo Handlers
# ---------------------------

def _do_history_step(
    local_latents, latents, history,
    *, can_check, step_callable, verb_past: str,
):
    """Shared core of :func:`handle_undo` / :func:`handle_redo`."""
    if history is None or not can_check(history):
        gr.Info(f"Nothing to {verb_past.lower().rstrip('ne').rstrip('do')}do — no recorded actions yet.")
        return gr.update(), gr.update(), history, _history_status(history), gr.update(), gr.update()

    desc = step_callable(local_latents, parent=latents)

    if not hasattr(local_latents, 'cluster') or not hasattr(local_latents, 'embedding'):
        gr.Info(f"Cannot {verb_past.lower().rstrip('ne').rstrip('do')}do: no valid state found.")
        return gr.update(), gr.update(), history, _history_status(history), gr.update(), gr.update()

    gr.Info(f"{verb_past}: {desc}")

    from castle.service.plotting_service import build_scatter_plot
    from castle.ui.cluster_tree import build_cluster_tree_html

    Z_plt, img = build_scatter_plot(local_latents)
    tree_html = gr.update()
    tree_sel = gr.update()
    if latents is not None and hasattr(latents, 'cluster_meta') and hasattr(latents, 'cluster'):
        html = build_cluster_tree_html(latents.cluster_meta, latents.cluster)
        tree_html = gr.update(value=html)
        tree_sel = gr.update(value="")

    return Z_plt, img, history, _history_status(history), tree_html, tree_sel


def handle_undo(local_latents, latents, history):
    """Undo the last clustering operation and redraw the plot."""
    return _do_history_step(
        local_latents, latents, history,
        can_check=lambda h: h.can_undo,
        step_callable=lambda ll, parent: history.undo(ll, parent=parent),
        verb_past="Undone",
    )


def handle_redo(local_latents, latents, history):
    """Redo the last undone clustering operation and redraw the plot."""
    return _do_history_step(
        local_latents, latents, history,
        can_check=lambda h: h.can_redo,
        step_callable=lambda ll, parent: history.redo(ll, parent=parent),
        verb_past="Redone",
    )


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


def export_representatives(storage_path, project_name, latents, aggregator,
                           n_per_cluster, selection):
    """Export representative frames + montage per cluster as a ZIP download.

    UX-02. Picks N frames per (non-noise, non-"init") labelled cluster,
    writes them as PNG plus a montage, zips the bundle, and returns the
    zip path for Gradio to expose as a download.
    """
    from castle.service.representatives_service import export_cluster_representatives

    if latents is None or aggregator is None:
        gr.Info(
            "Session not initialised. Open a project and click '⚙️ New Session' "
            "before exporting representatives."
        )
        return gr.update(value=None, visible=False), ""

    try:
        n_int = max(1, int(n_per_cluster))
    except (TypeError, ValueError):
        gr.Info(f"Frames per cluster must be a positive integer; got {n_per_cluster!r}")
        return gr.update(value=None, visible=False), ""

    if not isinstance(selection, str) or selection not in ("medoid", "random"):
        gr.Info("Selection must be 'medoid' or 'random'.")
        return gr.update(value=None, visible=False), ""

    cluster_meta = getattr(latents, "cluster_meta", {}) or {}
    labelled = [
        cid for cid, meta in cluster_meta.items()
        if cid != -1 and meta.get("name") != "init"
    ]
    if not labelled:
        gr.Info(
            "No labelled clusters yet. Run UMAP + DBSCAN, label clusters, then "
            "click Export Representatives."
        )
        return gr.update(value=None, visible=False), ""

    import shutil
    import tempfile

    tmpdir = Path(tempfile.mkdtemp(prefix="castle_repr_"))
    try:
        written = export_cluster_representatives(
            latents, aggregator,
            output_dir=tmpdir,
            n_per_cluster=n_int,
            selection=selection,
        )
        if not written:
            gr.Info("No representatives written (clusters may all be empty).")
            return gr.update(value=None, visible=False), ""

        out_root = os.path.join(storage_path, project_name, "cluster", "representatives")
        os.makedirs(out_root, exist_ok=True)
        archive_base = os.path.join(out_root, "representatives")
        zip_path = shutil.make_archive(archive_base, "zip", root_dir=str(tmpdir))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    total_pngs = sum(len(v) for v in written.values())
    gr.Info(
        f"Exported {len(written)} clusters × ~{n_int} frames ({total_pngs} PNGs)."
    )
    status_md = (
        f"✅ **Representatives exported.** "
        f"{len(written)} clusters, {total_pngs} PNG files. "
        f"Bundle: `{zip_path}`."
    )
    return gr.update(value=zip_path, visible=True), status_md
