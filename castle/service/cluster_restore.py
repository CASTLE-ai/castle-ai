"""Restore a clustering session / local latent from on-disk artifacts.

Extracted from clustering_service (god-module split). These rebuild in-memory
clustering state (LatentAggregator, Latent, LocalLatent, embedding) from a saved
session's id.csv / time_series CSVs / cluster_*.npz. Self-contained: nothing here
depends on clustering_service internals, so clustering_service re-exports these
names for backward compatibility.
"""

import os
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Any, Tuple

import numpy as np
import pandas as pd

from castle.core.logging_config import setup_logger
from castle.core.types import CastleDataError
from castle.service.session_manager import SessionManager
from castle.service.cluster_npz import (
    find_latest_cluster_npz,
    _extract_child_names_from_filename,
)
from castle.utils.latent_explorer import LocalLatent

logger = setup_logger(__name__)


def restore_local_latent_from_npz(
    npz_path: str,
    latents: Any,
    parent_cluster_name: Optional[str] = None,
) -> Tuple[Optional[Any], Optional[np.ndarray]]:
    """Reconstruct a :class:`LocalLatent` from a saved cluster ``.npz``.

    The npz holds three arrays:

    * ``emb``    — ``(N, 2)`` embedding with NaN for non-selected rows.
    * ``cls``    — ``(N,)`` integer labels with ``-1`` for non-selected rows.
    * ``config`` — UMAP config used to produce ``emb``.

    Args:
        npz_path: Path to the saved ``cluster_*.npz``.
        latents: Parent :class:`castle.utils.latent_explorer.Latent`
            object (provides ``data``, ``used_palette``, ``device``,
            ``cluster``, ``cluster_meta``).
        parent_cluster_name: Name of the parent node (e.g. ``'init'``).
            When supplied, the filename is parsed as a fallback to recover
            the original child names for any local cluster IDs whose current
            global counterpart has since been evicted from ``cluster_meta``
            due to deeper splits.

    Returns:
        ``(local_latents, embedding_array)`` or ``(None, None)`` on
        failure. ``embedding_array`` is the ``(M, 2)`` masked embedding
        ready to hand to ``EmbeddingScatterPlot``.

    Notes:
        The function is intentionally exception-tolerant — clustering
        sessions sometimes carry partially-written npz files from
        crashed runs, and restoring a session should fall back to "no
        embedding restored" rather than refusing to open the UI.
    """
    try:
        data = np.load(npz_path, allow_pickle=True)
        # Other npz artefacts (e.g. cluster_model.npz from save_cluster_model)
        # live in the same directory but use a different schema. Bail out
        # quietly rather than logging a full traceback.
        required_keys = {'emb', 'cls', 'config'}
        if not required_keys.issubset(set(data.files)):
            logger.debug(
                "Skipping %s: missing required keys (have %s)",
                npz_path, list(data.files),
            )
            return None, None
        emb_full = data['emb']
        cls_full = data['cls']
        config = data['config']

        valid_mask = ~np.isnan(emb_full[:, 0])
        masked_emb = emb_full[valid_mask]
        masked_cls = cls_full[valid_mask]

        # latents.data is the FULL latent (N rows); emb_full is the LOCAL
        # subset (M rows, M ≤ N). valid_mask has M entries, so indexing
        # latents.data with it fails when N ≠ M.  Use the embedding itself
        # as a stand-in when the sizes do not match.
        local_data = (
            latents.data[valid_mask]
            if hasattr(latents, 'data') and latents.data.shape[0] == emb_full.shape[0]
            else masked_emb
        )
        local_latents = LocalLatent(
            data=local_data,
            index_mask=valid_mask,
            color_avoid=latents.used_palette,
            device=latents.device,
        )
        local_latents.embedding = masked_emb
        local_latents.cluster = masked_cls
        local_latents.configs = config.tolist() if hasattr(config, 'tolist') else config

        # Step 1: try to recover historic child names from the filename.
        # The file is named cluster_{c1}_{c2}_..._{ck}_.npz in submission
        # order (c_i corresponds to local cluster ID i).  This gives us the
        # correct names even when deeper splits have evicted the original
        # children from cluster_meta.
        basename = os.path.basename(npz_path)
        filename_child_names: List[str] = []
        if parent_cluster_name:
            filename_child_names = _extract_child_names_from_filename(
                basename, parent_cluster_name,
            )

        # Step 2: build export — prefer filename-derived historic names;
        # fall back to current cluster_meta for any ID not covered.
        # Build a name→color lookup from current cluster_meta so we can
        # assign colours to historic clusters whose descendants are still live.
        name_to_color: Dict[str, str] = {
            meta['name']: meta['color']
            for meta in latents.cluster_meta.values()
        }

        def _find_color_for_historic(child_name: str) -> str:
            """Return color for a historic cluster, walking to descendants."""
            if child_name in name_to_color:
                return name_to_color[child_name]
            prefix = child_name + '_'
            for nm, col in name_to_color.items():
                if nm.startswith(prefix) and col:
                    return col
            return ''  # engine default — resolved live by name at render time

        # The filename encodes child names in sorted-cluster-ID order (see the
        # submit() writer). Re-pair them to cluster IDs by that order, NOT by
        # using the cluster ID as a positional index: labelled clusters can be
        # non-contiguous (e.g. 0 and 2), which previously attached the wrong
        # historic name to the wrong cluster.
        nonnoise_ids = sorted(int(c) for c in np.unique(masked_cls) if c != -1)
        name_by_id = {
            cid: filename_child_names[k]
            for k, cid in enumerate(nonnoise_ids)
            if k < len(filename_child_names)
        }

        for cid_local in np.unique(masked_cls):
            if cid_local == -1:
                continue
            cid_local = int(cid_local)
            # Prefer historic name from filename when available.
            if cid_local in name_by_id:
                child_name = name_by_id[cid_local]
                local_latents.export[cid_local] = {
                    'name': child_name,
                    'color': _find_color_for_historic(child_name),
                }
                continue
            # Fallback: map via current global cluster_meta.
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
        return local_latents, masked_emb
    except Exception:
        logger.exception("Failed to restore local latent from %s", npz_path)
        return None, None


@dataclass
class RestoredSessionArtifacts:
    """Pure result of restoring a clustering session from disk."""
    aggregator: Any
    latents: Any
    syllables_fig: Any
    cluster_choices: List[Tuple[str, int]]
    id_csv_path: str
    time_series_paths: List[str]
    local_latents: Optional[Any]
    embedding_array: Optional[np.ndarray]


def restore_session_from_disk(
    storage_path: str,
    project_name: str,
    *,
    select_roi_id: Any,
    bin_size: Any,
    select_model: str,
    session_id: Optional[str] = None,
    notify: Optional[Callable[[str, str], None]] = None,
) -> RestoredSessionArtifacts:
    """Restore a clustering session — Gradio-free version of ``_do_restore_session``.

    Replaces the dual responsibility of "build LatentAggregator + reload
    cluster_meta from id.csv + reload assignments from per-video CSVs +
    optionally restore UMAP embedding from npz" with a single typed call.

    Args:
        storage_path: Root storage directory.
        project_name: Project to restore.
        select_roi_id: ROI ID currently selected in the UI (may be
            overridden by the session's stored value).
        bin_size: Bin size from UI (may be overridden).
        select_model: Model name from UI (may be overridden).
        session_id: Explicit session to restore; if None, picks the most
            recently updated.
        notify: ``(msg, level)`` callback for LatentAggregator progress
            messages.

    Returns:
        :class:`RestoredSessionArtifacts` ready to map into Gradio state.
    """
    from castle.core.cluster import LatentAggregator
    from castle.service.plotting_service import plot_syllables_per_video
    from castle.ui.cluster_handlers import update_select_cluster_list

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

    prepare_id: Optional[str] = None
    k_prime: Optional[int] = None
    if session_info:
        select_model = session_info.model or select_model
        select_roi_id = (str(session_info.roi_id)
                         if session_info.roi_id else select_roi_id)
        bin_size = session_info.bin_size if session_info.bin_size else bin_size
        # Prepared sessions MUST rebuild the prepared aggregator — otherwise the
        # legacy path tries to load the raw latents (e.g. 362 GB) and produces a
        # bin axis that does not match the saved per-window labels.
        prepare_id = getattr(session_info, "prepare_id", None)
        k_prime = getattr(session_info, "k_prime", None)

    aggregator = LatentAggregator(
        storage_path, project_name, select_roi_id, int(bin_size),
        model_name=select_model,
        notify=notify,
        prepare_id=prepare_id,
        k_prime=k_prime,
    )
    latents = aggregator.get_latent_object()

    cluster_path = os.path.join(storage_path, project_name, 'cluster')

    id_csv_path = os.path.join(cluster_path, 'id.csv')
    df2_paths: List[str] = []
    if os.path.exists(id_csv_path):
        id_df = pd.read_csv(id_csv_path)
        for _, row in id_df.iterrows():
            cluster_id = int(row['Id'])
            # Empty (engine-default) colour cell reads back as NaN -> coerce to ''.
            raw = row.get('Color', '')
            color = raw if isinstance(raw, str) and raw.strip().lower() != 'nan' else ''
            latents.cluster_meta[cluster_id] = {'name': row['Name'], 'color': color}
            latents.behavior_name2cluster_id[row['Name']] = cluster_id
            if color and color != 'grey':
                latents.used_palette.add(color)
        latents.num_cluster = len(id_df)

        # Prepared sessions: a datapoint is a decimated window, not a uniform bin,
        # so recover per-window GLOBAL labels by sampling the authoritative
        # original-frame CSV through the window map (the npz only holds per-submit
        # LOCAL labels). Legacy sessions keep the historical bin downsample.
        fim = getattr(aggregator, "frame_index_map", None)
        prepared = bool(getattr(aggregator, "_prepared", False)) and fim is not None
        cum = 0
        for video_idx, (vn, v) in enumerate(aggregator.videos_meta):
            video_basename = os.path.splitext(os.path.basename(v))[0]
            ts_path = os.path.join(cluster_path, f'time_series_{video_basename}.csv')
            if os.path.exists(ts_path):
                ts_df = pd.read_csv(ts_path)
                behavior = ts_df['behavior'].values
                if prepared:
                    assert fim is not None  # `prepared` implies a window map
                    bin_clusters = fim.windowed_labels_from_orig(behavior, video_idx)
                else:
                    bin_clusters = behavior[::latents.time_window][:vn]
                if len(bin_clusters) != vn:
                    raise CastleDataError(
                        f"Session restore: {os.path.basename(ts_path)} downsamples "
                        f"to {len(bin_clusters)} bins but video {v!r} expects {vn}. "
                        f"The time_series CSV is likely truncated/corrupt. Assigning "
                        f"it would mislabel this and every subsequent video — refusing. "
                        f"Re-save the session or delete the corrupt CSV and re-cluster."
                    )
                if not np.isfinite(np.asarray(bin_clusters, dtype=np.float64)).all():
                    raise CastleDataError(
                        f"Session restore: {os.path.basename(ts_path)} has non-finite "
                        f"(NaN/inf) behavior values; refusing to coerce them into "
                        f"integer cluster labels. Fix or delete the CSV and re-cluster."
                    )
                latents.cluster[cum:cum + vn] = bin_clusters.astype(latents.cluster.dtype)
                df2_paths.append(ts_path)
            cum += vn
    else:
        logger.info(
            "No id.csv at %s — restoring aggregator only; cluster_meta is empty.",
            id_csv_path,
        )

    restored_local_latents: Optional[Any] = None
    embedding_array: Optional[np.ndarray] = None

    npz_path = find_latest_cluster_npz(cluster_path)
    if npz_path:
        restored_local_latents, embedding_array = restore_local_latent_from_npz(
            npz_path, latents,
        )

    fig = plot_syllables_per_video(latents, aggregator)
    choices = update_select_cluster_list(latents)

    return RestoredSessionArtifacts(
        aggregator=aggregator,
        latents=latents,
        syllables_fig=fig,
        cluster_choices=choices,
        id_csv_path=id_csv_path,
        time_series_paths=df2_paths,
        local_latents=restored_local_latents,
        embedding_array=embedding_array,
    )
