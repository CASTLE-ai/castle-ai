"""castle/service/cluster_persistence.py

Save / apply of CASTLE cluster-transfer models (longitudinal studies): export a
trained clustering as a portable model and apply it to another project. Extracted
out of the former clustering_service god-module. Heavy deps (cluster_transfer,
prepare, SessionManager) are imported lazily inside the functions, exactly as
before — behaviour is unchanged, only the location.
"""

import json
import os
from typing import Optional

import numpy as np
import pandas as pd

from castle.core.logging_config import setup_logger
from castle.core.types import CastleDataError
from castle.service.cluster_npz import _embedding_npz_files

logger = setup_logger(__name__)


def _save_prepared_cluster_model(project_path, cluster_dir, prepare_id, session_info,
                                 output_path, model_name, k) -> str:
    """Export a transfer model for a prepared (PCA-reduced) session.

    training_features are the per-decimated-frame reduced vectors (k' dims);
    each frame inherits its window's cluster label/embedding. The Prepare
    transform (L2 + PCA basis + k') is bundled so apply() can map a new
    project's RAW latents into the same space. See cluster_transfer.ClusterModel.

    NOTE (scale provenance): the bundled ``raw_feature_dim`` is the *combined*
    width of the SPP scales this cache was built on, but the scale identity is
    NOT carried into the transfer model. apply() only dim-checks. So a model
    built from a SCALE-SUBSET cache (e.g. only 2×2 → 3072-d) cannot be applied to
    a fresh project's full combined latent (16128-d) — it raises on the dim
    mismatch, and there is no scale info to auto-slice. Build transfer-export
    caches on the full scale set (or match the subset width on the new project).
    """
    from castle.core.cluster_transfer import save_cluster_model
    from castle.core.prepare import load_prepare, k_prime_for_variance

    # --- cluster names (id.csv) + per-window emb/cls (most recent npz) ---
    id_csv_path = os.path.join(cluster_dir, "id.csv")
    if not os.path.exists(id_csv_path):
        raise FileNotFoundError(f"No id.csv found: {id_csv_path}")
    id_df = pd.read_csv(id_csv_path)
    cluster_names = {int(r["Id"]): r["Name"] for _, r in id_df.iterrows()}
    emb_files = _embedding_npz_files(cluster_dir)
    if not emb_files:
        raise FileNotFoundError(f"No embedding .npz found in {cluster_dir}")
    emb_data = np.load(emb_files[0], allow_pickle=True)
    emb_full = emb_data["emb"].astype(np.float64)   # (n_windows, 2)
    cls_full = emb_data["cls"].astype(np.int32)     # (n_windows,)
    # Only TRUE DBSCAN members may train the transfer model — k-NN-propagated
    # rows (UMAP subsample) are interpolations, not density memberships, and
    # would make apply() a k-NN over k-NN-smoothed labels. Missing key => all
    # rows are real (legacy / non-subsampled), preserving prior behaviour.
    win_sampled = (emb_data["is_sampled"].astype(bool)
                   if "is_sampled" in emb_data.files else None)

    # --- prepared cache: reduced features + PCA basis ---
    prep_dir = os.path.join(cluster_dir, "prepared", prepare_id)
    pd_obj = load_prepare(prep_dir)
    if pd_obj.pca_components is None:
        raise CastleDataError(
            f"Prepared cache {prepare_id} has no PCA basis (PCA was off, or it "
            f"predates basis persistence). Rebuild the cache with PCA enabled to "
            f"export a transfer model."
        )
    W = int(getattr(session_info, "bin_size", 1) or 1)
    kp = int(getattr(session_info, "k_prime", 0) or 0) or k_prime_for_variance(pd_obj.meta, 0.95)
    kp = max(1, min(kp, pd_obj.width))

    wmap = pd_obj.index_map.for_window(W)
    if wmap.n_windows != len(cls_full):
        raise CastleDataError(
            f"Window count mismatch: cache has {wmap.n_windows} windows but the "
            f"saved embedding has {len(cls_full)}. Re-run clustering on this cache."
        )
    dp_win = wmap.datapoint_window_ids()            # (N_dp,) global window id or -1
    n_dp = int(pd_obj.reduced.shape[0])
    feats = np.asarray(pd_obj.reduced[:, :kp], dtype=np.float32)
    labels = np.full(n_dp, -1, dtype=np.int32)
    emb2 = np.full((n_dp, 2), np.nan, dtype=np.float64)
    valid = dp_win >= 0
    labels[valid] = cls_full[dp_win[valid]]
    emb2[valid] = emb_full[dp_win[valid]]
    keep = valid & np.isfinite(feats).all(axis=1) & np.isfinite(emb2).all(axis=1)
    if win_sampled is not None:
        dp_sampled = np.zeros(n_dp, dtype=bool)
        dp_sampled[valid] = win_sampled[dp_win[valid]]
        keep &= dp_sampled

    transform = {
        "components": pd_obj.pca_components,        # (K_full, D_raw)
        "mean": pd_obj.pca_mean,                    # (D_raw,)
        "normalize": pd_obj.meta.get("normalize", "l2"),
        "k_prime": kp,
        "raw_feature_dim": int(pd_obj.meta.get("n_features", pd_obj.pca_components.shape[1])),
        "scales": pd_obj.meta.get("scales"),  # SPP scale provenance (see ClusterModel.scales)
    }
    fps = float(pd_obj.index_map.raw_fps[0]) if pd_obj.index_map.n_videos else 30.0
    if output_path is None:
        output_path = os.path.join(cluster_dir, "cluster_model.npz")
    return save_cluster_model(
        output_path=output_path,
        umap_embedding=emb2[keep],
        training_features=feats[keep],
        cluster_labels=labels[keep],
        cluster_names=cluster_names,
        model_name=model_name,
        fps=fps,
        k=k,
        transform=transform,
        bin_size=W,
    )


def save_project_cluster_model(
    project_path: str,
    output_path: Optional[str] = None,
    model_name: str = "",
    k: int = 5,
) -> str:
    """Save a project's clustering model for transfer.

    Loads the UMAP embedding, cluster labels, and original latent features
    from the project's ``cluster/`` directory, then persists them as a
    ``.npz`` file that can be applied to new data.

    Args:
        project_path: Absolute path to the project directory.
        output_path: Where to write the model file.  Defaults to
            ``<project_path>/cluster/cluster_model.npz``.
        model_name: Descriptive name saved in the metadata.
        k: Number of neighbours for k-NN at apply time.

    Returns:
        Absolute path to the saved model file.

    Raises:
        FileNotFoundError: If required cluster/embedding files are missing.
    """
    from castle.core.cluster_transfer import save_cluster_model
    import glob

    cluster_dir = os.path.join(project_path, "cluster")
    if not os.path.isdir(cluster_dir):
        raise FileNotFoundError(f"No cluster directory found: {cluster_dir}")

    # Prepared (PCA-reduced) sessions take a dedicated export path: the transfer
    # model bundles the Prepare transform (raw -> L2 -> PCA -> k') so a new
    # project's raw latents can be mapped into the same reduced space at apply
    # time (per-frame; the source's temporal windowing is not re-applied).
    try:
        from castle.service.session_manager import SessionManager
        _mgr = SessionManager(os.path.dirname(project_path), os.path.basename(project_path))
        _sid = _mgr.get_active_session_id()
        _sinfo = _mgr.get_session(_sid) if _sid else None
        _prepare_id = getattr(_sinfo, "prepare_id", None) if _sinfo else None
    except Exception:  # noqa: BLE001 — never block legacy export on a probe error
        _sinfo, _prepare_id = None, None
    if _prepare_id:
        return _save_prepared_cluster_model(
            project_path, cluster_dir, _prepare_id, _sinfo, output_path, model_name, k,
        )

    # --- Load id.csv for cluster names ---
    id_csv_path = os.path.join(cluster_dir, "id.csv")
    if not os.path.exists(id_csv_path):
        raise FileNotFoundError(f"No id.csv found: {id_csv_path}")

    id_df = pd.read_csv(id_csv_path)
    cluster_names = {int(row["Id"]): row["Name"] for _, row in id_df.iterrows()}

    # --- Load embedding .npz (most recently modified, not arbitrary glob order) ---
    emb_files = _embedding_npz_files(cluster_dir)
    if not emb_files:
        raise FileNotFoundError(f"No embedding .npz found in {cluster_dir}")
    emb_path = emb_files[0]
    emb_data = np.load(emb_path, allow_pickle=True)
    emb_full = emb_data["emb"]        # (N, 2) with NaN for masked-out points
    cls_full = emb_data["cls"]        # (N,) with -1 for masked-out points
    # Train the transfer model only on TRUE DBSCAN members; exclude k-NN-
    # propagated rows (UMAP subsample). Missing key => all real (legacy).
    bin_sampled = (emb_data["is_sampled"].astype(bool)
                   if "is_sampled" in emb_data.files else None)

    # --- Load latent features from latent/ directory ---
    latent_dir = os.path.join(project_path, "latent")
    if not os.path.isdir(latent_dir):
        raise FileNotFoundError(f"No latent directory found: {latent_dir}")

    # Pick most-recently-modified model sub-directory (matches user's latest extraction).
    model_dirs = [
        os.path.join(latent_dir, d) for d in os.listdir(latent_dir)
        if os.path.isdir(os.path.join(latent_dir, d))
    ]
    if not model_dirs:
        raise FileNotFoundError(f"No model sub-directories in {latent_dir}")
    model_subdir = max(model_dirs, key=os.path.getmtime)

    # Concatenate latent files in the same order as the project config
    latent_files = sorted(glob.glob(os.path.join(model_subdir, "*.npz")))
    if not latent_files:
        raise FileNotFoundError(f"No latent .npz files in {model_subdir}")

    latent_chunks = []
    for lf in latent_files:
        loaded = np.load(lf)
        latent_chunks.append(loaded["latent"])
    all_features = np.concatenate(latent_chunks, axis=0)

    # The latent .npz rows are per-FRAME, but emb_full / cls_full are per-BIN
    # (the embedding was built from Latent's time_window binning). Naively
    # pairing the i-th FRAME's features with the i-th BIN's label mis-aligns
    # every training example whenever bin_size > 1 (the common case). Instead,
    # label each frame by the bin it belongs to (frame f -> bin f // bin_size)
    # and keep features at frame resolution, so the per-frame apply path matches
    # without a dimension change.
    n_bins = len(emb_full)
    bin_size = max(1, len(all_features) // n_bins) if n_bins > 0 else 1
    n_keep = n_bins * bin_size
    all_features = all_features[:n_keep]
    frame_emb = np.repeat(emb_full, bin_size, axis=0)[:n_keep]
    frame_cls = np.repeat(cls_full, bin_size)[:n_keep]

    # --- Build valid mask (non-NaN embedding rows), at frame resolution ---
    valid_mask = ~np.isnan(frame_emb).any(axis=1)
    if bin_sampled is not None:
        frame_sampled = np.repeat(bin_sampled, bin_size)[:n_keep]
        valid_mask &= frame_sampled
    umap_embedding = frame_emb[valid_mask]
    cluster_labels = frame_cls[valid_mask]
    training_features = all_features[valid_mask]

    if output_path is None:
        output_path = os.path.join(cluster_dir, "cluster_model.npz")

    # Determine fps from project config if available
    fps = 30.0
    config_path = os.path.join(project_path, "castle_config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path) as f:
                cfg = json.load(f)
            fps = cfg.get("fps", fps)
        except Exception:
            pass

    return save_cluster_model(
        output_path=output_path,
        umap_embedding=umap_embedding,
        training_features=training_features,
        cluster_labels=cluster_labels,
        cluster_names=cluster_names,
        model_name=model_name,
        fps=fps,
        k=k,
        bin_size=bin_size,
    )


def apply_cluster_model_to_project(
    model_path: str,
    project_path: str,
    method: str = "knn_feature",
) -> dict:
    """Apply a saved cluster model to a new project's latent features.

    Loads latent features from *project_path*, classifies them with the
    saved model, and writes ``transferred_labels.csv`` + ``id.csv`` into
    the project's ``cluster/`` directory.

    Args:
        model_path: Path to the saved model ``.npz``.
        project_path: Absolute path to the target project directory.
        method: ``"knn_feature"`` or ``"knn_umap"``.

    Returns:
        A dict with ``labels``, ``confidence``, ``cluster_names``,
        ``output_csv``, and ``n_frames``.
    """
    from castle.core.cluster_transfer import load_cluster_model, apply_cluster_model
    import glob

    model = load_cluster_model(model_path)

    # Transfer probe: the model was trained on bin_size-aggregated features, but
    # apply runs on the target's PER-FRAME latents. A bin_size>1 model applied
    # per-frame is a feature-distribution mismatch — warn so a degraded transfer
    # is not silently accepted as a valid result (scientific-output correctness).
    model_bin = int(getattr(model, "bin_size", 1) or 1)
    if model_bin > 1:
        logger.warning(
            "Transfer probe: this model was trained with bin_size=%d "
            "(bin-aggregated features) but is being applied to per-frame latents. "
            "Cross-bin transfer can degrade accuracy; prefer a bin_size=1 model, "
            "or interpret the transferred labels with this in mind.",
            model_bin,
        )

    # --- Load latent features from target project ---
    latent_dir = os.path.join(project_path, "latent")
    if not os.path.isdir(latent_dir):
        raise FileNotFoundError(f"No latent directory found: {latent_dir}")

    # Pick most-recently-modified model sub-directory (matches user's latest extraction).
    model_dirs = [
        os.path.join(latent_dir, d) for d in os.listdir(latent_dir)
        if os.path.isdir(os.path.join(latent_dir, d))
    ]
    if not model_dirs:
        raise FileNotFoundError(f"No model sub-directories in {latent_dir}")
    model_subdir = max(model_dirs, key=os.path.getmtime)

    latent_files = sorted(glob.glob(os.path.join(model_subdir, "*.npz")))
    if not latent_files:
        raise FileNotFoundError(f"No latent .npz files in {model_subdir}")

    latent_chunks = []
    for lf in latent_files:
        loaded = np.load(lf)
        latent_chunks.append(loaded["latent"])
    new_features = np.concatenate(latent_chunks, axis=0)

    # --- Apply ---
    result = apply_cluster_model(model, new_features, method=method)

    # --- Write results ---
    cluster_dir = os.path.join(project_path, "cluster")
    os.makedirs(cluster_dir, exist_ok=True)

    # transferred_id.csv (from the model's cluster names). Deliberately NOT the
    # project's own cluster/id.csv: overwriting it would destroy the native
    # clustering's names/colors that ethogram/export/restore read back, and
    # silently re-pair the project's existing time_series labels with the
    # transferred names. The transfer output is a self-contained pair
    # (transferred_id.csv + transferred_labels.csv).
    id_rows = sorted(result["cluster_names"].items())
    id_df = pd.DataFrame(
        [{"Id": cid, "Name": cname, "Color": "grey"} for cid, cname in id_rows]
    )
    id_csv_path = os.path.join(cluster_dir, "transferred_id.csv")
    id_df.to_csv(id_csv_path, index=False)

    # transferred_labels.csv
    labels_df = pd.DataFrame({
        "behavior": result["labels"],
        "confidence": result["confidence"],
    })
    labels_csv_path = os.path.join(cluster_dir, "transferred_labels.csv")
    labels_df.to_csv(labels_csv_path, index=False)

    logger.info(
        "Applied cluster model to %s: %d frames, %d unique labels",
        project_path,
        len(result["labels"]),
        len(np.unique(result["labels"])),
    )

    return {
        "labels": result["labels"],
        "confidence": result["confidence"],
        "cluster_names": result["cluster_names"],
        "output_csv": labels_csv_path,
        "id_csv": id_csv_path,
        "n_frames": len(result["labels"]),
        "mean_confidence": float(result["confidence"].mean()) if len(result["confidence"]) else 0.0,
    }
