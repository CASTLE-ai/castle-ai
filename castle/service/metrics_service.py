"""Service layer for clustering quality evaluation.

Loads cluster labels and embedding from a CASTLE project directory
and runs the full quality evaluation pipeline.
"""

import os
import logging
from dataclasses import asdict
from typing import Optional

import numpy as np
import pandas as pd

from castle.core.metrics import evaluate_clustering

logger = logging.getLogger(__name__)


def evaluate_project_clustering(
    project_path: str,
    ground_truth_path: Optional[str] = None,
) -> dict:
    """Evaluate clustering quality for a project.

    Loads cluster labels from time_series CSVs and embedding from saved NPZ
    in the project's ``cluster/`` directory, then runs the full quality
    evaluation and returns a JSON-serializable dict.

    Args:
        project_path: Absolute path to the project directory
            (e.g. ``/storage/my_project``).
        ground_truth_path: Optional path to a CSV with a ``behavior`` column
            containing ground-truth labels aligned to the cluster labels.

    Returns:
        dict representation of :class:`ClusterQualityReport`.
    """
    cluster_dir = os.path.join(project_path, "cluster")

    if not os.path.isdir(cluster_dir):
        return {"error": f"No cluster directory found at {cluster_dir}"}

    # ------------------------------------------------------------------
    # 1. Load cluster labels from time_series CSVs
    # ------------------------------------------------------------------
    ts_files = sorted(
        f
        for f in os.listdir(cluster_dir)
        if f.startswith("time_series_") and f.endswith(".csv")
    )
    if not ts_files:
        return {"error": "No time_series CSV files found in cluster directory"}

    label_chunks = []
    for fname in ts_files:
        df = pd.read_csv(os.path.join(cluster_dir, fname))
        if "behavior" not in df.columns:
            return {"error": f"Column 'behavior' not found in {fname}"}
        label_chunks.append(df["behavior"].values)

    labels = np.concatenate(label_chunks)
    logger.info("Loaded %d labels from %d time_series file(s)", len(labels), len(ts_files))

    # ------------------------------------------------------------------
    # 2. Load embedding from NPZ (if available)
    # ------------------------------------------------------------------
    # Initialise BOTH here so every downstream path has them defined. Previously
    # labels_for_emb was only set inside the branches, so an npz that loaded but
    # lacked an "emb" key left it unbound — only a short-circuit on
    # `embedding is not None` avoided a NameError. Any refactor would have
    # re-broken it. (PR3 Stage 7.5)
    embedding = None
    labels_for_emb = None
    npz_files = sorted(
        f for f in os.listdir(cluster_dir) if f.startswith("cluster_") and f.endswith(".npz")
    )
    if npz_files:
        npz_path = os.path.join(cluster_dir, npz_files[-1])  # latest
        try:
            data = np.load(npz_path, allow_pickle=True)
            if "emb" in data:
                emb_raw = data["emb"]
                # emb_raw may have NaN rows for filtered-out points
                # Only use rows that are not all-NaN
                valid_rows = ~np.isnan(emb_raw).all(axis=1)
                if valid_rows.sum() > 0:
                    embedding = emb_raw[valid_rows]
                    # Align labels to valid embedding rows if sizes match
                    if len(embedding) != len(labels):
                        # embedding is from a single LocalLatent subset,
                        # labels cover all videos — sizes may not match
                        # Use embedding only if they align
                        if "cls" in data:
                            cls_full = data["cls"]
                            valid_cls = cls_full[valid_rows]
                            # Use cls as labels aligned to embedding
                            labels_for_emb = valid_cls
                        else:
                            embedding = None
                            labels_for_emb = None
                    else:
                        labels_for_emb = labels
                else:
                    embedding = None
                    labels_for_emb = None
        except Exception as exc:
            logger.warning("Could not load embedding from %s: %s", npz_path, exc)
            embedding = None
            labels_for_emb = None
    else:
        labels_for_emb = None

    # ------------------------------------------------------------------
    # 3. Load ground truth (if provided)
    # ------------------------------------------------------------------
    ground_truth = None
    if ground_truth_path and os.path.isfile(ground_truth_path):
        gt_df = pd.read_csv(ground_truth_path)
        if "behavior" in gt_df.columns:
            ground_truth = gt_df["behavior"].values
            # Truncate or pad to match labels length
            min_len = min(len(ground_truth), len(labels))
            ground_truth = ground_truth[:min_len]
            labels_for_gt = labels[:min_len]
        else:
            logger.warning("Ground truth CSV has no 'behavior' column, skipping")
            labels_for_gt = labels
    else:
        labels_for_gt = labels

    # ------------------------------------------------------------------
    # 4. Run evaluation
    # ------------------------------------------------------------------
    # If we have embedding with aligned labels, run with embedding
    if embedding is not None and labels_for_emb is not None:
        if ground_truth is not None:
            min_len = min(len(labels_for_emb), len(ground_truth))
            labels_for_emb = labels_for_emb[:min_len]
            gt_for_emb = ground_truth[:min_len]
        else:
            gt_for_emb = None
        report = evaluate_clustering(
            labels_for_emb,
            embedding=embedding,
            ground_truth=gt_for_emb,
        )
    else:
        # Run without embedding, use full label array
        report = evaluate_clustering(
            labels_for_gt if ground_truth is not None else labels,
            ground_truth=ground_truth,
        )

    result = asdict(report)
    result["n_frames"] = int(len(labels))
    result["n_time_series_files"] = len(ts_files)
    return result
