"""Clustering quality metrics for behavioral analysis.

Provides both internal validation (no ground truth needed) and external
validation (with ground truth) metrics, plus behavior-specific metrics.
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ClusterQualityReport:
    """Complete clustering quality assessment."""

    # Internal validation (no ground truth)
    temporal_coherence: float  # fraction of matching neighbor labels
    calinski_harabasz: Optional[float]  # variance ratio (needs embedding)
    davies_bouldin: Optional[float]  # cluster similarity (needs embedding)
    silhouette_sample: Optional[float]  # sampled silhouette (needs embedding)

    # Bout-based metrics
    n_single_frame_bouts: int  # number of 1-frame "bouts" (noise indicator)
    single_frame_ratio: float  # fraction of bouts that are single-frame
    median_bout_duration_frames: float
    bout_duration_cv: float  # coefficient of variation across all bouts

    # External validation (with ground truth, optional)
    nmi: Optional[float] = None
    ari: Optional[float] = None
    v_measure: Optional[float] = None
    homogeneity: Optional[float] = None
    completeness: Optional[float] = None

    # Summary
    verdict: str = ""  # "GOOD" / "ACCEPTABLE" / "POOR"
    warnings: List[str] = field(default_factory=list)


def temporal_coherence(labels: np.ndarray, window: int = 1) -> float:
    """Fraction of frames where label matches the next frame's label.

    Args:
        labels: 1D array of cluster assignments.
        window: number of neighbors to check (default 1 = next frame only).

    Returns:
        float in [0, 1]. Higher = more temporally stable clusters.

    Good clustering: > 0.95 (long, stable bouts)
    Poor clustering: < 0.80 (flickering labels)
    """
    labels = np.asarray(labels)
    if len(labels) < 2:
        return 1.0
    n = len(labels) - window
    if n <= 0:
        return 1.0
    matches = np.sum(labels[:n] == labels[window:])
    return float(matches / n)


def bout_quality_metrics(labels: np.ndarray) -> dict:
    """Compute bout-based quality metrics.

    A "bout" is a maximal run of consecutive identical labels.

    Returns dict with:
        n_bouts: total number of bouts
        n_single_frame: number of single-frame bouts
        single_frame_ratio: fraction that are single-frame
        median_duration: median bout length in frames
        duration_cv: coefficient of variation of bout durations
    """
    labels = np.asarray(labels)
    if len(labels) == 0:
        return {
            "n_bouts": 0,
            "n_single_frame": 0,
            "single_frame_ratio": 0.0,
            "median_duration": 0.0,
            "duration_cv": 0.0,
        }

    # Find bout boundaries: where label changes
    changes = np.where(labels[1:] != labels[:-1])[0]
    # Bout starts: index 0 + (each change + 1)
    starts = np.concatenate([[0], changes + 1])
    # Bout ends: each change + 1, then len(labels)
    ends = np.concatenate([changes + 1, [len(labels)]])
    durations = ends - starts

    n_bouts = len(durations)
    n_single = int(np.sum(durations == 1))
    single_ratio = n_single / n_bouts if n_bouts > 0 else 0.0
    median_dur = float(np.median(durations))
    mean_dur = float(np.mean(durations))
    std_dur = float(np.std(durations))
    cv = std_dur / mean_dur if mean_dur > 0 else 0.0

    return {
        "n_bouts": n_bouts,
        "n_single_frame": n_single,
        "single_frame_ratio": single_ratio,
        "median_duration": median_dur,
        "duration_cv": cv,
    }


def compute_internal_metrics(
    labels: np.ndarray,
    embedding: np.ndarray = None,
    subsample: int = 5000,
) -> dict:
    """Compute internal validation metrics.

    Args:
        labels: cluster assignments (1D integer array).
        embedding: optional 2D UMAP/feature embedding for distance-based metrics.
        subsample: max samples for expensive metrics (silhouette, CH, DB).

    Returns:
        dict with temporal_coherence, bout metrics, and optionally
        silhouette_sample, calinski_harabasz, davies_bouldin.
    """
    labels = np.asarray(labels)
    result = {
        "temporal_coherence": temporal_coherence(labels),
    }
    result.update(bout_quality_metrics(labels))

    # Distance-based metrics require embedding + at least 2 distinct clusters
    result["silhouette_sample"] = None
    result["calinski_harabasz"] = None
    result["davies_bouldin"] = None

    if embedding is not None and len(labels) > 0:
        from sklearn.metrics import (
            silhouette_score,
            calinski_harabasz_score,
            davies_bouldin_score,
        )

        emb = np.asarray(embedding)
        labs = np.asarray(labels)

        # Ensure labels and embedding have matching lengths
        if len(labs) != len(emb):
            min_len = min(len(labs), len(emb))
            logger.warning(
                "Labels length (%d) != embedding length (%d); truncating to %d",
                len(labs), len(emb), min_len,
            )
            labs = labs[:min_len]
            emb = emb[:min_len]

        # Filter out noise labels (-1) for distance-based metrics
        valid = labs >= 0
        emb_valid = emb[valid]
        labs_valid = labs[valid]

        n_unique = len(np.unique(labs_valid))
        if n_unique >= 2 and len(labs_valid) >= 2:
            # Subsample for expensive O(n²) metrics
            if len(labs_valid) > subsample:
                rng = np.random.RandomState(42)
                idx = rng.choice(len(labs_valid), size=subsample, replace=False)
                emb_sub = emb_valid[idx]
                labs_sub = labs_valid[idx]
            else:
                emb_sub = emb_valid
                labs_sub = labs_valid

            # Only compute if subsample still has >= 2 clusters
            n_unique_sub = len(np.unique(labs_sub))
            if n_unique_sub >= 2:
                result["silhouette_sample"] = float(
                    silhouette_score(emb_sub, labs_sub)
                )
                result["calinski_harabasz"] = float(
                    calinski_harabasz_score(emb_sub, labs_sub)
                )
                result["davies_bouldin"] = float(
                    davies_bouldin_score(emb_sub, labs_sub)
                )

    return result


def compute_external_metrics(
    labels: np.ndarray, ground_truth: np.ndarray
) -> dict:
    """Compute external validation metrics against ground truth.

    Args:
        labels: predicted cluster assignments.
        ground_truth: reference labels (same length).

    Returns:
        dict with nmi, ari, v_measure, homogeneity, completeness.
    """
    from sklearn.metrics import (
        normalized_mutual_info_score,
        adjusted_rand_score,
        v_measure_score,
        homogeneity_score,
        completeness_score,
    )

    labels = np.asarray(labels)
    ground_truth = np.asarray(ground_truth)

    return {
        "nmi": float(normalized_mutual_info_score(ground_truth, labels)),
        "ari": float(adjusted_rand_score(ground_truth, labels)),
        "v_measure": float(v_measure_score(ground_truth, labels)),
        "homogeneity": float(homogeneity_score(ground_truth, labels)),
        "completeness": float(completeness_score(ground_truth, labels)),
    }


def evaluate_clustering(
    labels: np.ndarray,
    embedding: np.ndarray = None,
    ground_truth: np.ndarray = None,
    fps: float = 30.0,
) -> ClusterQualityReport:
    """Run complete clustering quality evaluation.

    This is the main entry point. Computes all applicable metrics
    and returns a structured report with verdict and warnings.

    Args:
        labels: 1D array of cluster assignments.
        embedding: optional 2D embedding for distance-based metrics.
        ground_truth: optional reference labels for external validation.
        fps: frame rate (for informational purposes / future use).

    Returns:
        ClusterQualityReport with all metrics and a verdict.
    """
    labels = np.asarray(labels)

    # Align embedding length to labels when they diverge
    if embedding is not None:
        emb_arr = np.asarray(embedding)
        if len(emb_arr) != len(labels):
            min_len = min(len(labels), len(emb_arr))
            logger.warning(
                "evaluate_clustering: labels length (%d) != embedding length (%d); "
                "truncating both to %d",
                len(labels), len(emb_arr), min_len,
            )
            labels = labels[:min_len]
            embedding = emb_arr[:min_len]

    # Internal metrics (always)
    internal = compute_internal_metrics(labels, embedding=embedding)

    tc = internal["temporal_coherence"]
    sfr = internal["single_frame_ratio"]

    # External metrics (optional)
    ext = {}
    if ground_truth is not None:
        ext = compute_external_metrics(labels, ground_truth)

    # Build warnings
    warnings: List[str] = []
    if tc < 0.80:
        warnings.append(
            f"Very low temporal coherence ({tc:.3f}). Labels are flickering."
        )
    elif tc < 0.90:
        warnings.append(
            f"Low temporal coherence ({tc:.3f}). Consider smoothing or larger eps."
        )
    if sfr > 0.3:
        warnings.append(
            f"High single-frame bout ratio ({sfr:.2f}). Many isolated label flips."
        )
    if internal["silhouette_sample"] is not None and internal["silhouette_sample"] < 0:
        warnings.append(
            f"Negative silhouette score ({internal['silhouette_sample']:.3f}). "
            "Clusters may overlap significantly."
        )
    if internal["davies_bouldin"] is not None and internal["davies_bouldin"] > 2.0:
        warnings.append(
            f"High Davies-Bouldin index ({internal['davies_bouldin']:.2f}). "
            "Clusters are not well separated."
        )

    n_unique = len(np.unique(labels[labels >= 0])) if len(labels) > 0 else 0
    if n_unique <= 1 and len(labels) > 0:
        warnings.append("Only one cluster (excluding noise). No meaningful clustering.")

    # Verdict
    if tc > 0.95 and sfr < 0.1:
        verdict = "GOOD"
    elif tc > 0.85 and sfr < 0.2:
        verdict = "ACCEPTABLE"
    else:
        verdict = "POOR"

    return ClusterQualityReport(
        temporal_coherence=tc,
        calinski_harabasz=internal["calinski_harabasz"],
        davies_bouldin=internal["davies_bouldin"],
        silhouette_sample=internal["silhouette_sample"],
        n_single_frame_bouts=internal["n_single_frame"],
        single_frame_ratio=sfr,
        median_bout_duration_frames=internal["median_duration"],
        bout_duration_cv=internal["duration_cv"],
        nmi=ext.get("nmi"),
        ari=ext.get("ari"),
        v_measure=ext.get("v_measure"),
        homogeneity=ext.get("homogeneity"),
        completeness=ext.get("completeness"),
        verdict=verdict,
        warnings=warnings,
    )
