"""Automated Behavior Microscope — parameter sweep and quality-based selection.

Runs through CASTLE's microscope presets (Raiso-optimized UMAP configurations),
tries multiple DBSCAN eps values, scores each with quality metrics, and selects
the best clustering.
"""

import numpy as np
import logging
from dataclasses import dataclass
from typing import List, Optional, Callable, Any

logger = logging.getLogger(__name__)

# ── Raiso-optimized Behavior Microscope presets ─────────────────────────

MICROSCOPE_PRESETS = {
    "low": {
        "description": "Low-magnification objective (1-stage UMAP, n_components=2)",
        "n_neighbors_options": [1000, 500, 300, 100, 50, 25],
        "build_config": lambda n: [
            {"n_neighbors": n, "min_dist": 0.0, "n_components": 2, "n_epochs": 5000}
        ],
    },
    "intermediate": {
        "description": "Intermediate-magnification objective (2-stage UMAP, 5→2)",
        "n_neighbors_options": [(1000, 500), (500, 300), (300, 100), (100, 50), (50, 25)],
        "build_config": lambda pair: [
            {"n_neighbors": pair[0], "min_dist": 0.0, "n_components": 5, "n_epochs": 5000},
            {"n_neighbors": pair[1], "min_dist": 0.0, "n_components": 2, "n_epochs": 5000},
        ],
    },
    "high": {
        "description": "High-magnification objective (2-stage UMAP, 10→2)",
        "n_neighbors_options": [(1000, 500), (500, 300), (300, 100), (100, 50), (50, 25)],
        "build_config": lambda pair: [
            {"n_neighbors": pair[0], "min_dist": 0.0, "n_components": 10, "n_epochs": 5000},
            {"n_neighbors": pair[1], "min_dist": 0.0, "n_components": 2, "n_epochs": 5000},
        ],
    },
    "super_high": {
        "description": "Super-high-magnification objective (3-stage UMAP, 15→5→2)",
        "n_neighbors_options": [(500, 300, 100), (300, 100, 50), (100, 50, 25)],
        "build_config": lambda triple: [
            {"n_neighbors": triple[0], "min_dist": 0.0, "n_components": 15, "n_epochs": 5000},
            {"n_neighbors": triple[1], "min_dist": 0.0, "n_components": 5, "n_epochs": 5000},
            {"n_neighbors": triple[2], "min_dist": 0.0, "n_components": 2, "n_epochs": 5000},
        ],
    },
}

# Default eps values to try for each UMAP embedding
DEFAULT_EPS_VALUES = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]


@dataclass
class ClusteringCandidate:
    """One candidate clustering result from the parameter sweep."""
    preset_name: str  # e.g., "low", "high"
    n_neighbors: Any  # int or tuple
    eps: float
    umap_config: list  # the actual UMAP config dicts
    n_clusters: int
    noise_ratio: float
    quality_score: float  # composite score
    temporal_coherence: float
    single_frame_ratio: float
    calinski_harabasz: float = 0.0
    davies_bouldin: float = float('inf')
    labels: Optional[np.ndarray] = None
    embedding: Optional[np.ndarray] = None


def score_clustering(labels: np.ndarray, embedding: np.ndarray = None,
                     features: np.ndarray = None) -> dict:
    """Score a clustering result using quality metrics.
    
    Uses castle.core.metrics.evaluate_clustering when available,
    with a fallback to basic metrics.
    
    Returns dict with quality_score and component scores.
    """
    from castle.core.metrics import (
        temporal_coherence as calc_tc,
        bout_quality_metrics,
    )
    
    # Filter out noise (-1) for internal metrics
    valid_mask = labels >= 0
    n_valid = valid_mask.sum()
    n_total = len(labels)
    noise_ratio = 1.0 - (n_valid / n_total) if n_total > 0 else 1.0
    
    # Temporal coherence (on full labels including noise)
    tc = calc_tc(labels)
    
    # Bout quality
    bout_q = bout_quality_metrics(labels)
    sfr = bout_q.get("single_frame_ratio", 1.0)
    
    # Number of clusters (excluding noise)
    unique_valid = np.unique(labels[valid_mask])
    n_clusters = len(unique_valid)
    
    # Calinski-Harabasz on valid points with embedding
    ch_score = 0.0
    db_score = float('inf')
    if embedding is not None and n_clusters >= 2 and n_valid > n_clusters:
        try:
            from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
            ch_score = calinski_harabasz_score(embedding[valid_mask], labels[valid_mask])
            db_score = davies_bouldin_score(embedding[valid_mask], labels[valid_mask])
        except Exception:
            pass
    
    # Composite score:
    # - temporal_coherence (higher = better, weight 0.35)
    # - 1 - single_frame_ratio (higher = better, weight 0.25)
    # - normalized CH score (higher = better, weight 0.2)
    # - 1 - noise_ratio (lower noise = better, weight 0.1)
    # - cluster count penalty: penalize too few (<3) or too many (>30)
    
    ch_normalized = min(ch_score / 1000.0, 1.0) if ch_score > 0 else 0.0
    
    cluster_penalty = 0.0
    if n_clusters < 2:
        cluster_penalty = -0.5  # bad: only one cluster
    elif n_clusters < 3:
        cluster_penalty = -0.1
    elif n_clusters > 50:
        cluster_penalty = -0.3
    elif n_clusters > 30:
        cluster_penalty = -0.1
    
    quality_score = (
        0.35 * tc +
        0.25 * (1.0 - sfr) +
        0.20 * ch_normalized +
        0.10 * (1.0 - noise_ratio) +
        0.10 * (1.0 + cluster_penalty)
    )
    
    return {
        "quality_score": quality_score,
        "temporal_coherence": tc,
        "single_frame_ratio": sfr,
        "calinski_harabasz": ch_score,
        "davies_bouldin": db_score,
        "noise_ratio": noise_ratio,
        "n_clusters": n_clusters,
    }


def auto_cluster(
    data: np.ndarray,
    presets: List[str] = None,
    eps_values: List[float] = None,
    n_neighbors_filter: Any = None,
    min_clusters: int = 2,
    max_clusters: int = 50,
    progress_callback: Optional[Callable] = None,
    device: str = "cpu",
) -> List[ClusteringCandidate]:
    """Run automated Behavior Microscope parameter sweep.
    
    Args:
        data: (N, D) latent features
        presets: Which presets to try. Default: all ["low", "intermediate", "high", "super_high"]
        eps_values: DBSCAN eps values to sweep. Default: [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
        n_neighbors_filter: If set, only use this specific n_neighbors value/tuple per preset.
        min_clusters: Minimum acceptable clusters (below this = skip)
        max_clusters: Maximum acceptable clusters (above this = skip)
        progress_callback: Optional (step, total, desc) callback
        device: "cpu" or "cuda:N"
    
    Returns:
        List of ClusteringCandidate sorted by quality_score (best first)
    """
    if presets is None:
        presets = ["low", "intermediate", "high", "super_high"]
    if eps_values is None:
        eps_values = DEFAULT_EPS_VALUES
    
    # Import UMAP and DBSCAN based on device
    if device == 'cpu' or device == 'mps':
        from umap import UMAP
        from sklearn.cluster import DBSCAN
    elif 'cuda' in device:
        try:
            from cuml.manifold import UMAP
            from cuml.cluster import DBSCAN
        except ImportError:
            try:
                from castle.utils.myumap import UMAP
            except ImportError:
                from umap import UMAP
            from sklearn.cluster import DBSCAN
    else:
        from umap import UMAP
        from sklearn.cluster import DBSCAN
    
    candidates = []
    
    # Count total steps for progress
    total_steps = 0
    for preset_name in presets:
        if preset_name not in MICROSCOPE_PRESETS:
            continue
        preset = MICROSCOPE_PRESETS[preset_name]
        n_options = preset["n_neighbors_options"]
        if n_neighbors_filter is not None:
            n_options = [n for n in n_options if n == n_neighbors_filter]
        total_steps += len(n_options) * len(eps_values)
    
    current_step = 0
    
    for preset_name in presets:
        if preset_name not in MICROSCOPE_PRESETS:
            logger.warning(f"Unknown preset: {preset_name}, skipping")
            continue
        
        preset = MICROSCOPE_PRESETS[preset_name]
        n_options = preset["n_neighbors_options"]
        if n_neighbors_filter is not None:
            n_options = [n for n in n_options if n == n_neighbors_filter]
        
        for n_opt in n_options:
            # Build UMAP config
            umap_config = preset["build_config"](n_opt)
            
            # Run multi-stage UMAP
            if progress_callback:
                progress_callback(current_step, total_steps, 
                                  f"{preset_name} n={n_opt}: UMAP...")
            
            try:
                Z = data.copy()
                # Filter NaN rows
                valid_rows = ~np.isnan(Z).any(axis=1)
                Z_clean = Z[valid_rows]
                
                if len(Z_clean) < 10:
                    logger.warning(f"Too few valid points ({len(Z_clean)}), skipping {preset_name} n={n_opt}")
                    current_step += len(eps_values)
                    continue
                
                for stage_cfg in umap_config:
                    Z_clean = UMAP(**stage_cfg).fit_transform(Z_clean)
                
                embedding = np.array(Z_clean)
            except Exception as e:
                logger.warning(f"UMAP failed for {preset_name} n={n_opt}: {e}")
                current_step += len(eps_values)
                continue
            
            # Try each eps
            for eps in eps_values:
                if progress_callback:
                    progress_callback(current_step, total_steps,
                                      f"{preset_name} n={n_opt} eps={eps}: DBSCAN...")
                
                try:
                    db_labels = DBSCAN(eps=eps).fit_predict(embedding)
                except Exception as e:
                    logger.warning(f"DBSCAN failed: {e}")
                    current_step += 1
                    continue
                
                # Reconstruct full labels (with -1 for NaN rows)
                full_labels = np.full(len(data), -1, dtype=int)
                full_labels[valid_rows] = db_labels
                
                # Score
                scores = score_clustering(full_labels, embedding, Z[valid_rows])
                n_clusters = scores["n_clusters"]
                
                # Filter by cluster count
                if n_clusters < min_clusters or n_clusters > max_clusters:
                    current_step += 1
                    continue
                
                candidate = ClusteringCandidate(
                    preset_name=preset_name,
                    n_neighbors=n_opt,
                    eps=eps,
                    umap_config=umap_config,
                    n_clusters=n_clusters,
                    noise_ratio=scores["noise_ratio"],
                    quality_score=scores["quality_score"],
                    temporal_coherence=scores["temporal_coherence"],
                    single_frame_ratio=scores["single_frame_ratio"],
                    calinski_harabasz=scores["calinski_harabasz"],
                    davies_bouldin=scores["davies_bouldin"],
                    labels=full_labels,
                    embedding=embedding,
                )
                candidates.append(candidate)
                current_step += 1
    
    # Sort by quality score (best first)
    candidates.sort(key=lambda c: c.quality_score, reverse=True)
    
    return candidates


def select_best(candidates: List[ClusteringCandidate],
                min_tc: float = 0.8) -> Optional[ClusteringCandidate]:
    """Select the best candidate with minimum temporal coherence threshold."""
    for c in candidates:
        if c.temporal_coherence >= min_tc:
            return c
    # If none meets threshold, return the best overall
    return candidates[0] if candidates else None
