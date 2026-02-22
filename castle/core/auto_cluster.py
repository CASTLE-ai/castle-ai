"""Automated Behavior Microscope — recursive hierarchical clustering.

Implements CASTLE's multi-level Behavior Microscope workflow as an automated
pipeline. Mirrors the manual process where a user:
1. Selects a cluster → runs UMAP → runs DBSCAN → gets sub-clusters
2. Decides which sub-clusters are "done" (leaf) vs need further splitting
3. Recurses on each non-leaf sub-cluster with higher magnification

The automated version replaces human judgment with quality metrics to
decide when to stop splitting.

Raiso-optimized UMAP presets at each depth:
- Depth 0:   Low magnification   (1-stage, n_neighbors=300, n_components=2)
- Depth 1+:  Intermediate magnification (2-stage, 300→100, 5→2)
- Small clusters (<500 frames): Low magnification (n_neighbors=100, 2D)
"""

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Any

logger = logging.getLogger(__name__)

# ── Raiso-optimized Behavior Microscope presets ─────────────────────────
# Derived from Raiso's ctrl30 OFT project (76 leaf clusters, max depth 7)
# and the 6-OHDA benchmark project NPZ configs.

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

# Default eps values to try at each UMAP level
DEFAULT_EPS_VALUES = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]


# ── Data structures ─────────────────────────────────────────────────────

@dataclass
class ClusteringCandidate:
    """One candidate clustering result from the parameter sweep."""
    preset_name: str
    n_neighbors: Any
    eps: float
    umap_config: list
    n_clusters: int
    noise_ratio: float
    quality_score: float
    temporal_coherence: float
    single_frame_ratio: float
    calinski_harabasz: float = 0.0
    davies_bouldin: float = float('inf')
    labels: Optional[np.ndarray] = None
    embedding: Optional[np.ndarray] = None


@dataclass
class TreeNode:
    """One node in the hierarchical clustering tree."""
    name: str           # e.g., "root_a0_b1"
    depth: int
    n_frames: int
    is_leaf: bool       # True = final behavior, False = split further
    stop_reason: str = ""  # why we stopped: "leaf", "min_frames", "max_depth", etc.
    children: List['TreeNode'] = field(default_factory=list)
    umap_config: Optional[list] = None
    eps: Optional[float] = None
    quality_score: Optional[float] = None


# ── UMAP config selection by depth ──────────────────────────────────────

def select_umap_config(depth: int, n_frames: int) -> list:
    """Select UMAP config based on depth and cluster size.
    
    Follows Raiso's observed pattern from ctrl30 project:
    - Depth 0: Low magnification, n_neighbors=300, 2D (see big structure)
    - Depth 1+: Intermediate, 300→100, 5D→2D (refine sub-clusters)
    - Small clusters (<500): Low with smaller n_neighbors
    
    The n_neighbors is clamped to n_frames//3 to avoid UMAP errors.
    """
    if n_frames < 500:
        # Small cluster: use low magnification with appropriate n_neighbors
        n = min(100, max(15, n_frames // 5))
        return [{"n_neighbors": n, "min_dist": 0.0, "n_components": 2, "n_epochs": 5000}]
    
    if depth == 0:
        # First level: Low magnification
        n = min(300, n_frames // 3)
        return [{"n_neighbors": n, "min_dist": 0.0, "n_components": 2, "n_epochs": 5000}]
    
    # Depth 1+: Intermediate magnification (2-stage)
    n1 = min(300, n_frames // 3)
    n2 = min(100, n_frames // 5)
    return [
        {"n_neighbors": n1, "min_dist": 0.0, "n_components": 5, "n_epochs": 5000},
        {"n_neighbors": n2, "min_dist": 0.0, "n_components": 2, "n_epochs": 5000},
    ]


# ── Scoring ─────────────────────────────────────────────────────────────

def score_clustering(labels: np.ndarray, embedding: np.ndarray = None,
                     features: np.ndarray = None) -> dict:
    """Score a clustering result using quality metrics.
    
    Returns dict with quality_score and component scores.
    """
    from castle.core.metrics import (
        temporal_coherence as calc_tc,
        bout_quality_metrics,
    )
    
    # Align embedding to labels length when they diverge
    if embedding is not None and len(embedding) != len(labels):
        min_len = min(len(labels), len(embedding))
        logger.warning(
            "score_clustering: labels length (%d) != embedding length (%d); "
            "truncating both to %d",
            len(labels), len(embedding), min_len,
        )
        labels = labels[:min_len]
        embedding = embedding[:min_len]

    valid_mask = labels >= 0
    n_valid = valid_mask.sum()
    n_total = len(labels)
    noise_ratio = 1.0 - (n_valid / n_total) if n_total > 0 else 1.0
    
    tc = calc_tc(labels)
    bout_q = bout_quality_metrics(labels)
    sfr = bout_q.get("single_frame_ratio", 1.0)
    
    unique_valid = np.unique(labels[valid_mask])
    n_clusters = len(unique_valid)
    
    ch_score = 0.0
    db_score = float('inf')
    if embedding is not None and n_clusters >= 2 and n_valid > n_clusters:
        try:
            from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
            ch_score = calinski_harabasz_score(embedding[valid_mask], labels[valid_mask])
            db_score = davies_bouldin_score(embedding[valid_mask], labels[valid_mask])
        except Exception as exc:
            logger.debug("sklearn cluster metrics unavailable: %s", exc)
    
    ch_normalized = min(ch_score / 1000.0, 1.0) if ch_score > 0 else 0.0
    
    cluster_penalty = 0.0
    if n_clusters < 2:
        cluster_penalty = -0.5
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


# ── Single-level sweep (used at each node) ──────────────────────────────

def find_best_eps(
    embedding: np.ndarray,
    labels_full: np.ndarray,
    valid_mask: np.ndarray,
    eps_values: List[float] = None,
    device: str = "cpu",
) -> Optional[ClusteringCandidate]:
    """Try multiple eps values on an existing UMAP embedding.
    
    Returns the best ClusteringCandidate or None.
    """
    if eps_values is None:
        eps_values = DEFAULT_EPS_VALUES
    
    if device == 'cpu' or device == 'mps':
        from sklearn.cluster import DBSCAN
    elif 'cuda' in device:
        try:
            from cuml.cluster import DBSCAN
        except ImportError:
            from sklearn.cluster import DBSCAN
    else:
        from sklearn.cluster import DBSCAN
    
    best = None
    for eps in eps_values:
        try:
            db_labels = DBSCAN(eps=eps).fit_predict(embedding)
        except Exception as e:
            logger.warning(f"DBSCAN(eps={eps}) failed: {e}")
            continue
        
        full_labels = np.full(len(labels_full), -1, dtype=int)
        full_labels[valid_mask] = db_labels
        
        scores = score_clustering(full_labels, embedding)
        n_clusters = scores["n_clusters"]
        
        if n_clusters < 2:
            continue
        
        candidate = ClusteringCandidate(
            preset_name="auto",
            n_neighbors=0,
            eps=eps,
            umap_config=[],
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
        
        if best is None or candidate.quality_score > best.quality_score:
            best = candidate
    
    return best


# ── Legacy flat sweep (backward compat) ─────────────────────────────────

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
    """Run flat (non-recursive) parameter sweep. Kept for backward compat.
    
    For the recursive hierarchical version, use the service layer's
    ``ClusteringService.auto_cluster_recursive()``.
    """
    if presets is None:
        presets = ["low", "intermediate", "high", "super_high"]
    if eps_values is None:
        eps_values = DEFAULT_EPS_VALUES
    
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
            continue
        preset = MICROSCOPE_PRESETS[preset_name]
        n_options = preset["n_neighbors_options"]
        if n_neighbors_filter is not None:
            n_options = [n for n in n_options if n == n_neighbors_filter]
        
        for n_opt in n_options:
            umap_config = preset["build_config"](n_opt)
            if progress_callback:
                progress_callback(current_step, total_steps, f"{preset_name} n={n_opt}: UMAP...")
            try:
                Z = data.copy()
                valid_rows = ~np.isnan(Z).any(axis=1)
                Z_clean = Z[valid_rows]
                if len(Z_clean) < 10:
                    current_step += len(eps_values)
                    continue
                for stage_cfg in umap_config:
                    Z_clean = UMAP(**stage_cfg).fit_transform(Z_clean)
                embedding = np.array(Z_clean)
            except Exception as e:
                logger.warning(f"UMAP failed for {preset_name} n={n_opt}: {e}")
                current_step += len(eps_values)
                continue
            
            for eps in eps_values:
                if progress_callback:
                    progress_callback(current_step, total_steps, f"{preset_name} n={n_opt} eps={eps}")
                try:
                    db_labels = DBSCAN(eps=eps).fit_predict(embedding)
                except Exception:
                    current_step += 1
                    continue
                full_labels = np.full(len(data), -1, dtype=int)
                full_labels[valid_rows] = db_labels
                scores = score_clustering(full_labels, embedding)
                n_clusters = scores["n_clusters"]
                if n_clusters < min_clusters or n_clusters > max_clusters:
                    current_step += 1
                    continue
                candidates.append(ClusteringCandidate(
                    preset_name=preset_name, n_neighbors=n_opt, eps=eps,
                    umap_config=umap_config, n_clusters=n_clusters,
                    noise_ratio=scores["noise_ratio"],
                    quality_score=scores["quality_score"],
                    temporal_coherence=scores["temporal_coherence"],
                    single_frame_ratio=scores["single_frame_ratio"],
                    calinski_harabasz=scores["calinski_harabasz"],
                    davies_bouldin=scores["davies_bouldin"],
                    labels=full_labels, embedding=embedding,
                ))
                current_step += 1
    
    candidates.sort(key=lambda c: c.quality_score, reverse=True)
    return candidates


def select_best(candidates: List[ClusteringCandidate],
                min_tc: float = 0.8) -> Optional[ClusteringCandidate]:
    """Select the best candidate with minimum temporal coherence threshold."""
    for c in candidates:
        if c.temporal_coherence >= min_tc:
            return c
    return candidates[0] if candidates else None
