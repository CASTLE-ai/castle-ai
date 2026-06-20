"""castle/service/cluster_params.py

Heuristic clustering-parameter suggestion for first-time users. Extracted out of
the former ``clustering_service`` god-module — a small, pure heuristic with no
clustering state.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class ClusteringParamSuggestion:
    """Heuristic clustering parameters for first-time users.

    Attributes:
        n_samples: Sample count the suggestion was computed for.
        min_cluster_size: HDBSCAN ``min_cluster_size`` suggestion. Sized
            so the smallest accepted cluster represents ~0.5% of the
            data (a B-SOiD / MoSeq convention).
        min_samples: HDBSCAN ``min_samples`` suggestion. Always smaller
            than ``min_cluster_size``.
        eps_range: DBSCAN ``eps`` values worth sweeping interactively.
    """

    n_samples: int
    min_cluster_size: int
    min_samples: int
    eps_range: List[float] = field(default_factory=list)


def suggest_clustering_params(n_samples: int) -> ClusteringParamSuggestion:
    """Suggest HDBSCAN/DBSCAN starting parameters for ``n_samples`` bins.

    Args:
        n_samples: Total number of latent samples (bins) the user is
            about to cluster.

    Returns:
        :class:`ClusteringParamSuggestion`. Values are heuristics —
        researchers should sweep ``eps_range`` interactively in the
        Behavior Microscope rather than trust the suggestion blindly.

    Notes:
        Rationale: ``min_cluster_size = max(10, n//200)`` keeps the
        smallest cluster ≥ 0.5% of the data. ``min_samples = max(5,
        n//500)`` keeps DBSCAN's k-neighbour requirement lower than
        ``min_cluster_size`` (HDBSCAN expects this). The eps sweep is
        anchored at 1.0 (the global default, see
        :data:`castle.defaults.DBSCAN_EPS`) and brackets two octaves on
        each side.
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}.")
    return ClusteringParamSuggestion(
        n_samples=int(n_samples),
        min_cluster_size=max(10, n_samples // 200),
        min_samples=max(5, n_samples // 500),
        eps_range=[0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
    )
