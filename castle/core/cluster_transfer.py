"""Cluster transfer: save and apply clustering models to new data."""
import numpy as np
import json
import logging
from dataclasses import dataclass
from typing import Dict

logger = logging.getLogger(__name__)

@dataclass
class ClusterModel:
    """Saved clustering model."""
    umap_embedding: np.ndarray    # (N, 2)
    training_features: np.ndarray  # (N, D)
    cluster_labels: np.ndarray     # (N,)
    cluster_names: Dict[int, str]
    k: int = 5
    model_name: str = ""
    fps: float = 30.0
    feature_dim: int = 768
    n_clusters: int = 0

def save_cluster_model(output_path, umap_embedding, training_features, cluster_labels, 
                       cluster_names=None, model_name="", fps=30.0, k=5) -> str:
    """Save as .npz with metadata JSON."""
    metadata = json.dumps({
        "cluster_names": {str(k): v for k, v in (cluster_names or {}).items()},
        "model_name": model_name, "fps": fps, "k": k,
        "feature_dim": int(training_features.shape[1]),
        "n_clusters": int(len(set(cluster_labels[cluster_labels >= 0]))),
    })
    np.savez_compressed(output_path, 
                        umap_embedding=umap_embedding,
                        training_features=training_features,
                        cluster_labels=cluster_labels,
                        metadata=np.array([metadata]))
    return output_path

def load_cluster_model(model_path) -> ClusterModel:
    """Load saved model."""
    data = np.load(model_path, allow_pickle=False)
    meta = json.loads(str(data["metadata"][0]))
    return ClusterModel(
        umap_embedding=data["umap_embedding"],
        training_features=data["training_features"],
        cluster_labels=data["cluster_labels"],
        cluster_names={int(k): v for k, v in meta["cluster_names"].items()},
        k=meta.get("k", 5), model_name=meta.get("model_name", ""),
        fps=meta.get("fps", 30.0), feature_dim=meta.get("feature_dim", 768),
        n_clusters=meta.get("n_clusters", 0),
    )

def _majority_vote_excluding_noise(neighbor_labels: np.ndarray) -> tuple[int, float]:
    """Majority vote over neighbour labels, ignoring -1 (noise) training points.

    DBSCAN/HDBSCAN noise (-1) is not a behavioral class: it must not win votes
    nor dilute a real winner. Returns ``(predicted_label, confidence)``; when
    every neighbour is noise the sample is genuinely unclassifiable → ``(-1,
    0.0)``. Confidence is the winning count over the number of *valid*
    (non-noise) neighbours, so it reflects the real support.
    """
    valid = np.asarray(neighbor_labels)
    valid = valid[valid >= 0]
    if valid.size == 0:
        return -1, 0.0
    unique, counts = np.unique(valid, return_counts=True)
    winner = int(unique[int(np.argmax(counts))])
    confidence = float(counts.max()) / float(valid.size)
    return winner, confidence


def apply_cluster_model(model, new_features, method="knn_feature") -> dict:
    """Apply saved model to new features.
    
    Methods:
      knn_feature: k-NN in original D-dim feature space (cosine). Recommended.
      knn_umap: Approximate UMAP projection via weighted interpolation, then k-NN in 2D.
    """
    from sklearn.neighbors import NearestNeighbors
    
    if new_features.shape[1] != model.training_features.shape[1]:
        raise ValueError(f"Feature dim mismatch: {new_features.shape[1]} vs {model.training_features.shape[1]}")
    if len(new_features) == 0:
        return {"labels": np.array([], dtype=int), "confidence": np.array([]), "cluster_names": model.cluster_names}
    
    nn = NearestNeighbors(n_neighbors=model.k, metric='cosine')
    nn.fit(model.training_features)
    distances, indices = nn.kneighbors(new_features)
    
    if method == "knn_feature":
        predicted, confidences = [], []
        for i in range(len(new_features)):
            label, conf = _majority_vote_excluding_noise(model.cluster_labels[indices[i]])
            predicted.append(label)
            confidences.append(conf)
        return {"labels": np.array(predicted), "confidence": np.array(confidences), "cluster_names": model.cluster_names}
    
    elif method == "knn_umap":
        weights = 1.0 / (distances + 1e-8)
        weights /= weights.sum(axis=1, keepdims=True)
        projected = np.zeros((len(new_features), 2))
        for i in range(len(new_features)):
            projected[i] = np.average(model.umap_embedding[indices[i]], weights=weights[i], axis=0)
        nn_umap = NearestNeighbors(n_neighbors=model.k)
        nn_umap.fit(model.umap_embedding)
        _, umap_indices = nn_umap.kneighbors(projected)
        predicted, confidences = [], []
        for i in range(len(new_features)):
            label, conf = _majority_vote_excluding_noise(model.cluster_labels[umap_indices[i]])
            predicted.append(label)
            confidences.append(conf)
        return {"labels": np.array(predicted), "confidence": np.array(confidences),
                "umap_projection": projected, "cluster_names": model.cluster_names}
    else:
        raise ValueError(f"Unknown method: {method}")
