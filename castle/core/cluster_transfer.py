"""Cluster transfer: save and apply clustering models to new data."""
import numpy as np
import json
import logging
from dataclasses import dataclass
from typing import Dict, Optional

logger = logging.getLogger(__name__)

@dataclass
class ClusterModel:
    """Saved clustering model.

    ``training_features`` are in whatever space the source clustering used:
    raw DINO features for a legacy session, or PCA-reduced features for a
    prepared session. When the source was a *prepared* session, the model also
    carries the Prepare transform (per-sample normalisation + PCA basis +
    ``k_prime`` slice) so a new project's RAW latents can be mapped into the same
    reduced space before k-NN. Transfer is per-frame (the source's temporal
    windowing was smoothing for clustering; it is not re-applied here).
    """
    umap_embedding: np.ndarray    # (N, 2)
    training_features: np.ndarray  # (N, D) — raw (legacy) or reduced (prepared)
    cluster_labels: np.ndarray     # (N,)
    cluster_names: Dict[int, str]
    k: int = 5
    model_name: str = ""
    fps: float = 30.0
    feature_dim: int = 768         # dim of training_features (the k-NN space)
    n_clusters: int = 0
    # --- Prepare transform (prepared sessions only; all None for legacy) ---
    pca_components: Optional[np.ndarray] = None  # (K_full, D_raw)
    pca_mean: Optional[np.ndarray] = None        # (D_raw,)
    normalize: str = ""                          # "l2" | "none" | "" (legacy)
    k_prime: int = 0                             # slice width into PCA dims (0 = none)
    raw_feature_dim: int = 0                     # expected RAW input dim for apply

    @property
    def has_transform(self) -> bool:
        return self.pca_components is not None


def save_cluster_model(output_path, umap_embedding, training_features, cluster_labels,
                       cluster_names=None, model_name="", fps=30.0, k=5,
                       transform=None) -> str:
    """Save as .npz with metadata JSON.

    ``transform`` (prepared sessions): dict with ``components`` (K_full, D_raw),
    ``mean`` (D_raw,), ``normalize`` ("l2"/"none"), ``k_prime`` (int),
    ``raw_feature_dim`` (int). Omitted for legacy raw-feature models.
    """
    arrays = dict(
        umap_embedding=umap_embedding,
        training_features=training_features,
        cluster_labels=cluster_labels,
    )
    meta = {
        "cluster_names": {str(k): v for k, v in (cluster_names or {}).items()},
        "model_name": model_name, "fps": fps, "k": k,
        "feature_dim": int(training_features.shape[1]),
        "n_clusters": int(len(set(cluster_labels[cluster_labels >= 0]))),
        "has_transform": bool(transform),
    }
    if transform:
        arrays["pca_components"] = np.asarray(transform["components"], dtype=np.float32)
        arrays["pca_mean"] = np.asarray(transform["mean"], dtype=np.float32)
        meta["normalize"] = str(transform.get("normalize", "l2"))
        meta["k_prime"] = int(transform.get("k_prime", 0))
        meta["raw_feature_dim"] = int(transform.get("raw_feature_dim", 0))
    arrays["metadata"] = np.array([json.dumps(meta)])
    np.savez_compressed(output_path, **arrays)
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
        pca_components=(data["pca_components"] if "pca_components" in data.files else None),
        pca_mean=(data["pca_mean"] if "pca_mean" in data.files else None),
        normalize=meta.get("normalize", ""),
        k_prime=meta.get("k_prime", 0),
        raw_feature_dim=meta.get("raw_feature_dim", 0),
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


def _transform_new_features(model: "ClusterModel", new_features: np.ndarray) -> np.ndarray:
    """Map RAW new-project features into the model's reduced space.

    Reproduces the Prepare transform stored in the model: per-sample L2 (if
    used) → centre → PCA basis → slice ``k_prime``. No-op when the model has no
    transform (legacy raw-feature model). NaN rows pass through as NaN so they
    are handled the same as during training.
    """
    if not model.has_transform:
        return new_features
    X = np.asarray(new_features, dtype=np.float32)
    if X.shape[1] != model.raw_feature_dim:
        raise ValueError(
            f"Raw feature dim mismatch: {X.shape[1]} vs expected {model.raw_feature_dim} "
            f"(the model was built from a prepared cache of {model.raw_feature_dim}-d latents)."
        )
    if model.normalize == "l2":
        norm = np.linalg.norm(X, axis=1, keepdims=True)
        X = X / np.maximum(norm, 1e-8)
    reduced = (X - model.pca_mean) @ model.pca_components.T
    if model.k_prime:
        reduced = reduced[:, : model.k_prime]
    return np.asarray(reduced, dtype=np.float32)


def apply_cluster_model(model, new_features, method="knn_feature") -> dict:
    """Apply saved model to new features.

    For a prepared-session model the RAW ``new_features`` are first mapped into
    the model's reduced space (L2 → PCA basis → k') before k-NN; legacy models
    use the raw features directly.

    Methods:
      knn_feature: k-NN in the model's feature space (cosine). Recommended.
      knn_umap: Approximate UMAP projection via weighted interpolation, then k-NN in 2D.
    """
    from sklearn.neighbors import NearestNeighbors

    new_features = _transform_new_features(model, new_features)
    if new_features.shape[1] != model.training_features.shape[1]:
        raise ValueError(f"Feature dim mismatch: {new_features.shape[1]} vs {model.training_features.shape[1]}")
    n = len(new_features)
    if n == 0:
        return {"labels": np.array([], dtype=int), "confidence": np.array([]), "cluster_names": model.cluster_names}

    # Tracking-loss frames arrive as whole-NaN rows (and the prepared transform
    # propagates that NaN). sklearn k-NN rejects NaN, so classify only the finite
    # rows and leave the rest as -1 (unclustered), keeping the output the same
    # length as the input so transferred_labels.csv stays frame-aligned.
    valid = np.isfinite(np.asarray(new_features)).all(axis=1)
    labels = np.full(n, -1, dtype=int)
    confidence = np.zeros(n, dtype=float)
    if not valid.any():
        out = {"labels": labels, "confidence": confidence, "cluster_names": model.cluster_names}
        if method == "knn_umap":
            out["umap_projection"] = np.full((n, 2), np.nan)
        return out

    feats = np.asarray(new_features)[valid]
    nn = NearestNeighbors(n_neighbors=model.k, metric='cosine')
    nn.fit(model.training_features)
    distances, indices = nn.kneighbors(feats)

    if method == "knn_feature":
        for j, i in enumerate(np.where(valid)[0]):
            label, conf = _majority_vote_excluding_noise(model.cluster_labels[indices[j]])
            labels[i] = label
            confidence[i] = conf
        return {"labels": labels, "confidence": confidence, "cluster_names": model.cluster_names}

    elif method == "knn_umap":
        weights = 1.0 / (distances + 1e-8)
        weights /= weights.sum(axis=1, keepdims=True)
        projected_v = np.zeros((len(feats), 2))
        for j in range(len(feats)):
            projected_v[j] = np.average(model.umap_embedding[indices[j]], weights=weights[j], axis=0)
        nn_umap = NearestNeighbors(n_neighbors=model.k)
        nn_umap.fit(model.umap_embedding)
        _, umap_indices = nn_umap.kneighbors(projected_v)
        for j, i in enumerate(np.where(valid)[0]):
            label, conf = _majority_vote_excluding_noise(model.cluster_labels[umap_indices[j]])
            labels[i] = label
            confidence[i] = conf
        projected = np.full((n, 2), np.nan)
        projected[valid] = projected_v
        return {"labels": labels, "confidence": confidence,
                "umap_projection": projected, "cluster_names": model.cluster_names}
    else:
        raise ValueError(f"Unknown method: {method}")
