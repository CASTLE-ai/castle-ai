"""Tests for cluster transfer: save/load/apply clustering models."""
import pytest
import numpy as np
import tempfile
import os
from castle.core.cluster_transfer import (
    save_cluster_model,
    load_cluster_model,
    apply_cluster_model,
    ClusterModel,
)


@pytest.fixture
def simple_model():
    """Create a simple 2-cluster model for testing."""
    np.random.seed(42)
    n_train = 100
    feature_dim = 64
    
    # Cluster 0: centered at (0, 0)
    features_0 = np.random.randn(50, feature_dim) * 0.5
    # Cluster 1: centered at (5, 5)
    features_1 = np.random.randn(50, feature_dim) * 0.5 + 2.0
    
    training_features = np.vstack([features_0, features_1])
    cluster_labels = np.array([0] * 50 + [1] * 50)
    
    # Simulate UMAP embedding in 2D
    umap_embedding = np.random.randn(n_train, 2)
    umap_embedding[:50] = np.random.randn(50, 2) * 0.5
    umap_embedding[50:] = np.random.randn(50, 2) * 0.5 + 5.0
    
    cluster_names = {0: "rest", 1: "active"}
    
    return ClusterModel(
        umap_embedding=umap_embedding,
        training_features=training_features,
        cluster_labels=cluster_labels,
        cluster_names=cluster_names,
        k=5,
        model_name="test_model",
        fps=30.0,
        feature_dim=feature_dim,
        n_clusters=2,
    )


def test_save_load_roundtrip(simple_model):
    """Test that save/load preserves model data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.npz")
        
        save_cluster_model(
            output_path=model_path,
            umap_embedding=simple_model.umap_embedding,
            training_features=simple_model.training_features,
            cluster_labels=simple_model.cluster_labels,
            cluster_names=simple_model.cluster_names,
            model_name=simple_model.model_name,
            fps=simple_model.fps,
            k=simple_model.k,
        )
        
        loaded = load_cluster_model(model_path)
        
        assert np.allclose(loaded.umap_embedding, simple_model.umap_embedding)
        assert np.allclose(loaded.training_features, simple_model.training_features)
        assert np.array_equal(loaded.cluster_labels, simple_model.cluster_labels)
        assert loaded.cluster_names == simple_model.cluster_names
        assert loaded.k == simple_model.k
        assert loaded.model_name == simple_model.model_name
        assert loaded.fps == simple_model.fps
        assert loaded.feature_dim == simple_model.feature_dim
        assert loaded.n_clusters == simple_model.n_clusters


def test_knn_feature_exact_training_data(simple_model):
    """Applying model to its own training data should return mostly same labels with high confidence."""
    result = apply_cluster_model(simple_model, simple_model.training_features, method="knn_feature")
    
    # Most labels should match (clusters might have some overlap at boundaries)
    accuracy = np.mean(result["labels"] == simple_model.cluster_labels)
    assert accuracy >= 0.9, f"Expected >=90% accuracy on training data, got {accuracy:.2%}"
    
    # Mean confidence should be high
    assert np.mean(result["confidence"]) >= 0.8
    assert result["cluster_names"] == simple_model.cluster_names


def test_knn_feature_new_data(simple_model):
    """Test applying model to new data similar to training clusters."""
    np.random.seed(43)
    # New data similar to cluster 0 (centered at 0)
    new_features = np.random.randn(10, simple_model.feature_dim) * 0.5
    
    result = apply_cluster_model(simple_model, new_features, method="knn_feature")
    
    assert len(result["labels"]) == 10
    assert len(result["confidence"]) == 10
    # Most should be classified as cluster 0
    assert np.sum(result["labels"] == 0) >= 7
    assert np.all((result["confidence"] >= 0.0) & (result["confidence"] <= 1.0))


def test_knn_umap_returns_projection(simple_model):
    """Test that knn_umap method returns umap_projection key."""
    new_features = np.random.randn(5, simple_model.feature_dim) * 0.5
    
    result = apply_cluster_model(simple_model, new_features, method="knn_umap")
    
    assert "umap_projection" in result
    assert result["umap_projection"].shape == (5, 2)
    assert len(result["labels"]) == 5
    assert len(result["confidence"]) == 5


def test_different_k_values():
    """Test that different k values work correctly."""
    np.random.seed(44)
    n_train = 60
    feature_dim = 32
    training_features = np.random.randn(n_train, feature_dim)
    cluster_labels = np.array([0] * 30 + [1] * 30)
    umap_embedding = np.random.randn(n_train, 2)
    cluster_names = {0: "a", 1: "b"}
    
    for k in [1, 3, 5, 10, 15]:
        model = ClusterModel(
            umap_embedding=umap_embedding,
            training_features=training_features,
            cluster_labels=cluster_labels,
            cluster_names=cluster_names,
            k=k,
            feature_dim=feature_dim,
            n_clusters=2,
        )
        
        new_features = np.random.randn(5, feature_dim)
        result = apply_cluster_model(model, new_features, method="knn_feature")
        
        assert len(result["labels"]) == 5
        assert len(result["confidence"]) == 5


def test_single_cluster_edge_case():
    """Test model with only one cluster."""
    np.random.seed(45)
    n_train = 50
    feature_dim = 32
    training_features = np.random.randn(n_train, feature_dim)
    cluster_labels = np.zeros(n_train, dtype=int)  # All in cluster 0
    umap_embedding = np.random.randn(n_train, 2)
    cluster_names = {0: "single"}
    
    model = ClusterModel(
        umap_embedding=umap_embedding,
        training_features=training_features,
        cluster_labels=cluster_labels,
        cluster_names=cluster_names,
        k=5,
        feature_dim=feature_dim,
        n_clusters=1,
    )
    
    new_features = np.random.randn(10, feature_dim)
    result = apply_cluster_model(model, new_features, method="knn_feature")
    
    # All should be classified as cluster 0 with confidence 1.0
    assert np.all(result["labels"] == 0)
    assert np.all(result["confidence"] == 1.0)


def test_feature_dim_mismatch_error(simple_model):
    """Test that mismatched feature dimensions raise error."""
    wrong_features = np.random.randn(10, 128)  # Wrong dim
    
    with pytest.raises(ValueError, match="Feature dim mismatch"):
        apply_cluster_model(simple_model, wrong_features, method="knn_feature")


def test_empty_new_features(simple_model):
    """Test handling of empty new feature array."""
    empty_features = np.empty((0, simple_model.feature_dim))
    
    result = apply_cluster_model(simple_model, empty_features, method="knn_feature")
    
    assert len(result["labels"]) == 0
    assert len(result["confidence"]) == 0
    assert result["cluster_names"] == simple_model.cluster_names


def test_cluster_names_preserved(simple_model):
    """Test that cluster names are preserved through save/load/apply."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.npz")
        
        save_cluster_model(
            output_path=model_path,
            umap_embedding=simple_model.umap_embedding,
            training_features=simple_model.training_features,
            cluster_labels=simple_model.cluster_labels,
            cluster_names=simple_model.cluster_names,
            k=5,
        )
        
        loaded = load_cluster_model(model_path)
        result = apply_cluster_model(loaded, simple_model.training_features[:5], method="knn_feature")
        
        assert result["cluster_names"] == simple_model.cluster_names


def test_unknown_method_error(simple_model):
    """Test that unknown method raises error."""
    new_features = np.random.randn(5, simple_model.feature_dim)
    
    with pytest.raises(ValueError, match="Unknown method"):
        apply_cluster_model(simple_model, new_features, method="invalid_method")


def test_large_data_performance():
    """Test that transfer works efficiently with larger datasets."""
    import time
    np.random.seed(46)
    
    n_train = 1000
    n_new = 500
    feature_dim = 768
    
    training_features = np.random.randn(n_train, feature_dim)
    cluster_labels = np.random.randint(0, 5, n_train)
    umap_embedding = np.random.randn(n_train, 2)
    cluster_names = {i: f"cluster_{i}" for i in range(5)}
    
    model = ClusterModel(
        umap_embedding=umap_embedding,
        training_features=training_features,
        cluster_labels=cluster_labels,
        cluster_names=cluster_names,
        k=5,
        feature_dim=feature_dim,
        n_clusters=5,
    )
    
    new_features = np.random.randn(n_new, feature_dim)
    
    start = time.time()
    result = apply_cluster_model(model, new_features, method="knn_feature")
    elapsed = time.time() - start
    
    assert len(result["labels"]) == n_new
    assert elapsed < 1.0  # Should complete in less than 1 second


def test_confidence_range(simple_model):
    """Test that confidence values are always in [0, 1]."""
    np.random.seed(47)
    new_features = np.random.randn(20, simple_model.feature_dim)
    
    result = apply_cluster_model(simple_model, new_features, method="knn_feature")
    
    assert np.all(result["confidence"] >= 0.0)
    assert np.all(result["confidence"] <= 1.0)


def test_no_allow_pickle_security():
    """Test that saved models don't require allow_pickle=True (security)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.npz")
        np.random.seed(48)
        
        save_cluster_model(
            output_path=model_path,
            umap_embedding=np.random.randn(50, 2),
            training_features=np.random.randn(50, 64),
            cluster_labels=np.array([0] * 25 + [1] * 25),
            cluster_names={0: "a", 1: "b"},
            k=5,
        )
        
        # This should work without allow_pickle=True
        loaded = load_cluster_model(model_path)
        assert loaded.n_clusters == 2


def test_knn_feature_vs_knn_umap_consistency(simple_model):
    """Test that both methods return valid results."""
    np.random.seed(49)
    new_features = np.random.randn(10, simple_model.feature_dim)
    
    result_feature = apply_cluster_model(simple_model, new_features, method="knn_feature")
    result_umap = apply_cluster_model(simple_model, new_features, method="knn_umap")
    
    # Both should return labels and confidence
    assert len(result_feature["labels"]) == 10
    assert len(result_umap["labels"]) == 10
    assert len(result_feature["confidence"]) == 10
    assert len(result_umap["confidence"]) == 10
    
    # Only umap should have projection
    assert "umap_projection" not in result_feature
    assert "umap_projection" in result_umap


def test_metadata_all_fields():
    """Test that all metadata fields are saved and loaded correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.npz")
        np.random.seed(50)
        
        save_cluster_model(
            output_path=model_path,
            umap_embedding=np.random.randn(30, 2),
            training_features=np.random.randn(30, 128),
            cluster_labels=np.array([0] * 10 + [1] * 10 + [2] * 10),
            cluster_names={0: "walk", 1: "groom", 2: "rear"},
            model_name="experiment_123",
            fps=60.0,
            k=7,
        )
        
        loaded = load_cluster_model(model_path)
        
        assert loaded.model_name == "experiment_123"
        assert loaded.fps == 60.0
        assert loaded.k == 7
        assert loaded.feature_dim == 128
        assert loaded.n_clusters == 3
        assert loaded.cluster_names == {0: "walk", 1: "groom", 2: "rear"}


def test_negative_cluster_ids_ignored():
    """Test that negative cluster IDs (noise) are handled correctly."""
    np.random.seed(51)
    n_train = 60
    feature_dim = 32
    training_features = np.random.randn(n_train, feature_dim)
    # Include noise points (-1)
    cluster_labels = np.array([0] * 20 + [1] * 20 + [-1] * 20)
    umap_embedding = np.random.randn(n_train, 2)
    cluster_names = {0: "a", 1: "b"}  # -1 not in names
    
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.npz")
        
        save_cluster_model(
            output_path=model_path,
            umap_embedding=umap_embedding,
            training_features=training_features,
            cluster_labels=cluster_labels,
            cluster_names=cluster_names,
            k=5,
        )
        
        loaded = load_cluster_model(model_path)
        # n_clusters should only count non-negative clusters
        assert loaded.n_clusters == 2
        
        # Applying should still work — noise points can be neighbors
        new_features = np.random.randn(5, feature_dim)
        result = apply_cluster_model(loaded, new_features, method="knn_feature")
        assert len(result["labels"]) == 5


# ---------------------------------------------------------------------------
# PR1 Stage 1.4: -1 (noise) neighbours must not win votes or dilute confidence
# ---------------------------------------------------------------------------

def _tiny_model(cluster_labels, feature_dim=4, k=None):
    """A model whose k == n_train, so every training point neighbours any query
    (deterministic regardless of the cosine metric)."""
    n = len(cluster_labels)
    rng = np.random.default_rng(0)
    return ClusterModel(
        umap_embedding=rng.standard_normal((n, 2)),
        training_features=rng.standard_normal((n, feature_dim)),
        cluster_labels=np.asarray(cluster_labels),
        cluster_names={0: "rest", 1: "active"},
        k=k or n,
        feature_dim=feature_dim,
        n_clusters=len({int(c) for c in cluster_labels if c >= 0}),
    )


def test_noise_neighbours_excluded_from_vote():
    """A noise majority among neighbours must not make the prediction -1."""
    model = _tiny_model([-1, -1, 0])  # k=3, 2 noise + 1 real
    q = np.random.default_rng(1).standard_normal((1, 4))
    res = apply_cluster_model(model, q, method="knn_feature")
    assert res["labels"][0] == 0           # the real label, not the noise majority
    assert res["confidence"][0] == 1.0     # 1 valid neighbour, unanimous


def test_all_noise_neighbours_unclassified():
    """When every neighbour is noise the sample is genuinely unclassifiable."""
    model = _tiny_model([-1, -1, -1])
    q = np.random.default_rng(2).standard_normal((1, 4))
    res = apply_cluster_model(model, q, method="knn_feature")
    assert res["labels"][0] == -1
    assert res["confidence"][0] == 0.0


def test_confidence_uses_valid_neighbour_denominator():
    """Confidence is winning count / number of VALID neighbours (noise excluded)."""
    model = _tiny_model([-1, -1, 0, 0, 1])  # k=5; valid = [0,0,1]
    q = np.random.default_rng(3).standard_normal((1, 4))
    res = apply_cluster_model(model, q, method="knn_feature")
    assert res["labels"][0] == 0
    assert res["confidence"][0] == pytest.approx(2 / 3)  # 2 wins / 3 valid


def test_noise_excluded_in_knn_umap_too():
    model = _tiny_model([-1, -1, 0])
    q = np.random.default_rng(4).standard_normal((1, 4))
    res = apply_cluster_model(model, q, method="knn_umap")
    assert res["labels"][0] == 0
