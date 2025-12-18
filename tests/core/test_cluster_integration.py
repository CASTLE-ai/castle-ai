
import pytest
import numpy as np
import tempfile
import os
import shutil
from unittest.mock import patch
from castle.core.cluster import LatentAggregator

@pytest.mark.integration
@patch('castle.core.cluster.get_project_config')
def test_latent_aggregator_clustering(mock_get_project_config, device):
    """
    Test K-Means clustering in LatentAggregator.
    """
    # Create fake latents: 2 distinct clusters
    # Cluster A: around [1, 1, ... 1]
    # Cluster B: around [-1, -1, ... -1]
    dim = 128
    n_samples = 50
    
    cluster_a = np.ones((n_samples, dim), dtype=np.float32) + np.random.normal(0, 0.1, (n_samples, dim))
    cluster_b = -1 * np.ones((n_samples, dim), dtype=np.float32) + np.random.normal(0, 0.1, (n_samples, dim))
    
    latents = np.vstack([cluster_a, cluster_b]) # (100, 128)
    
    mock_get_project_config.return_value = ("/tmp/test_project", {})
    # Init Aggregator
    aggregator = LatentAggregator(storage_path="/tmp", project_name="test_project", select_roi_id=1, bin_size=10)
    # Manually set latents (usually done via __init__ loop over files, but we test logic)
    aggregator.latents = latents
    # Also mock videos_meta to allow generate_subtitles to run
    aggregator.videos_meta = [(len(latents) // aggregator.bin_size, "dummy_video.mp4")]
    
    # Generate Subtitles (Clustering)
    # We need a dummy output path
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a dummy project path to satisfy generate_subtitles
        dummy_project_path = os.path.join(temp_dir, "test_project")
        os.makedirs(os.path.join(dummy_project_path, "subtitles"), exist_ok=True)
        aggregator.project_path = dummy_project_path # Override the path

        # Re-introduce behavior_labels as a local variable
        behavior_labels = {0: "Behavior A", 1: "Behavior B"}

        syllables = np.zeros(len(latents), dtype=int) # Dummy syllables
        meta = {str(k): {'name': v} for k, v in behavior_labels.items()} # Convert behavior_labels to meta format
        
        generated_files = aggregator.generate_subtitles(syllables=syllables, meta=meta)
        
        # Check that a file was generated
        assert len(generated_files) > 0
        output_path = generated_files[0]
        assert os.path.exists(output_path)
        
        # Read SRT
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
            assert "Behavior A" in content
            
