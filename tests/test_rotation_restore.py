import unittest
import numpy as np
import sys
import os
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from castle.core.extractor import extract_roi_rotation_latent_from_video
from castle.core.data import Preprocess

class TestRotationLatent(unittest.TestCase):

    @patch('castle.core.extractor.VideoReader')
    @patch('castle.core.extractor.H5IO')
    @patch('castle.core.extractor._get_observer')
    @patch('castle.core.extractor.get_project_config')
    @patch('os.makedirs')
    @patch('numpy.savez_compressed')
    @patch('castle.core.extractor.save_project_config')
    @patch('os.path.exists')
    def test_rotation_extraction_logic(self, mock_exists, mock_save_project_config, mock_savez, mock_makedirs, mock_get_config, mock_get_observer, mock_h5io, mock_videoreader):
        
        # Setup Mocks
        mock_config = {'latent': {}}
        mock_get_config.return_value = ('/tmp/project', mock_config)
        
        # Mock Observer
        mock_observer = MagicMock()
        mock_observer.n_feature = 10
        mock_observer.extract_tensor_batch.side_effect = lambda frames, masks, roi: np.zeros((len(frames), 10)) # Returns (N*24, 10)
        mock_get_observer.return_value = mock_observer
        
        # Mock Video and Tracker
        mock_vr_instance = MagicMock()
        mock_vr_instance.__len__.return_value = 2 # 2 frames
        mock_vr_instance.get_frame.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_videoreader.return_value.__enter__.return_value = mock_vr_instance
        
        mock_tracker_instance = MagicMock()
        mock_tracker_instance.read_mask.return_value = np.zeros((100, 100), dtype=np.uint8)
        mock_h5io.return_value = mock_tracker_instance

        mock_exists.return_value = True # Simulate mask file existing

        # Mock Preprocess
        preprocess = MagicMock()
        preprocess.transform.return_value = (np.zeros((30, 30, 3)), np.zeros((30, 30))) # Return dummy crop
        
        # Run Function
        result_path = extract_roi_rotation_latent_from_video(
            storage_path='/tmp',
            project_name='test_project',
            video_name='test.mp4',
            roi_id=1,
            model_name='dinov2_vitb14',
            batch_size=2,
            preprocess_config=preprocess,
            skip_existing=False
        )
        
        # Verify
        # Total frames = 2. Batch size = 2.
        # Loop runs once.
        # Inside loop: 2 frames.
        # Each frame -> 7 rotations (Current impl).
        # Total items sent to observer: 2 * 7 = 14.
        
        # Check observer call
        self.assertTrue(mock_observer.extract_tensor_batch.called)
        call_args = mock_observer.extract_tensor_batch.call_args
        frames_arg = call_args[0][0]
        self.assertEqual(len(frames_arg), 14) 
        
        # Check save
        self.assertTrue(mock_savez.called)
        saved_latent = mock_savez.call_args[1]['latent']
        # Expected shape: (2, 10) because we averaged 24 rotations
        self.assertEqual(saved_latent.shape, (2, 10))
        
        print("Test passed: Rotation logic verified.")

if __name__ == '__main__':
    unittest.main()
