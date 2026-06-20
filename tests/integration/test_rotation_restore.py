import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from castle.core.extractor import extract_roi_rotation_latent_from_video


class TestRotationLatent(unittest.TestCase):

    @patch('castle.core.extractor.VideoReader')
    @patch('castle.core.extractor.H5IO')
    @patch('castle.core.extractor._get_observer')
    @patch('castle.core.extractor.get_project_config')
    def test_rotation_extraction_logic(
        self, mock_get_config, mock_get_observer, mock_h5io, mock_videoreader
    ):
        """Rotation augmentation sends 7 rotated views per frame to the observer
        and averages them back to one latent row per frame.

        Only the compute boundary (video reader / tracker / observer) is mocked;
        filesystem writes are real (temp dir) so this exercises the actual
        atomic save + config-update path rather than mocking it away.
        """
        with tempfile.TemporaryDirectory() as root:
            storage_path = root
            project_name = "test_project"
            project_path = os.path.join(root, project_name)
            os.makedirs(project_path, exist_ok=True)
            # update_config does an atomic read-modify-write on config.json.
            with open(os.path.join(project_path, "config.json"), "w") as f:
                f.write("{}")
            mock_get_config.return_value = (project_path, {'latent': {}})

            # A real (empty) mask file so the existence check passes without
            # globally patching os.path.exists; H5IO itself is mocked.
            mask_path = os.path.join(root, "mask_list.h5")
            open(mask_path, "w").close()

            # Observer: returns a (N, 10) latent for each batch of N rotated views.
            mock_observer = MagicMock()
            mock_observer.n_feature = 10
            mock_observer.extract_tensor_batch.side_effect = (
                lambda frames, masks, roi: np.zeros((len(frames), 10))
            )
            mock_get_observer.return_value = mock_observer

            # Video: 2 frames.
            mock_vr_instance = MagicMock()
            mock_vr_instance.__len__.return_value = 2
            mock_vr_instance.get_frame.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
            mock_videoreader.return_value.__enter__.return_value = mock_vr_instance

            mock_tracker_instance = MagicMock()
            mock_tracker_instance.read_mask.return_value = np.zeros((100, 100), dtype=np.uint8)
            mock_h5io.return_value = mock_tracker_instance

            preprocess = MagicMock()
            preprocess.remove_background_switch = False
            preprocess.transform.return_value = (
                np.zeros((30, 30, 3)), np.zeros((30, 30))
            )

            result_path = extract_roi_rotation_latent_from_video(
                storage_path=storage_path,
                project_name=project_name,
                video_name='test.mp4',
                roi_id=1,
                model_name='dinov2_vitb14',
                batch_size=2,
                preprocess_config=preprocess,
                skip_existing=False,
                mask_path_override=mask_path,
            )

            # 2 frames x 7 rotated views = 14 items sent to the observer.
            self.assertTrue(mock_observer.extract_tensor_batch.called)
            frames_arg = mock_observer.extract_tensor_batch.call_args[0][0]
            self.assertEqual(len(frames_arg), 14)

            # The 7 rotations per frame are averaged -> (2, 10) saved latent.
            self.assertTrue(os.path.exists(result_path))
            saved = np.load(result_path)
            self.assertEqual(saved['latent'].shape, (2, 10))


if __name__ == '__main__':
    unittest.main()
