"""
tests/unit/test_stabilized_camera.py

Comprehensive unit tests for castle.core.stabilized_camera.
Written against the expected API spec; tests may import-fail until
the implementation agent finishes — that is expected.
"""

import h5py
import numpy as np
import numpy.testing as npt
import pytest

# --------------------------------------------------------------------------- #
# Import guard: collect gracefully if not yet implemented                      #
# --------------------------------------------------------------------------- #
from castle.core.stabilized_camera import (
    StabilizedCamera,
    extract_centroids_from_masks,
    extract_orientations_from_masks,
)


# =========================================================================== #
# Shared fixtures                                                               #
# =========================================================================== #


@pytest.fixture
def fps():
    return 30.0


@pytest.fixture
def static_trajectory(fps):
    """N=200, animal perfectly still at (500, 400)."""
    N = 200
    positions = np.full((N, 2), [500.0, 400.0])
    angles = np.zeros(N)
    return positions, angles, fps


@pytest.fixture
def moving_trajectory(fps):
    """N=300, animal moving with both slow drift and fast jitter."""
    rng = np.random.default_rng(42)
    N = 300
    t = np.arange(N) / fps
    # slow drift (0.05 Hz) + fast jitter (8 Hz)
    x = 500 + 50 * np.sin(2 * np.pi * 0.05 * t) + 3 * np.sin(2 * np.pi * 8 * t)
    y = 400 + 30 * np.sin(2 * np.pi * 0.07 * t) + 3 * np.cos(2 * np.pi * 8 * t)
    positions = np.stack([x, y], axis=1)
    angles = np.degrees(np.unwrap(rng.uniform(0, 2 * np.pi, N)))
    return positions, angles, fps


@pytest.fixture
def static_cam(static_trajectory):
    positions, angles, fps = static_trajectory
    return StabilizedCamera(positions, angles, fps)


@pytest.fixture
def moving_cam(moving_trajectory):
    positions, angles, fps = moving_trajectory
    return StabilizedCamera(positions, angles, fps)


@pytest.fixture
def synthetic_frame():
    """800×800 uint8 BGR frame: white rectangle on black background."""
    frame = np.zeros((800, 800, 3), dtype=np.uint8)
    frame[350:450, 450:550] = 255  # white square ~centred at (500, 400)
    return frame


# =========================================================================== #
# Filter correctness                                                            #
# =========================================================================== #


class TestFilterCorrectness:
    def test_zero_phase_no_delay(self, fps):
        """
        Zero-phase (filtfilt) should not shift the step-response midpoint.
        A causal filter introduces group delay; filtfilt cancels it.
        """
        N = 500
        positions = np.zeros((N, 2))
        positions[N // 2 :, 0] = 100.0  # step at sample 250
        angles = np.zeros(N)

        cam = StabilizedCamera(positions, angles, fps, fc=4.0, order=2)
        filt_x = cam.pos_filtered[:, 0]

        # Find where filtered signal crosses 50 % of step amplitude
        half = 50.0
        crossings = np.where(np.diff(np.sign(filt_x - half)))[0]
        assert len(crossings) >= 1, "filtered step should cross the midpoint"

        midpoint_idx = crossings[0]
        # Allow ±5 samples tolerance around the true step at N//2
        assert abs(midpoint_idx - N // 2) <= 5, (
            f"filtfilt midpoint {midpoint_idx} too far from true step at {N // 2}"
        )

    def test_lowpass_removes_high_freq(self, fps):
        """
        Signal = slow sinusoid (0.1 Hz) + fast sinusoid (5 Hz).
        After lowpass at fc=0.25 Hz, the 5 Hz component should be >90 % attenuated.
        """
        N = 1000
        t = np.arange(N) / fps
        slow = np.sin(2 * np.pi * 0.1 * t)
        fast = np.sin(2 * np.pi * 5.0 * t)
        signal = slow + fast

        positions = np.column_stack([signal, np.zeros(N)])
        angles = np.zeros(N)
        cam = StabilizedCamera(positions, angles, fps, fc=0.25, order=2)

        filt = cam.pos_filtered[:, 0]

        # Residual high-freq: subtract reconstructed slow component
        residual = filt - slow
        rms_fast_in = np.sqrt(np.mean(fast**2))
        rms_residual = np.sqrt(np.mean(residual**2))

        attenuation = rms_residual / rms_fast_in
        assert attenuation < 0.10, (
            f"5 Hz component should be >90 % attenuated, got {attenuation:.3f}"
        )

    def test_filter_short_signal(self, fps):
        """N=10 — should not crash even though filtfilt padlen may exceed signal length."""
        N = 10
        positions = np.random.default_rng(0).uniform(0, 100, (N, 2))
        angles = np.zeros(N)
        cam = StabilizedCamera(positions, angles, fps, fc=0.25, order=2)
        assert cam.pos_filtered.shape == (N, 2)
        assert cam.angle_filtered.shape == (N,)


# =========================================================================== #
# Crop size                                                                    #
# =========================================================================== #


class TestCropSize:
    def test_static_position_min_crop(self, static_cam):
        """Static position → zero displacement → every frame at min_crop."""
        assert np.all(static_cam.crop_sizes == static_cam.crop_sizes[0])
        # Default min_crop is 300
        assert static_cam.crop_sizes[0] >= 300

    def test_fast_movement_large_crop(self, fps):
        """Large positional jumps should produce crop sizes above min_crop."""
        N = 200
        positions = np.zeros((N, 2))
        # Alternate between two distant points to create large high-freq motion
        positions[::2] = [100, 100]
        positions[1::2] = [700, 700]
        angles = np.zeros(N)
        cam = StabilizedCamera(positions, angles, fps, fc=0.25, min_crop=300)
        # At least some frames should exceed min_crop
        assert np.any(cam.crop_sizes > 300)

    def test_crop_size_bounds(self, moving_cam):
        """crop_size must never fall below min_crop."""
        min_crop = 300  # default
        assert np.all(moving_cam.crop_sizes >= min_crop)

    def test_get_crop_size(self, static_cam):
        """get_crop_size(idx) should match crop_sizes[idx]."""
        for idx in [0, 10, len(static_cam.crop_sizes) - 1]:
            assert static_cam.get_crop_size(idx) == int(static_cam.crop_sizes[idx])


# =========================================================================== #
# Frame generation                                                             #
# =========================================================================== #


class TestFrameGeneration:
    def test_output_shape(self, static_cam, synthetic_frame):
        out = static_cam.generate_frame(synthetic_frame, 0)
        # The output frame must be square at the camera's configured output_size
        # (the attribute is `output_size`, not `_output_size`).
        expected = static_cam.output_size
        assert out.shape == (expected, expected, 3)

    def test_output_dtype(self, static_cam, synthetic_frame):
        out = static_cam.generate_frame(synthetic_frame, 0)
        assert out.dtype == np.uint8

    def test_static_frame_centered(self, fps):
        """
        Static animal at image centre → output frame centroid of bright region
        should be near the output centre.
        """
        H, W = 800, 800
        output_size = 518
        cx, cy = W // 2, H // 2  # 400, 400 — true centre

        frame = np.zeros((H, W, 3), dtype=np.uint8)
        r = 20
        frame[cy - r : cy + r, cx - r : cx + r] = 255

        N = 50
        positions = np.full((N, 2), [float(cx), float(cy)])
        angles = np.zeros(N)
        cam = StabilizedCamera(
            positions, angles, fps, min_crop=300, output_size=output_size
        )

        out = cam.generate_frame(frame, 25)
        gray = out[:, :, 0].astype(float)
        total = gray.sum()
        if total > 0:
            col_c = (gray * np.arange(output_size)[None, :]).sum() / total
            row_c = (gray * np.arange(output_size)[:, None]).sum() / total
            centre = output_size / 2
            assert abs(col_c - centre) < output_size * 0.2
            assert abs(row_c - centre) < output_size * 0.2

    def test_generate_frame_all_indices(self, static_cam, synthetic_frame):
        """generate_frame should not crash for any valid frame index."""
        N = len(static_cam.crop_sizes)
        for idx in [0, N // 2, N - 1]:
            out = static_cam.generate_frame(synthetic_frame, idx)
            assert out.ndim == 3


# =========================================================================== #
# Diagnostics                                                                  #
# =========================================================================== #


class TestDiagnostics:
    EXPECTED_KEYS = {
        "crop_sizes",
        "hp_residual_rms",
        "pct_at_min_crop",
        "speed_crop_correlation",
    }

    def test_diagnostics_keys(self, moving_cam):
        diag = moving_cam.get_diagnostics()
        assert isinstance(diag, dict)
        assert self.EXPECTED_KEYS <= diag.keys()

    def test_diagnostics_static(self, static_cam):
        """
        Static trajectory:
        - hp_residual_rms should be ~0 (no high-frequency residual)
        - pct_at_min_crop should be 100 (always at minimum crop)
        """
        diag = static_cam.get_diagnostics()
        assert diag["hp_residual_rms"] == pytest.approx(0.0, abs=1e-3)
        assert diag["pct_at_min_crop"] == pytest.approx(100.0, abs=1.0)

    def test_diagnostics_types(self, moving_cam):
        diag = moving_cam.get_diagnostics()
        assert isinstance(diag["hp_residual_rms"], float)
        assert isinstance(diag["pct_at_min_crop"], float)
        assert isinstance(diag["speed_crop_correlation"], float)


# =========================================================================== #
# Edge cases                                                                   #
# =========================================================================== #


class TestEdgeCases:
    def test_single_frame(self, fps):
        """N=1: filtering not possible, should handle gracefully."""
        positions = np.array([[500.0, 400.0]])
        angles = np.array([0.0])
        cam = StabilizedCamera(positions, angles, fps)
        assert cam.pos_filtered.shape == (1, 2)
        assert cam.angle_filtered.shape == (1,)
        assert cam.crop_sizes.shape == (1,)

    def test_nan_handling(self, fps):
        """NaN values in positions should not raise an exception."""
        N = 100
        rng = np.random.default_rng(7)
        positions = rng.uniform(300, 700, (N, 2))
        angles = np.zeros(N)
        # Inject some NaNs
        positions[10:15, 0] = np.nan
        positions[50, 1] = np.nan

        # Should not crash
        cam = StabilizedCamera(positions, angles, fps)
        assert cam.pos_filtered.shape == (N, 2)
        assert not np.any(np.isnan(cam.pos_filtered))  # NaNs should be handled

    def test_different_fc(self, fps):
        """Different fc values should produce different crop size distributions."""
        N = 300
        t = np.arange(N) / fps
        x = 500 + 200 * np.sin(2 * np.pi * 3 * t)
        y = 400 + 200 * np.sin(2 * np.pi * 3 * t)
        positions = np.stack([x, y], axis=1)
        angles = np.zeros(N)

        cam_low = StabilizedCamera(positions, angles, fps, fc=0.1)
        cam_high = StabilizedCamera(positions, angles, fps, fc=5.0)

        # Lower fc → more high-freq content in residual → larger crop sizes on average
        assert not np.array_equal(cam_low.crop_sizes, cam_high.crop_sizes), (
            "Different fc values should produce different crop distributions"
        )

    def test_two_frames(self, fps):
        """N=2: minimal viable trajectory — should not crash."""
        positions = np.array([[500.0, 400.0], [502.0, 401.0]])
        angles = np.array([0.0, 1.0])
        cam = StabilizedCamera(positions, angles, fps)
        assert cam.pos_filtered.shape == (2, 2)

    def test_large_margin_no_crash(self, fps):
        """Very large margin parameter: should not crash."""
        N = 100
        positions = np.full((N, 2), 500.0)
        angles = np.zeros(N)
        cam = StabilizedCamera(positions, angles, fps, margin=200)
        assert cam.pos_filtered.shape == (N, 2)


# =========================================================================== #
# Helper functions                                                             #
# =========================================================================== #


class TestExtractCentroids:
    @staticmethod
    def _make_h5io_masks(path, masks):
        """Write masks in H5IO format: keys are string frame indices."""
        with h5py.File(path, "w") as f:
            for i in range(len(masks)):
                f.create_dataset(str(i), data=masks[i], dtype="uint8",
                                 compression="gzip", compression_opts=3)

    def test_extract_centroids_from_masks_known_position(self, tmp_path):
        """
        Single white filled circle at known pixel → centroid should match.
        """
        H, W = 200, 200
        N = 5
        roi_id = 1
        masks = np.zeros((N, H, W), dtype=np.uint8)
        masks[:, 70:91, 90:111] = roi_id  # 21×21 square, centroid ≈ (100, 80)

        h5_path = str(tmp_path / "masks.h5")
        self._make_h5io_masks(h5_path, masks)

        centroids = extract_centroids_from_masks(h5_path, roi_id=roi_id, n_frames=N)
        assert centroids.shape == (N, 2)

        expected_x = 100.0
        expected_y = 80.0
        npt.assert_allclose(centroids[:, 0], expected_x, atol=1.0)
        npt.assert_allclose(centroids[:, 1], expected_y, atol=1.0)

    def test_extract_centroids_empty_mask(self, tmp_path):
        """All-zero mask → should raise ValueError (no valid centroids)."""
        H, W, N = 100, 100, 3
        masks = np.zeros((N, H, W), dtype=np.uint8)

        h5_path = str(tmp_path / "empty.h5")
        self._make_h5io_masks(h5_path, masks)

        with pytest.raises(ValueError, match="No valid centroids"):
            extract_centroids_from_masks(h5_path, roi_id=1, n_frames=N)

    def test_extract_centroids_multiple_frames(self, tmp_path):
        """Each frame has centroid at a different location — all should be recovered."""
        H, W, N = 300, 300, 4
        roi_id = 1
        masks = np.zeros((N, H, W), dtype=np.uint8)
        expected_centres = [(50, 60), (150, 100), (200, 250), (80, 200)]
        for i, (cx, cy) in enumerate(expected_centres):
            masks[i, cy - 5 : cy + 6, cx - 5 : cx + 6] = roi_id

        h5_path = str(tmp_path / "multi.h5")
        self._make_h5io_masks(h5_path, masks)

        centroids = extract_centroids_from_masks(h5_path, roi_id=roi_id, n_frames=N)
        for i, (cx, cy) in enumerate(expected_centres):
            npt.assert_allclose(centroids[i, 0], cx, atol=1.0)
            npt.assert_allclose(centroids[i, 1], cy, atol=1.0)


class TestExtractOrientations:
    @staticmethod
    def _make_h5io_masks(path, masks):
        """Write masks in H5IO format: keys are string frame indices."""
        with h5py.File(path, "w") as f:
            for i in range(len(masks)):
                f.create_dataset(str(i), data=masks[i], dtype="uint8",
                                 compression="gzip", compression_opts=3)

    def test_extract_orientations_rightward(self, tmp_path):
        """
        Body centroid at (100,100), head centroid at (150,100) → angle ≈ 0°
        """
        H, W, N = 200, 200, 3
        body_id, head_id = 1, 2
        masks = np.zeros((N, H, W), dtype=np.uint8)
        masks[:, 95:106, 95:106] = body_id   # body centroid ≈ (100, 100)
        masks[:, 95:106, 145:156] = head_id   # head centroid ≈ (150, 100)

        h5_path = str(tmp_path / "orient.h5")
        self._make_h5io_masks(h5_path, masks)

        angles = extract_orientations_from_masks(
            h5_path, body_roi_id=body_id, head_roi_id=head_id, n_frames=N
        )
        assert angles.shape == (N,)
        npt.assert_allclose(angles % 360, 0.0, atol=5.0)

    def test_extract_orientations_upward(self, tmp_path):
        """
        Head above body → angle should be consistent and not NaN.
        """
        H, W, N = 200, 200, 3
        body_id, head_id = 1, 2
        masks = np.zeros((N, H, W), dtype=np.uint8)
        masks[:, 120:131, 95:106] = body_id
        masks[:, 70:81, 95:106] = head_id

        h5_path = str(tmp_path / "up.h5")
        self._make_h5io_masks(h5_path, masks)

        angles = extract_orientations_from_masks(
            h5_path, body_roi_id=body_id, head_roi_id=head_id, n_frames=N
        )
        assert angles.shape == (N,)
        assert not np.any(np.isnan(angles))

    def test_extract_orientations_returns_unwrapped(self, tmp_path):
        """
        Returned angles should be unwrapped (no ±180° jumps).
        """
        H, W = 200, 200
        N = 20
        body_id, head_id = 1, 2

        masks_list = []
        for i in range(N):
            rad = np.deg2rad(i * 9)
            bx, by = 100, 100
            hx = int(bx + 30 * np.cos(rad))
            hy = int(by - 30 * np.sin(rad))
            mask = np.zeros((H, W), dtype=np.uint8)
            mask[by - 5 : by + 6, bx - 5 : bx + 6] = body_id
            hx = max(5, min(W - 6, hx))
            hy = max(5, min(H - 6, hy))
            mask[hy - 5 : hy + 6, hx - 5 : hx + 6] = head_id
            masks_list.append(mask)

        h5_path = str(tmp_path / "unwrap.h5")
        self._make_h5io_masks(h5_path, np.array(masks_list))

        angles = extract_orientations_from_masks(
            h5_path, body_roi_id=body_id, head_roi_id=head_id, n_frames=N
        )
        jumps = np.abs(np.diff(angles))
        assert np.all(jumps <= 15.0), f"Max jump {jumps.max():.1f}° suggests wrapping"


# =========================================================================== #
# StabilizedCamera attribute shapes                                            #
# =========================================================================== #


class TestAttributeShapes:
    def test_pos_filtered_shape(self, moving_cam, moving_trajectory):
        positions, _, _ = moving_trajectory
        N = len(positions)
        assert moving_cam.pos_filtered.shape == (N, 2)

    def test_angle_filtered_shape(self, moving_cam, moving_trajectory):
        positions, _, _ = moving_trajectory
        N = len(positions)
        assert moving_cam.angle_filtered.shape == (N,)

    def test_crop_sizes_shape(self, moving_cam, moving_trajectory):
        positions, _, _ = moving_trajectory
        N = len(positions)
        assert moving_cam.crop_sizes.shape == (N,)

    def test_crop_sizes_dtype_int(self, moving_cam):
        assert np.issubdtype(moving_cam.crop_sizes.dtype, np.integer)

    def test_pos_filtered_smooth(self, moving_trajectory):
        """Filtered positions should be smoother than raw (lower std of diff)."""
        positions, angles, fps = moving_trajectory
        cam = StabilizedCamera(positions, angles, fps, fc=0.25)
        raw_diff_std = np.std(np.diff(positions, axis=0))
        filt_diff_std = np.std(np.diff(cam.pos_filtered, axis=0))
        assert filt_diff_std < raw_diff_std


def test_get_diagnostics_constant_series_is_nan_no_warning(static_cam):
    """A perfectly still animal has zero-variance speed and crop size, so the
    speed-vs-crop correlation is undefined: report NaN and do NOT emit a
    RuntimeWarning from np.corrcoef dividing by a zero std (PR2 Stage 5)."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        diag = static_cam.get_diagnostics()
    assert np.isnan(diag["speed_crop_correlation"])
