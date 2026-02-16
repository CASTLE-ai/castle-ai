"""Unit tests for NWB export functionality."""

import os
import tempfile

import numpy as np
import pytest

from castle.core.nwb_export import HAS_NWB

pytestmark = pytest.mark.skipif(not HAS_NWB, reason="pynwb not installed")


# ---------------------------------------------------------------------------
# Core export tests
# ---------------------------------------------------------------------------

class TestExportToNwb:
    def test_minimal_export(self, tmp_path):
        """Export with only cluster labels — minimal data."""
        from castle.core.nwb_export import export_to_nwb

        labels = np.array([0, 0, 1, 1, 2, 2, 0, 0], dtype=np.int32)
        output = str(tmp_path / "minimal.nwb")
        result = export_to_nwb(output, labels, fps=30.0)

        assert os.path.exists(result)
        assert result.endswith(".nwb")

    def test_full_export(self, tmp_path):
        """Export with labels + bout stats + transition matrix."""
        from castle.core.nwb_export import export_to_nwb

        K = 3
        labels = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 0, 0], dtype=np.int32)
        cluster_names = {0: "walking", 1: "grooming", 2: "resting"}
        bout_stats = {
            "0": {
                "cluster_name": "walking",
                "n_bouts": 3,
                "frequency": 0.45,
                "mean_duration_s": 0.1,
                "median_duration_s": 0.1,
                "cv_duration": 0.5,
            },
            "1": {
                "cluster_name": "grooming",
                "n_bouts": 1,
                "frequency": 0.18,
                "mean_duration_s": 0.067,
                "median_duration_s": 0.067,
                "cv_duration": 0.0,
            },
            "2": {
                "cluster_name": "resting",
                "n_bouts": 1,
                "frequency": 0.36,
                "mean_duration_s": 0.133,
                "median_duration_s": 0.133,
                "cv_duration": 0.0,
            },
        }
        transition_matrix = np.array([
            [0.0, 0.5, 0.5],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ])

        output = str(tmp_path / "full.nwb")
        result = export_to_nwb(
            output,
            labels,
            fps=30.0,
            cluster_names=cluster_names,
            bout_stats=bout_stats,
            transition_matrix=transition_matrix,
            session_description="Test session",
            experimenter="Test User",
            subject_id="mouse_001",
        )

        assert os.path.exists(result)

    def test_roundtrip_labels(self, tmp_path):
        """Export → read back → verify cluster labels match."""
        from castle.core.nwb_export import export_to_nwb
        from pynwb import NWBHDF5IO

        labels = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0], dtype=np.int32)
        output = str(tmp_path / "roundtrip.nwb")
        export_to_nwb(output, labels, fps=10.0)

        with NWBHDF5IO(output, "r") as io:
            nwbfile = io.read()
            behavior = nwbfile.processing["behavior"]
            bts = behavior.data_interfaces["behavioral_clusters"]
            ts = bts.time_series["cluster_labels"]
            read_labels = ts.data[:]

        np.testing.assert_array_equal(read_labels, labels)

    def test_roundtrip_transition_matrix(self, tmp_path):
        """Export → read back → verify transition matrix."""
        from castle.core.nwb_export import export_to_nwb
        from pynwb import NWBHDF5IO

        labels = np.array([0, 0, 1, 1], dtype=np.int32)
        tm = np.array([[0.0, 1.0], [1.0, 0.0]])
        output = str(tmp_path / "tm_roundtrip.nwb")
        export_to_nwb(output, labels, fps=30.0, transition_matrix=tm)

        with NWBHDF5IO(output, "r") as io:
            nwbfile = io.read()
            read_tm = nwbfile.scratch["transition_matrix"].data[:]

        np.testing.assert_array_almost_equal(read_tm, tm)

    def test_bout_intervals_created(self, tmp_path):
        """Bouts should be present as time intervals."""
        from castle.core.nwb_export import export_to_nwb
        from pynwb import NWBHDF5IO

        labels = np.array([0, 0, 0, 1, 1, 2, 0, 0], dtype=np.int32)
        output = str(tmp_path / "bouts.nwb")
        export_to_nwb(output, labels, fps=10.0,
                       cluster_names={0: "walk", 1: "groom", 2: "rest"})

        with NWBHDF5IO(output, "r") as io:
            nwbfile = io.read()
            behavior = nwbfile.processing["behavior"]
            bouts = behavior.data_interfaces["behavioral_bouts"]
            # Should have 4 bouts: [0,0,0], [1,1], [2], [0,0]
            assert len(bouts) == 4

    def test_output_path_creation(self, tmp_path):
        """Should create parent directories automatically."""
        from castle.core.nwb_export import export_to_nwb

        labels = np.array([0, 1, 0], dtype=np.int32)
        deep_path = str(tmp_path / "a" / "b" / "c" / "output.nwb")
        result = export_to_nwb(deep_path, labels, fps=30.0)
        assert os.path.exists(result)

    def test_single_cluster(self, tmp_path):
        """All frames same cluster — should still export cleanly."""
        from castle.core.nwb_export import export_to_nwb

        labels = np.zeros(100, dtype=np.int32)
        output = str(tmp_path / "single.nwb")
        result = export_to_nwb(output, labels, fps=30.0)
        assert os.path.exists(result)

    def test_fps_stored_correctly(self, tmp_path):
        """Verify that fps is stored in the time series."""
        from castle.core.nwb_export import export_to_nwb
        from pynwb import NWBHDF5IO

        labels = np.array([0, 1, 0, 1], dtype=np.int32)
        output = str(tmp_path / "fps.nwb")
        export_to_nwb(output, labels, fps=25.0)

        with NWBHDF5IO(output, "r") as io:
            nwbfile = io.read()
            behavior = nwbfile.processing["behavior"]
            bts = behavior.data_interfaces["behavioral_clusters"]
            ts = bts.time_series["cluster_labels"]
            assert ts.rate == pytest.approx(25.0)


# ---------------------------------------------------------------------------
# Missing pynwb graceful error
# ---------------------------------------------------------------------------

class TestMissingPynwb:
    def test_require_pynwb_when_available(self):
        """_require_pynwb should not raise when pynwb is installed."""
        from castle.core.nwb_export import _require_pynwb
        # Should not raise
        _require_pynwb()

    def test_has_nwb_flag(self):
        """HAS_NWB should be True when pynwb is available."""
        assert HAS_NWB is True


# ---------------------------------------------------------------------------
# Bout extraction helper
# ---------------------------------------------------------------------------

class TestExtractBouts:
    def test_simple_sequence(self):
        from castle.core.nwb_export import _extract_bouts

        labels = np.array([0, 0, 1, 1, 1, 2])
        bouts = _extract_bouts(labels, fps=10.0)
        assert len(bouts) == 3
        assert bouts[0]["cluster_id"] == 0
        assert bouts[0]["start_frame"] == 0
        assert bouts[0]["stop_frame"] == 1
        assert bouts[1]["cluster_id"] == 1
        assert bouts[2]["cluster_id"] == 2

    def test_empty_labels(self):
        from castle.core.nwb_export import _extract_bouts

        bouts = _extract_bouts(np.array([]), fps=30.0)
        assert len(bouts) == 0

    def test_single_frame(self):
        from castle.core.nwb_export import _extract_bouts

        bouts = _extract_bouts(np.array([5]), fps=30.0)
        assert len(bouts) == 1
        assert bouts[0]["cluster_id"] == 5


# ---------------------------------------------------------------------------
# CLI help works even without full pipeline
# ---------------------------------------------------------------------------

class TestCLIHelp:
    def test_ethogram_export_nwb_in_help(self):
        """The export-nwb command should be registered."""
        from castle.cli.ethogram_cmd import app
        # Check that the command exists
        command_names = [cmd.name for cmd in app.registered_commands]
        assert "export-nwb" in command_names
