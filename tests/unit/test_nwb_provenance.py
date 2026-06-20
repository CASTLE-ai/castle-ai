"""NWB session_start_time + provenance (audit: nwb-session-start-export-time).

session_start_time must be the RECORDING start (NWB spec), not the export
wall-clock; the exported file should also record which CASTLE version produced it.
"""

import logging
import os
from datetime import datetime, timezone

import numpy as np
import pytest

from castle.core.nwb_export import HAS_NWB

pytestmark = pytest.mark.skipif(not HAS_NWB, reason="pynwb not installed")


def test_session_start_time_uses_supplied_recording_time(tmp_path):
    from pynwb import NWBHDF5IO

    from castle.core.nwb_export import export_to_nwb

    rec = datetime(2024, 5, 1, 9, 30, 0, tzinfo=timezone.utc)
    out = str(tmp_path / "p.nwb")
    export_to_nwb(
        out, np.array([0, 0, 1, 1], dtype=np.int32), fps=30.0,
        subject_id="m1", session_start_time=rec,
    )
    with NWBHDF5IO(out, "r") as io:
        nwb = io.read()
        assert nwb.session_start_time == rec                 # recording time, not "now"
        assert "castle-ai" in (nwb.source_script or "")       # CASTLE provenance present
        assert "m1" in nwb.identifier                         # stable, subject-scoped id


def test_missing_recording_time_warns_and_still_exports(tmp_path, caplog):
    from castle.core.nwb_export import export_to_nwb

    out = str(tmp_path / "p2.nwb")
    with caplog.at_level(logging.WARNING):
        export_to_nwb(out, np.array([0, 1], dtype=np.int32), fps=30.0)
    assert os.path.exists(out)
    assert any("session_start_time" in r.getMessage() for r in caplog.records)
