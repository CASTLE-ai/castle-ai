"""P0-2 regression: a tolerated batch failure must NOT shift the frame timeline.

Before the fix, a batch that raised inside the extraction loop was silently
dropped, so the saved latent array had fewer rows than the video had frames —
every frame after the gap was mis-indexed, and downstream cluster labels were
assigned to the wrong frames (a wrong-but-plausible result).

The fix keeps the timeline aligned: a tolerated failure becomes a NaN placeholder
of the exact row count, and the failed frame range is recorded in the metadata
sidecar. This test forces a middle batch to fail and asserts the contract.
"""

import numpy as np
import pytest


def test_failed_middle_batch_becomes_nan_placeholder(dummy_project, monkeypatch):
    from castle.core import extractor as extractor_mod
    from castle.core.data import Preprocess
    from castle.utils.latent_metadata import load_latent_metadata

    storage_path, project_name, video_name = dummy_project

    # Deterministic single-process loader (no worker spawn / pickling).
    monkeypatch.setattr(extractor_mod, "get_num_workers", lambda *a, **k: 0)

    FEATURE_DIM = 8

    class _FakeObserver:
        """Returns constant features, but raises on the 2nd (middle) batch."""

        def __init__(self):
            self.calls = 0

        def extract_tensor_batch(self, frames, masks, roi_id,
                                 pooling="weighted_average", scales=None, layers=None):
            self.calls += 1
            n = int(frames.shape[0])
            if self.calls == 2:
                raise RuntimeError("simulated transient GPU error on middle batch")
            return np.full((n, FEATURE_DIM), float(self.calls), dtype=np.float32)

    monkeypatch.setattr(extractor_mod, "_get_observer", lambda model_name: _FakeObserver())

    # No centering -> raw frames; ROI 1 is present in every dummy mask.
    pre = Preprocess()

    out_path = extractor_mod.extract_roi_latent_from_video(
        storage_path, project_name, video_name,
        roi_id=1, model_name="fakemodel", batch_size=10,
        preprocess_config=pre, skip_existing=False,
        on_frame_error="skip", max_batch_failure_rate=0.5,
    )

    arr = np.load(out_path)["latent"]

    # 30 frames / batch_size 10 -> 3 batches; the middle batch (rows 10:20) failed.
    assert arr.shape == (30, FEATURE_DIM), "row count must equal the frame count"
    assert np.isnan(arr[10:20]).all(), "failed batch -> NaN placeholder, not dropped"
    assert np.isfinite(arr[0:10]).all(), "frames before the gap stay aligned"
    assert np.isfinite(arr[20:30]).all(), "frames after the gap stay aligned"

    meta = load_latent_metadata(out_path)
    assert meta["tags"]["failed_frame_ranges"] == [[10, 20]]

    # Extract-config provenance is recorded in the sidecar (so a latent is
    # self-describing: was background removed? which preprocess session? etc.).
    tags = meta["tags"]
    assert "remove_background" in tags and isinstance(tags["remove_background"], bool)
    assert tags["preprocess_session_id"] is None  # raw source, no session
    assert tags["device"] is None                 # single-GPU default path
    assert tags["batch_size"] == 10
