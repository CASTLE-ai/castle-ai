"""Regression tests for the P0 scientific-correctness fixes (2026-06 audit).

Each test locks one confirmed P0 finding so the fix cannot silently regress:

* P0-1  error-frame masks must be 2D (not blank_page's 3D frame).
* P0-3  session restore must reject a truncated time_series CSV instead of
        silently mis-aligning every subsequent video's labels.
* P0-4  multi-video subtitles must use each video's own fps.
* P0-5  DINOv3 mask alignment must mirror the frame geometry (resize +
        center-crop) for non-square input, while staying bit-identical to the
        old stretch for square input (no behaviour change for preprocessed crops).

These are deliberately lightweight (no model load, no GPU). The batch-failure
timeline-integrity fix (P0-2) needs a real video + mask store and lives in
tests/integration/test_p0_timeline_integrity.py.
"""

import os
from types import SimpleNamespace

import numpy as np
import pytest


# --------------------------------------------------------------------------- #
# P0-1: error-frame mask is 2D
# --------------------------------------------------------------------------- #

class _FakeReader:
    def __init__(self, frame):
        self._frame = frame

    def __getitem__(self, idx):
        return self._frame


class _FakeTracker:
    def __init__(self, mask):
        self._mask = mask

    def read_mask(self, idx):
        return self._mask

    def close(self):
        pass


def test_videodataset_error_frame_mask_is_2d():
    """A skipped (error) frame must yield a 2D mask, not the 3D blank_page frame."""
    from castle.core.data import Preprocess, VideoDataset

    pre = Preprocess(
        center_roi_switch=True, center_roi_id=7,
        center_roi_crop_width=64, center_roi_crop_height=48,
    )
    ds = VideoDataset("v.mp4", 1, "m.h5", pre, select_roi=7, on_frame_error="skip")
    # ROI 7 is absent from the mask -> ROINotFoundError -> the skip/blank path.
    ds.reader = _FakeReader(np.zeros((100, 120, 3), dtype=np.uint8))
    ds.tracker = _FakeTracker(np.zeros((100, 120), dtype=np.uint8))

    pf, pm = ds[0]

    assert pm.ndim == 2, "error-frame mask must be 2D (H, W), not a 3D frame"
    assert pm.shape == (48, 64)
    assert pf.ndim == 3 and pf.shape == (48, 64, 3)
    assert not pm.any(), "placeholder mask should be all background"


# --------------------------------------------------------------------------- #
# P0-3: session restore rejects a truncated time_series CSV
# --------------------------------------------------------------------------- #

def test_restore_rejects_truncated_time_series(tmp_path):
    """A short/corrupt time_series CSV must raise, not silently mis-align labels."""
    import pandas as pd

    from castle.core.types import CastleDataError
    from castle.service.clustering_service import ClusteringSession

    storage = str(tmp_path)
    project = "proj"
    cluster_dir = os.path.join(storage, project, "cluster")
    os.makedirs(cluster_dir, exist_ok=True)

    pd.DataFrame([{"Id": 0, "Name": "init", "Color": "grey"}]).to_csv(
        os.path.join(cluster_dir, "id.csv"), index=False
    )
    # Video "vid.mp4" expects vn=4 bins at time_window=5 -> a healthy CSV has 20
    # frame rows. Write a TRUNCATED CSV (14 rows): [::5] yields 3 bins != 4.
    pd.DataFrame({"behavior": [0] * 14}).to_csv(
        os.path.join(cluster_dir, "time_series_vid.csv"), index=False
    )

    sess = object.__new__(ClusteringSession)
    sess.storage_path = storage
    sess.project_name = project
    sess.latents = SimpleNamespace(
        cluster=np.zeros(4, dtype=int),
        time_window=5,
        cluster_meta={},
        behavior_name2cluster_id={},
        used_palette=set(),
        num_cluster=0,
    )
    sess.aggregator = SimpleNamespace(videos_meta=[(4, "vid.mp4")])

    with pytest.raises(CastleDataError):
        sess.restore()


def test_restore_accepts_intact_time_series(tmp_path):
    """A correctly-sized CSV restores without error and assigns all bins."""
    import pandas as pd

    from castle.service.clustering_service import ClusteringSession

    storage = str(tmp_path)
    project = "proj"
    cluster_dir = os.path.join(storage, project, "cluster")
    os.makedirs(cluster_dir, exist_ok=True)

    pd.DataFrame([{"Id": 0, "Name": "init", "Color": "grey"}]).to_csv(
        os.path.join(cluster_dir, "id.csv"), index=False
    )
    # 4 bins * time_window 5 = 20 frame rows; bins alternate 0/1 every 5 frames.
    behavior = np.repeat([0, 1, 0, 1], 5)
    pd.DataFrame({"behavior": behavior}).to_csv(
        os.path.join(cluster_dir, "time_series_vid.csv"), index=False
    )

    sess = object.__new__(ClusteringSession)
    sess.storage_path = storage
    sess.project_name = project
    sess.latents = SimpleNamespace(
        cluster=np.full(4, -9, dtype=int),
        time_window=5,
        cluster_meta={},
        behavior_name2cluster_id={},
        used_palette=set(),
        num_cluster=0,
    )
    sess.aggregator = SimpleNamespace(videos_meta=[(4, "vid.mp4")])

    result = sess.restore()
    assert result["success"] is True
    np.testing.assert_array_equal(sess.latents.cluster, [0, 1, 0, 1])


# --------------------------------------------------------------------------- #
# P0-4: per-video fps in subtitle generation
# --------------------------------------------------------------------------- #

def test_generate_subtitles_uses_per_video_fps(tmp_path):
    """Each video's .srt must use its own fps, not the first video's."""
    from castle.core.cluster import LatentAggregator, frame_to_timestamp

    agg = object.__new__(LatentAggregator)
    agg._video_reader_cache = {}  # __del__/close() touches this; we skipped __init__
    agg.project_path = str(tmp_path)
    agg.bin_size = 1
    agg.videos_meta = [(2, "fast.mp4"), (2, "slow.mp4")]
    agg.fps = 30.0  # first-video fallback that the OLD code wrongly used for all
    agg.fps_per_video = {"fast.mp4": 60.0, "slow.mp4": 24.0}

    # 4 bins (1 frame/bin): video1 bins [0,1], video2 bins [2,3].
    syllables = np.array([0, 1, 0, 1], dtype=int)
    meta = {0: {"name": "A"}, 1: {"name": "B"}}

    files = agg.generate_subtitles(syllables, meta)
    assert len(files) == 2

    fast_txt = open(next(f for f in files if "fast" in f), encoding="utf-8").read()
    slow_txt = open(next(f for f in files if "slow" in f), encoding="utf-8").read()

    # Frame 1's timestamp differs by fps: 1/60 s vs 1/24 s.
    assert frame_to_timestamp(1, 60.0) in fast_txt
    assert frame_to_timestamp(1, 24.0) in slow_txt
    # The slow video must NOT inherit the fast (first) video's fps.
    assert frame_to_timestamp(1, 60.0) not in slow_txt


# --------------------------------------------------------------------------- #
# P0-5: DINOv3 mask alignment mirrors frame geometry
# --------------------------------------------------------------------------- #

def test_dinov3_mask_alignment_square_identity_nonsquare_crops():
    """Square input -> identical to old stretch; non-square -> resize+center-crop."""
    import torch
    import torch.nn.functional as F

    from castle.core.models import DINOv2Encoder, DINOv3Encoder

    torch.manual_seed(0)
    v3 = DINOv3Encoder()  # __init__ sets constants only; no model load
    img = v3.image_size   # 592

    # Square input: DINOv3 alignment must equal the naive stretch (bit-identical),
    # so preprocessed (square) crops -- 孟炫's data -- are unaffected.
    msq = (torch.rand(2, 100, 100) > 0.5).float()
    stretch_sq = F.interpolate(msq[:, None], size=(img, img), mode="nearest")[:, 0]
    assert torch.equal(v3._align_mask_to_input(msq, img), stretch_sq)

    # Non-square input: alignment must differ from the naive stretch (crop active)
    # and produce the correct (B, img, img) shape.
    mns = (torch.rand(2, 80, 160) > 0.5).float()
    aligned = v3._align_mask_to_input(mns, img)
    assert aligned.shape == (2, img, img)
    stretch_ns = F.interpolate(mns[:, None], size=(img, img), mode="nearest")[:, 0]
    assert not torch.equal(aligned, stretch_ns)

    # The DINOv2 path (base implementation) keeps the anisotropic stretch.
    v2 = DINOv2Encoder()
    res = v2.resolution  # 518
    stretch_v2 = F.interpolate(mns[:, None], size=(res, res), mode="nearest")[:, 0]
    assert torch.equal(v2._align_mask_to_input(mns, res), stretch_v2)


# --------------------------------------------------------------------------- #
# Follow-up (manual-test feedback): session-latent path resolution
# --------------------------------------------------------------------------- #

def test_resolve_latent_path_handles_session_prefix(tmp_path):
    """KIT session latents register a logical '{session_id}/{file}' key but are
    stored flat (disambiguated by a _pre-{session} suffix); the loader must
    resolve to the flat file, not a phantom '{session_id}/' sub-directory."""
    from castle.core.cluster import _resolve_latent_path

    latent_dir = tmp_path / "latent" / "dinov3_vitb16"
    latent_dir.mkdir(parents=True)
    fname = "mouse06_ROI_1_dinov3_vitb16_rmbg_spp1x2x4_pre-02dfd768.npz"
    (latent_dir / fname).write_bytes(b"x")  # flat file on disk

    # Logical key with the session prefix (what config['latent'] stores).
    key = f"02dfd768/{fname}"
    resolved = _resolve_latent_path(str(latent_dir), key)
    assert os.path.exists(resolved)
    assert os.path.basename(resolved) == fname
    # Resolves flat in latent_dir, not under a phantom '02dfd768/' sub-directory.
    assert os.path.dirname(resolved) == str(latent_dir)

    # Non-session (flat) keys still resolve directly.
    flat = "mouse06_ROI_1_dinov3_vitb16.npz"
    (latent_dir / flat).write_bytes(b"y")
    assert os.path.exists(_resolve_latent_path(str(latent_dir), flat))

    # A genuinely missing latent resolves to a non-existent path (caller warns).
    assert not os.path.exists(_resolve_latent_path(str(latent_dir), "nope/ghost.npz"))
