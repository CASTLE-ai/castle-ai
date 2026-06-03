"""P2b unit tests: opt-in multi-GPU dispatch routing + canonical latent filename.

The numerical equivalence (single full-range == 2-GPU split-merge) and speedup
are verified end-to-end on real video/GPUs separately; here we lock the routing
gate (default off → single-GPU) and the shared filename helper (single & 2-GPU
must produce identical names so skip_existing / config keys match).
"""
import pytest


def test_auto_dispatch_routing(monkeypatch):
    import torch
    from castle.core import extractor as ex

    monkeypatch.setattr(ex, "extract_roi_latent_from_video", lambda *a, **k: "single")
    monkeypatch.setattr(ex, "extract_roi_latent_from_video_2gpu", lambda *a, **k: "2gpu")

    # Default (env unset) → single-GPU.
    monkeypatch.delenv("CASTLE_MULTI_GPU", raising=False)
    assert ex.extract_roi_latent_from_video_auto() == "single"

    # Falsy values → single-GPU.
    for val in ("0", "false", "no", "off", ""):
        monkeypatch.setenv("CASTLE_MULTI_GPU", val)
        assert ex.extract_roi_latent_from_video_auto() == "single"

    # Opt-in + >1 CUDA device → multi-GPU.
    monkeypatch.setenv("CASTLE_MULTI_GPU", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    assert ex.extract_roi_latent_from_video_auto() == "2gpu"

    # Opt-in but only 1 device → single-GPU (no benefit).
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    assert ex.extract_roi_latent_from_video_auto() == "single"


def test_latent_filename_canonical():
    from castle.core.data import Preprocess
    from castle.core.extractor import _latent_filename

    pre_rmbg = Preprocess(remove_background_switch=True)
    fn = _latent_filename(
        "mouse06.mp4", 1, "dinov3_vitb16", pre_rmbg,
        "multiscale", [1, 2, 4], None, "02dfd768",
    )
    assert fn == "mouse06_ROI_1_dinov3_vitb16_rmbg_spp1x2x4_pre-02dfd768.npz"

    # Plain case: no rmbg, weighted-average, no session.
    fn2 = _latent_filename(
        "v.mp4", 2, "dinov3_vitb16", Preprocess(),
        "weighted_average", None, None, None,
    )
    assert fn2 == "v_ROI_2_dinov3_vitb16.npz"
