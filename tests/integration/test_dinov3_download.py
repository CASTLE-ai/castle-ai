"""Smoke test for the DINOv3 HuggingFace loading path (post-P0-A').

The previous version of this file tested the legacy gdown + torch.hub
loading flow, which was retired on 2026-05-16. The new path goes through
``transformers.AutoModel.from_pretrained``. This test verifies that the
encoder can be constructed and (optionally, when network + HF cache are
available) that the model can actually load.

To run the full network-backed smoke test:

    CASTLE_TEST_HF_DOWNLOAD=1 pytest tests/integration/test_dinov3_download.py

Without that env var, only the offline encoder-construction test runs.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skip(
    reason="DINOv3 HuggingFace auto-download is intentionally NOT used — CASTLE "
    "loads DINOv3 from a local checkpoint via gdown by design (project decision "
    "2026-06-20: HF needs a user account + Meta license acceptance for no UX gain). "
    "These tests describe that unimplemented HF path."
)


def test_dinov3_encoder_construct_known_variant() -> None:
    """DINOv3Encoder can be initialised for the documented variants."""
    from castle.core.models import DINOv3Encoder

    for variant in ("dinov3_vitb16", "dinov3_vitl16", "dinov3_vits16"):
        enc = DINOv3Encoder(variant, device="cpu")
        assert enc.model_type == variant
        assert enc.hf_id.startswith("facebook/")


@pytest.mark.skipif(
    not os.environ.get("CASTLE_TEST_HF_DOWNLOAD"),
    reason="Set CASTLE_TEST_HF_DOWNLOAD=1 to opt in to the network-backed HF download test",
)
def test_dinov3_loading_from_huggingface() -> None:
    """End-to-end: load weights + processor from HF, run a tiny forward pass."""
    import torch
    from castle.core.models import DINOv3Encoder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = DINOv3Encoder("dinov3_vitb16", device=device)
    enc.load_model()

    assert enc.model is not None
    assert enc.processor is not None
    assert enc.n_feature == enc.model.config.hidden_size

    # Tiny synthetic forward pass — just verify the contract, not numeric quality.
    x = torch.zeros(1, 3, enc.image_size, enc.image_size, device=device)
    feats = enc.extract_features(x, layers=None)
    expected_patches = enc.target_patches ** 2  # 37*37 = 1369
    assert feats.shape == (1, expected_patches, enc.n_feature), (
        f"Unexpected feature shape {tuple(feats.shape)}; "
        f"expected (1, {expected_patches}, {enc.n_feature})"
    )
