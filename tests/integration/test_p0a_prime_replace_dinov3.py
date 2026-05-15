"""Integration tests verifying the DINOv3 HuggingFace migration (P0-A').

The DINOv3 encoder was migrated from a private Google Drive checkpoint
(via gdown) to the official HuggingFace repository on 2026-05-16. These
tests verify the new code path without actually downloading model weights
(which is gated behind network + HF cache).

Behaviour covered:
- ``DINOV3_HF_MAP`` exists in config with the documented variants
- ``DINOv3Encoder`` accepts the standard variants and exposes the HF id
- Unknown variants raise a clear error
- Attention backend selection picks ``eager`` on CPU (no GPU expected
  in CI without a CUDA runtime)
- Legacy ``MODEL_TO_CKPT_FILENAME`` / ``MODEL_TO_NUM_LAYERS`` keys are
  removed from ``DINOV3_CONSTANTS``
- ``CKPT_DINO_IDS`` no longer holds DINOv3 entries
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CASTLE_DIR = REPO_ROOT / "castle"


def test_dinov3_hf_map_exposes_expected_variants() -> None:
    from castle.core.config import DINOV3_HF_MAP

    assert "dinov3_vitb16" in DINOV3_HF_MAP
    assert "dinov3_vitl16" in DINOV3_HF_MAP
    assert DINOV3_HF_MAP["dinov3_vitb16"] == "facebook/dinov3-vitb16-pretrain-lvd1689m"


def test_legacy_gdown_keys_are_removed() -> None:
    from castle.core.config import CKPT_DINO_IDS, DINOV3_CONSTANTS

    # CKPT_DINO_IDS may still hold DINOv2 placeholders but must not hold DINOv3
    for key in CKPT_DINO_IDS:
        assert not key.startswith("dinov3_"), (
            f"CKPT_DINO_IDS still contains DINOv3 key {key!r}; "
            "should have moved to DINOV3_HF_MAP"
        )

    # Legacy lookup tables should be gone — variant info now comes from HF config
    assert "MODEL_TO_CKPT_FILENAME" not in DINOV3_CONSTANTS
    assert "MODEL_TO_NUM_LAYERS" not in DINOV3_CONSTANTS


def test_dinov3_encoder_init_known_variant() -> None:
    from castle.core.models import DINOv3Encoder

    enc = DINOv3Encoder("dinov3_vitb16", device="cpu")
    assert enc.model_type == "dinov3_vitb16"
    assert enc.hf_id == "facebook/dinov3-vitb16-pretrain-lvd1689m"
    assert enc.image_size == 592
    assert enc.patch_size == 16
    assert enc.target_patches == 37


def test_dinov3_encoder_rejects_unknown_variant() -> None:
    from castle.core.models import DINOv3Encoder

    with pytest.raises(ValueError, match="Unknown DINOv3 variant"):
        DINOv3Encoder("dinov3_does_not_exist", device="cpu")


def test_dinov3_encoder_cpu_uses_eager_attention() -> None:
    from castle.core.models import DINOv3Encoder

    enc = DINOv3Encoder("dinov3_vitb16", device="cpu")
    assert enc._select_attn_impl() == "eager"
    assert enc._supports_bf16() is False


def test_no_gdown_call_remains_in_dinov3_path() -> None:
    """The DINOv3 code path should no longer reference gdown helpers."""
    models_src = (CASTLE_DIR / "core" / "models.py").read_text()
    # download_with_gdown can still be imported by AOT/visual_latent_extract,
    # but the DINOv3 class block itself must not reference it.
    dinov3_start = models_src.find("class DINOv3Encoder")
    assert dinov3_start != -1, "DINOv3Encoder class not found"
    dinov3_block = models_src[dinov3_start:]
    assert "gdown" not in dinov3_block.lower(), (
        "DINOv3Encoder still references gdown — should be HF-only after P0-A'"
    )


def test_transformers_is_pinned_in_requirements() -> None:
    req = (REPO_ROOT / "requirements.txt").read_text()
    assert "transformers" in req, (
        "requirements.txt must declare transformers (DINOv3 HF migration)"
    )


def test_no_legacy_download_dinov3_ckpt_helper() -> None:
    """The obsolete download_dinov3_ckpt helper should be gone."""
    result = subprocess.run(
        ["grep", "-rn", "download_dinov3_ckpt", str(CASTLE_DIR), "--include=*.py"],
        capture_output=True,
        text=True,
    )
    assert not result.stdout, (
        f"download_dinov3_ckpt should be removed:\n{result.stdout}"
    )
