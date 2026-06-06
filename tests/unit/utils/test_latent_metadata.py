"""Tests for :mod:`castle.utils.latent_metadata` (BUG-14 / P3-C)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from castle.utils.latent_metadata import (
    extract_metadata_from_npz,
    load_latent_metadata,
    save_latent_with_metadata,
)


def _latent() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.standard_normal((50, 8)).astype(np.float32)


def test_save_writes_npz_and_sidecar(tmp_path: Path) -> None:
    arr = _latent()
    npz = tmp_path / "demo_ROI_1_dinov3_vitb16.npz"
    save_latent_with_metadata(
        str(npz), arr,
        video_name="demo.mp4", roi_id=1, model_name="dinov3_vitb16",
        seed=42, tags={"pooling_method": "weighted_average"},
    )
    sidecar = npz.with_suffix(npz.suffix + ".json")
    assert npz.exists()
    assert sidecar.exists()
    text = json.loads(sidecar.read_text())
    assert text["video_name"] == "demo.mp4"
    assert text["roi_id"] == 1
    assert text["model_name"] == "dinov3_vitb16"
    assert text["n_frames"] == 50
    assert text["feature_dim"] == 8
    assert text["seed"] == 42
    assert text["tags"]["pooling_method"] == "weighted_average"
    assert "castle_version" in text
    assert text["schema_version"] == 1


def test_extract_metadata_reads_embedded_dict(tmp_path: Path) -> None:
    arr = _latent()
    npz = tmp_path / "x.npz"
    save_latent_with_metadata(str(npz), arr, video_name="v", roi_id=2, model_name="m")
    out = extract_metadata_from_npz(str(npz))
    assert out is not None
    assert out["video_name"] == "v"
    assert out["roi_id"] == 2


def test_extract_returns_none_for_legacy_npz(tmp_path: Path) -> None:
    """An old npz with only the 'latent' key returns None — not an error."""
    npz = tmp_path / "legacy.npz"
    np.savez_compressed(npz, latent=_latent())
    assert extract_metadata_from_npz(str(npz)) is None


def test_extract_returns_none_for_missing_file(tmp_path: Path) -> None:
    assert extract_metadata_from_npz(str(tmp_path / "nope.npz")) is None


def test_load_latent_metadata_prefers_sidecar(tmp_path: Path) -> None:
    """Sidecar is faster to read; use it when available."""
    arr = _latent()
    npz = tmp_path / "x.npz"
    save_latent_with_metadata(str(npz), arr, video_name="v", roi_id=3, model_name="m")
    sidecar = npz.with_suffix(npz.suffix + ".json")

    # Mutate the sidecar so we can prove the loader read it (not the npz).
    fake = json.loads(sidecar.read_text())
    fake["video_name"] = "mutated_sidecar.mp4"
    sidecar.write_text(json.dumps(fake))

    out = load_latent_metadata(str(npz))
    assert out is not None
    assert out["video_name"] == "mutated_sidecar.mp4"


def test_load_latent_metadata_falls_back_to_npz_when_sidecar_missing(tmp_path: Path) -> None:
    arr = _latent()
    npz = tmp_path / "x.npz"
    save_latent_with_metadata(str(npz), arr, video_name="v", roi_id=4, model_name="m")
    sidecar = npz.with_suffix(npz.suffix + ".json")
    sidecar.unlink()

    out = load_latent_metadata(str(npz))
    assert out is not None
    assert out["video_name"] == "v"
    assert out["roi_id"] == 4


def test_saved_npz_still_loadable_by_safe_load(tmp_path: Path) -> None:
    """Verify that adding the metadata key didn't break safe_load."""
    from castle.utils.safe_load import load_latent_safe

    arr = _latent()
    npz = tmp_path / "x.npz"
    save_latent_with_metadata(str(npz), arr, video_name="v", roi_id=5, model_name="m")
    out = load_latent_safe(npz)
    np.testing.assert_array_equal(out, arr)


# --- Atomic write (crash-safe) ----------------------------------------------
# A torn .npz on a flaky CephFS would be mistaken for a finished extraction by
# skip_existing. The write must go tmp -> fsync -> os.replace so the final path
# is never partial.


def test_memmap_branch_round_trips(tmp_path: Path) -> None:
    """The uncompressed (memmap-backed) np.savez streaming path stays correct."""
    backing = tmp_path / "backing.dat"
    mm = np.memmap(backing, dtype=np.float32, mode="w+", shape=(40, 6))
    mm[:] = np.random.default_rng(1).standard_normal((40, 6)).astype(np.float32)
    mm.flush()
    npz = tmp_path / "mm_ROI_1_dinov3_vitb16.npz"
    save_latent_with_metadata(
        str(npz), mm, video_name="m.mp4", roi_id=1, model_name="dinov3_vitb16",
    )
    assert npz.exists()
    np.testing.assert_allclose(np.load(npz)["latent"], np.asarray(mm))


def test_failed_write_leaves_no_torn_npz(tmp_path: Path, monkeypatch) -> None:
    import castle.utils.latent_metadata as lm

    def boom(*_a, **_k):
        raise RuntimeError("storage vanished mid-write")

    monkeypatch.setattr(lm.np, "savez_compressed", boom)
    npz = tmp_path / "fail_ROI_1_m.npz"
    with pytest.raises(RuntimeError):
        save_latent_with_metadata(str(npz), _latent(), video_name="v", roi_id=1, model_name="m")

    # No truncated final file, no orphaned temp, no sidecar — skip_existing is safe.
    assert not npz.exists()
    assert list(tmp_path.glob("*.tmp")) == []
    assert list(tmp_path.glob("*.json")) == []


def test_write_uses_same_dir_tmp_then_replace(tmp_path: Path, monkeypatch) -> None:
    """The temp file must sit in the destination dir so os.replace is atomic."""
    import os

    seen: dict = {}
    real_replace = os.replace

    def spy_replace(src, dst):
        seen["src"], seen["dst"] = src, dst
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", spy_replace)
    npz = tmp_path / "loc_ROI_1_m.npz"
    save_latent_with_metadata(str(npz), _latent(), video_name="v", roi_id=1, model_name="m")

    assert Path(seen["src"]).parent == npz.parent  # same filesystem → atomic rename
    assert Path(seen["dst"]) == npz
    assert npz.exists()
