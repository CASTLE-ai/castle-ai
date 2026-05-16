"""Latent ``.npz`` metadata sidecar helpers (BUG-14 / P3-C).

Historically CASTLE saved feature arrays as plain ``np.savez_compressed(
path, latent=array)`` and recovered the model / ROI / video info by
splitting the filename. Filename schema drift therefore breaks every
downstream consumer. This module saves the same data plus a structured
metadata dict so consumers can trust the npz itself.

The metadata is stored two ways for resilience:

* **Inside the npz** under key ``metadata`` (JSON string, one element
  numpy array). Always present when an npz is written by these helpers.
* **Sidecar ``.json``** next to the npz. Human-readable, lets external
  tools (`jq` / `cat`) inspect a latent without numpy. Sidecar is
  best-effort — write failures are logged but never raise.

Loaders should ``try`` the metadata path first and fall back to filename
parsing for npz files written before this helper landed.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

import castle

logger = logging.getLogger(__name__)

__all__ = [
    "save_latent_with_metadata",
    "load_latent_metadata",
    "extract_metadata_from_npz",
]


def _build_metadata(
    video_name: str,
    roi_id: int,
    model_name: str,
    latent_array: np.ndarray,
    *,
    seed: Optional[int] = None,
    tags: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble the canonical metadata dict for a latent npz."""
    meta = {
        "schema_version": 1,
        "castle_version": getattr(castle, "__version__", "unknown"),
        "video_name": video_name,
        "roi_id": int(roi_id),
        "model_name": model_name,
        "n_frames": int(latent_array.shape[0]),
        "feature_dim": int(latent_array.shape[1]) if latent_array.ndim >= 2 else None,
        "dtype": str(latent_array.dtype),
    }
    if seed is not None:
        meta["seed"] = int(seed)
    if tags:
        meta["tags"] = dict(tags)
    return meta


def save_latent_with_metadata(
    latent_path: str,
    latent_array: np.ndarray,
    *,
    video_name: str,
    roi_id: int,
    model_name: str,
    seed: Optional[int] = None,
    tags: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save a latent npz with embedded + sidecar metadata.

    Args:
        latent_path: Destination ``.npz`` path.
        latent_array: ``(T, F)`` feature array.
        video_name: Source video filename (basename, no path).
        roi_id: ROI integer ID.
        model_name: Feature extractor model name (e.g. ``'dinov3_vitb16'``).
        seed: Master seed used for this extraction, if known.
        tags: Optional extra key/value metadata (pooling scales, feature
            layers, etc.). Must be JSON-serialisable.

    Returns:
        Path to the npz that was written.

    Notes:
        Always writes the npz. The sidecar ``.json`` is best-effort —
        if the directory is read-only the npz still succeeds and a
        warning is logged.
    """
    latent_path = Path(latent_path)
    meta = _build_metadata(
        video_name=video_name,
        roi_id=roi_id,
        model_name=model_name,
        latent_array=latent_array,
        seed=seed,
        tags=tags,
    )
    meta_json = json.dumps(meta, ensure_ascii=False)
    np.savez_compressed(
        latent_path,
        latent=latent_array,
        metadata=np.array([meta_json]),
    )

    sidecar = latent_path.with_suffix(latent_path.suffix + ".json")
    try:
        sidecar.write_text(meta_json + "\n", encoding="utf-8")
    except OSError as exc:
        logger.warning(
            "Could not write sidecar %s (npz itself was saved successfully): %s",
            sidecar, exc,
        )

    return latent_path


def extract_metadata_from_npz(npz_path: str) -> Optional[Dict[str, Any]]:
    """Read the embedded metadata dict from a CASTLE latent npz.

    Args:
        npz_path: Path to the ``.npz``.

    Returns:
        The metadata dict, or ``None`` if the npz predates this helper
        (no ``metadata`` key) or the JSON is malformed. Callers can fall
        back to filename parsing in either case.
    """
    try:
        with np.load(npz_path, allow_pickle=False) as data:
            if "metadata" not in data.files:
                return None
            raw = data["metadata"]
            if raw.ndim == 0:
                meta_json = str(raw.item())
            else:
                meta_json = str(raw[0])
    except (OSError, ValueError, EOFError) as exc:
        logger.debug("Could not read metadata from %s: %s", npz_path, exc)
        return None

    try:
        return json.loads(meta_json)
    except json.JSONDecodeError as exc:
        logger.debug("Metadata JSON malformed in %s: %s", npz_path, exc)
        return None


def load_latent_metadata(npz_path: str) -> Optional[Dict[str, Any]]:
    """Prefer the sidecar ``.json`` for speed; fall back to the npz.

    Loading the sidecar avoids unzipping the (large) npz when callers
    only need metadata — used by the project-config refresh path.
    """
    npz = Path(npz_path)
    sidecar = npz.with_suffix(npz.suffix + ".json")
    if sidecar.exists():
        try:
            return json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.debug("Sidecar %s unreadable, falling back to npz: %s", sidecar, exc)
    return extract_metadata_from_npz(str(npz))
