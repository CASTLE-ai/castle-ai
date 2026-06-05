"""
castle/service/auto_config.py
Auto-configuration service — recommends pipeline parameters based on video
properties, animal size, and available GPU resources.

No gradio imports. All functions take/return plain Python types.
"""

from __future__ import annotations

import logging
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------

def get_gpu_info() -> dict:
    """Detect available GPU and VRAM.

    Returns
    -------
    dict with keys:
        - ``available`` (bool)
        - ``name`` (str)       — GPU name or "CPU only"
        - ``vram_mb`` (int)    — VRAM in MB (0 if CPU only)
        - ``vram_free_mb`` (int) — free VRAM in MB (0 if CPU only)
    """
    info: dict = {
        "available": False,
        "name": "CPU only",
        "vram_mb": 0,
        "vram_free_mb": 0,
    }

    # Try PyTorch first (most reliable in this project's env)
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            info["available"] = True
            info["name"] = torch.cuda.get_device_name(idx)
            props = torch.cuda.get_device_properties(idx)
            info["vram_mb"] = props.total_memory // (1024 * 1024)
            # Free memory (may be approximate)
            free, _ = torch.cuda.mem_get_info(idx)
            info["vram_free_mb"] = free // (1024 * 1024)
            return info
    except Exception:
        pass

    # Fallback: nvidia-smi
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        lines = out.decode().strip().splitlines()
        if lines:
            parts = [p.strip() for p in lines[0].split(",")]
            if len(parts) >= 3:
                info["available"] = True
                info["name"] = parts[0]
                info["vram_mb"] = int(parts[1])
                info["vram_free_mb"] = int(parts[2])
    except Exception:
        pass

    return info


# ---------------------------------------------------------------------------
# Video introspection (without heavy import at module level)
# ---------------------------------------------------------------------------

def _get_video_meta(video_path: str) -> dict:
    """Return basic video metadata using cv2 (or ffprobe fallback)."""
    meta = {"width": 0, "height": 0, "fps": 30.0, "frame_count": 0}

    try:
        import cv2  # noqa: PLC0415

        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            meta["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            meta["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            meta["fps"] = cap.get(cv2.CAP_PROP_FPS) or 30.0
            meta["frame_count"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return meta
    except Exception:
        pass

    # ffprobe fallback
    try:
        import json  # noqa: PLC0415

        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_streams",
            video_path,
        ]
        out = subprocess.check_output(cmd, timeout=10)
        data = json.loads(out)
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                meta["width"] = int(stream.get("width", 0))
                meta["height"] = int(stream.get("height", 0))
                # fps may be "30/1" or "30000/1001"
                fps_str = stream.get("avg_frame_rate", "30/1")
                try:
                    num, den = fps_str.split("/")
                    meta["fps"] = float(num) / float(den)
                except Exception:
                    pass
                meta["frame_count"] = int(stream.get("nb_frames", 0))
                break
    except Exception:
        pass

    return meta


# ---------------------------------------------------------------------------
# Main recommendation function
# ---------------------------------------------------------------------------

def recommend_config(
    video_path: str,
    mask_info: Optional[dict] = None,
    gpu_info: Optional[dict] = None,
    model_name: str = "dinov3_vitb16",
) -> dict:
    """Automatically recommend preprocessing and pipeline parameters.

    Based on:
    - Video resolution & fps → ``fc``, crop size, ``margin``
    - Animal size from mask (if available) → ``min_crop``
    - Available GPU VRAM → ``batch_size``

    Parameters
    ----------
    video_path : str
        Path to the source video file.
    mask_info : dict, optional
        Keys: ``animal_width_px``, ``animal_height_px`` (bounding box in px).
    gpu_info : dict, optional
        Output of :func:`get_gpu_info`. Auto-detected if *None*.

    Returns
    -------
    dict
        Nested dict with recommended params for each pipeline step::

            {
                "video_meta": {...},
                "gpu_info": {...},
                "preprocessing": {
                    "fc": float,          # Butterworth cutoff (0–0.5)
                    "margin": int,        # px margin around crop
                    "min_crop": int,      # minimum crop dimension in px
                    "output_size": int,   # square output frame px
                },
                "extraction": {
                    "batch_size": int,
                    "center_roi_crop_width": int,
                    "center_roi_crop_height": int,
                },
                "clustering": {
                    "n_clusters": int,    # initial suggestion
                },
            }
    """
    if gpu_info is None:
        gpu_info = get_gpu_info()

    meta = _get_video_meta(video_path)

    # ------------------------------------------------------------------
    # Cutoff frequency: higher fps → more temporal smoothing needed
    # Rule: fc ∝ 1/fps, capped at [0.05, 0.4]
    # ------------------------------------------------------------------
    fps = meta["fps"] or 30.0
    fc = max(0.05, min(0.4, 5.0 / fps))

    # ------------------------------------------------------------------
    # Crop margin: scale with resolution
    # ------------------------------------------------------------------
    short_side = min(meta["width"], meta["height"]) or 480
    margin = max(50, min(200, int(short_side * 0.12)))

    # ------------------------------------------------------------------
    # Min crop & output size from animal mask size (if available)
    # ------------------------------------------------------------------
    if mask_info:
        animal_px = max(
            mask_info.get("animal_width_px", 0),
            mask_info.get("animal_height_px", 0),
        )
        min_crop = max(150, int(animal_px * 1.5))
        output_size = max(224, min(518, int(animal_px * 2.5)))
    else:
        # Resolution-based defaults
        if short_side >= 1080:
            min_crop, output_size = 400, 518
        elif short_side >= 720:
            min_crop, output_size = 300, 518
        else:
            min_crop, output_size = 200, 336

    # ------------------------------------------------------------------
    # Batch size: single source of truth is memory_guard.suggest_batch_size,
    # which scales with free VRAM (so a 24 GB cloud card is not capped at 32),
    # accounts for the rotation multiplier, and applies a safety margin.
    # ------------------------------------------------------------------
    if not gpu_info.get("available"):
        batch_size = 1
    else:
        try:
            from castle.core.memory_guard import suggest_batch_size
            batch_size = suggest_batch_size(model_name, "cuda", rotate=False)
        except Exception:
            batch_size = 8

    # ------------------------------------------------------------------
    # Cluster count: heuristic from video length
    # More frames → more potential behaviours discovered
    # ------------------------------------------------------------------
    n_frames = meta["frame_count"] or int(fps * 60)
    n_clusters = max(5, min(30, int(n_frames / (fps * 20))))

    return {
        "video_meta": meta,
        "gpu_info": gpu_info,
        "preprocessing": {
            "fc": round(fc, 3),
            "margin": margin,
            "min_crop": min_crop,
            "output_size": output_size,
        },
        "extraction": {
            "batch_size": batch_size,
            "center_roi_crop_width": min_crop,
            "center_roi_crop_height": min_crop,
        },
        "clustering": {
            "n_clusters": n_clusters,
        },
    }


# ---------------------------------------------------------------------------
# ETA estimator
# ---------------------------------------------------------------------------

def estimate_pipeline_time(video_path: str, config: dict) -> float:
    """Rough ETA for the full pipeline in seconds.

    Uses a simple heuristic model based on video length, resolution, and
    batch size. Does NOT actually run any model.

    Parameters
    ----------
    video_path : str
        Path to the video file.
    config : dict
        Output of :func:`recommend_config` (or compatible subset).

    Returns
    -------
    float
        Estimated time in seconds.
    """
    meta = config.get("video_meta") or _get_video_meta(video_path)
    fps = meta.get("fps") or 30.0
    n_frames = meta.get("frame_count") or int(fps * 60)
    short_side = min(meta.get("width", 480), meta.get("height", 480)) or 480
    batch_size = config.get("extraction", {}).get("batch_size", 8)
    gpu_available = config.get("gpu_info", {}).get("available", False)

    # --- preprocessing (camera stabilisation) ---
    # ~0.5 s/frame for CPU, ~0.1 s/frame for GPU at 480p; scale with resolution
    res_factor = max(1.0, short_side / 480.0) ** 1.5
    preprocess_fps = (10.0 if gpu_available else 2.0) / res_factor
    t_preprocess = n_frames / preprocess_fps

    # --- SAM tracking ---
    # ~0.5 s/frame on GPU, ~5 s/frame on CPU
    t_tracking = n_frames * (0.5 if gpu_available else 5.0)

    # --- feature extraction ---
    # ~ (1/batch_size) * 0.05 s/frame on GPU, 0.5 s on CPU
    t_extract = n_frames * (0.05 / batch_size if gpu_available else 0.5)

    # --- clustering (fast, negligible) ---
    t_cluster = max(10.0, n_frames * 0.001)

    total = t_preprocess + t_tracking + t_extract + t_cluster
    return round(total, 1)
