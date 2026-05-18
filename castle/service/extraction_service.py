"""
castle/service/extraction_service.py
Service layer for latent extraction operations.

All functions take simple types and return strings/dicts.
No gradio imports.
"""

import json
import logging
import os
from pathlib import Path
from typing import Callable, Optional

from castle.core.data import Preprocess
from castle.core.extractor import (
    extract_roi_latent_from_video,
    extract_roi_crop_video,
    extract_roi_rotation_latent_from_video,
)
from castle.core.project import get_project_config
from castle.defaults import EXTRACTION_BATCH_SIZE

logger = logging.getLogger(__name__)


def make_preprocess_config(
    center_roi_switch: bool = False,
    center_roi_id: int = 1,
    center_roi_crop_width: int = 300,
    center_roi_crop_height: int = 300,
    rotate_roi_tail_switch: bool = False,
    rotate_roi_tail_id: int = 2,
    remove_background_switch: bool = False,
) -> Preprocess:
    """
    Create a Preprocess config object from simple parameters.
    
    This wraps the Preprocess dataclass so callers don't need to import
    castle.core.data directly.
    
    Returns:
        Preprocess config object
    """
    return Preprocess(
        center_roi_switch=center_roi_switch,
        center_roi_id=center_roi_id,
        center_roi_crop_width=center_roi_crop_width,
        center_roi_crop_height=center_roi_crop_height,
        rotate_roi_tail_switch=rotate_roi_tail_switch,
        rotate_roi_tail_id=rotate_roi_tail_id,
        remove_background_switch=remove_background_switch,
    )


def extract_latent(
    storage_path: str,
    project_name: str,
    video_name: str,
    model: str,
    roi: int,
    batch_size: int = EXTRACTION_BATCH_SIZE,
    preprocess_config: Optional[Preprocess] = None,
    skip_existing: bool = True,
    progress_callback: Optional[Callable] = None,
    pooling_method: str = 'weighted_average',
    pooling_scales: Optional[list] = None,
    feature_layers: Optional[list] = None,
) -> str:
    """
    Extract latent features from a tracked video ROI.
    
    Args:
        storage_path: Root storage directory
        project_name: Project name
        video_name: Video filename (or 'All' for all videos)
        model: Model name (e.g., 'dinov3_vitb16')
        roi: ROI ID
        batch_size: Batch size for extraction
        preprocess_config: Preprocess configuration. If None, uses defaults.
        skip_existing: Skip if latent file already exists
        progress_callback: Optional progress callback(fraction, description)
        pooling_method: 'weighted_average' (default) or 'multiscale'
        pooling_scales: Grid scales for multiscale pooling, e.g. [1, 2, 4]
        feature_layers: Layer indices for multi-layer extraction. None = last only.
    
    Returns:
        Path to saved latent file, or empty string on failure.
        If video_name is 'All', returns semicolon-separated paths.
    """
    if preprocess_config is None:
        preprocess_config = Preprocess()
    
    # Handle "All" videos
    _, config = get_project_config(storage_path, project_name)
    video_list = sorted(config.get('source', [])) if video_name == 'All' else [video_name]
    
    paths = []
    for vname in video_list:
        try:
            path = extract_roi_latent_from_video(
                storage_path=storage_path,
                project_name=project_name,
                video_name=vname,
                roi_id=roi,
                model_name=model,
                batch_size=batch_size,
                preprocess_config=preprocess_config,
                skip_existing=skip_existing,
                progress_callback=progress_callback,
                pooling_method=pooling_method,
                pooling_scales=pooling_scales,
                feature_layers=feature_layers,
            )
            if path:
                paths.append(path)
        except Exception as e:
            logger.error(f"Extraction failed for {vname}: {e}", exc_info=True)
    
    return ';'.join(paths)


# ---------------------------------------------------------------------------
# KIT on-the-fly latent extraction
# ---------------------------------------------------------------------------


def extract_latent_with_kit(
    storage_path: str,
    project_name: str,
    video_name: str,
    roi_id: int,
    model_name: str,
    batch_size: int = EXTRACTION_BATCH_SIZE,
    kit_params: Optional[dict] = None,
    skip_existing: bool = True,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> str:
    """On-the-fly KIT extraction: frame AND mask transformed by StabilizedCamera.

    Runs the 3-stage ``ParallelExtractor`` pipeline with both ``mask_path`` and
    ``stabilized_camera`` set, so the same affine matrix is applied to every
    frame/mask pair (Stage 2) before ROI-weighted pooling (Stage 3).  No
    intermediate stabilised video is written to disk.

    A metadata sidecar JSON is saved alongside the ``.npz`` file to record
    that KIT was applied and which parameters were used (without embedding them
    in the filename).

    Args:
        storage_path: Root storage directory.
        project_name: Project name.
        video_name: Source video filename.
        roi_id: ROI id for pooling.
        model_name: Visual encoder name (e.g. ``'dinov2_vitb14'``).
        batch_size: GPU inference batch size.
        kit_params: KIT parameter dict.  Required keys: ``body_roi_id``,
            ``head_roi_id``.  Optional: ``fc``, ``order``, ``margin``,
            ``min_crop``, ``output_size``.  If ``None`` the function raises
            ``ValueError`` — callers should load params via
            :func:`castle.core.project.load_kit_params` first.
        skip_existing: Skip if the ``.npz`` already exists.
        progress_callback: Called as ``callback(current, total, message)``.

    Returns:
        Absolute path to the saved ``.npz`` latent file.

    Raises:
        ValueError: If ``kit_params`` is ``None`` or missing required keys.
        FileNotFoundError: If the mask HDF5 file does not exist.
    """
    import numpy as np

    from castle.core.stabilized_camera import (
        StabilizedCamera,
        extract_centroids_from_masks,
        extract_orientations_from_masks,
    )
    from castle.core.pipeline_parallel import ParallelExtractor
    from castle.core.project import get_project_config

    if kit_params is None:
        raise ValueError(
            "kit_params is required for extract_latent_with_kit. "
            "Call load_kit_params() first and pass the result."
        )
    for key in ("body_roi_id", "head_roi_id"):
        if key not in kit_params:
            raise ValueError(f"kit_params is missing required key: '{key}'")

    project_path, _ = get_project_config(storage_path, project_name)
    project_dir = Path(project_path)
    latent_dir = project_dir / "latent" / model_name
    latent_dir.mkdir(parents=True, exist_ok=True)

    base_name = os.path.splitext(video_name)[0]
    latent_path = latent_dir / f"{base_name}_ROI_{roi_id}_{model_name}.npz"
    sidecar_path = Path(str(latent_path) + ".json")

    if skip_existing and latent_path.exists():
        logger.info("extract_latent_with_kit: skipping %s (already exists)", latent_path)
        return str(latent_path)

    source_path = str(project_dir / "sources" / video_name)
    mask_path = str(project_dir / "track" / video_name / "mask_list.h5")
    if not Path(mask_path).exists():
        raise FileNotFoundError(
            f"Mask file not found: {mask_path}. "
            f"Run tracking for '{video_name}' first."
        )

    from castle.utils.video_io import ReadArray
    with ReadArray(source_path) as vr:
        n_frames: int = len(vr)
        fps: float = vr.fps

    if progress_callback:
        progress_callback(0, n_frames, "Extracting centroids…")

    positions = extract_centroids_from_masks(
        mask_path, kit_params["body_roi_id"], n_frames
    )
    angles = extract_orientations_from_masks(
        mask_path, kit_params["body_roi_id"], kit_params["head_roi_id"], n_frames
    )

    cam = StabilizedCamera(
        positions=positions,
        angles=angles,
        fps=fps,
        fc=kit_params.get("fc", 0.25),
        order=kit_params.get("order", 2),
        margin=kit_params.get("margin", 75),
        min_crop=kit_params.get("min_crop", 300),
        output_size=kit_params.get("output_size", 518),
    )

    from castle.core.models import get_visual_encoder  # type: ignore
    model_obj = get_visual_encoder(model_name)

    extractor = ParallelExtractor(
        video_path=source_path,
        stabilized_camera=cam,
        mask_path=mask_path,
        model=model_obj,
        batch_size=batch_size,
        roi_id=roi_id,
    )

    if progress_callback:
        progress_callback(0, n_frames, "Running KIT extraction…")

    latents = extractor.run(progress_callback=progress_callback)

    np.savez_compressed(str(latent_path), latent=latents)
    logger.info(
        "extract_latent_with_kit: saved %s  shape=%s", latent_path, latents.shape
    )

    sidecar = {
        "video_name": video_name,
        "roi_id": roi_id,
        "model_name": model_name,
        "n_frames": int(latents.shape[0]),
        "feature_dim": int(latents.shape[1]) if latents.ndim == 2 else None,
        "kit_applied": True,
        "kit_params": kit_params,
    }
    sidecar_path.write_text(json.dumps(sidecar, indent=2))

    return str(latent_path)


# NOTE: Not yet exposed via CLI or UI
def extract_crop_video(
    storage_path: str,
    project_name: str,
    video_name: str,
    roi: int,
    preprocess_config: Optional[Preprocess] = None,
    skip_existing: bool = True,
    progress_callback: Optional[Callable] = None,
) -> str:
    """
    Extract cropped/preprocessed video for a tracked ROI.
    
    Args:
        storage_path: Root storage directory
        project_name: Project name
        video_name: Video filename (or 'All')
        roi: ROI ID
        preprocess_config: Preprocess configuration. If None, uses defaults.
        skip_existing: Skip if crop video already exists
        progress_callback: Optional progress callback(fraction, description)
    
    Returns:
        Path to saved crop video, or empty string on failure.
        If video_name is 'All', returns semicolon-separated paths.
    """
    if preprocess_config is None:
        preprocess_config = Preprocess()
    
    _, config = get_project_config(storage_path, project_name)
    video_list = sorted(config.get('source', [])) if video_name == 'All' else [video_name]
    
    paths = []
    for vname in video_list:
        try:
            path = extract_roi_crop_video(
                storage_path=storage_path,
                project_name=project_name,
                video_name=vname,
                roi_id=roi,
                preprocess_config=preprocess_config,
                skip_existing=skip_existing,
                progress_callback=progress_callback,
            )
            if path:
                paths.append(path)
        except Exception as e:
            logger.error(f"Crop extraction failed for {vname}: {e}", exc_info=True)
    
    return ';'.join(paths)


# NOTE: Not yet exposed via CLI or UI
def extract_rotation_latent(
    storage_path: str,
    project_name: str,
    video_name: str,
    model: str,
    roi: int,
    batch_size: int = EXTRACTION_BATCH_SIZE,
    preprocess_config: Optional[Preprocess] = None,
    skip_existing: bool = True,
    progress_callback: Optional[Callable] = None,
) -> str:
    """
    Extract rotation-invariant latent features.
    
    Generates 7 rotated views and averages the latent representations.
    
    Args:
        storage_path: Root storage directory
        project_name: Project name
        video_name: Video filename (or 'All')
        model: Model name
        roi: ROI ID
        batch_size: Batch size for extraction
        preprocess_config: Preprocess configuration. If None, uses defaults.
        skip_existing: Skip if latent file already exists
        progress_callback: Optional progress callback(fraction, description)
    
    Returns:
        Path to saved latent file, or empty string on failure.
    """
    if preprocess_config is None:
        preprocess_config = Preprocess()
    
    _, config = get_project_config(storage_path, project_name)
    video_list = sorted(config.get('source', [])) if video_name == 'All' else [video_name]
    
    paths = []
    for vname in video_list:
        try:
            path = extract_roi_rotation_latent_from_video(
                storage_path=storage_path,
                project_name=project_name,
                video_name=vname,
                roi_id=roi,
                model_name=model,
                batch_size=batch_size,
                preprocess_config=preprocess_config,
                skip_existing=skip_existing,
                progress_callback=progress_callback,
            )
            if path:
                paths.append(path)
        except Exception as e:
            logger.error(f"Rotation extraction failed for {vname}: {e}", exc_info=True)
    
    return ';'.join(paths)
