"""
castle/service/extraction_service.py
Service layer for latent extraction operations.

All functions take simple types and return strings/dicts.
No gradio imports.
"""

import logging
import os
from pathlib import Path
from typing import Callable, Optional

from castle.core.data import Preprocess
from castle.core.extractor import (
    extract_roi_latent_from_video,
    extract_roi_latent_from_video_auto,
    extract_roi_crop_video,
    extract_roi_rotation_latent_from_video,
)
from castle.core.project import get_project_config, save_project_config
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
    session_id: Optional[str] = None,
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
        session_id: Pre-process session ID. If provided, uses preprocessed video and
            mask from that session. Raises CastleIOError if the video is not in the session.

    Returns:
        Path to saved latent file, or empty string on failure.
        If video_name is 'All', returns semicolon-separated paths.
    """
    from castle.core.types import CastleIOError, ExtractionError
    from castle.core.preprocess_session import get_preprocessed_paths

    if preprocess_config is None:
        preprocess_config = Preprocess()

    is_batch = (video_name == 'All')

    _, config = get_project_config(storage_path, project_name)
    video_list = sorted(config.get('source', [])) if is_batch else [video_name]

    paths: list[str] = []
    successes: list[str] = []
    failures: list[tuple[str, str]] = []

    for vname in video_list:
        source_video_path: Optional[str] = None
        mask_path_override: Optional[str] = None
        if session_id:
            try:
                vpath, mpath = get_preprocessed_paths(storage_path, project_name, session_id, vname)
                source_video_path = str(vpath)
                mask_path_override = str(mpath)
            except FileNotFoundError as exc:
                msg = (
                    f"not preprocessed in session '{session_id}'. "
                    f"Run Pre-process first. ({exc})"
                )
                if is_batch:
                    failures.append((vname, msg))
                    logger.error("Extraction failed for %s: %s", vname, msg)
                    continue
                raise CastleIOError(f"'{vname}' {msg}") from exc

        try:
            path = extract_roi_latent_from_video_auto(
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
                source_video_path=source_video_path,
                mask_path_override=mask_path_override,
                session_id=session_id,
            )
            if path:
                paths.append(path)
                successes.append(vname)
            else:
                # extract_roi_latent_from_video returned empty: treat as failure in batch mode.
                if is_batch:
                    failures.append((vname, "extractor returned empty path (see logs)"))
                else:
                    raise ExtractionError(
                        f"Extraction for '{vname}' returned no latent. See logs."
                    )
        except Exception as e:
            logger.error("Extraction failed for %s: %s", vname, e, exc_info=True)
            if is_batch:
                failures.append((vname, str(e)))
                continue
            # Single-video mode: re-raise immediately so the caller sees the real error.
            raise

    if is_batch and failures:
        # Build per-video ✅/❌ summary in original video_list order.
        succ_set = set(successes)
        fail_map = dict(failures)
        lines = [
            f"Extraction complete. {len(successes)} succeeded, {len(failures)} failed."
        ]
        for v in video_list:
            if v in succ_set:
                lines.append(f"✅ {v}")
            elif v in fail_map:
                lines.append(f"❌ {v} — {fail_map[v]}")
        lines.append("Fix the above errors and re-run.")
        raise ExtractionError("\n".join(lines))

    return ';'.join(paths)


def delete_session_with_latent_cleanup(
    storage_path: str, project_name: str, session_id: str
) -> None:
    """Delete a pre-process session directory and remove its latent entries from config.

    Args:
        storage_path: Root storage directory.
        project_name: Project name.
        session_id: Session ID (8-char hash) to delete.
    """
    from castle.core.preprocess_session import delete_session

    delete_session(storage_path, project_name, session_id)

    # Atomic RMW so a concurrent extraction registering a new latent doesn't
    # get clobbered by this delete (3-F).
    from castle.core.project import update_config
    prefix = f"{session_id}/"
    with update_config(storage_path, project_name) as config:
        config["latent"] = {
            k: v for k, v in config.get("latent", {}).items()
            if not k.startswith(prefix)
        }


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
