"""
castle/service/extraction_service.py
Service layer for latent extraction operations.

All functions take simple types and return strings/dicts.
No gradio imports.
"""

import logging
from typing import Callable, Optional

from castle.core.data import Preprocess
from castle.core.extractor import (
    extract_roi_latent_from_video,
    extract_roi_latent_from_video_auto,
    extract_roi_crop_video,
    extract_roi_rotation_latent_from_video,
)
from castle.core.project import get_project_config
from castle.defaults import EXTRACTION_BATCH_SIZE

logger = logging.getLogger(__name__)


def _auto_batch_size(model: str, preprocess_config: Preprocess) -> int:
    """Auto-size the extraction batch from free VRAM (single source of truth:
    memory_guard.suggest_batch_size, which already accounts for the 7x rotation
    multiplier and a 0.75 safety margin). Falls back to the static default if
    memory info is unavailable. ``auto_retry_on_oom`` in the extractor is the
    backstop if the estimate is still too optimistic.
    """
    try:
        import torch
        from castle.core.memory_guard import suggest_batch_size
        device = "cuda" if torch.cuda.is_available() else "cpu"
        rotate = bool(getattr(preprocess_config, "rotate_roi_tail_switch", False))
        bs = suggest_batch_size(model, device, rotate=rotate)
        logger.info(
            "extract_latent: auto batch_size=%d (model=%s, device=%s, rotate=%s)",
            bs, model, device, rotate,
        )
        return bs
    except Exception:
        return EXTRACTION_BATCH_SIZE


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
    batch_size: Optional[int] = EXTRACTION_BATCH_SIZE,
    preprocess_config: Optional[Preprocess] = None,
    skip_existing: bool = True,
    progress_callback: Optional[Callable] = None,
    pooling_method: str = 'weighted_average',
    pooling_scales: Optional[list] = None,
    feature_layers: Optional[list] = None,
    session_id: Optional[str] = None,
    latent_dtype: str = 'float32',
    cancel_event=None,
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

    # batch_size=None → auto-size from free VRAM (CLI default). An explicit int
    # (UI / tests / callers using the static default) is honoured as-is.
    if batch_size is None:
        batch_size = _auto_batch_size(model, preprocess_config)

    is_batch = (video_name == 'All')

    _, config = get_project_config(storage_path, project_name)
    video_list = sorted(config.get('source', [])) if is_batch else [video_name]

    paths: list[str] = []
    successes: list[str] = []
    failures: list[tuple[str, str]] = []

    # Video-level multi-GPU: when opted in (CASTLE_MULTI_GPU + >1 CUDA device) and
    # there are >= 2 videos, run one whole video per GPU concurrently (each pinned
    # single-GPU via the device arg). A single video (or flag off) falls through to
    # the sequential path below, where `_auto` still applies the within-video
    # 2-GPU split when multi-GPU is on. Per-video outputs are independent; they
    # match sequential exactly except for fp16-autocast rounding (~1e-2) on videos
    # that run on the second GPU (accepted as negligible for downstream UMAP).
    from castle.core.gpu_pool import (
        resolve_device_ids, run_on_device_pool, deterministic_ctx_if_enabled,
        host_ram_available_bytes,
    )
    device_ids = resolve_device_ids() if is_batch else []

    if is_batch and len(device_ids) >= 2 and len(video_list) >= 2:
        from castle.core.environment import get_num_workers
        n_gpu = len(device_ids)
        per_worker = max(1, get_num_workers('extraction') // n_gpu)
        total = len(video_list)
        completed = {'n': 0}

        def _worker(vname: str, device: str):
            # Session-path resolution per video (a missing session video raises here
            # and the pool records it as this item's failure).
            svp = mpo = None
            if session_id:
                vpath, mpath = get_preprocessed_paths(storage_path, project_name, session_id, vname)
                svp, mpo = str(vpath), str(mpath)
            return extract_roi_latent_from_video(
                storage_path=storage_path, project_name=project_name, video_name=vname,
                roi_id=roi, model_name=model, batch_size=batch_size,
                preprocess_config=preprocess_config, skip_existing=skip_existing,
                progress_callback=None,  # per-video % would interleave; report on completion
                pooling_method=pooling_method, pooling_scales=pooling_scales,
                feature_layers=feature_layers,
                source_video_path=svp, mask_path_override=mpo, session_id=session_id,
                device=device, num_workers=per_worker, latent_dtype=latent_dtype,
                cancel_event=cancel_event,
            )

        def _on_done(vname: str, res) -> None:
            completed['n'] += 1
            if progress_callback is not None:
                ok = (not isinstance(res, BaseException)) and bool(res)
                progress_callback(completed['n'] / total,
                                  f"[{completed['n']}/{total}] {vname} {'✅' if ok else '❌'}")

        logger.info("extract_latent: video-level multi-GPU over %s for %d video(s) (%d workers/GPU)",
                    list(device_ids), len(video_list), per_worker)
        free = host_ram_available_bytes()
        if free is not None and free < 2 * (1024 ** 3) * n_gpu:
            logger.warning("extract_latent: low free host RAM (%.1f GB) for %d-GPU extraction.",
                           free / 1024 ** 3, n_gpu)
        # Speed by default (fast cuDNN benchmark + fp16, like single-GPU). Opt in
        # to per-GPU-reproducible cuDNN-deterministic via CASTLE_MULTI_GPU_DETERMINISTIC.
        with deterministic_ctx_if_enabled():
            pool_out = run_on_device_pool(video_list, _worker, device_ids, on_done=_on_done,
                                          cancel_event=cancel_event)
        for vname, res in zip(video_list, pool_out):
            if isinstance(res, BaseException):
                logger.error("Extraction failed for %s: %s", vname, res)
                failures.append((vname, str(res)))
            elif res:
                paths.append(res)
                successes.append(vname)
            else:
                failures.append((vname, "extractor returned empty path (see logs)"))
        # Free the secondary-GPU encoders built during the pool — they live in
        # extractor._device_encoder_cache (separate from the model singleton) and
        # would otherwise stay resident on GPU1 for the process lifetime.
        try:
            from castle.core.extractor import clear_device_encoder_cache
            clear_device_encoder_cache()
        except Exception:  # noqa: BLE001 - teardown must never fail the batch
            logger.debug("clear_device_encoder_cache after pool failed", exc_info=True)
    else:
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
                    latent_dtype=latent_dtype,
                    cancel_event=cancel_event,
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


def latent_gap_summary(latent_path: str) -> Optional[dict]:
    """Per-video skipped-frame summary for a saved latent ``.npz``.

    Frames whose tracker produced no usable mask are stored as all-NaN gap rows
    and the count is recorded in the latent metadata (``n_skipped_frames`` /
    ``n_total_frames``). This reads that back from the cheap metadata sidecar so
    the CLI / UI can tell the user how many frames were skipped without
    re-scanning the array.

    Returns:
        ``{'n_skipped': int, 'n_total': int, 'frac': float}`` or ``None`` when
        the file has no such metadata (e.g. an older latent).
    """
    if not latent_path:
        return None
    try:
        from castle.utils.latent_metadata import load_latent_metadata
        meta = load_latent_metadata(latent_path)
    except Exception:  # noqa: BLE001 - a summary must never break the caller
        return None
    if not meta:
        return None
    tags = meta.get("tags") or {}
    n_skipped = tags.get("n_skipped_frames")
    n_total = tags.get("n_total_frames") or meta.get("n_frames")
    if n_skipped is None or n_total is None:
        return None
    n_skipped, n_total = int(n_skipped), int(n_total)
    return {
        "n_skipped": n_skipped,
        "n_total": n_total,
        "frac": (n_skipped / n_total) if n_total else 0.0,
    }


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
