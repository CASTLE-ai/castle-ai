"""
castle/service/tracking_service.py
Service layer for ROI tracking operations.

All functions take simple types and return dicts.
No gradio imports.
"""

import logging
from pathlib import Path
from typing import Optional, Callable

from castle.utils.video_io import ReadArray
from castle.utils.h5_io import H5IO
from castle.utils.tracking_manager import ROITracker

logger = logging.getLogger(__name__)


def track_video(storage_path: str, project_name: str, video_name: str,
                model: str = 'r50_deaotl',
                start: int = 0, stop: int = -1,
                skip_existing: bool = True,
                progress_callback: Optional[Callable] = None,
                device: Optional[str] = None,
                num_workers: Optional[int] = None,
                pin_memory: bool = True) -> str:
    """
    Execute tracking on a single video.

    Args:
        storage_path: Root storage directory
        project_name: Project name
        video_name: Video filename
        model: Tracking model type ('r50_deaotl' or 'swinb_deaotl')
        start: Start frame (0-based)
        stop: Stop frame (-1 for end)
        skip_existing: Skip if mask_list.h5 already exists
        progress_callback: Optional progress callback(progress_fraction, description)
        device: CUDA device to run on (e.g. ``'cuda:1'``). ``None`` defers to the
            module default (single-GPU). Set by :func:`track_videos` workers.
        num_workers: DataLoader worker override (``None`` = ``get_num_workers('tracking')``).
        pin_memory: DataLoader ``pin_memory`` (set ``False`` for concurrent multi-GPU).

    Returns:
        Status string: 'Done', 'Skipped', 'Cancel', or error message
    """
    project_path = Path(storage_path) / project_name
    track_dir = project_path / 'track' / video_name
    mask_path = track_dir / 'mask_list.h5'

    if skip_existing and mask_path.exists():
        return 'Skipped'

    video_path = project_path / 'sources' / video_name
    if not video_path.exists():
        return f'Error: Video not found: {video_path}'

    try:
        with ReadArray(str(video_path)) as source_video:
            total_frames = len(source_video)

            if stop < 0:
                stop = total_frames - 1

            tracker = ROITracker(
                storage_path=storage_path,
                project_name=project_name,
                video_source=source_video,
                start_frame=start,
                stop_frame=stop,
                model_type=model,
                device=device,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )

            return tracker.track(progress=None)

    except Exception as e:
        logger.error(f"Tracking failed for {video_name}: {e}", exc_info=True)
        return f'Error: {e}'


def track_videos(storage_path: str, project_name: str, video_names,
                 model: str = 'r50_deaotl',
                 *, start: int = 0, stop: int = -1, skip_existing: bool = True,
                 device_ids=None,
                 progress_callback: Optional[Callable] = None,
                 on_video_done: Optional[Callable] = None,
                 cancel_event=None) -> dict:
    """Track multiple videos, spreading **whole videos across GPUs** when opted in.

    DeAOT is sequential *within* a video (memory propagation), so it cannot be
    frame-split like DINO extraction. But videos are independent, so when
    ``CASTLE_MULTI_GPU`` is on and >1 CUDA device is visible we run one whole
    video per GPU concurrently via :func:`castle.core.gpu_pool.run_on_device_pool`.
    Otherwise (flag off / single GPU / single video) we fall back to the existing
    sequential path, so every batch caller behaves identically.

    Args:
        video_names: iterable of video filenames in the project.
        skip_existing: skip videos that already have ``mask_list.h5`` (pre-flight).
        device_ids: override the GPU set; defaults to
            :func:`castle.core.gpu_pool.resolve_device_ids`.
        progress_callback: ``(fraction, desc)`` fired on each *video completion*
            (completed/total — not per-video internal %, which would interleave
            incoherently across concurrent videos).
        on_video_done: ``(video_name, status)`` fired as each video finishes — used
            by the Gradio batch UI to run its per-video post-tracking analysis.
        cancel_event: optional ``threading.Event``; once set, unstarted videos are
            reported as ``'Cancel'``.

    Returns:
        ``{video_name: status}`` where status is 'Done'/'Skip'/'Skipped'/'Cancel'/'Error: …'.
    """
    from castle.core.gpu_pool import (
        resolve_device_ids, run_on_device_pool, deterministic_ctx_if_enabled,
        host_ram_available_bytes, CANCELLED,
    )

    results: dict = {}
    todo = []
    for v in list(video_names):
        mask_path = Path(storage_path) / project_name / 'track' / v / 'mask_list.h5'
        if skip_existing and mask_path.exists():
            results[v] = 'Skipped'
        else:
            todo.append(v)
    if not todo:
        return results

    if device_ids is None:
        device_ids = resolve_device_ids()

    total = len(todo)
    completed = {'n': 0}

    def _norm(res) -> str:
        if res is CANCELLED:
            return 'Cancel'
        if isinstance(res, BaseException):
            return f'Error: {res}'
        return res

    def _announce(video: str, status: str) -> None:
        completed['n'] += 1
        if progress_callback is not None:
            progress_callback(completed['n'] / total, f"[{completed['n']}/{total}] {video} → {status}")
        if on_video_done is not None:
            on_video_done(video, status)

    if len(device_ids) >= 2 and len(todo) >= 2:
        from castle.core.environment import get_num_workers
        n_gpu = len(device_ids)
        # Divide the tracking DataLoader-worker budget across the concurrent GPUs
        # so N trackers don't oversubscribe CPU / pinned host RAM (the OOM cause).
        per_gpu_workers = max(1, get_num_workers('tracking') // n_gpu)
        logger.info("track_videos: video-level multi-GPU over %s for %d video(s) "
                    "(%d workers/GPU, pin_memory=False)", list(device_ids), len(todo), per_gpu_workers)

        # Best-effort host-RAM heads-up (DeAOT models + decode buffers x N).
        free = host_ram_available_bytes()
        if free is not None and free < 2 * (1024 ** 3) * n_gpu:
            logger.warning("track_videos: low free host RAM (%.1f GB) for %d-GPU tracking; "
                           "consider closing other apps.", free / 1024 ** 3, n_gpu)

        def _worker(video: str, device: str) -> str:
            assert device and str(device).startswith("cuda"), (
                f"track_videos worker requires an explicit cuda device, got {device!r}"
            )
            return track_video(storage_path, project_name, video, model=model,
                               start=start, stop=stop, skip_existing=False, device=device,
                               num_workers=per_gpu_workers, pin_memory=False)

        def _on_done(video: str, res) -> None:
            _announce(video, _norm(res))

        # Speed by default; opt in to per-GPU-reproducible cuDNN-deterministic via
        # CASTLE_MULTI_GPU_DETERMINISTIC. Masks are near-identical either way
        # (verified mean-IoU ~0.9999).
        with deterministic_ctx_if_enabled():
            pool_out = run_on_device_pool(todo, _worker, device_ids,
                                          on_done=_on_done, cancel_event=cancel_event)
        for v, r in zip(todo, pool_out):
            results[v] = _norm(r)
    else:
        for v in todo:
            if cancel_event is not None and cancel_event.is_set():
                results[v] = 'Cancel'
                continue
            status = track_video(storage_path, project_name, v, model=model,
                                 start=start, stop=stop, skip_existing=False)
            results[v] = status
            _announce(v, status)

    return results


def get_tracking_status(storage_path: str, project_name: str, video_name: str) -> dict:
    """
    Check tracking status for a video.
    
    Args:
        storage_path: Root storage directory
        project_name: Project name
        video_name: Video filename
    
    Returns:
        dict with keys:
            'tracked': bool — whether mask_list.h5 exists
            'mask_path': str — path to mask file
            'n_rois': int — number of ROIs found (0 if not tracked)
            'n_frames': int — number of tracked frames (0 if not tracked)
            'csv_path': str — path to kinematic CSV if exists
            'mix_video_path': str — path to mix video if exists
    """
    project_path = Path(storage_path) / project_name
    track_dir = project_path / 'track' / video_name
    mask_path = track_dir / 'mask_list.h5'
    
    result = {
        'tracked': mask_path.exists(),
        'mask_path': str(mask_path),
        'n_rois': 0,
        'n_frames': 0,
        'csv_path': '',
        'mix_video_path': '',
    }
    
    if not mask_path.exists():
        return result
    
    try:
        h5 = H5IO(str(mask_path), read_only=True)
        try:
            result['n_rois'] = h5.get_n_rois()
            result['n_frames'] = len(h5)
        finally:
            h5.close()
    except Exception as e:
        logger.warning(f"Could not read mask file {mask_path}: {e}")
    
    # Check for generated files
    video_basename = Path(video_name).stem
    csv_path = track_dir / f'{video_basename}-basic-information.csv'
    mix_path = track_dir / f'{video_basename}-mix.mp4'
    
    if csv_path.exists():
        result['csv_path'] = str(csv_path)
    if mix_path.exists():
        result['mix_video_path'] = str(mix_path)
    
    return result
