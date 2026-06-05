"""
castle/service/tracking_service.py
Service layer for ROI tracking operations.

All functions take simple types and return dicts.
No gradio imports.
"""

import logging
import os
import threading
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
                pin_memory: bool = True,
                cancel_event=None,
                frame_callback: Optional[Callable] = None) -> str:
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
        cancel_event: optional ``threading.Event``; checked per batch inside the
            tracker so an in-flight video aborts mid-track (partial output removed).
        frame_callback: optional ``(fraction, desc)`` fired by the tracker after
            each frame-batch — used by the batch UI for a frame-granular bar.

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

            return tracker.track(progress=None, cancel_event=cancel_event,
                                 frame_callback=frame_callback)

    except Exception as e:
        logger.error(f"Tracking failed for {video_name}: {e}", exc_info=True)
        return f'Error: {e}'


def track_video_2gpu(storage_path: str, project_name: str, video_name: str,
                     model: str = 'r50_deaotl',
                     device_ids=(0, 1), skip_existing: bool = True,
                     frame_callback: Optional[Callable] = None,
                     cancel_event=None, warmup: Optional[int] = None) -> str:
    """Track ONE video across GPUs by a warmup-overlap frame-split.

    DeAOT propagates a per-frame memory bank, so a naive split would break at the
    seam. But the ROI prompts are a shared project-wide reference pool, so we can
    re-seed each half from them and use a warmup region: GPU0 tracks ``[0, mid)``
    and writes it; GPU1 tracks ``[mid-K, N)`` but only WRITES ``[mid, N)`` — the
    first K (=warmup) frames re-propagate from the references to rebuild the memory
    so the boundary at ``mid`` is near-seamless. Both halves write disjoint frame
    keys into ONE shared, lock-guarded H5 writer. Not bit-identical to a single
    sequential pass, but reference-anchored IDs make label swaps unlikely.

    Falls back to single-GPU :func:`track_video` when <2 GPUs or the clip is too
    short to be worth splitting. Returns 'Done'/'Skipped'/'Cancel'/'Error: ...'.
    """
    from castle.core.environment import get_num_workers
    from castle.core.gpu_pool import deterministic_ctx_if_enabled

    device_ids = list(device_ids)
    project_path = Path(storage_path) / project_name
    track_dir = project_path / 'track' / video_name
    mask_path = track_dir / 'mask_list.h5'
    if skip_existing and mask_path.exists():
        return 'Skipped'
    video_path = project_path / 'sources' / video_name
    if not video_path.exists():
        return f'Error: Video not found: {video_path}'

    _MIN_SPLIT = 2000
    K = warmup if warmup is not None else int(os.environ.get('CASTLE_TRACK_WARMUP_FRAMES', '256') or 256)

    try:
        # One reader per half so concurrent decode never shares a single av handle.
        with ReadArray(str(video_path)) as src0, ReadArray(str(video_path)) as src1:
            N = len(src0)
            if len(device_ids) < 2 or N < max(2 * K, _MIN_SPLIT):
                # Not worth / not safe to split → single GPU (one reader is enough).
                return track_video(storage_path, project_name, video_name, model=model,
                                   skip_existing=skip_existing, cancel_event=cancel_event,
                                   frame_callback=frame_callback)

            mid = N // 2
            per_gpu = max(1, get_num_workers('tracking') // len(device_ids))
            t0 = ROITracker(storage_path=storage_path, project_name=project_name,
                            video_source=src0, start_frame=0, stop_frame=mid - 1,
                            model_type=model, device=f'cuda:{device_ids[0]}',
                            num_workers=per_gpu, pin_memory=False)
            t1 = ROITracker(storage_path=storage_path, project_name=project_name,
                            video_source=src1, start_frame=max(0, mid - K), stop_frame=N - 1,
                            model_type=model, device=f'cuda:{device_ids[1]}',
                            num_workers=per_gpu, pin_memory=False)
            if not t0.reference_frames:
                return ("Error: No ROI prompts found in this project. Create at least one "
                        "ROI in the 'Label ROI' tab first — prompts are shared across videos.")

            track_dir.mkdir(parents=True, exist_ok=True)
            lock = threading.Lock()
            half_frac = [0.0, 0.0]
            weights = [mid / N, (N - mid) / N]
            results: dict = {}
            errors: dict = {}

            def _agg(h):
                def cb(frac, desc=''):
                    with lock:
                        half_frac[h] = frac
                        combined = half_frac[0] * weights[0] + half_frac[1] * weights[1]
                    if frame_callback:
                        frame_callback(min(1.0, combined), desc)
                return cb

            def _run(h, trk, write_start):
                try:
                    results[h] = trk.track(progress=None, mask_writer=writer,
                                           write_start=write_start, cancel_event=cancel_event,
                                           frame_callback=_agg(h))
                except Exception as e:  # noqa: BLE001 — surfaced after join
                    errors[h] = e

            logger.info("track_video_2gpu: %s — frame-split N=%d at mid=%d (warmup K=%d) over GPUs %s",
                        video_name, N, mid, K, device_ids)
            with deterministic_ctx_if_enabled():
                with H5IO(str(mask_path)) as writer:
                    first = src0[0]
                    writer.write_config('n_rois', t0.n_rois)
                    writer.write_config('total_frames', N)
                    writer.write_config('height', first.shape[0])
                    writer.write_config('width', first.shape[1])
                    threads = [
                        threading.Thread(target=_run, args=(0, t0, 0), daemon=True),
                        threading.Thread(target=_run, args=(1, t1, mid), daemon=True),
                    ]
                    for t in threads:
                        t.start()
                    for t in threads:
                        t.join()

        # Writer closed. Clean up the partial file on any error/cancel so a re-run
        # (skip_existing) re-tracks cleanly instead of treating a stub as complete.
        cancelled = (cancel_event is not None and cancel_event.is_set()) \
            or any(r == 'Cancel' for r in results.values())
        if errors or cancelled:
            try:
                os.remove(mask_path)
            except OSError:
                pass
            if errors:
                return f'Error: {next(iter(errors.values()))}'
            return 'Cancel'
        return 'Done'

    except Exception as e:  # noqa: BLE001
        logger.error("track_video_2gpu failed for %s: %s", video_name, e, exc_info=True)
        return f'Error: {e}'


def track_videos(storage_path: str, project_name: str, video_names,
                 model: str = 'r50_deaotl',
                 *, start: int = 0, stop: int = -1, skip_existing: bool = True,
                 device_ids=None,
                 progress_callback: Optional[Callable] = None,
                 on_video_done: Optional[Callable] = None,
                 cancel_event=None,
                 frame_callback: Optional[Callable] = None) -> dict:
    """Track multiple videos, spreading **whole videos across GPUs** when opted in.

    DeAOT is sequential *within* a video (memory propagation). With ≥2 GPUs and
    ≥2 videos we run one whole video per GPU concurrently via
    :func:`castle.core.gpu_pool.run_on_device_pool`. With ≥2 GPUs and a SINGLE
    video we frame-split it across the GPUs with a warmup overlap
    (:func:`track_video_2gpu`) — re-seeding each half from the shared reference
    prompts. Otherwise (flag off / single GPU) we fall back to the sequential path.

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
        frame_callback: optional ``(video_name, fraction)`` fired after each
            frame-batch of *every* video (the per-video tracker fraction, tagged
            with the video name) — lets the batch UI aggregate a frame-granular
            overall progress bar across (possibly concurrent) videos.

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
            fcb = (lambda frac, desc, v=video: frame_callback(v, frac)) if frame_callback else None
            return track_video(storage_path, project_name, video, model=model,
                               start=start, stop=stop, skip_existing=False, device=device,
                               num_workers=per_gpu_workers, pin_memory=False,
                               cancel_event=cancel_event, frame_callback=fcb)

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
    elif len(device_ids) >= 2 and len(todo) == 1 and start == 0 and stop == -1:
        # A single video can't go to the video-level pool — frame-split it across
        # the GPUs (warmup-overlap) so both cards work instead of one idling.
        v = todo[0]
        fcb = (lambda frac, desc, vv=v: frame_callback(vv, frac)) if frame_callback else None
        status = track_video_2gpu(storage_path, project_name, v, model=model,
                                  device_ids=device_ids, skip_existing=False,
                                  frame_callback=fcb, cancel_event=cancel_event)
        results[v] = status
        _announce(v, status)
    else:
        for v in todo:
            if cancel_event is not None and cancel_event.is_set():
                results[v] = 'Cancel'
                continue
            fcb = (lambda frac, desc, v=v: frame_callback(v, frac)) if frame_callback else None
            status = track_video(storage_path, project_name, v, model=model,
                                 start=start, stop=stop, skip_existing=False,
                                 cancel_event=cancel_event, frame_callback=fcb)
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
