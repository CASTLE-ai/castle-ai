"""Video-level multi-GPU work-queue primitive.

DeAOT tracking is *sequential within a video* (the AOT engine propagates
segmentation frame->frame via long/short-term memory), so it cannot be
frame-split across GPUs the way DINO latent extraction can. Whole videos,
however, are independent — so the right parallelism is **video-level**: run
different videos on different GPUs concurrently.

This module provides the shared work-queue used by both
:func:`castle.service.tracking_service.track_videos` and the batch branch of
:func:`castle.service.extraction_service.extract_latent`.

Threads, not processes: each worker owns one GPU, builds its own
model/encoder/DataLoader on that device, and writes its own per-video output
file, so there is no shared mutable GPU state. PyTorch releases the GIL during
CUDA kernels, so the GPU work of different workers overlaps. CPU-bound work
(decode / resize / DataLoader / save) still contends on the GIL, so callers
should size each worker's DataLoader at ``get_num_workers(task) // n_gpu`` to
avoid oversubscribing the CPU and starving the GPUs.
"""

from __future__ import annotations

import contextlib
import logging
import os
import queue
import threading
from typing import Any, Callable, List, Optional, Sequence

logger = logging.getLogger(__name__)

# Falsey spellings of CASTLE_MULTI_GPU (kept identical to the extractor's
# auto-dispatcher so the opt-in semantics are uniform across the codebase).
_FALSEY = ("", "0", "false", "no", "off")

# Serialises the process-global cuDNN flag save/restore in cross_gpu_deterministic.
# cudnn.benchmark/.deterministic are process-wide; if two deterministic blocks
# (or a block and another handler) interleaved their save/restore, the wrong
# baseline could be restored, leaving the long-lived Gradio process stuck in
# deterministic mode. Holding this for the block makes save→set→restore atomic.
_cudnn_flag_lock = threading.Lock()


class _Cancelled:
    """Sentinel result for an item that was never started (pool cancelled)."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return "<cancelled>"


CANCELLED = _Cancelled()


def multi_gpu_enabled() -> bool:
    """True when ``CASTLE_MULTI_GPU`` is opted-in AND >1 CUDA device is visible.

    This is the single source of truth for the multi-GPU opt-in gate; the
    extractor's ``extract_roi_latent_from_video_auto`` dispatcher uses it too.
    """
    flag = os.environ.get("CASTLE_MULTI_GPU", "").strip().lower()
    if flag in _FALSEY:
        return False
    try:
        import torch
        return bool(torch.cuda.is_available() and torch.cuda.device_count() > 1)
    except Exception as exc:  # noqa: BLE001 - never let the check break the caller
        logger.warning("Multi-GPU check failed (%s); assuming single GPU.", exc)
        return False


@contextlib.contextmanager
def cross_gpu_deterministic():
    """Pin deterministic cuDNN for the duration, then restore previous settings.

    Sets ``cudnn.deterministic=True`` + ``benchmark=False`` while a multi-GPU pool
    runs and restores afterwards (so a long-lived process such as the Gradio app
    is not left in the slower deterministic mode). This makes a *given* GPU
    reproducible run-to-run and keeps tracking masks bit-stable.

    IMPORTANT — it does **not** make extraction bit-identical *across different
    physical GPUs*: DINO extraction runs under ``torch.autocast(float16)`` and the
    two dies round fp16 reductions slightly differently (a reproducible ~1e-2
    delta that cuDNN flags cannot remove). We accept that as fp16-level noise
    (single-GPU also runs fp16; this delta is negligible for downstream UMAP). For exact cross-GPU
    bit-identity you would have to drop the fp16 autocast (fp32) — slower + more
    VRAM. Uses cuDNN flags only (not ``torch.use_deterministic_algorithms``) to
    avoid raising on ops that lack a deterministic kernel.
    """
    try:
        import torch
    except Exception:  # noqa: BLE001 - torch is always present in practice
        yield
        return
    cudnn = torch.backends.cudnn
    # Hold the flag lock for the whole block so the save→set→restore is atomic
    # w.r.t. any other cuDNN-flag mutation; the long-lived process is never left
    # in deterministic mode by an interleaved restore.
    with _cudnn_flag_lock:
        prev_benchmark, prev_deterministic = cudnn.benchmark, cudnn.deterministic
        cudnn.benchmark = False
        cudnn.deterministic = True
        logger.info("cross_gpu_deterministic: cudnn.benchmark=False, deterministic=True")
        try:
            yield
        finally:
            cudnn.benchmark = prev_benchmark
            cudnn.deterministic = prev_deterministic


def multi_gpu_deterministic_enabled() -> bool:
    """True when ``CASTLE_MULTI_GPU_DETERMINISTIC`` opts into the slower, per-GPU
    reproducible cuDNN-deterministic path.

    **Default OFF (speed).** By default multi-GPU keeps the fast cuDNN benchmark +
    fp16 autocast — same numerical behaviour as single-GPU (results vary at the
    fp16 level run-to-run / across GPUs, which is negligible for downstream
    clustering). Set the var (``1``/``true``/…) to force determinism when exact
    per-GPU reproducibility matters more than throughput.
    """
    return os.environ.get("CASTLE_MULTI_GPU_DETERMINISTIC", "").strip().lower() not in _FALSEY


def deterministic_ctx_if_enabled():
    """Return :func:`cross_gpu_deterministic` if opted in, else a no-op context.

    Lets the pools write ``with deterministic_ctx_if_enabled():`` unconditionally.
    """
    if multi_gpu_deterministic_enabled():
        return cross_gpu_deterministic()
    return contextlib.nullcontext()


def host_ram_available_bytes() -> Optional[int]:
    """Best-effort free system RAM in bytes (psutil), or ``None`` if unavailable."""
    try:
        from castle.core.memory_guard import get_available_bytes
        return get_available_bytes("cpu")
    except Exception:  # noqa: BLE001
        return None


def available_cuda_devices() -> List[int]:
    """CUDA device indices usable for multi-GPU, **ignoring the env gate**.

    ``list(range(device_count()))`` when CUDA is available *and* ≥2 devices are
    visible, else ``[]``. Unlike :func:`resolve_device_ids` (which honours the
    ``CASTLE_MULTI_GPU`` env opt-in for CLI/headless), this reports raw hardware
    capability so the Gradio UI can offer a multi-GPU checkbox and pass the ids
    explicitly to ``track_videos(device_ids=…)`` regardless of the env var.
    """
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            return list(range(torch.cuda.device_count()))
    except Exception as exc:  # noqa: BLE001 - never let the check break the caller
        logger.warning("available_cuda_devices check failed (%s); assuming none.", exc)
    return []


def resolve_device_ids() -> List[int]:
    """CUDA device indices to spread work across, or ``[]`` when multi-GPU is off.

    Returns ``range(device_count())`` when :func:`multi_gpu_enabled` is true,
    otherwise ``[]`` — the caller then runs sequentially on the default device.
    """
    if not multi_gpu_enabled():
        return []
    try:
        import torch
        return list(range(torch.cuda.device_count()))
    except Exception:  # noqa: BLE001
        return []


def run_on_device_pool(
    items: Sequence[Any],
    worker: Callable[[Any, str], Any],
    device_ids: Sequence[int],
    *,
    on_done: Optional[Callable[[Any, Any], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> List[Any]:
    """Run ``worker(item, "cuda:N")`` over ``items`` with one thread per GPU.

    One worker thread is started per id in ``device_ids``; each pulls the next
    item off a shared queue and runs ``worker(item, device)`` on its assigned
    GPU. A GPU that finishes early immediately pulls the next item, so the queue
    self-balances and there is **no barrier** between items.

    Args:
        items: work items (e.g. video names), consumed in queue (input) order.
        worker: ``worker(item, device_str) -> result``; runs on a worker thread
            and MUST confine all its GPU work to ``device_str`` (e.g. ``"cuda:1"``).
        device_ids: CUDA device indices (one worker thread each). Must be
            non-empty — callers use :func:`resolve_device_ids` and take a
            sequential path when it returns ``[]``.
        on_done: optional ``on_done(item, result_or_exc)`` fired (serialized under
            a lock, on the finishing worker's thread) as each item completes —
            for progress reporting or per-item post-processing. Exceptions raised
            inside the hook are logged, not propagated.
        cancel_event: optional :class:`threading.Event`; once set, workers stop
            pulling new items. Not-yet-started items get :data:`CANCELLED`.

    Returns:
        A list aligned with ``items``: each entry is the worker's return value,
        the exception it raised (per-item isolation — one failure never kills the
        pool), or :data:`CANCELLED` if the item was skipped due to cancellation.
    """
    if not device_ids:
        raise ValueError("run_on_device_pool requires at least one device id")

    n = len(items)
    results: List[Any] = [CANCELLED] * n  # overwritten when an item actually runs
    index_q: "queue.Queue[int]" = queue.Queue()
    for i in range(n):
        index_q.put(i)
    done_lock = threading.Lock()

    def _drain(device: str) -> None:
        while True:
            if cancel_event is not None and cancel_event.is_set():
                break
            try:
                idx = index_q.get_nowait()
            except queue.Empty:
                break
            item = items[idx]
            try:
                res: Any = worker(item, device)
            except Exception as exc:  # noqa: BLE001 - isolate per item (let
                # KeyboardInterrupt/SystemExit propagate so a batch can abort)
                logger.exception("gpu_pool worker failed for %r on %s", item, device)
                res = exc
            results[idx] = res
            if on_done is not None:
                with done_lock:
                    try:
                        on_done(item, res)
                    except Exception:  # noqa: BLE001
                        logger.exception("gpu_pool on_done hook failed for %r", item)

    threads = [
        threading.Thread(
            target=_drain, args=(f"cuda:{d}",), daemon=True, name=f"gpupool-cuda{d}"
        )
        for d in device_ids
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results
