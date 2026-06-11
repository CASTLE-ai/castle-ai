"""castle/service/prepare_service.py — orchestration for the clustering Prepare step.

Resolves the user's latent-file selection into :class:`SourceSpec` (physical
path, source video, fps, ROI), keys a settings-deterministic cache, builds it
once (atomic dir swap + cross-process filelock), registers it under
``config['prepare']`` so multiple caches coexist like preprocess/extract
sessions, and loads it for the Explore stage.

Pure numerics live in :mod:`castle.core.prepare`; this layer owns project
config, fps probing, the cache directory, and concurrency.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence

from filelock import FileLock

from castle.core import prepare as _prepare
from castle.core import runtime_env
from castle.core.cluster import _resolve_latent_path
from castle.core.latent_scales import _spp_scales_of
from castle.core.prepare import SourceSpec, compute_prepare_id, is_stale, load_prepare, run_prepare
from castle.core.project import get_project_config, update_config

logger = logging.getLogger(__name__)

_BUILD_LOCK_TIMEOUT = 600.0  # a full build can take minutes (two passes over many GB)
_ROI_RE = re.compile(r"_ROI_(\d+)_ROI_(\d+)_")


def _prepared_root(storage_path: str, project_name: str) -> str:
    return os.path.join(storage_path, project_name, "cluster", "prepared")


def _latent_dir(storage_path: str, project_name: str, model_name: str) -> str:
    return os.path.join(storage_path, project_name, "latent", model_name)


def _roi_from_key(key: str) -> int:
    """Body-part (extraction) ROI parsed from the latent filename, fallback 1.

    Filenames look like ``..._ROI_{field}_ROI_{body}_{model}...`` — the second
    token is the extraction ROI the latent represents.
    """
    m = _ROI_RE.search(os.path.basename(key))
    if m:
        return int(m.group(2))
    return 1


def _read_fps(video_path: str, notify: Callable[[str], None]) -> float:
    try:
        from castle.utils.video_io import VideoReader

        with VideoReader(video_path) as vr:
            fps = float(vr.fps)
        if fps > 0:
            return fps
    except Exception as exc:  # noqa: BLE001 — fps probe is best-effort
        notify(f"Prepare: could not read fps from {os.path.basename(video_path)} ({exc}); using 30.0.")
    return 30.0


def resolve_sources(
    storage_path: str,
    project_name: str,
    model_name: str,
    selected_keys: Sequence[str],
    notify: Callable[[str], None] = logger.info,
    scales: Optional[Sequence[int]] = None,
) -> List[SourceSpec]:
    """Resolve config['latent'] keys into SourceSpec (path, video, fps, roi).

    When ``scales`` is given AND the selection contains SPP (multiscale) files,
    the per-video files are grouped and a single scale-combination SourceSpec is
    built per video (the requested scale blocks are column-concatenated before
    Prepare's PCA — so each scale-combo is its own cache). A video missing a
    requested scale is skipped with a warning. Otherwise (no scales, or
    weighted-average files) one SourceSpec is built per file, as before.
    """
    latent_dir = _latent_dir(storage_path, project_name, model_name)
    _, config = get_project_config(storage_path, project_name)
    latent_map = config.get("latent", {})
    sources_dir = os.path.join(storage_path, project_name, "sources")

    def _video_of(key: str) -> str:
        v = latent_map.get(key)
        if v is None:
            stem = os.path.basename(key)
            m = _ROI_RE.search(stem)
            v = (stem[: m.start()] + f"_ROI_{m.group(1)}.mp4") if m else stem
        return str(v)

    resolved = []  # (key, npz_path, video_name, raw_fps, roi, file_scales)
    for key in selected_keys:
        npz_path = _resolve_latent_path(latent_dir, key)
        video_name = _video_of(key)
        roi = _roi_from_key(key)
        raw_fps = _read_fps(os.path.join(sources_dir, video_name), notify)
        resolved.append((key, npz_path, video_name, raw_fps, roi, _spp_scales_of(key)))

    want = sorted({int(s) for s in scales}) if scales else None
    spp_present = any(fs for (*_, fs) in resolved)
    if not want or not spp_present:
        # No scale selection (or nothing multiscale): one source per file, whole.
        return [
            SourceSpec(key=k, npz_path=p, video_name=v, raw_fps=f, roi=roi,
                       file_scales=(sorted(fs) if fs else None))
            for (k, p, v, f, roi, fs) in resolved
        ]

    available = sorted({s for (*_, fs) in resolved for s in fs})
    req = [s for s in want if s in available]
    if not req:
        raise ValueError(
            f"None of the requested SPP scales {want} are available "
            f"(have {available}). Pick available scales or re-extract."
        )

    specs: List[SourceSpec] = []
    for (k, p, v, f, roi, fs) in resolved:
        if not fs:
            notify(f"Prepare: '{v}' is a weighted_average latent; "
                   f"skipped while SPP scales are selected.")
            continue
        avail = sorted(fs)
        missing = [s for s in req if s not in avail]
        if missing:
            notify(f"Prepare: '{v}' missing SPP scale(s) {missing}; skipped.")
            continue
        # Selecting every available scale needs no slicing — load the whole file
        # (req_scales=None), so "all scales" and "no selection" share one cache.
        req_scales = None if req == avail else req
        specs.append(SourceSpec(
            key=k, npz_path=p, video_name=v, raw_fps=f, roi=roi,
            file_scales=avail, req_scales=req_scales,
        ))
    return specs


def build_prepare(
    storage_path: str,
    project_name: str,
    model_name: str,
    selected_keys: Sequence[str],
    *,
    downsample: bool = True,
    target_fps_cap: float = 60.0,
    normalize: str = "l2",
    pca: bool = True,
    K: int = 1024,
    fit_fraction: float = 1.0,
    seed: int = 0,
    scales: Optional[Sequence[int]] = None,
    force: bool = False,
    notify: Callable[[str], None] = logger.info,
    progress_cb: Optional[Callable[[float, str], None]] = None,
    should_cancel: Callable[[], bool] = lambda: False,
) -> str:
    """Build (or reuse) a prepare cache; returns its ``prepare_id``.

    Atomic: builds into ``{id}.tmp/`` then ``os.replace`` to ``{id}/`` under a
    per-id ``FileLock`` so concurrent identical runs don't corrupt the dir.
    Reuses an existing, non-stale cache unless ``force``. ``progress_cb`` and
    ``should_cancel`` are forwarded to :func:`run_prepare`; on cancellation the
    partial ``{id}.tmp`` dir is removed and :class:`BuildCancelled` propagates.
    """
    specs = resolve_sources(storage_path, project_name, model_name, selected_keys, notify, scales)
    if not specs:
        raise ValueError("build_prepare: no latent files selected.")

    pid = compute_prepare_id(
        specs, downsample=downsample, target_fps_cap=target_fps_cap, normalize=normalize,
        pca=pca, K=K, fit_fraction=fit_fraction, model_name=model_name, seed=seed,
    )
    root = _prepared_root(storage_path, project_name)
    final_dir = os.path.join(root, pid)
    latent_dir = _latent_dir(storage_path, project_name, model_name)

    def _resolve(key: str) -> Optional[str]:
        p = _resolve_latent_path(latent_dir, key)
        return p if os.path.exists(p) else None

    if os.path.isdir(final_dir) and not force and not is_stale(final_dir, _resolve):
        notify(f"Prepare: reusing existing cache {pid}.")
        return pid

    os.makedirs(root, exist_ok=True)
    lock = FileLock(os.path.join(root, f"{pid}.lock"), timeout=_BUILD_LOCK_TIMEOUT)
    with lock:
        # Re-check after acquiring the lock: another process may have built it.
        if os.path.isdir(final_dir) and not force and not is_stale(final_dir, _resolve):
            notify(f"Prepare: cache {pid} was built by another process; reusing.")
            return pid
        tmp_dir = final_dir + ".tmp"
        shutil.rmtree(tmp_dir, ignore_errors=True)
        try:
            meta = run_prepare(
                tmp_dir, specs,
                downsample=downsample, target_fps_cap=target_fps_cap, normalize=normalize,
                pca=pca, K=K, fit_fraction=fit_fraction, model_name=model_name, seed=seed,
                avail_ram_bytes=runtime_env.available_ram_bytes(), notify=notify,
                progress_cb=progress_cb, should_cancel=should_cancel,
            )
        except _prepare.BuildCancelled:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            notify(f"Prepare: build {pid} cancelled; partial cache removed.")
            raise
        meta["created_at"] = datetime.now(timezone.utc).isoformat()
        with open(os.path.join(tmp_dir, _prepare.META_FILENAME), "w", encoding="utf-8") as f:
            import json

            json.dump(meta, f, indent=2, ensure_ascii=False)
        if os.path.isdir(final_dir):
            shutil.rmtree(final_dir)
        os.replace(tmp_dir, final_dir)
        notify(f"Prepare: built cache {pid} ({meta['n_dp_total']} datapoints x {meta['width']}).")

    # Register a summary so the UI can list/pick caches (multiple coexist).
    with update_config(storage_path, project_name) as config:
        reg = config.setdefault("prepare", {})
        reg[pid] = {
            "created_at": meta["created_at"],
            "model_name": model_name,
            "n_sources": len(specs),
            "n_dp_total": meta["n_dp_total"],
            "width": meta["width"],
            "scales": meta.get("scales"),  # SPP scales this cache represents (provenance)
            "downsample": meta["downsample"],
            "normalize": meta["normalize"],
            "pca_on": meta["pca"]["on"],
            "K": meta["pca"]["K"],
        }
    return pid


def prepared_dir(storage_path: str, project_name: str, prepare_id: str) -> str:
    return os.path.join(_prepared_root(storage_path, project_name), prepare_id)


def load_prepared(storage_path: str, project_name: str, prepare_id: str) -> "_prepare.PreparedData":
    """Load a prepare cache (see :func:`castle.core.prepare.load_prepare`)."""
    return load_prepare(prepared_dir(storage_path, project_name, prepare_id))


def list_prepared(storage_path: str, project_name: str) -> Dict[str, Any]:
    """Return the ``config['prepare']`` registry (id -> summary)."""
    _, config = get_project_config(storage_path, project_name)
    return dict(config.get("prepare", {}))


def delete_prepared(storage_path: str, project_name: str, prepare_id: str) -> None:
    """Remove a prepare cache directory + its registry entry."""
    shutil.rmtree(prepared_dir(storage_path, project_name, prepare_id), ignore_errors=True)
    with update_config(storage_path, project_name) as config:
        config.get("prepare", {}).pop(prepare_id, None)
