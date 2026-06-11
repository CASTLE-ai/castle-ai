"""castle/core/prepare.py — cached dimensionality-reduction "Prepare" step for clustering.

The Behavior-Microscope clustering stage cannot hold raw DINOv3 latents for a
large project in RAM (e.g. 26 mice x 215784 frames x 16128 features = 362 GB
float32). The "Prepare" step runs a one-time, cacheable, **independently
toggleable** pipeline that shrinks the data once so the interactive UMAP/DBSCAN
explore can run cheaply many times on the cache:

    select latent files
        -> (downsample)  per-video nearest-frame decimation to a target fps
        -> (normalize)   per-sample L2 (cosine geometry; magnitude is nuisance)
        -> (pca)         IncrementalPCA, center-only, NO whitening, top-K
        -> reduced (N_dp, K) float32 + FrameIndexMap + meta.json

Each stage is optional; with all three off the cache is the raw (decimated)
features, equivalent to the legacy path. This module is **pure computation** —
it takes resolved source paths + per-video fps and does not touch project
config, Gradio, or VideoReader. Orchestration (file selection, fps probing,
config registry, atomic dir swap, filelock) lives in
:mod:`castle.service.prepare_service`.

Design notes that matter:
* ``load_latent_safe`` already normalises +/-Inf -> NaN, and non-finite values
  appear as **whole rows** (tracking-loss frames). We keep that row-alignment
  end to end: NaN rows are excluded from the PCA *fit* but pass through
  *transform* as NaN, so downstream ``Latent.__init__`` still marks them
  ``cluster = -1``.
* Windowing (``time_window``) is an Explore-time op and is **not** applied here.
  The cache is at decimated-frame resolution; :class:`FrameIndexMap` is
  W-agnostic and gains a ``for_window(W)`` view at explore time.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, cast

import numpy as np
import numpy.typing as npt

from castle.utils.safe_load import load_latent_safe

FloatArr = npt.NDArray[np.float32]
IntArr = npt.NDArray[np.int64]
F64Arr = npt.NDArray[np.float64]

logger = logging.getLogger(__name__)

PREPARE_SCHEMA_VERSION = 1
_L2_EPS = 1e-8


class BuildCancelled(Exception):
    """Raised inside :func:`run_prepare` when ``should_cancel()`` turns True.

    The service layer catches it, removes the partial ``{id}.tmp`` dir, and
    re-raises so the UI can report a clean cancellation.
    """

# meta.json / array filenames inside a prepare dir.
META_FILENAME = "meta.json"
PCA_FILENAME = "reduced.dat"
INDEX_MAP_FILENAME = "frame_index_map.npz"
PCA_COMPONENTS_FILENAME = "pca_components.npy"
PCA_MEAN_FILENAME = "pca_mean.npy"


# --------------------------------------------------------------------------- #
# Decimation                                                                  #
# --------------------------------------------------------------------------- #
def decimate_indices(n_orig: int, raw_fps: Optional[float], target_fps: Optional[float]) -> IntArr:
    """Nearest-frame index resample of ``n_orig`` frames to ``target_fps``.

    Picks *real* frames (no averaging) at the target time grid, so every kept
    point lies on the true behaviour manifold. Returns indices into the
    original frame axis, monotonically non-decreasing.

    No-op (returns ``arange(n_orig)``) when downsampling is disabled
    (``target_fps`` is None) or the source is already at/below the target
    (``raw_fps <= target_fps``). For an integer ratio (e.g. 120->60) this is an
    exact uniform stride; for a non-integer ratio (e.g. 100->60) the gaps are
    near-uniform (+/- half a frame), which is negligible at behaviour
    timescales.
    """
    n_orig = int(n_orig)
    if n_orig <= 0:
        return np.zeros(0, dtype=np.int64)
    if target_fps is None or raw_fps is None or raw_fps <= target_fps:
        return np.arange(n_orig, dtype=np.int64)
    n_out = int(np.floor(n_orig * float(target_fps) / float(raw_fps)))
    if n_out <= 0:
        return np.zeros(0, dtype=np.int64)
    idx = np.round(np.arange(n_out) * (float(raw_fps) / float(target_fps))).astype(np.int64)
    return np.asarray(np.clip(idx, 0, n_orig - 1), dtype=np.int64)


def effective_target_fps(raw_fps: Optional[float], cap: float) -> Optional[float]:
    """Target fps for a video = ``min(raw_fps, cap)`` (raw<=cap -> no decimation)."""
    if raw_fps is None:
        return None
    return min(float(raw_fps), float(cap))


# --------------------------------------------------------------------------- #
# FrameIndexMap                                                               #
# --------------------------------------------------------------------------- #
@dataclass
class FrameIndexMap:
    """Maps datapoint indices back to original video frames.

    Generalises both ``videos_meta`` and the legacy hard-coded
    ``index*bin_size + bin_size//2`` / ``np.repeat(., time_window)`` contract so
    decimation + windowing route through one place.

    Resolution here is **decimated** frames (W-agnostic). Call
    :meth:`for_window` to get a window-aware view whose datapoint index matches
    what UMAP/DBSCAN see at explore time.

    Attributes:
        video_names: ``(V,)`` source video filenames, in cache row order.
        dp_offsets: ``(V+1,)`` cumulative decimated-frame counts; rows
            ``dp_offsets[v]:dp_offsets[v+1]`` of the reduced cache belong to
            video ``v``.
        orig_frame_idx: ``(N_dp,)`` original (pre-decimation) frame index within
            each video for every decimated row; monotonic within a video slice.
        raw_fps: ``(V,)`` each video's own fps (for bout durations).
        n_orig_frames: ``(V,)`` original frame count per video (post any
            truncation), so labels can be expanded back to full resolution.
        source_roi: ``(V,)`` ROI id each video's latent came from (for grid
            mask overlay).
    """

    video_names: List[str]
    dp_offsets: IntArr
    orig_frame_idx: IntArr
    raw_fps: F64Arr
    n_orig_frames: IntArr
    source_roi: IntArr

    # --- persistence ------------------------------------------------------- #
    def save(self, path: str) -> None:
        np.savez(
            path,
            video_names=np.array(self.video_names, dtype=object),
            dp_offsets=self.dp_offsets,
            orig_frame_idx=self.orig_frame_idx,
            raw_fps=self.raw_fps,
            n_orig_frames=self.n_orig_frames,
            source_roi=self.source_roi,
        )

    @classmethod
    def load(cls, path: str) -> "FrameIndexMap":
        with np.load(path, allow_pickle=True) as d:
            return cls(
                video_names=[str(v) for v in d["video_names"].tolist()],
                dp_offsets=d["dp_offsets"].astype(np.int64),
                orig_frame_idx=d["orig_frame_idx"].astype(np.int64),
                raw_fps=d["raw_fps"].astype(np.float64),
                n_orig_frames=d["n_orig_frames"].astype(np.int64),
                source_roi=d["source_roi"],
            )

    @property
    def n_videos(self) -> int:
        return len(self.video_names)

    @property
    def n_datapoints(self) -> int:
        return int(self.dp_offsets[-1]) if len(self.dp_offsets) else 0

    def for_window(self, window: int) -> "WindowedFrameIndexMap":
        """Return a window-aware view for non-overlapping windows of size W.

        Each windowed datapoint = W consecutive decimated frames *within one
        video* (windowing is per-video so a window never straddles two videos).
        """
        return WindowedFrameIndexMap(self, max(1, int(window)))


@dataclass
class WindowedFrameIndexMap:
    """Window-aware view of a :class:`FrameIndexMap` (per-video, stride-W)."""

    base: FrameIndexMap
    window: int
    # derived
    n_windows_per_video: IntArr = field(init=False)
    win_offsets: IntArr = field(init=False)

    def __post_init__(self) -> None:
        base, W = self.base, self.window
        counts = np.diff(base.dp_offsets)  # decimated frames per video
        self.n_windows_per_video = counts // W
        self.win_offsets = np.concatenate([[0], np.cumsum(self.n_windows_per_video)]).astype(np.int64)

    @property
    def n_windows(self) -> int:
        return int(self.win_offsets[-1]) if len(self.win_offsets) else 0

    def _video_of_window(self, global_w: int) -> int:
        return int(np.searchsorted(self.win_offsets, global_w, side="right") - 1)

    def _window_orig_span(self, v: int, local_w: int) -> Tuple[int, int]:
        """Original-frame span ``[start, stop)`` a window covers within video v."""
        base, W = self.base, self.window
        dp_start = int(base.dp_offsets[v])
        nwin = int(self.n_windows_per_video[v])
        start = int(base.orig_frame_idx[dp_start + local_w * W])
        if local_w + 1 < nwin:
            stop = int(base.orig_frame_idx[dp_start + (local_w + 1) * W])
        else:
            stop = int(base.n_orig_frames[v])
        return start, stop

    def dp_to_orig_frame(self, global_w: int) -> Tuple[int, int]:
        """Windowed datapoint -> (video_idx, representative original frame).

        Representative = the midpoint of the window's original-frame span, which
        reduces to the legacy ``idx*bin_size + bin_size//2`` bin centre and to a
        sensible centre frame for decimated windows.
        """
        v = self._video_of_window(global_w)
        local_w = global_w - int(self.win_offsets[v])
        start, stop = self._window_orig_span(v, local_w)
        return v, min((start + stop) // 2, stop - 1)

    def dp_to_orig_span(self, global_w: int) -> Tuple[int, int, int]:
        """Windowed datapoint -> ``(video_idx, first_orig, last_orig_exclusive)``.

        The original-frame interval the datapoint actually *covers* (its window's
        decimated span expanded to original frames), as opposed to
        :meth:`dp_to_orig_frame`'s single representative midpoint. Used to draw a
        preview clip's ROI overlay only on the datapoint's true extent (leaving
        the surrounding buffer frames un-marked). Legacy (window=1) returns the
        bin's frame span.
        """
        v = self._video_of_window(global_w)
        local_w = global_w - int(self.win_offsets[v])
        start, stop = self._window_orig_span(v, local_w)
        return v, start, stop

    def windowed_row_range(self, video_idx: int) -> Tuple[int, int]:
        """Global windowed-datapoint row range ``[start, stop)`` for a video."""
        return int(self.win_offsets[video_idx]), int(self.win_offsets[video_idx + 1])

    def datapoint_window_ids(self) -> IntArr:
        """Global window id each decimated cache row belongs to (-1 for the tail).

        Maps the per-decimated-frame cache (``dp_offsets`` order) to the
        per-window label/embedding arrays (``win_offsets`` order). Rows in a
        video's truncated tail (``n_decimated % W`` frames) get -1. Used to
        expand per-window cluster labels/embeddings back to per-frame for the
        transfer-model export.
        """
        base, W = self.base, self.window
        out = np.full(base.n_datapoints, -1, dtype=np.int64)
        for v in range(base.n_videos):
            dp_s = int(base.dp_offsets[v])
            nwin = int(self.n_windows_per_video[v])
            win_s = int(self.win_offsets[v])
            for u in range(nwin):
                rs = dp_s + u * W
                out[rs:rs + W] = win_s + u
        return out

    def expand_labels_to_orig(self, per_window_labels: npt.ArrayLike, video_idx: int) -> IntArr:
        """Expand one video's per-window labels back to per-original-frame.

        Every original frame inherits the label of the window whose decimated
        span covers it (gap-filling between successive windows' first frames).
        Frames before the first / after the last window stay ``-1``.
        """
        base, W = self.base, self.window
        v = video_idx
        n_orig = int(base.n_orig_frames[v])
        out = np.full(n_orig, -1, dtype=np.int64)
        nwin = int(self.n_windows_per_video[v])
        if nwin == 0:
            return out
        per_window_labels = np.asarray(per_window_labels)
        dp_start = int(base.dp_offsets[v])
        # First original frame represented by each window (its first decimated frame).
        win_first_orig = base.orig_frame_idx[dp_start : dp_start + nwin * W : W]
        for u in range(nwin):
            start = int(win_first_orig[u])
            stop = int(win_first_orig[u + 1]) if u + 1 < nwin else n_orig
            out[start:stop] = int(per_window_labels[u])
        return out

    def windowed_labels_from_orig(self, per_orig_labels: npt.ArrayLike, video_idx: int) -> IntArr:
        """Sample one video's per-window labels from a per-original-frame array.

        Inverse of :meth:`expand_labels_to_orig`: each window takes the label at
        its first represented original frame. Used to recover per-window GLOBAL
        cluster labels from the authoritative original-frame ``time_series`` CSV
        on session restore / annotator load (the cache npz only stores per-submit
        LOCAL labels, so the CSV is the source of truth). Reproduces the legacy
        ``values[::bin_size]`` contract when ``window == 1`` and the base map is a
        legacy bin map. Out-of-range frames (truncated CSV) stay ``-1``.
        """
        base, W = self.base, self.window
        v = video_idx
        nwin = int(self.n_windows_per_video[v])
        out = np.full(nwin, -1, dtype=np.int64)
        if nwin == 0:
            return out
        per = np.asarray(per_orig_labels)
        dp_start = int(base.dp_offsets[v])
        win_first_orig = base.orig_frame_idx[dp_start : dp_start + nwin * W : W]
        n = len(per)
        for u in range(nwin):
            f = int(win_first_orig[u])
            if 0 <= f < n:
                out[u] = int(per[f])
        return out


def build_legacy_index_map(
    videos_meta: Sequence[Tuple[int, str]],
    bin_size: int,
    raw_fps: Optional[Dict[str, float]] = None,
    source_roi: Optional[Dict[str, int]] = None,
) -> FrameIndexMap:
    """Build a FrameIndexMap that reproduces the legacy ``bin_size`` contract.

    Legacy aggregation truncates each video to a multiple of ``bin_size`` and
    bins are contiguous ``bin_size`` original frames, so the decimated-frame
    axis here == bin axis, ``orig_frame_idx`` is a plain stride, and
    ``for_window(1)`` reproduces ``np.repeat(., bin_size)`` /
    ``idx*bin_size + bin_size//2`` exactly.
    """
    names: List[str] = []
    offsets = [0]
    orig: List[IntArr] = []
    fps_list: List[float] = []
    n_orig_list: List[int] = []
    roi_list: List[int] = []
    for n_bins, name in videos_meta:
        names.append(name)
        offsets.append(offsets[-1] + int(n_bins))
        # bin b spans original frames [b*bin_size, (b+1)*bin_size); store the
        # span START so for_window(1) reproduces np.repeat (expand) and the
        # span-midpoint representative == the legacy b*bin_size + bin_size//2.
        orig.append(np.arange(int(n_bins), dtype=np.int64) * bin_size)
        fps_list.append(float((raw_fps or {}).get(name, 30.0)))
        n_orig_list.append(int(n_bins) * int(bin_size))
        roi_list.append(int((source_roi or {}).get(name, 1)))
    return FrameIndexMap(
        video_names=names,
        dp_offsets=np.array(offsets, dtype=np.int64),
        orig_frame_idx=np.concatenate(orig) if orig else np.zeros(0, dtype=np.int64),
        raw_fps=np.array(fps_list, dtype=np.float64),
        n_orig_frames=np.array(n_orig_list, dtype=np.int64),
        source_roi=np.array(roi_list, dtype=np.int64),
    )


# --------------------------------------------------------------------------- #
# Normalisation helpers                                                       #
# --------------------------------------------------------------------------- #
def l2_normalize_rows(x: npt.ArrayLike, eps: float = _L2_EPS) -> FloatArr:
    """Per-sample L2 normalisation (eps-guarded). NaN rows stay NaN."""
    arr = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(arr, axis=1, keepdims=True)
    norm = np.maximum(norm, eps)
    return np.asarray(arr / norm, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Cache key + provenance                                                      #
# --------------------------------------------------------------------------- #
@dataclass
class SourceSpec:
    """One selected latent source for a video, resolved by the service layer.

    Normally a single ``.npz`` file. For multiscale scale-combination the source
    instead column-concatenates the requested SPP scale blocks (``req_scales``)
    gathered from ``scale_files`` — each ``(npz_path, file_scales)`` — *before*
    Prepare's L2/PCA, so the cache is built on exactly the chosen scales. The
    block for a scale is read from a per-scale file or sliced from a legacy
    combined file (see :func:`castle.core.latent_scales._scale_block`).
    """

    key: str          # config['latent'] logical key (the primary file)
    npz_path: str     # physical .npz path (the primary / first file)
    video_name: str   # source mp4 filename
    raw_fps: float
    roi: int          # source ROI id (for grid mask overlay)
    scale_files: Optional[List[Tuple[str, List[int]]]] = None  # (path, file_scales)
    req_scales: Optional[List[int]] = None                     # ascending subset to combine


def _load_scale_combined_latent(s: "SourceSpec") -> npt.NDArray[Any]:
    """Column-concatenate ``s.req_scales`` blocks for one source, ascending.

    Each contributing file is loaded once (native dtype, no Inf→NaN rewrite —
    matching :func:`_load_raw_decidx`); its blocks for any requested scales are
    sliced out and the per-scale blocks are hstacked in ascending-scale order.
    """
    from castle.core.latent_scales import _scale_block
    assert s.req_scales and s.scale_files
    req = sorted(s.req_scales)
    blocks: Dict[int, npt.NDArray[Any]] = {}
    for path, file_scales in s.scale_files:
        needed = [sc for sc in req if sc in file_scales and sc not in blocks]
        if not needed:
            continue
        arr = load_latent_safe(path, fix_nonfinite=False)
        for sc in needed:
            blocks[sc] = _scale_block(arr, file_scales, sc)
    missing = [sc for sc in req if sc not in blocks]
    if missing:
        raise ValueError(
            f"Source '{s.video_name}' is missing SPP scale(s) {missing} for the "
            f"requested combination {req}."
        )
    n = min(blocks[sc].shape[0] for sc in req)
    return np.hstack([np.ascontiguousarray(blocks[sc][:n]) for sc in req])


def _round6(x: float) -> float:
    return round(float(x), 6)


def compute_prepare_id(
    sources: Sequence[SourceSpec],
    *,
    downsample: bool,
    target_fps_cap: float,
    normalize: str,
    pca: bool,
    K: int,
    fit_fraction: float,
    model_name: str,
) -> str:
    """Deterministic 8-char id from the selection + every toggle/param.

    Includes each source's mtime+size so editing/re-extracting a latent
    invalidates the cache. Excludes ``castle_version`` (use
    ``prepare_schema_version`` for format bumps).
    """
    parts: List[str] = [
        f"schema={PREPARE_SCHEMA_VERSION}",
        f"model={model_name}",
        f"downsample={int(bool(downsample))}",
        f"target_fps_cap={_round6(target_fps_cap)}",
        f"normalize={normalize}",
        f"pca={int(bool(pca))}",
        f"K={int(K)}",
        f"fit_fraction={_round6(fit_fraction)}",
    ]
    for s in sorted(sources, key=lambda x: x.key):
        # Hash every contributing file (a scale-combination source has several),
        # plus the requested scale subset, so different scale combos / edited
        # files map to different caches.
        files = s.scale_files if s.scale_files else [(s.npz_path, [])]
        sigs = []
        for path, _fs in sorted(files):
            try:
                st = os.stat(path)
                sigs.append(f"{path}|{_round6(st.st_mtime)}|{st.st_size}")
            except OSError:
                sigs.append(f"{path}|missing|missing")
        scales_tag = (
            "x".join(str(x) for x in sorted(s.req_scales)) if s.req_scales else "-"
        )
        parts.append(f"{s.key}|scales={scales_tag}|" + ";".join(sigs))
    payload = "\n".join(parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]


# --------------------------------------------------------------------------- #
# I/O prefetch + device selection                                             #
# --------------------------------------------------------------------------- #
def _peek_header(npz_path: str, key: str = "latent") -> Tuple[int, int, int]:
    """``(n_rows, n_features, itemsize_bytes)`` of the ``latent`` member, cheaply.

    Reads only the ``.npy`` header of the zip member (numpy.lib.format) — for a
    compressed npz this inflates just the few header bytes, not the multi-GB
    array — so we learn the exact shape AND dtype size (needed to bound RAM)
    without decompressing. Falls back to a full load for non-standard npz.
    """
    import zipfile

    import numpy.lib.format as _nfmt
    try:
        with zipfile.ZipFile(npz_path) as z:
            member = f"{key}.npy"
            with z.open(member) as f:
                version = _nfmt.read_magic(f)
                shape, _fortran, dtype = _nfmt._read_array_header(f, version)  # type: ignore[attr-defined]  # noqa: SLF001
        if len(shape) >= 2:
            return int(shape[0]), int(shape[1]), int(dtype.itemsize)
    except Exception:  # noqa: BLE001 — fall back to a (slow) full load
        pass
    arr = load_latent_safe(npz_path)
    n0, nf, isz = int(arr.shape[0]), int(arr.shape[1]), int(arr.dtype.itemsize)
    del arr
    return n0, nf, isz


def _max_concurrent_loads(sources: Sequence[SourceSpec], avail_ram_bytes: Optional[int]) -> int:
    """How many files may be resident at once so loaders can't OOM the box.

    Each in-flight file holds up to its *raw* (pre-decimation) bytes during
    inflate. We keep ``max_concurrent x largest_raw_file <= available_ram -
    margin`` so the worst case (all loaders mid-inflate) still leaves the margin
    free. Margin default 10 GiB; overridable via ``CASTLE_PREPARE_RAM_MARGIN_GB``
    and ``CASTLE_PREPARE_LOADERS`` (hard cap on threads).
    """
    try:
        margin = int(float(os.environ.get("CASTLE_PREPARE_RAM_MARGIN_GB", "10")) * (1 << 30))
    except ValueError:
        margin = 10 << 30
    # Conservative fallback when RAM is unknown (non-Linux / unreadable procfs):
    # assume a modest 8 GiB so we don't over-prefetch on a small box.
    avail = int(avail_ram_bytes) if avail_ram_bytes else (8 << 30)
    max_raw = 1
    for s in sources:
        try:
            # A scale-combination source holds several files resident while it
            # combines; sum their raw bytes for the worst-case footprint.
            files = [p for p, _ in s.scale_files] if s.scale_files else [s.npz_path]
            raw = 0
            for p in files:
                n0, nf, isz = _peek_header(p)
                raw += n0 * nf * isz
            max_raw = max(max_raw, raw)
        except Exception:  # noqa: BLE001
            pass
    usable = max(0, avail - margin)
    by_ram = max(1, usable // max_raw)
    try:
        hard_cap = int(os.environ.get("CASTLE_PREPARE_LOADERS", "4"))
    except ValueError:
        hard_cap = 4
    return int(max(1, min(by_ram, max(1, hard_cap), len(sources))))


def _parallel_decimated(
    sources: Sequence[SourceSpec],
    *,
    downsample: bool,
    target_fps_cap: float,
    should_cancel: Callable[[], bool],
    max_concurrent: int,
) -> Iterator[Tuple[SourceSpec, npt.NDArray[Any], IntArr]]:
    """Yield ``(source, raw_array, dec_idx)`` in **source order**, with a single
    background thread prefetching up to ``max_concurrent`` files ahead.

    The consumer slices ``raw_array[dec_idx[i:i+W]]`` one block at a time, so no
    full decimated copy is ever made (that raw+copy coexistence is what blew the
    RAM budget). A single loader thread inflates files serially (numpy's
    compressed-npz read is GIL-bound, so parallel inflate is slower) but runs
    ahead of the consumer, overlapping the load with the GPU work. Two safety
    invariants:

    * **Bounded RAM**: a per-file semaphore slot is acquired *before* inflating
      and released by the consumer *after* the file is used, so at most
      ``max_concurrent`` raw arrays are resident (the OOM guard — sized so the
      resident raws stay under ``available_ram - margin``).
    * **Deterministic order**: results are yielded strictly in source order, so
      the PCA Gram accumulation (a sum of per-file contributions) is
      bit-reproducible run-to-run.

    ``CASTLE_PREPARE_PREFETCH=0`` disables prefetch (one resident file, no overlap).
    """
    import threading

    # ONE loader thread, not N: numpy's compressed-npz read is GIL-bound (it
    # ping-pongs the GIL on small inflate chunks), so two threads inflating at
    # once are *slower* than one (measured 14.8 vs 10.2 s/file) AND double the
    # resident RAM. Instead a single thread inflates serially and prefetches up
    # to ``conc`` files ahead, overlapping the load with the GPU work. ``conc``
    # (RAM-bounded) is only the resident cap / prefetch depth.
    serial = os.environ.get("CASTLE_PREPARE_PREFETCH", "").strip() in ("0", "false", "no")
    n = len(sources)
    conc = 1 if serial else max(1, min(int(max_concurrent), n))
    n_threads = 1

    sem = threading.Semaphore(conc)
    cv = threading.Condition()
    results: Dict[int, Tuple[SourceSpec, npt.NDArray[Any], IntArr]] = {}
    state: Dict[str, Any] = {"err": None}
    abort = threading.Event()
    dispatch = {"next": 0}
    dispatch_lock = threading.Lock()

    def _worker() -> None:
        while not abort.is_set() and not should_cancel():
            with dispatch_lock:
                idx = dispatch["next"]
                if idx >= n:
                    return
                dispatch["next"] = idx + 1
            # Bound concurrency: block until a resident slot frees (consumer
            # releases one per file). Time out so we re-check abort/cancel.
            while not sem.acquire(timeout=0.5):
                if abort.is_set() or should_cancel():
                    return
            try:
                raw, dec_idx, _n = _load_raw_decidx(
                    sources[idx], downsample=downsample, target_fps_cap=target_fps_cap
                )
            except Exception as exc:  # noqa: BLE001 — surfaced to the consumer
                sem.release()
                with cv:
                    if state["err"] is None:
                        state["err"] = exc
                    cv.notify_all()
                return
            with cv:
                results[idx] = (sources[idx], raw, dec_idx)
                cv.notify_all()
            # Drop the worker's own refs immediately — otherwise this local
            # variable keeps the just-loaded array alive while the worker inflates
            # its NEXT file, so a worker transiently holds TWO raws and the
            # resident-bytes bound (and the box) blow up. results/consumer own it now.
            del raw, dec_idx

    threads = [threading.Thread(target=_worker, daemon=True, name=f"prep-load-{i}")
               for i in range(n_threads)]
    for t in threads:
        t.start()
    try:
        for idx in range(n):
            with cv:
                while idx not in results and state["err"] is None and not should_cancel():
                    cv.wait(timeout=0.5)
                if state["err"] is not None:
                    raise state["err"]
                if should_cancel():
                    raise BuildCancelled()
                s, raw, dec_idx = results.pop(idx)
            yield s, raw, dec_idx
            del raw, dec_idx
            sem.release()  # free a resident slot now this file is consumed
    finally:
        abort.set()
        with cv:
            cv.notify_all()


def _to_device_l2(block: npt.ArrayLike, normalize: str, dev: Any, torch: Any) -> Tuple[Any, Any]:
    """Upload a native-dtype block to ``dev``, upcast to float32, L2-normalise.

    The upcast + L2 run on ``dev`` (GPU, or multi-threaded CPU torch) instead of
    single-threaded numpy on the host — that host-side ``_prep_block`` was the
    7.8 s/file CPU bottleneck. Transfers the block in its native (e.g. float16)
    dtype, so only half the bytes cross the PCIe bus. NaN rows stay NaN and are
    reported via the ``finite`` mask. Returns ``(xb_float32_on_dev, finite_mask)``.
    """
    t = torch.from_numpy(np.ascontiguousarray(block)).to(dev)  # native dtype (e.g. fp16)
    xb = t.to(torch.float32)
    finite = torch.isfinite(xb).all(dim=1)
    if normalize == "l2":
        norm = torch.linalg.vector_norm(xb, dim=1, keepdim=True).clamp_min(_L2_EPS)
        xb = xb / norm  # NaN rows: norm is NaN -> row stays NaN (finite already False)
    return xb, finite


def _select_pca_device(n_features: int, notify: Callable[[str], None]) -> str:
    """Pick 'cuda:N' (largest free VRAM) when there's room for the D×D solve, else 'cpu'.

    The covariance + ``eigh`` solve needs roughly the Gram matrix plus the
    eigensolver workspace on the device (~2·D²·4 bytes + a few GB). When no GPU
    has that free (e.g. llama-server is resident), fall back to CPU. Honour
    ``CASTLE_PREPARE_DEVICE=cpu|cuda`` to force.
    """
    forced = os.environ.get("CASTLE_PREPARE_DEVICE", "").strip().lower()
    try:
        import torch
    except Exception:  # noqa: BLE001 — torch missing -> CPU
        return "cpu"
    if forced == "cpu":
        return "cpu"
    if not torch.cuda.is_available():
        if forced == "cuda":
            notify("Prepare: CUDA requested but unavailable; using CPU.")
        return "cpu"
    from castle.core import runtime_env
    need = 2 * n_features * n_features * 4 + (3 << 30)  # ~2·Gram(f32) + 3 GiB slack
    if forced == "cuda":
        idx = runtime_env.idlest_gpu()
        return f"cuda:{idx}" if idx is not None else "cuda"
    idx = runtime_env.idlest_gpu(min_free_bytes=need)
    if idx is not None:
        return f"cuda:{idx}"
    # No GPU has enough room — report the largest free amount, then fall to CPU.
    try:
        largest_free = max(
            (int(cast(int, d["free_bytes"])) for d in runtime_env.gpu_info()),
            default=0,
        )
    except Exception:  # noqa: BLE001
        largest_free = 0
    if largest_free:
        notify(
            f"Prepare: largest GPU has {largest_free / 1e9:.1f} GB free "
            f"(< ~{need / 1e9:.1f} GB needed for {n_features}-d PCA); using CPU. "
            f"Stop other GPU jobs (or set CASTLE_PREPARE_DEVICE=cuda) to force GPU."
        )
    return "cpu"


# --------------------------------------------------------------------------- #
# The pipeline                                                                #
# --------------------------------------------------------------------------- #
@dataclass
class PreparedData:
    """Loaded prepare cache."""

    reduced: FloatArr            # (N_dp, width) float32 memmap (or array)
    index_map: FrameIndexMap
    meta: Dict[str, Any]
    # PCA basis (present only when pca was on AND the cache was built after the
    # basis-persistence change); needed to bundle the transform into a transfer
    # model. None for normalize/decimate-only caches or older caches.
    pca_components: Optional[FloatArr] = None  # (K_full, n_features)
    pca_mean: Optional[FloatArr] = None        # (n_features,)

    @property
    def width(self) -> int:
        return int(self.reduced.shape[1])


_BLOCK_ROWS = 8192  # rows per processing block; the float32 upcast + L2 happen here


def _load_raw_decidx(
    s: SourceSpec, *, downsample: bool, target_fps_cap: float
) -> Tuple[npt.NDArray[Any], IntArr, int]:
    """Load the full latent at its NATIVE dtype + compute the decimation indices.

    Crucially does NOT materialise a separate decimated copy: returning the raw
    array (in its stored precision, e.g. float16) plus ``dec_idx`` lets the
    consumer slice ``raw[dec_idx[i:i+W]]`` one small block at a time. That keeps
    peak RAM per in-flight file at ONE raw array (~7 GB) instead of raw + a
    decimated copy (~10 GB) — the coexistence that previously blew the budget
    when two loaders ran at once.

    Loads with ``fix_nonfinite=False`` (skips the Inf->NaN rewrite that costs
    ~3x RAM for files that contain non-finite values — which is all of them).
    Non-finite rows (+/-Inf or NaN) are caught downstream by ``torch.isfinite``
    in :func:`_to_device_l2` and written to the cache as NaN, so the result is
    identical to the eager conversion.
    """
    if s.req_scales:
        arr = _load_scale_combined_latent(s)  # column-combine selected scales
    else:
        arr = load_latent_safe(s.npz_path, fix_nonfinite=False)  # native dtype; ~1x RAM
    n_orig = int(arr.shape[0])
    tgt = effective_target_fps(s.raw_fps, target_fps_cap) if downsample else None
    if tgt is not None and s.raw_fps is not None and s.raw_fps > tgt:
        dec_idx = decimate_indices(n_orig, s.raw_fps, tgt)
    else:
        dec_idx = np.arange(n_orig, dtype=np.int64)
    return arr, dec_idx, n_orig


def _prep_block(block: npt.ArrayLike, normalize: str) -> Tuple[FloatArr, npt.NDArray[np.bool_]]:
    """Upcast one native-dtype row block to float32 (+ optional L2); flag finite rows.

    The ONLY place a float32 copy is made, and it is at most ``_BLOCK_ROWS`` wide,
    so peak RAM stays ~(native array + one block) rather than a whole float32
    duplicate. NaN rows pass through as NaN and report as non-finite.
    """
    b = np.asarray(block, dtype=np.float32)
    if normalize == "l2":
        b = l2_normalize_rows(b)  # NaN rows stay NaN; finite rows eps-guarded
    finite = np.isfinite(b).all(axis=1)
    return b, finite


def _source_geometry(s: SourceSpec) -> Tuple[int, int]:
    """``(n_orig_frames, n_features)`` for a source, cheaply from the npy header.

    Reads the exact shape from the ``.npy`` header (see :func:`_peek_header`)
    without decompressing the multi-GB array. Preferred over the metadata sidecar
    because the sidecar's recorded ``dtype`` can be stale (it says float32 while
    the array is stored float16); the header is authoritative.
    """
    if s.req_scales:
        # Combined width = C · Σ(requested s²), with C derived from one file.
        req = sorted(s.req_scales)
        n0 = base_c = 0
        for path, file_scales in s.scale_files or []:
            if any(sc in file_scales for sc in req):
                rows, nf, _isz = _peek_header(path)
                units = sum(s2 * s2 for s2 in sorted(int(x) for x in file_scales))
                if units and nf % units == 0:
                    n0, base_c = rows, nf // units
                    break
        return n0, base_c * sum(s2 * s2 for s2 in req)
    n0, nf, _itemsize = _peek_header(s.npz_path)
    return n0, nf


def run_prepare(
    out_dir: str,
    sources: Sequence[SourceSpec],
    *,
    downsample: bool = True,
    target_fps_cap: float = 60.0,
    normalize: str = "l2",
    pca: bool = True,
    K: int = 1024,
    fit_fraction: float = 1.0,
    model_name: str = "",
    seed: int = 0,
    avail_ram_bytes: Optional[int] = None,
    notify: Callable[[str], None] = logger.info,
    progress_cb: Optional[Callable[[int, int, int, int], None]] = None,
    should_cancel: Callable[[], bool] = lambda: False,
) -> Dict[str, Any]:
    """Run the (toggleable) Prepare pipeline and write the cache into ``out_dir``.

    Writes ``reduced.dat`` (memmap), ``frame_index_map.npz`` and ``meta.json``.
    Returns the meta dict. Caller (service) owns the atomic dir swap + filelock.

    NaN rows (tracking loss) are excluded from the PCA fit and pass through
    transform as NaN so downstream marks them ``cluster = -1``.

    ``progress_cb(frames_done, total_frames, steps_done, total_steps)`` is called
    after each source in each pass (total = n_dp x passes / n_sources x passes);
    ``should_cancel()`` is polled per source and per block — when it turns True a
    :class:`BuildCancelled` is raised so the caller can clean up the partial cache.
    """
    if normalize not in ("l2", "none"):
        raise ValueError(f"normalize must be 'l2' or 'none', got {normalize!r}")
    if not sources:
        raise ValueError("run_prepare: no sources selected.")
    os.makedirs(out_dir, exist_ok=True)

    # --- Geometry pass: per-video decimated counts + the index map ----------
    # Shapes come from each npz's .npy header, so NO latent array is decompressed
    # here (one fewer full read per file than the old three-pass design).
    names: List[str] = []
    offsets = [0]
    orig_idx_parts: List[IntArr] = []
    raw_fps_list: List[float] = []
    n_orig_list: List[int] = []
    roi_list: List[int] = []
    n_features: Optional[int] = None
    for s in sources:
        n_orig, nf = _source_geometry(s)
        if n_features is None:
            n_features = nf
        tgt = effective_target_fps(s.raw_fps, target_fps_cap) if downsample else None
        dec_idx = decimate_indices(n_orig, s.raw_fps, tgt)
        names.append(s.video_name)
        offsets.append(offsets[-1] + len(dec_idx))
        orig_idx_parts.append(dec_idx.astype(np.int64))
        raw_fps_list.append(float(s.raw_fps))
        n_orig_list.append(int(n_orig))
        roi_list.append(int(s.roi))
    n_dp = offsets[-1]
    assert n_features is not None
    if n_dp == 0:
        # Every source decimated to zero rows (e.g. a 1-frame video downsampled
        # to a lower fps). An empty reduced.dat can't be memmapped on load, so
        # fail early with a clear message instead of writing a broken cache.
        raise ValueError(
            "run_prepare: no datapoints after decimation — all selected sources are "
            "empty or too short for the target fps. Disable downsampling or pick longer videos."
        )
    index_map = FrameIndexMap(
        video_names=names,
        dp_offsets=np.array(offsets, dtype=np.int64),
        orig_frame_idx=np.concatenate(orig_idx_parts) if orig_idx_parts else np.zeros(0, np.int64),
        raw_fps=np.array(raw_fps_list, dtype=np.float64),
        n_orig_frames=np.array(n_orig_list, dtype=np.int64),
        source_roi=np.array(roi_list, dtype=np.int64),
    )
    notify(f"Prepare: {len(sources)} videos -> {n_dp} datapoints x {n_features} features.")

    # Progress accounting: the fit and transform passes each scan all decimated
    # rows, so total "frames" = n_dp x (2 with PCA, 1 without) and total "steps"
    # = n_sources x passes. progress_cb(frames_done, total_frames, steps_done,
    # total_steps) lets the UI render the same frames/sources/bar/ETA as the
    # extract & pre-process tabs.
    dp_per_source = [int(offsets[i + 1] - offsets[i]) for i in range(len(sources))]
    pass_count = 2 if pca else 1
    total_frames_units = max(1, n_dp * pass_count)
    total_steps = max(1, len(sources) * pass_count)
    done_frames = 0
    done_steps = 0

    def _emit_progress() -> None:
        if progress_cb is not None:
            progress_cb(done_frames, total_frames_units, done_steps, total_steps)

    _emit_progress()  # show 0 / total immediately
    n_src = len(sources)

    # RAM-bounded prefetch depth: how many files a single loader thread may hold
    # resident (1 being consumed + the rest prefetched ahead) without risking OOM.
    max_concurrent = _max_concurrent_loads(sources, avail_ram_bytes)
    notify(f"Prepare: 1 loader thread, prefetch depth {max_concurrent}.")

    width = int(n_features)
    evr: List[float] = []
    n_components_kept = width
    rank_limited = False
    n_finite_fit = 0
    components: Optional[FloatArr] = None  # (n_components_kept, D) float32
    mean_vec: Optional[FloatArr] = None    # (D,) float32

    if pca:
        # Streaming PCA by accumulating the mean + Gram (XᵀX) over the (finite,
        # optionally sub-sampled) fit rows, then a single top-K eigendecomposition
        # of the covariance. The accumulation is one big matmul per block — GPU
        # BLAS (or multi-threaded CPU BLAS) instead of sklearn's serial-ish
        # per-batch SVD — and the eigh runs on-device. Mathematically equivalent
        # to centre-only, no-whiten PCA (components are eigenvectors of the
        # covariance, ordered by descending eigenvalue).
        import torch

        K_eff = int(min(K, n_features))
        device = _select_pca_device(n_features, notify)
        dev = torch.device(device)
        # GPUs are slow at float64; use float32 there. CPU float64 is cheap + safer.
        acc_dtype = torch.float32 if device.startswith("cuda") else torch.float64
        S1 = torch.zeros(n_features, dtype=acc_dtype, device=dev)
        S2 = torch.zeros((n_features, n_features), dtype=acc_dtype, device=dev)
        rng = np.random.default_rng(seed)

        notify(f"Prepare: fitting PCA on {device} (pass 1/2)…")
        for si, (s, raw, dec_idx) in enumerate(_parallel_decimated(
            sources, downsample=downsample, target_fps_cap=target_fps_cap,
            should_cancel=should_cancel, max_concurrent=max_concurrent,
        )):
            if should_cancel():
                raise BuildCancelled()
            notify(f"Prepare: PCA fit {si + 1}/{n_src} — {s.video_name}")
            for i in range(0, len(dec_idx), _BLOCK_ROWS):
                if should_cancel():
                    raise BuildCancelled()
                # Slice the decimated block out of the resident raw array (small
                # copy), then upcast + L2 on the device (not host numpy).
                block = raw[dec_idx[i:i + _BLOCK_ROWS]]
                xb, finite = _to_device_l2(block, normalize, dev, torch)
                rows = xb[finite]
                n_rows = int(rows.shape[0])
                if fit_fraction < 1.0 and n_rows > 0:
                    k = max(1, int(round(fit_fraction * n_rows)))
                    sel = np.sort(rng.choice(n_rows, size=k, replace=False))
                    rows = rows.index_select(0, torch.as_tensor(sel, device=dev))
                    n_rows = k
                if n_rows:
                    r = rows.to(acc_dtype)
                    S1 += r.sum(dim=0)
                    S2 += r.T @ r
                    n_finite_fit += n_rows
                    del r
                del xb, rows
            del raw, dec_idx
            done_frames += dp_per_source[si]
            done_steps += 1
            _emit_progress()

        # Need MORE finite fit rows than components: centering drops one degree of
        # freedom, so n rows give a rank-(n-1) covariance. <= K_eff would yield a
        # rank-deficient fit (spurious near-zero eigenvalues mislabelled as kept).
        if n_finite_fit <= K_eff:
            raise ValueError(
                f"Too few finite frames ({n_finite_fit}) to fit PCA with K={K_eff} "
                f"(need > {K_eff}). Lower K, raise fit_fraction, or select more / "
                f"less-decimated videos."
            )
        notify("Prepare: solving principal components (eigendecomposition)…")
        n_fit_f = float(n_finite_fit)
        mean_t = S1 / n_fit_f
        cov = (S2 - n_fit_f * torch.outer(mean_t, mean_t)) / (n_fit_f - 1.0)
        cov = 0.5 * (cov + cov.T)  # enforce exact symmetry before eigh
        # Free the Gram before eigh allocates its (sizable) workspace, and make
        # sure cov/mean are released even if eigh raises (else a failed build
        # leaks GPU memory in the long-running app).
        del S1, S2
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        try:
            evals, evecs = torch.linalg.eigh(cov)             # ascending eigenvalues
            order = torch.argsort(evals, descending=True)[:K_eff]
            comp_t = evecs[:, order].T.contiguous()           # (K_eff, D), desc. variance
            ev_top = torch.clamp(evals[order], min=0.0)
            total_var = float(torch.clamp(evals, min=0.0).sum().item())
            if total_var <= 0.0:
                notify("Prepare: WARNING — near-zero total variance (degenerate "
                       "covariance); explained-variance ratios will be ~0. Check inputs.")
                total_var = 1.0
            evr = [float(x) for x in (ev_top / total_var).cpu().numpy()]
            components = comp_t.to("cpu", dtype=torch.float32).numpy()
            mean_vec = mean_t.to("cpu", dtype=torch.float32).numpy()
            n_components_kept = int(components.shape[0])
            rank_limited = n_components_kept < K
            width = n_components_kept
            del evals, evecs, comp_t
        finally:
            del cov, mean_t
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

    # --- Transform / write pass --------------------------------------------
    # Write reduced.dat SEQUENTIALLY via a buffered file handle (rows are produced
    # in global order). A np.memmap(mode="w+") would instead accumulate dirty
    # anonymous-ish pages in RSS (~width*n_dp*4 = up to ~11.5 GB for 26 mice),
    # which tripped the OOM guard; a streamed write lands in reclaimable page
    # cache instead. load_prepare mmaps the identical bytes read-only.
    reduced_path = os.path.join(out_dir, PCA_FILENAME)
    notify(f"Prepare: writing reduced cache {n_dp}x{width} ({'pass 2/2' if pca else 'single pass'})...")
    comp_dev = None
    mean_dev = None
    if pca:
        assert components is not None and mean_vec is not None
        import torch
        comp_dev = torch.from_numpy(components).to(dev)   # (K, D) float32
        mean_dev = torch.from_numpy(mean_vec).to(dev)     # (D,)  float32
    rows_written = 0
    with open(reduced_path, "wb", buffering=8 * 1024 * 1024) as fout:
        for si, (s, raw, dec_idx) in enumerate(_parallel_decimated(
            sources, downsample=downsample, target_fps_cap=target_fps_cap,
            should_cancel=should_cancel, max_concurrent=max_concurrent,
        )):
            if should_cancel():
                raise BuildCancelled()
            notify(f"Prepare: transform+write {si + 1}/{n_src} — {s.video_name}")
            m = len(dec_idx)
            for i in range(0, m, _BLOCK_ROWS):
                if should_cancel():
                    raise BuildCancelled()
                block = raw[dec_idx[i:i + _BLOCK_ROWS]]  # decimated block out of resident raw
                if pca:
                    assert comp_dev is not None and mean_dev is not None
                    # Upcast + L2 + project on the device (host stays cheap).
                    xb, finite = _to_device_l2(block, normalize, dev, torch)
                    nb = int(xb.shape[0])
                    out = np.full((nb, width), np.nan, dtype=np.float32)
                    if bool(finite.any()):
                        proj = (xb[finite] - mean_dev) @ comp_dev.T
                        out[finite.cpu().numpy()] = proj.to("cpu", dtype=torch.float32).numpy()
                        del proj
                    del xb, finite
                else:
                    out, finite = _prep_block(block, normalize)
                    out[~finite] = np.nan  # non-finite rows -> NaN (loader no longer does this)
                fout.write(np.ascontiguousarray(out, dtype=np.float32).tobytes())
                rows_written += out.shape[0]
            del raw, dec_idx
            done_frames += dp_per_source[si]
            done_steps += 1
            _emit_progress()
    assert rows_written == n_dp, f"reduced.dat row count {rows_written} != n_dp {n_dp}"
    if pca:
        del comp_dev, mean_dev
        import torch
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    index_map.save(os.path.join(out_dir, INDEX_MAP_FILENAME))

    # Persist the PCA basis so a transfer model can reproduce the transform
    # (raw -> L2 -> centre -> PCA -> k') on a new project's raw latents.
    if pca and components is not None and mean_vec is not None:
        np.save(os.path.join(out_dir, PCA_COMPONENTS_FILENAME), np.asarray(components, dtype=np.float32))
        np.save(os.path.join(out_dir, PCA_MEAN_FILENAME), np.asarray(mean_vec, dtype=np.float32))

    meta = {
        "prepare_schema_version": PREPARE_SCHEMA_VERSION,
        "created_at": None,  # stamped by service (no clock in pure core)
        "model_name": model_name,
        "downsample": {"on": bool(downsample), "target_fps_cap": float(target_fps_cap)},
        "normalize": normalize,
        "pca": {
            "on": bool(pca),
            "K": int(K),
            "center": True,
            "whiten": False,
            "n_components_kept": int(n_components_kept),
            "rank_limited": bool(rank_limited),
            "n_finite_fit_rows": int(n_finite_fit),
            "explained_variance_ratio": evr,
        },
        "fit_fraction": float(fit_fraction),
        "decimation_method": "nearest_frame_resample",
        "width": int(width),
        "n_dp_total": int(n_dp),
        "n_features": int(n_features),
        "seed": int(seed),
        "sources": [
            {
                "key": s.key,
                "mtime": _round6(os.stat(s.npz_path).st_mtime) if os.path.exists(s.npz_path) else None,
                "size": os.stat(s.npz_path).st_size if os.path.exists(s.npz_path) else None,
                "roi": int(s.roi),
                "raw_fps": float(s.raw_fps),
                "video_name": s.video_name,
                "n_orig_frames": int(n),
                "n_decimated": int(index_map.dp_offsets[i + 1] - index_map.dp_offsets[i]),
            }
            for i, (s, n) in enumerate(zip(sources, n_orig_list))
        ],
    }
    with open(os.path.join(out_dir, META_FILENAME), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return meta


# --------------------------------------------------------------------------- #
# Load / staleness / k'                                                       #
# --------------------------------------------------------------------------- #
def load_meta(prepare_dir: str) -> Dict[str, Any]:
    with open(os.path.join(prepare_dir, META_FILENAME), encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)
        return data


def load_prepare(prepare_dir: str) -> PreparedData:
    """Load a prepare cache directory (mmaps the reduced array read-only)."""
    meta = load_meta(prepare_dir)
    n_dp, width = int(meta["n_dp_total"]), int(meta["width"])
    reduced = np.memmap(
        os.path.join(prepare_dir, PCA_FILENAME), dtype=np.float32, mode="r", shape=(n_dp, width)
    )
    index_map = FrameIndexMap.load(os.path.join(prepare_dir, INDEX_MAP_FILENAME))
    comp_path = os.path.join(prepare_dir, PCA_COMPONENTS_FILENAME)
    mean_path = os.path.join(prepare_dir, PCA_MEAN_FILENAME)
    pca_components = np.load(comp_path) if os.path.exists(comp_path) else None
    pca_mean = np.load(mean_path) if os.path.exists(mean_path) else None
    return PreparedData(reduced=reduced, index_map=index_map, meta=meta,
                        pca_components=pca_components, pca_mean=pca_mean)


def is_stale(prepare_dir: str, resolve_path: Callable[[str], Optional[str]]) -> bool:
    """True if any source latent's mtime/size differs from the cache's record.

    ``resolve_path(key)`` maps a recorded ``config['latent']`` key to its
    current physical ``.npz`` path (or None if gone). The service supplies it
    (it knows the project's latent dir), keeping this module config-free.
    """
    try:
        meta = load_meta(prepare_dir)
    except (OSError, ValueError):
        return True
    for src in meta.get("sources", []):
        path = resolve_path(src.get("key", ""))
        if not path or not os.path.exists(path):
            return True
        st = os.stat(path)
        if _round6(st.st_mtime) != src.get("mtime") or st.st_size != src.get("size"):
            return True
    return False


def variance_pct_to_fraction(pct: Optional[float], default: float = 0.95) -> float:
    """Map a user-entered explained-variance **percent** to a ``(0, 1]`` fraction.

    The Explore UI lets the user request a target explained variance (e.g. ``95``)
    instead of a raw PCA-dim count; this normalises that box value before it is
    fed to :func:`k_prime_for_variance`. Blank / non-positive / unparseable →
    *default* (95%). Clamped to ``(0, 1]`` (so ``100`` and over → ``1.0`` = full
    width). Accepts the int the Number box yields or a float.
    """
    try:
        p = float(pct)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    if not (p > 0.0):
        return default
    return min(p / 100.0, 1.0)


def k_prime_for_variance(meta: Dict[str, Any], frac: float = 0.95) -> int:
    """Smallest k whose cumulative explained variance reaches ``frac``.

    Falls back to the full width when PCA was off or evr is unavailable.
    """
    evr = meta.get("pca", {}).get("explained_variance_ratio") or []
    width = int(meta.get("width", 0))
    if not evr:
        return width
    cum = np.cumsum(np.asarray(evr, dtype=np.float64))
    k = int(np.searchsorted(cum, float(frac)) + 1)
    return int(min(max(1, k), len(evr)))
