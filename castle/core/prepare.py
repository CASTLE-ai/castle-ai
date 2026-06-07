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
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

from castle.utils.latent_metadata import load_latent_metadata
from castle.utils.safe_load import load_latent_safe

FloatArr = npt.NDArray[np.float32]
IntArr = npt.NDArray[np.int64]
F64Arr = npt.NDArray[np.float64]

logger = logging.getLogger(__name__)

PREPARE_SCHEMA_VERSION = 1
_L2_EPS = 1e-8

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
    """One selected latent file, resolved by the service layer."""

    key: str          # config['latent'] logical key
    npz_path: str     # physical .npz path
    video_name: str   # source mp4 filename
    raw_fps: float
    roi: int          # source ROI id (for grid mask overlay)


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
        try:
            st = os.stat(s.npz_path)
            sig = f"{_round6(st.st_mtime)}|{st.st_size}"
        except OSError:
            sig = "missing|missing"
        parts.append(f"{s.key}|{sig}")
    payload = "\n".join(parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]


# --------------------------------------------------------------------------- #
# IncrementalPCA batch sizing                                                 #
# --------------------------------------------------------------------------- #
def _pca_batch_size(K: int, n_features: int, avail_bytes: Optional[int]) -> int:
    """Streaming batch size: >= K (sklearn requirement), RAM-bounded, capped."""
    floor = max(int(K), 1)
    if avail_bytes:
        by_ram = int(0.25 * avail_bytes / (n_features * 4))
    else:
        by_ram = 8192
    return int(min(max(floor, by_ram), 16384))


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


def _load_decimated(
    s: SourceSpec, *, downsample: bool, target_fps_cap: float
) -> Tuple[npt.NDArray[Any], IntArr, int]:
    """Load + decimate one source at its NATIVE dtype (NO float32 upcast).

    Keeps the bulk array in its stored precision (e.g. float16) so a whole
    project's latents are never doubled to float32 in RAM — the upcast happens
    per row-block in :func:`_prep_block`. A no-op decimation (raw_fps <= target)
    returns the array itself (no copy). ``Inf`` is already ``NaN`` and NaN rows
    are preserved.
    """
    arr = load_latent_safe(s.npz_path)  # (n_orig, F) native dtype, Inf already -> NaN
    n_orig = int(arr.shape[0])
    tgt = effective_target_fps(s.raw_fps, target_fps_cap) if downsample else None
    if tgt is not None and s.raw_fps is not None and s.raw_fps > tgt:
        dec_idx = decimate_indices(n_orig, s.raw_fps, tgt)
        dec = arr[dec_idx]            # native-dtype copy of the kept rows
    else:
        dec_idx = np.arange(n_orig, dtype=np.int64)
        dec = arr                     # no decimation -> no copy
    return dec, dec_idx, n_orig


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
    """``(n_orig_frames, n_features)`` for a source, cheaply from its metadata.

    Avoids decompressing the (multi-GB) latent array just to learn its shape
    (the old geometry pass did a full load per file). Falls back to a full load
    only for legacy npz that predate the metadata sidecar.
    """
    md = load_latent_metadata(s.npz_path)
    if md and md.get("n_frames") is not None and md.get("feature_dim") is not None:
        return int(md["n_frames"]), int(md["feature_dim"])
    arr = load_latent_safe(s.npz_path)
    n0, nf = int(arr.shape[0]), int(arr.shape[1])
    del arr
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
) -> Dict[str, Any]:
    """Run the (toggleable) Prepare pipeline and write the cache into ``out_dir``.

    Writes ``reduced.dat`` (memmap), ``frame_index_map.npz`` and ``meta.json``.
    Returns the meta dict. Caller (service) owns the atomic dir swap + filelock.

    NaN rows (tracking loss) are excluded from the PCA fit and pass through
    transform as NaN so downstream marks them ``cluster = -1``.
    """
    if normalize not in ("l2", "none"):
        raise ValueError(f"normalize must be 'l2' or 'none', got {normalize!r}")
    if not sources:
        raise ValueError("run_prepare: no sources selected.")
    os.makedirs(out_dir, exist_ok=True)

    # --- Geometry pass: per-video decimated counts + the index map ----------
    # Shapes come from the metadata sidecar, so NO latent array is decompressed
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
    index_map = FrameIndexMap(
        video_names=names,
        dp_offsets=np.array(offsets, dtype=np.int64),
        orig_frame_idx=np.concatenate(orig_idx_parts) if orig_idx_parts else np.zeros(0, np.int64),
        raw_fps=np.array(raw_fps_list, dtype=np.float64),
        n_orig_frames=np.array(n_orig_list, dtype=np.int64),
        source_roi=np.array(roi_list, dtype=np.int64),
    )
    notify(f"Prepare: {len(sources)} videos -> {n_dp} datapoints x {n_features} features.")

    width = int(n_features)
    evr: List[float] = []
    n_components_kept = width
    rank_limited = False
    n_finite_fit = 0
    ipca = None

    if pca:
        from sklearn.decomposition import IncrementalPCA

        K_eff = int(min(K, n_features))
        batch = _pca_batch_size(K_eff, n_features, avail_ram_bytes)
        ipca = IncrementalPCA(n_components=K_eff)
        rng = np.random.default_rng(seed)
        buf: List[FloatArr] = []
        buf_n = 0

        def _flush(force: bool = False) -> None:
            nonlocal buf, buf_n, n_finite_fit
            while buf_n >= batch or (force and buf_n >= K_eff):
                chunk = np.concatenate(buf, axis=0) if len(buf) > 1 else buf[0]
                take = chunk[:batch] if not force else chunk
                ipca.partial_fit(take)
                n_finite_fit += take.shape[0]
                rest = chunk[take.shape[0]:]
                buf = [rest] if rest.shape[0] else []
                buf_n = rest.shape[0]
                if force:
                    break

        notify("Prepare: fitting IncrementalPCA (pass 1/2)...")
        for s in sources:
            dec, _dec_idx, _n_orig = _load_decimated(
                s, downsample=downsample, target_fps_cap=target_fps_cap
            )
            for i in range(0, dec.shape[0], _BLOCK_ROWS):
                b, finite = _prep_block(dec[i:i + _BLOCK_ROWS], normalize)
                rows = b[finite]
                if fit_fraction < 1.0 and rows.shape[0] > 0:
                    k = max(1, int(round(fit_fraction * rows.shape[0])))
                    sel = rng.choice(rows.shape[0], size=k, replace=False)
                    rows = rows[np.sort(sel)]
                if rows.shape[0]:
                    buf.append(rows)
                    buf_n += rows.shape[0]
                    _flush(force=False)
            del dec
        _flush(force=True)

        if n_finite_fit < K_eff:
            raise ValueError(
                f"Too few finite frames ({n_finite_fit}) to fit PCA with K={K_eff}. "
                f"Lower K, raise fit_fraction, or select more / less-decimated videos."
            )
        n_components_kept = int(ipca.n_components_)
        rank_limited = n_components_kept < K
        evr = [float(x) for x in np.asarray(ipca.explained_variance_ratio_)]
        width = n_components_kept

    # --- Transform / write pass --------------------------------------------
    reduced_path = os.path.join(out_dir, PCA_FILENAME)
    reduced = np.memmap(reduced_path, dtype=np.float32, mode="w+", shape=(n_dp, width))
    notify(f"Prepare: writing reduced cache {n_dp}x{width} ({'pass 2/2' if pca else 'single pass'})...")
    cursor = 0
    for s in sources:
        dec, _dec_idx, _n_orig = _load_decimated(
            s, downsample=downsample, target_fps_cap=target_fps_cap
        )
        m = dec.shape[0]
        for i in range(0, m, _BLOCK_ROWS):
            b, finite = _prep_block(dec[i:i + _BLOCK_ROWS], normalize)
            nb = b.shape[0]
            if pca:
                assert ipca is not None  # set in the fit pass above when pca is on
                out = np.full((nb, width), np.nan, dtype=np.float32)
                if finite.any():
                    out[finite] = ipca.transform(b[finite])
                reduced[cursor + i:cursor + i + nb] = out
            else:
                reduced[cursor + i:cursor + i + nb] = b  # raw (decimated, maybe L2'd), NaN preserved
        cursor += m
        del dec
    reduced.flush()
    del reduced

    index_map.save(os.path.join(out_dir, INDEX_MAP_FILENAME))

    # Persist the PCA basis so a transfer model can reproduce the transform
    # (raw -> L2 -> centre -> PCA -> k') on a new project's raw latents.
    if pca and ipca is not None:
        np.save(os.path.join(out_dir, PCA_COMPONENTS_FILENAME),
                np.asarray(ipca.components_, dtype=np.float32))
        np.save(os.path.join(out_dir, PCA_MEAN_FILENAME),
                np.asarray(ipca.mean_, dtype=np.float32))

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
