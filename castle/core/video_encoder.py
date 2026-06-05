"""Shared H.264 encoder selection for CASTLE video outputs.

Prefers NVENC (GPU hardware encode) with a RUNTIME fallback to threaded libx264 —
a compile-time ``Codec`` probe only proves PyAV was built with NVENC, not that it
works here (driver mismatch, session limit, unsupported option). Used by both the
pre-process stabilised-video encode and the post-track mix-overlay encode.

Env:
  CASTLE_VIDEO_ENCODER ∈ {auto (default), nvenc, x264}
  CASTLE_PREPROCESS_ENCODER  honored as a fallback alias (back-compat)
"""

import os

from castle.core.logging_config import setup_logger

logger = setup_logger(__name__)

NVENC_OPTS = {"preset": "p5", "rc": "vbr", "cq": "19"}
X264_OPTS = {"crf": "18", "preset": "fast"}
_probe_cache = {}  # (w, h) -> bool : does NVENC actually encode at this size?


def _nvenc_ok(fps, w: int, h: int) -> bool:
    """RUNTIME check that h264_nvenc actually encodes AT THE REAL (w, h) — a
    compile-time Codec probe (or `add_stream` alone) is not enough: NVENC only
    fails at the first `encode()` (``avcodec_open2``), e.g. for frames below its
    minimum dimension or with an unsupported option. Encodes a couple of black
    frames at w×h to a throwaway file; cached per size."""
    key = (int(w), int(h))
    if key in _probe_cache:
        return _probe_cache[key]
    import av  # type: ignore
    import numpy as np
    import tempfile
    from castle.core import runtime_env
    # Probe file goes to node-local scratch, never a network FS (CephFS), and
    # uses mkstemp (mktemp is deprecated / racy).
    fd, out = tempfile.mkstemp(suffix=".mp4", dir=runtime_env.scratch_dir())
    os.close(fd)
    ok = False
    try:
        c = av.open(out, mode="w")
        s = c.add_stream("h264_nvenc", rate=int(fps) or 30)
        s.width = int(w); s.height = int(h); s.pix_fmt = "yuv420p"; s.options = NVENC_OPTS
        blk = av.VideoFrame.from_ndarray(np.zeros((int(h), int(w), 3), np.uint8), format="bgr24")
        for _ in range(2):
            for pkt in s.encode(blk):
                c.mux(pkt)
        for pkt in s.encode():
            c.mux(pkt)
        c.close()
        ok = True
        logger.info("video encoder: h264_nvenc OK at %dx%d (GPU)", w, h)
    except Exception as e:  # noqa: BLE001
        # Log the real reason (driver/ffmpeg mismatch, session limit, unsupported
        # option) AND that we are falling back to CPU encode, which is much slower
        # — users on cloud boxes with a new driver vs older ffmpeg often hit this
        # silently and wonder why pre-process is slow.
        logger.warning(
            "video encoder: h264_nvenc unusable at %dx%d → falling back to libx264 "
            "(CPU, slower). Reason: %s: %s. Set CASTLE_VIDEO_ENCODER=x264 to skip "
            "this probe.", w, h, type(e).__name__, e,
        )
    finally:
        try:
            if os.path.exists(out):
                os.remove(out)
        except OSError:
            pass
    _probe_cache[key] = ok
    return ok


def select_video_encoder(fps, w: int, h: int) -> str:
    """Codec to use at (w, h). ``CASTLE_VIDEO_ENCODER`` (or legacy
    ``CASTLE_PREPROCESS_ENCODER``) ∈ {auto,nvenc,x264}; both 'auto' and 'nvenc'
    validate NVENC at the real size and fall back to libx264 if it can't encode."""
    mode = (os.environ.get("CASTLE_VIDEO_ENCODER")
            or os.environ.get("CASTLE_PREPROCESS_ENCODER")
            or "auto").strip().lower()
    if mode in ("x264", "libx264", "cpu"):
        return "libx264"
    if mode in ("nvenc", "h264_nvenc", "gpu", "auto", ""):
        return "h264_nvenc" if _nvenc_ok(fps, w, h) else "libx264"
    return "libx264"


def open_encoder(out_path: str, fps, w: int, h: int):
    """Open an output container + H.264 stream, preferring NVENC (validated at the
    real (w, h)) with a fallback to (threaded) libx264. Returns
    ``(container, stream, codec_name)``. A final ``add_stream`` try/except retries
    libx264 should the chosen codec still fail to open."""
    import av  # type: ignore
    chosen = select_video_encoder(int(fps) if fps else 30, w, h)
    candidates = [chosen] if chosen == "libx264" else [chosen, "libx264"]
    last_exc = None
    for codec in candidates:
        container = None
        try:
            container = av.open(out_path, mode="w")
            stream = container.add_stream(codec, rate=int(fps) or 30)
            stream.width = int(w); stream.height = int(h); stream.pix_fmt = "yuv420p"
            if codec == "h264_nvenc":
                stream.options = dict(NVENC_OPTS)
            else:
                stream.options = dict(X264_OPTS)
                try:  # let libx264 use its own internal threads
                    stream.thread_type = "AUTO"
                    stream.codec_context.thread_count = 0
                except Exception:  # noqa: BLE001
                    pass
            logger.info("video encode: using %s (%dx%d @ %sfps)", codec, w, h, int(fps) or 30)
            return container, stream, codec
        except Exception as e:  # noqa: BLE001 — runtime NVENC failure → next candidate
            last_exc = e
            logger.warning("video encode: %s failed at open (%s)", codec, e)
            try:
                if container is not None:
                    container.close()
            except Exception:  # noqa: BLE001
                pass
            try:
                if os.path.exists(out_path):
                    os.remove(out_path)
            except OSError:
                pass
    raise RuntimeError(f"no usable video encoder (last error: {last_exc})")
