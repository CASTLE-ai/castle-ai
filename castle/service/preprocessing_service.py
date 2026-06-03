"""
castle/service/preprocessing_service.py
Service layer for Pre-process tab operations.

All functions take simple types and return dicts.  No gradio imports.
Session management is delegated to castle.core.preprocess_session.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from pathlib import Path
from typing import Callable, Dict, Iterator, Optional

logger = logging.getLogger(__name__)


def _threaded_iter(producer, maxsize: int = 16) -> Iterator:
    """Run a generator ``producer()`` on a background thread, yielding its items
    in order on the calling thread.

    This overlaps a CPU/I-O producer (e.g. video decode + warpAffine, which
    release the GIL) with whatever the caller does per item (e.g. the libx264
    encode that dominates KIT wall-clock). Items are delivered through a bounded
    FIFO queue, so ordering is preserved exactly; the caller drives the encode at
    its own pace with unchanged settings, keeping output byte-identical to the
    serial version. A producer exception is re-raised on the calling thread.
    """
    q: "queue.Queue" = queue.Queue(maxsize=maxsize)
    sentinel = object()
    state: dict = {}
    stop = threading.Event()

    def _put(item) -> bool:
        # Block on a full queue but stay responsive to an early consumer exit,
        # so we never deadlock if the consumer stops draining.
        while not stop.is_set():
            try:
                q.put(item, timeout=0.25)
                return True
            except queue.Full:
                continue
        return False

    def _run() -> None:
        try:
            for item in producer():
                if not _put(item):
                    return
        except BaseException as exc:  # surfaced to the consumer thread below
            state["exc"] = exc
        finally:
            _put(sentinel)

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    try:
        while True:
            item = q.get()
            if item is sentinel:
                break
            yield item
    finally:
        # On early exit (consumer raised / broke), signal + drain so a producer
        # blocked on a full queue can wake up and stop.
        stop.set()
        try:
            while True:
                q.get_nowait()
        except queue.Empty:
            pass
        worker.join(timeout=5.0)
    if "exc" in state:
        raise state["exc"]


def preprocess_stabilized_camera(
    storage_path: str,
    project_name: str,
    video_name: str,
    kit_params: dict,
    skip_existing: bool = True,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Dict[str, object]:
    """Run KIT stabilized-camera preprocessing and save output into a session directory.

    Parameters
    ----------
    storage_path : str
        Root storage directory.
    project_name : str
        Project name.
    video_name : str
        Source video filename.
    kit_params : dict
        Required keys: ``anterior_roi_id``, ``posterior_roi_id``.
        Optional: ``fc`` (0.25), ``order`` (2), ``margin`` (75),
        ``min_crop`` (300), ``output_size`` (592).
    skip_existing : bool
        Skip if ``stabilized.mp4`` already exists in the session dir.
    progress_callback : callable, optional
        Called as ``callback(fraction, description)`` where fraction ∈ [0, 1].

    Returns
    -------
    dict with keys:
        session_id : str
        session_name : str
        preprocessed_video_path : str  — absolute path to stabilized.mp4
        mask_path : str               — absolute path to mask_list.h5
        diagnostics : dict
        n_frames : int
    """
    from castle.core.stabilized_camera import (
        StabilizedCamera,
        extract_centroids_from_masks,
        extract_orientations_from_masks,
    )
    from castle.core.preprocess_session import (
        find_or_create_session,
        get_session_dir,
        add_video_to_session,
    )
    from castle.utils.video_io import ReadArray

    body_roi_id = int(kit_params["anterior_roi_id"])
    head_roi_id = int(kit_params["posterior_roi_id"])
    fc = float(kit_params.get("fc", 0.25))
    order = int(kit_params.get("order", 2))
    margin = int(kit_params.get("margin", 75))
    min_crop = int(kit_params.get("min_crop", 300))
    output_size = int(kit_params.get("output_size", 592))

    # Normalise params dict for session naming (use canonical keys)
    session_params = {
        "anterior_roi_id": body_roi_id,
        "posterior_roi_id": head_roi_id,
        "fc": fc,
        "order": order,
        "margin": margin,
        "min_crop": min_crop,
        "output_size": output_size,
    }
    session_id = find_or_create_session(storage_path, project_name, "KIT", session_params)
    from castle.core.preprocess_session import load_session_meta
    session_name = (load_session_meta(storage_path, project_name, session_id) or {}).get(
        "session_name", session_id
    )

    project_dir = Path(storage_path) / project_name
    session_dir = get_session_dir(storage_path, project_name, session_id)
    video_out_dir = session_dir / video_name
    video_out_dir.mkdir(parents=True, exist_ok=True)

    out_video_path = video_out_dir / "stabilized.mp4"
    out_mask_path = video_out_dir / "mask_list.h5"

    if skip_existing and out_video_path.exists() and out_mask_path.exists():
        logger.info(
            "preprocess_stabilized_camera: skipping %s (session %s already complete)",
            video_name, session_id,
        )
        return {
            "session_id": session_id,
            "session_name": session_name,
            "preprocessed_video_path": str(out_video_path),
            "mask_path": str(out_mask_path),
            "diagnostics": {},
            "n_frames": 0,
        }

    source_path = str(project_dir / "sources" / video_name)
    mask_h5_path = str(project_dir / "track" / video_name / "mask_list.h5")

    logger.info(
        "preprocess_stabilized_camera: project=%s video=%s session=%s",
        project_name, video_name, session_id,
    )

    with ReadArray(source_path) as reader:
        n_frames: int = len(reader)
        fps: float = reader.fps

    if progress_callback:
        progress_callback(0.0, "Extracting centroids…")

    _t_centroid_start = time.perf_counter()
    positions = extract_centroids_from_masks(mask_h5_path, body_roi_id, n_frames)

    if progress_callback:
        progress_callback(0.05, "Extracting orientations…")

    # Reuse the body centroids we just computed (avoids a second full H5 sweep
    # + connected-components pass over the body ROI inside extract_orientations).
    angles = extract_orientations_from_masks(
        mask_h5_path, body_roi_id, head_roi_id, n_frames, body_pos=positions,
    )
    _t_centroid = time.perf_counter() - _t_centroid_start

    if progress_callback:
        progress_callback(0.10, "Computing stabilised trajectory…")

    cam = StabilizedCamera(
        positions=positions,
        angles=angles,
        fps=fps,
        fc=fc,
        order=order,
        margin=margin,
        min_crop=min_crop,
        output_size=output_size,
    )

    if progress_callback:
        progress_callback(0.12, "Encoding stabilised video…")

    import h5py
    from castle.utils.h5_io import H5IO

    try:
        _t_encode_start = time.perf_counter()
        _encode_stabilized_video(
            video_path=source_path,
            cam=cam,
            out_path=str(out_video_path),
            fps=fps,
            n_frames=n_frames,
            output_size=output_size,
            progress_callback=progress_callback,
            progress_start=0.12,
            progress_end=0.80,
        )
        _t_encode = time.perf_counter() - _t_encode_start

        if progress_callback:
            progress_callback(0.80, "Saving transformed masks…")

        _t_mask_start = time.perf_counter()
        with h5py.File(mask_h5_path, "r") as f_in, H5IO(str(out_mask_path)) as h5_out:
            keys = sorted(f_in.keys(), key=lambda x: int(x) if x.isdigit() else 0)
            for i, key in enumerate(keys):
                if not key.isdigit():
                    continue
                frame_idx = int(key)
                orig_mask = f_in[key][:]
                transformed = cam.generate_mask(orig_mask, frame_idx)
                h5_out.write_mask(frame_idx, transformed)

                if progress_callback and i % 500 == 0:
                    frac = 0.80 + 0.18 * i / max(len(keys), 1)
                    progress_callback(frac, f"Mask {i}/{len(keys)}")
        _t_mask = time.perf_counter() - _t_mask_start

        logger.info(
            "KIT timing (%s, %d frames): centroids+orient=%.1fs, encode=%.1fs, "
            "mask-transform=%.1fs",
            video_name, n_frames, _t_centroid, _t_encode, _t_mask,
        )
    except BaseException:
        # Remove partial artifacts so skip_existing won't silently reuse corrupt output.
        for p in (out_video_path, out_mask_path):
            try:
                if p.exists():
                    p.unlink()
            except OSError as cleanup_exc:
                logger.warning(
                    "preprocess_stabilized_camera: failed to clean up %s: %s",
                    p, cleanup_exc,
                )
        raise

    add_video_to_session(storage_path, project_name, session_id, video_name)

    if progress_callback:
        progress_callback(1.0, "Done.")

    diagnostics = cam.get_diagnostics()
    logger.info(
        "preprocess_stabilized_camera: done session=%s hp_rms=%.2f px",
        session_id, diagnostics["hp_residual_rms"],
    )

    return {
        "session_id": session_id,
        "session_name": session_name,
        "preprocessed_video_path": str(out_video_path),
        "mask_path": str(out_mask_path),
        "diagnostics": diagnostics,
        "n_frames": n_frames,
    }


def preprocess_center_crop(
    storage_path: str,
    project_name: str,
    video_name: str,
    roi_id: int,
    crop_width: int,
    crop_height: int,
    skip_existing: bool = True,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Dict[str, object]:
    """Crop and centre video frames around a tracked ROI and save as a session.

    Produces ``cropped.mp4`` + ``mask_list.h5`` in the session directory so
    Extract Latent can use the processed video and aligned masks.

    Parameters
    ----------
    storage_path : str
        Root storage directory.
    project_name : str
        Project name.
    video_name : str
        Source video filename.
    roi_id : int
        ROI to centre on.
    crop_width : int
        Output frame width in pixels.
    crop_height : int
        Output frame height in pixels.
    skip_existing : bool
        Skip if artifacts already exist in the session directory.
    progress_callback : callable, optional
        Called as ``callback(fraction, description)``.

    Returns
    -------
    dict with keys: session_id, session_name, preprocessed_video_path,
        mask_path, n_frames.
    """
    import av  # type: ignore
    import h5py

    from castle.core.data import Preprocess
    from castle.core.preprocess_session import (
        find_or_create_session,
        get_session_dir,
        load_session_meta,
        add_video_to_session,
    )
    from castle.utils.h5_io import H5IO
    from castle.utils.video_io import ReadArray

    session_params = {
        "roi_id": int(roi_id),
        "crop_width": int(crop_width),
        "crop_height": int(crop_height),
    }
    session_id = find_or_create_session(
        storage_path, project_name, "CenterROI", session_params
    )
    session_name = (load_session_meta(storage_path, project_name, session_id) or {}).get(
        "session_name", session_id
    )

    project_dir = Path(storage_path) / project_name
    session_dir = get_session_dir(storage_path, project_name, session_id)
    video_out_dir = session_dir / video_name
    video_out_dir.mkdir(parents=True, exist_ok=True)

    out_video_path = video_out_dir / "cropped.mp4"
    out_mask_path = video_out_dir / "mask_list.h5"

    if skip_existing and out_video_path.exists() and out_mask_path.exists():
        logger.info(
            "preprocess_center_crop: skipping %s (session %s already complete)",
            video_name, session_id,
        )
        return {
            "session_id": session_id,
            "session_name": session_name,
            "preprocessed_video_path": str(out_video_path),
            "mask_path": str(out_mask_path),
            "n_frames": 0,
        }

    source_path = str(project_dir / "sources" / video_name)
    mask_h5_path = str(project_dir / "track" / video_name / "mask_list.h5")

    preprocess = Preprocess(
        center_roi_switch=True,
        center_roi_id=int(roi_id),
        center_roi_crop_width=int(crop_width),
        center_roi_crop_height=int(crop_height),
    )

    with ReadArray(source_path) as reader:
        n_frames: int = len(reader)
        fps: float = reader.fps

    logger.info(
        "preprocess_center_crop: project=%s video=%s session=%s n_frames=%d",
        project_name, video_name, session_id, n_frames,
    )

    in_container = av.open(source_path)
    in_stream = in_container.streams.video[0]
    out_container = av.open(str(out_video_path), mode="w")
    out_stream = out_container.add_stream("h264", rate=int(fps))
    out_stream.width = int(crop_width)
    out_stream.height = int(crop_height)
    out_stream.pix_fmt = "yuv420p"
    out_stream.options = {"crf": "18", "preset": "fast"}

    encode_failed = False
    try:
        with h5py.File(mask_h5_path, "r") as f_in, H5IO(str(out_mask_path)) as h5_out:
            def _produce():
                # Decode + read source mask + center-crop transform on a
                # background thread, overlapping the encode below. Frames are
                # yielded in order; skipped frames (missing mask / failed
                # transform) are simply not yielded — identical to the serial
                # path, so output stays byte-identical.
                for frame_idx, pkt_frame in enumerate(in_container.decode(in_stream)):
                    if frame_idx >= n_frames:
                        break
                    img_bgr = pkt_frame.to_ndarray(format="bgr24")
                    key = str(frame_idx)
                    orig_mask = f_in[key][:] if key in f_in else None
                    if orig_mask is None:
                        logger.warning(
                            "preprocess_center_crop: mask missing at frame %d — skipping",
                            frame_idx,
                        )
                        continue
                    try:
                        cropped_frame, cropped_mask = preprocess.transform(img_bgr, orig_mask)
                    except Exception:
                        logger.debug(
                            "preprocess_center_crop: transform failed at frame %d — skipping",
                            frame_idx, exc_info=True,
                        )
                        continue
                    yield frame_idx, cropped_frame, cropped_mask

            try:
                for frame_idx, cropped_frame, cropped_mask in _threaded_iter(_produce):
                    out_frame = av.VideoFrame.from_ndarray(cropped_frame, format="bgr24")
                    for packet in out_stream.encode(out_frame):
                        out_container.mux(packet)

                    h5_out.write_mask(frame_idx, cropped_mask)

                    if progress_callback and frame_idx % 30 == 0:
                        progress_callback(frame_idx / n_frames, f"Frame {frame_idx}/{n_frames}")

                for packet in out_stream.encode():
                    out_container.mux(packet)
            finally:
                out_container.close()
                in_container.close()
    except BaseException:
        encode_failed = True
        raise
    finally:
        if encode_failed:
            for p in (out_video_path, out_mask_path):
                try:
                    if p.exists():
                        p.unlink()
                except OSError as cleanup_exc:
                    logger.warning(
                        "preprocess_center_crop: failed to clean up %s: %s",
                        p, cleanup_exc,
                    )

    add_video_to_session(storage_path, project_name, session_id, video_name)

    if progress_callback:
        progress_callback(1.0, "Done.")

    logger.info("preprocess_center_crop: done session=%s", session_id)

    return {
        "session_id": session_id,
        "session_name": session_name,
        "preprocessed_video_path": str(out_video_path),
        "mask_path": str(out_mask_path),
        "n_frames": n_frames,
    }


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _encode_stabilized_video(
    video_path: str,
    cam: object,
    out_path: str,
    fps: float,
    n_frames: int,
    output_size: int,
    max_frames: Optional[int] = None,
    progress_callback: Optional[Callable] = None,
    progress_start: float = 0.0,
    progress_end: float = 1.0,
) -> None:
    """Encode KIT-stabilised frames to an H.264 MP4 via PyAV."""
    import av  # type: ignore

    limit = min(max_frames, n_frames) if max_frames is not None else n_frames

    in_container = av.open(video_path)
    in_stream = in_container.streams.video[0]
    out_container = av.open(out_path, mode="w")
    out_stream = out_container.add_stream("h264", rate=int(fps))
    out_stream.width = output_size
    out_stream.height = output_size
    out_stream.pix_fmt = "yuv420p"
    out_stream.options = {"crf": "18", "preset": "fast"}

    def _produce():
        # Decode + warpAffine on a background thread (both release the GIL),
        # overlapping the libx264 encode below. Frames are yielded strictly in
        # order, so the encoder sees the same sequence + settings as before →
        # byte-identical output.
        for i, pkt_frame in enumerate(in_container.decode(in_stream)):
            if i >= limit:
                break
            img_bgr = pkt_frame.to_ndarray(format="bgr24")
            yield i, cam.generate_frame(img_bgr, i)

    ok = False
    try:
        for i, result in _threaded_iter(_produce):
            out_frame = av.VideoFrame.from_ndarray(result, format="bgr24")
            for packet in out_stream.encode(out_frame):
                out_container.mux(packet)
            if progress_callback and i % 30 == 0:
                frac = progress_start + (progress_end - progress_start) * i / limit
                progress_callback(frac, f"Frame {i}/{limit}")
        for packet in out_stream.encode():
            out_container.mux(packet)
        ok = True
    finally:
        out_container.close()
        in_container.close()
        # Don't leave a truncated MP4 that a later skip_existing could treat as
        # complete (mirrors the center-crop path's partial-output cleanup).
        if not ok and os.path.exists(out_path):
            try:
                os.remove(out_path)
            except OSError:
                pass
