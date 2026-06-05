"""
castle/ui/extract_ui.py
UI Layer for Extraction.
Delegates all logic to castle.service.extraction_service and castle.core.extractor.
"""

import logging
import os
import shutil
import threading
import time

import gradio as gr
from tqdm import tqdm

from castle.ui.progress_ui import status_md, init_cancel_event, request_cancel
from castle.utils.video_io import ReadArray
from castle.core.data import Preprocess
from castle.core.extractor import (
    extract_roi_latent_from_video,
    extract_roi_latent_from_video_2gpu,
    extract_roi_latent_from_video_auto,
    extract_roi_rotation_latent_from_video,
    clear_device_encoder_cache,
    ExtractionCancelled,
)
from castle.core.gpu_pool import (
    available_cuda_devices, run_on_device_pool, deterministic_ctx_if_enabled,
)
from castle.core.environment import get_num_workers

_EXTRACT_BTN_IDLE = "Extract"
_CANCEL_BTN_IDLE = "Cancel"
from castle.utils.video_manager import get_project_config
from castle.utils.h5_io import H5IO
from castle.ui.video_select import (
    build_video_selector, wire_video_selector, resolve_selected,
)

logger = logging.getLogger(__name__)


# ---------------------------
# Helpers
# ---------------------------

def handle_assertion_error(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (AssertionError, ValueError) as e:
            gr.Warning(f"Error: {e}")
            raise
        except Exception as e:
            gr.Warning(f"Unexpected Error: {e}")
            raise
    return wrapper


def _parse_session_id_from_display(session_display: str) -> str | None:
    """Extract session_id from a display string like 'KIT_a1_p2_fc0.25_sz592 | KIT | 2 videos'."""
    if not session_display or "(None" in session_display:
        return None
    try:
        session_name = session_display.split(" | ")[0].strip()
        from castle.core.preprocess_session import session_id_from_name
        return session_id_from_name(session_name)
    except Exception:
        return None


def _on_session_selected(
    storage_path: str, project_name: str, session_display: str
) -> tuple:
    """Return (session_status_text, era_kit_warning_update) for the selected session."""
    if not session_display or "(None" in session_display:
        return (
            gr.update(value="", visible=False),
            gr.update(visible=False, value=""),
        )

    try:
        session_name = session_display.split(" | ")[0].strip()
        from castle.core.preprocess_session import (
            session_id_from_name, load_session_meta, video_is_preprocessed
        )
        from castle.core.project import get_project_config as _gpc
        session_id = session_id_from_name(session_name)
        meta = load_session_meta(storage_path, project_name, session_id)
        if meta is None:
            return f"⚠️ Session '{session_name}' not found.", gr.update(visible=False)

        _, config = _gpc(storage_path, project_name)
        source_videos = sorted(config.get("source", []))

        per_video = []
        for vname in source_videos:
            ok = video_is_preprocessed(storage_path, project_name, session_id, vname)
            icon = "✅" if ok else "⚠️"
            per_video.append(f"{icon} {vname}")

        method = meta.get("method", "?")
        status = (
            f"Session: **{session_name}** | {method}\n" +
            "  ".join(per_video)
        )
        return status, gr.update(visible=False)
    except Exception as exc:
        return f"❌ Error reading session: {exc}", gr.update(visible=False)


def _on_era_kit_warning(
    session_display: str, era_enabled: bool
) -> gr.update:
    """Show ERA + KIT informational note when both are active."""
    if not era_enabled or not session_display or "(None" in session_display:
        return gr.update(visible=False, value="")
    try:
        method = session_display.split(" | ")[1].strip()
        if method == "KIT":
            return gr.update(
                visible=True,
                value=(
                    "ℹ️ KIT already aligns the body axis, so Eliminate Rotation "
                    "Asymmetry adds little here — but it does not affect correctness."
                ),
            )
    except (IndexError, AttributeError):
        pass
    return gr.update(visible=False, value="")


def _list_sessions_for_extract(storage_path: str, project_name: str) -> gr.update:
    """Return session selector choices: '(None — use raw source)' + sessions list."""
    choices = ["(None — use raw source)"]
    if storage_path and project_name:
        try:
            from castle.core.preprocess_session import list_sessions
            metas = list_sessions(storage_path, project_name)
            choices += [
                f"{m['session_name']} | {m['method']} | {len(m.get('videos', []))} videos"
                for m in metas
            ]
        except Exception as exc:
            logger.exception("Failed to list preprocess sessions for extract")
            gr.Warning(
                "Could not read preprocessing sessions (they may be corrupted) — "
                f"falling back to raw source. Details: {exc}"
            )
    return gr.update(choices=choices, value=choices[0])


def init_select_video_list(storage_path, project_name):
    """Populate the extract tab from current project state."""
    # Default: hide everything
    # Slot ordering MUST match all_ui_elements_to_control below.
    # NOTE: extract_crop_video_btn was removed from this list (2-D); it has no
    # handler wired and stayed visible=False permanently.  See comment near
    # its declaration.
    updates = []
    updates.extend([
        gr.update(visible=False),  # 0  session_selector
        gr.update(visible=False),  # 1  session_status
        gr.update(visible=False),  # 2  select_model
        gr.update(visible=False),  # 3  select_roi_id
        gr.update(visible=False),  # 4  batch_size
        gr.update(choices=[], value=[], visible=False),  # 5  video_select group
        gr.update(value=0, visible=False),  # 6  video_count
        gr.update(visible=False),  # 7  skip_existing
        gr.update(visible=False),  # 8  remove_background_switch
        gr.update(visible=False),  # 9  adv_accordion
        gr.update(visible=False),  # 10 extract_btn
        gr.update(visible=False),  # 11 latent_file_list
        gr.update(visible=False),  # 12 auto_batch_btn
        gr.update(value="", visible=False),  # 13 mem_warning
        gr.update(visible=False, interactive=False),  # 14 extract_cancel_btn
        gr.update(visible=False),  # 15 hdr_model
        gr.update(visible=False),  # 16 hdr_source
        gr.update(visible=False),  # 17 video_select btn_row
        [],                        # 18 video_select all_state (raw value)
        gr.update(visible=False),  # 19 video_select accordion (list container)
        gr.update(visible=False),  # 20 extract_status (live bar)
        gr.update(visible=False),  # 21 multi_gpu_toggle
    ])

    if not storage_path or not project_name:
        gr.Warning("No project selected. Please create or open a project in the 'Project' tab first.")
        return updates

    try:
        _, config = get_project_config(storage_path, project_name)
        video_list_from_config = config.get('source', [])
        choices = sorted(video_list_from_config)
        video_count_val = len(choices)

        if video_count_val > 0:
            # Session selector choices
            session_choices = ["(None — use raw source)"]
            try:
                from castle.core.preprocess_session import list_sessions
                metas = list_sessions(storage_path, project_name)
                session_choices += [
                    f"{m['session_name']} | {m['method']} | {len(m.get('videos', []))} videos"
                    for m in metas
                ]
            except Exception as e:
                logger.warning("Could not list preprocess sessions for %s: %s", project_name, e)

            updates[0] = gr.update(choices=session_choices, value=session_choices[0], visible=True)  # session_selector
            updates[1] = gr.update(value="", visible=False)  # session_status — hidden when using raw source
            updates[2] = gr.update(visible=True)   # select_model
            updates[3] = gr.update(visible=True)   # select_roi_id
            updates[4] = gr.update(visible=True)   # batch_size
            updates[5] = gr.update(choices=choices, value=choices, visible=True)  # video_select group (all checked)
            updates[6] = gr.update(value=video_count_val, visible=True)  # video_count
            updates[17] = gr.update(visible=True)  # video_select btn_row
            updates[18] = list(choices)            # video_select all_state
            updates[19] = gr.update(visible=True)  # video_select accordion
            updates[20] = gr.update(visible=True)  # extract_status
            updates[21] = gr.update(visible=True)  # multi_gpu_toggle
            updates[7] = gr.update(visible=True)   # skip_existing
            updates[8] = gr.update(visible=True)   # remove_background_switch
            updates[9] = gr.update(visible=True)   # adv_accordion
            updates[10] = gr.update(visible=True)  # extract_btn
            updates[11] = gr.update(visible=True)  # latent_file_list
            updates[12] = gr.update(visible=True)  # auto_batch_btn
            # mem_warning (13) stays hidden until reactive check triggers
            updates[14] = gr.update(visible=True, interactive=False)  # extract_cancel_btn
            updates[15] = gr.update(visible=True)  # hdr_model
            updates[16] = gr.update(visible=True)  # hdr_source
        else:
            gr.Warning(
                "No videos found in this project. Please add videos in the "
                "'Source' tab before extracting features."
            )
    except Exception as e:
        gr.Warning(
            f"Failed to load video list. Please check that the project is correctly "
            f"configured. Details: {e}"
        )

    return updates


# ---------------------------
# Main Action Handlers
# ---------------------------

# NOTE: no @handle_assertion_error here — this is a generator; the decorator would
# wrap the call and return the generator object (Gradio then sees 1 value, not the
# yielded tuples). Validation errors are surfaced via gr.Warning + a graceful final
# yield inside the body instead.
def ui_extract_roi_latent(
    storage_path: str,
    project_name: str,
    select_model: str,
    select_roi: str,
    selected_videos,
    batch_size: str,
    skip_existing: bool,
    remove_background_switch: bool = False,
    era_switch: bool = False,
    era_roi_id: int = 2,
    pooling_method: str = 'weighted_average',
    pooling_scales_list: list = None,
    feature_layers_str: str = '',
    latent_dtype: str = 'float32',
    use_multi_gpu: bool = False,
    session_display: str = "(None — use raw source)",
    cancel_event=None,
):
    """Extract latent features for the selected videos (generator → live UI).

    Mirrors the tracking/pre-process tabs: runs on a background thread, polls every
    ~0.5 s, and yields ``(log, extract_btn, cancel_btn, status_md)`` with a
    frame-granular unicode bar in a dedicated component. Multi-GPU (toggle): a
    video-level pool across GPUs for ≥2 videos, else frame-split for a single video.
    Cancel is batch-granular (the extractor aborts at its next batch).
    """
    # First yield: running state.
    yield (
        "",
        gr.update(interactive=False),
        gr.update(value=_CANCEL_BTN_IDLE, interactive=True),
        "🚀 Extracting latents…",
    )

    messages: list = []
    video_frac: dict = {}
    video_total_frames: dict = {}
    frac_lock = threading.Lock()
    completed = {"n": 0}
    success = {"n": 0}
    failed: list = []

    def _final(status_text):
        return (
            "\n".join(messages),
            gr.update(value=_EXTRACT_BTN_IDLE, interactive=True),
            gr.update(value=_CANCEL_BTN_IDLE, interactive=False),
            status_text,
        )

    preprocess_args = Preprocess(remove_background_switch=bool(remove_background_switch))
    parsed_scales = [int(s) for s in pooling_scales_list] if pooling_scales_list else [1, 2, 4]
    parsed_layers = None
    if feature_layers_str and feature_layers_str.strip():
        try:
            parsed_layers = [int(x.strip()) for x in feature_layers_str.split(',') if x.strip()]
        except ValueError:
            gr.Warning(f"Invalid feature layers format: '{feature_layers_str}'.")
            yield _final("⛔ Invalid feature layers format.")
            return

    session_id = _parse_session_id_from_display(session_display)
    try:
        select_roi = int(select_roi)
    except (TypeError, ValueError):
        gr.Warning(f"ROI ID must be an integer (got {select_roi!r}).")
        yield _final("⛔ ROI ID must be an integer.")
        return

    _, config = get_project_config(storage_path, project_name)
    video_list = resolve_selected(config.get('source', []), selected_videos)
    if not video_list:
        gr.Warning("No videos selected — tick at least one video to extract.")
        yield _final("No videos selected.")
        return

    if session_id:
        messages.append(f"Using pre-process session: {session_display.split(' | ')[0]}")

    # --- Pre-flight: drop videos that already have a latent file ---
    videos_to_process = []
    messages.append(f"Starting pre-flight check for {len(video_list)} videos...")
    for video_name in video_list:
        tags = []
        if preprocess_args.remove_background_switch:
            tags.append("rmbg")
        if pooling_method == 'multiscale' and parsed_scales:
            tags.append("spp" + "x".join(str(s) for s in sorted(parsed_scales)))
        if parsed_layers:
            tags.append("L" + "x".join(str(lay) for lay in sorted(parsed_layers)))
        suffix = "_".join([select_model] + tags)
        pre_tag = f"_pre-{session_id}" if session_id else ""
        latent_filename = f'{os.path.splitext(video_name)[0]}_ROI_{select_roi}_{suffix}{pre_tag}.npz'
        output_path = os.path.join(storage_path, project_name, 'latent', select_model, latent_filename)
        if skip_existing and os.path.exists(output_path):
            messages.append(f"  ⏩ Skipping existing: {video_name}")
            continue
        videos_to_process.append(video_name)

    if not videos_to_process:
        messages.append("\n✅ All videos already have latent files. Nothing to extract.")
        yield _final("✅ Nothing to extract.")
        return

    total = len(videos_to_process)
    messages.append(f"\nFound {total} new videos to process.")

    def _resolve_paths(vname):
        """(source_path, mask_path) for a video — preprocessed paths if a session
        is selected, else (None, None) so the extractor uses the raw source."""
        if session_id:
            from castle.core.preprocess_session import get_preprocessed_paths
            vpath, mpath = get_preprocessed_paths(storage_path, project_name, session_id, vname)
            return str(vpath), str(mpath)
        return None, None

    # Pre-count frames for the frame-granular bar (best-effort; 0 → video-granular).
    total_frames = 0
    for v in videos_to_process:
        try:
            svp, _ = _resolve_paths(v)
            src = svp or os.path.join(storage_path, project_name, 'sources', v)
            with ReadArray(src) as _r:
                n = len(_r)
        except Exception:  # noqa: BLE001
            n = 0
        video_total_frames[v] = n
        total_frames += n

    def _make_cb(vname):
        # Only writes the number under the lock — formatting happens in the poll thread.
        def _cb(frac, desc=None, v=vname):
            with frac_lock:
                video_frac[v] = max(0.0, min(1.0, float(frac))) if frac is not None else 0.0
        return _cb

    # Resolve the GPU set from the toggle (explicit — independent of CASTLE_MULTI_GPU).
    device_ids = available_cuda_devices() if use_multi_gpu else []
    use_pool = len(device_ids) >= 2 and total >= 2

    # DISK guard: each video's latents now stream to a temp memmap on disk (RAM is
    # ~one batch), and the pool runs n_gpu concurrently → n_gpu temp memmaps live at
    # once on the latent dir's filesystem. If disk won't hold them, drop to single
    # GPU rather than fill the disk. (Static dim from output_dim_for — no probe.)
    if use_pool:
        from castle.core.models import output_dim_for
        dim = output_dim_for(select_model, pooling_method,
                             parsed_scales if pooling_method == 'multiscale' else None,
                             parsed_layers)
        max_frames = max(video_total_frames.values() or [0])
        per_video_bytes = max_frames * dim * 4  # fp32 memmap on disk
        try:
            free_disk = shutil.disk_usage(os.path.join(storage_path, project_name, 'latent')
                                          if os.path.isdir(os.path.join(storage_path, project_name))
                                          else storage_path).free
        except Exception:  # noqa: BLE001
            free_disk = None
        if free_disk is not None and per_video_bytes * len(device_ids) > 0.85 * free_disk:
            messages.append(
                f"  ⚠️ Low free disk (~{free_disk / 1e9:.1f} GB) for {len(device_ids)}-GPU "
                f"extraction (~{per_video_bytes / 1e9:.1f} GB temp/video) — falling back to single GPU.")
            use_pool = False
            device_ids = device_ids[:1]

    worker_error: dict = {}
    done = threading.Event()

    def _extract(vname, device, num_workers=None):
        svp, mpo = _resolve_paths(vname)
        return extract_roi_latent_from_video(
            storage_path=storage_path, project_name=project_name, video_name=vname,
            roi_id=select_roi, model_name=select_model, batch_size=int(batch_size),
            preprocess_config=preprocess_args, skip_existing=skip_existing,
            progress_callback=_make_cb(vname),
            pooling_method=pooling_method,
            pooling_scales=parsed_scales if pooling_method == 'multiscale' else None,
            feature_layers=parsed_layers,
            source_video_path=svp, mask_path_override=mpo, session_id=session_id,
            device=device, num_workers=num_workers, latent_dtype=latent_dtype,
            cancel_event=cancel_event,
        )

    def _finish_video(vname, res):
        completed["n"] += 1
        with frac_lock:
            video_frac[vname] = 1.0
        if isinstance(res, BaseException):
            if isinstance(res, ExtractionCancelled):
                messages.append(f"  🛑 Cancelled: {vname}")
            else:
                failed.append(vname)
                messages.append(f"  ❌ {vname}: {res}")
        elif res:
            success["n"] += 1
            messages.append(f"  ✅ {vname}: {os.path.basename(res)}")
        else:
            failed.append(vname)
            messages.append(f"  ⚠️ {vname}: no path returned")

    def _run_era():
        import numpy as _np
        _era_roi_id = int(era_roi_id)
        _first = videos_to_process[0]
        _mp = os.path.join(storage_path, project_name, 'track', _first, 'mask_list.h5')
        try:
            with H5IO(_mp, read_only=True) as _h5:
                _gm = _h5.read_mask(0)
            _ids = set(_np.unique(_gm[_gm > 0]).tolist()) if _gm is not None else set()
            if _era_roi_id not in _ids or len(_ids) < 2:
                messages.append(f"\n⚠️ ERA skipped: reference ROI (ID {_era_roi_id}) not found "
                                f"(detected: {sorted(_ids) or 'none'}).")
                return
        except Exception as _e:  # noqa: BLE001
            messages.append(f"\n⚠️ Could not validate ERA ROI ({_e}); proceeding.")
        messages.append("\n--- Eliminate Rotation Asymmetry ---")
        rs = 0
        for vname in videos_to_process:
            if cancel_event is not None and cancel_event.is_set():
                break
            try:
                svp, mpo = _resolve_paths(vname)
                messages.append(f"ERA: {vname}...")
                rpath = extract_roi_rotation_latent_from_video(
                    storage_path=storage_path, project_name=project_name, video_name=vname,
                    roi_id=select_roi, model_name=select_model, batch_size=int(batch_size),
                    preprocess_config=preprocess_args, skip_existing=skip_existing,
                    progress_callback=None,  # ERA frames aren't in the bar budget
                    source_video_path=svp, mask_path_override=mpo, session_id=session_id,
                )
                if rpath:
                    rs += 1
                    messages.append(f"  ✅ {os.path.basename(rpath)}")
            except Exception as e:  # noqa: BLE001
                messages.append(f"  ❌ ERA error for {vname}: {e}")
        messages.append(f"\n🎉 ERA complete: {rs}/{total} succeeded.")

    def _run():
        try:
            if use_pool:
                n_gpu = len(device_ids)
                per_gpu = max(1, get_num_workers('extraction') // n_gpu)
                messages.append(f"🖥️ Multi-GPU: one video per GPU across {n_gpu} GPUs ({per_gpu} workers/GPU).")

                def _worker(vname, device):
                    return _extract(vname, device, num_workers=per_gpu)

                with deterministic_ctx_if_enabled():
                    pool_out = run_on_device_pool(
                        videos_to_process, _worker, device_ids,
                        on_done=lambda v, r: _finish_video(v, r), cancel_event=cancel_event,
                    )
                # on_done already accounted every result; (pool_out kept for parity)
                _ = pool_out
                try:
                    clear_device_encoder_cache()
                except Exception:  # noqa: BLE001
                    pass
            elif len(device_ids) >= 2 and total == 1:
                vname = videos_to_process[0]
                messages.append(f"🖥️ Multi-GPU: frame-split {vname} across {len(device_ids)} GPUs.")
                svp, mpo = _resolve_paths(vname)
                try:
                    rpath = extract_roi_latent_from_video_2gpu(
                        storage_path=storage_path, project_name=project_name, video_name=vname,
                        roi_id=select_roi, model_name=select_model, batch_size=int(batch_size),
                        preprocess_config=preprocess_args, skip_existing=skip_existing,
                        progress_callback=_make_cb(vname),
                        pooling_method=pooling_method,
                        pooling_scales=parsed_scales if pooling_method == 'multiscale' else None,
                        feature_layers=parsed_layers,
                        source_video_path=svp, mask_path_override=mpo, session_id=session_id,
                        device_ids=device_ids, latent_dtype=latent_dtype, cancel_event=cancel_event,
                    )
                    _finish_video(vname, rpath)
                except BaseException as e:  # noqa: BLE001
                    _finish_video(vname, e)
            else:
                for vname in videos_to_process:
                    if cancel_event is not None and cancel_event.is_set():
                        break
                    messages.append(f"\nProcessing {vname}...")
                    try:
                        svp, mpo = _resolve_paths(vname)
                        path = extract_roi_latent_from_video_auto(
                            storage_path=storage_path, project_name=project_name, video_name=vname,
                            roi_id=select_roi, model_name=select_model, batch_size=int(batch_size),
                            preprocess_config=preprocess_args, skip_existing=skip_existing,
                            progress_callback=_make_cb(vname),
                            pooling_method=pooling_method,
                            pooling_scales=parsed_scales if pooling_method == 'multiscale' else None,
                            feature_layers=parsed_layers,
                            source_video_path=svp, mask_path_override=mpo, session_id=session_id,
                            latent_dtype=latent_dtype, cancel_event=cancel_event,
                        )
                        _finish_video(vname, path)
                    except ExtractionCancelled as e:
                        _finish_video(vname, e)
                        break
                    except BaseException as e:  # noqa: BLE001
                        _finish_video(vname, e)

            messages.append(f"\n🎉 Extraction done: {success['n']}/{total} videos.")
            if failed:
                messages.append(f"⚠️ Failed: {', '.join(failed)}")

            if era_switch and not (cancel_event is not None and cancel_event.is_set()):
                _run_era()
        except Exception as e:  # noqa: BLE001
            logging.getLogger(__name__).exception("Extraction crashed")
            worker_error["e"] = e
        finally:
            done.set()

    worker = threading.Thread(target=_run, daemon=True, name="extract")
    t0 = time.time()
    worker.start()

    last_msg_count = 0
    try:
        while not done.wait(timeout=0.5):
            cancelling = cancel_event is not None and cancel_event.is_set()
            with frac_lock:
                frames_done = sum(video_frac.get(v, 0.0) * video_total_frames.get(v, 0)
                                  for v in video_total_frames)
            status = status_md(frames_done, total_frames, completed["n"], total, t0, cancelling)
            if len(messages) != last_msg_count:
                last_msg_count = len(messages)
                log_update = "\n".join(messages[-14:])
            else:
                log_update = gr.update()
            yield (log_update, gr.update(), gr.update(), status)
    except GeneratorExit:
        if cancel_event is not None:
            cancel_event.set()
        raise

    worker.join()
    if "e" in worker_error:
        messages.append(f"\n❌ Extraction crashed: {worker_error['e']}")
    cancelled = cancel_event is not None and cancel_event.is_set()
    if failed:
        gr.Warning(f"{len(failed)} video(s) failed during extraction. See the log.")
    final_status = ("🛑 Cancelled." if cancelled
                    else f"✅ Done — {success['n']}/{total} videos.")
    yield _final(final_status)


# ---------------------------
# UI Construction
# ---------------------------

def create_extract_ui(storage_path, project_name, extract_tab):
    ui = {}

    # ------------------------------------------------------------------
    # Session selector (top of tab)
    # ------------------------------------------------------------------
    ui["session_selector"] = gr.Dropdown(
        label="Pre-process Session",
        choices=["(None — use raw source)"],
        value="(None — use raw source)",
        visible=False,
    )
    ui["session_status"] = gr.Markdown(value="", visible=False)

    # ── Model ──────────────────────────────────────────────────────────
    ui['hdr_model'] = gr.Markdown("### Model", visible=False)
    with gr.Row():
        ui['select_model'] = gr.Dropdown(
            label="Visual Model",
            choices=["dinov2_vitb14_reg4_pretrain", "dinov3_vitb16", "dinov3_vitl16"],
            value="dinov3_vitb16",
            visible=False,
            scale=3,
        )
        ui['select_roi_id'] = gr.Number(
            label="ROI ID",
            value=1,
            precision=0,
            visible=False,
            scale=1,
        )
    with gr.Row(equal_height=True):
        ui['batch_size'] = gr.Textbox(
            label="Batch Size",
            value="32",
            visible=False,
            scale=3,
        )
        ui['auto_batch_btn'] = gr.Button("Auto Batch Size", size="sm", visible=False, scale=1)

    # ── Source & Options ───────────────────────────────────────────────
    ui['hdr_source'] = gr.Markdown("### Source & Options", visible=False)
    ui['video_count'] = gr.Number(
        label="Videos in project",
        value=0,
        interactive=False,
        visible=False,
    )
    # Per-video selection (split a project across machines).
    ui['video_select'] = build_video_selector(label="Videos to extract")
    with gr.Row():
        ui['skip_existing'] = gr.Checkbox(
            label="Skip existing files",
            value=True,
            visible=False,
        )
        ui['remove_background_switch'] = gr.Checkbox(
            label="Remove Background",
            value=False,
            visible=False,
        )

    # Advanced — full width so opening it expands downward (no side void).
    with gr.Accordion("Advanced Extraction Options", open=False, visible=False) as adv_accordion:
        ui['eliminate_rotation_asymmetry'] = gr.Checkbox(
            label="Eliminate Rotation Asymmetry",
            value=False,
            info=(
                "Extract 7-angle rotation latent after the main extraction. "
                "Averages rotation to reduce orientation bias in the latent space."
            ),
        )
        ui['era_roi_id'] = gr.Number(
            label="Reference ROI ID",
            value=2,
            info="ROI ID used to determine body orientation for rotation alignment.",
        )
        ui['era_kit_warning'] = gr.Markdown(value="", visible=False)
        ui['pooling_method'] = gr.Radio(
            choices=['weighted_average', 'multiscale'],
            value='weighted_average',
            label='Pooling Method',
            info='weighted_average: single vector; multiscale: spatial pyramid pooling.',
        )
        ui['pooling_scales'] = gr.CheckboxGroup(
            choices=['1', '2', '4', '8'],
            value=['1', '2', '4'],
            label='Multiscale Grid Sizes',
            info='Only used when Pooling Method is multiscale.',
        )
        ui['feature_layers'] = gr.Textbox(
            value='',
            label='Feature Layers',
            info='Comma-separated layer indices to concatenate (e.g. "3,7,11"). Empty = last layer only.',
            placeholder='Leave empty for default (last layer)',
        )
        ui['latent_precision'] = gr.Radio(
            choices=['float32', 'float16'],
            value='float32',
            label='Latent Precision',
            info='float16 halves the .npz size with negligible effect on clustering; '
                 'float32 is full precision. The Behavior Microscope reads either.',
        )
    ui['adv_accordion'] = adv_accordion

    ui['mem_warning'] = gr.HTML(value="", visible=False)
    _avail_gpus = available_cuda_devices()
    ui['multi_gpu_toggle'] = gr.Checkbox(
        label="Use multiple GPUs",
        value=bool(_avail_gpus),
        interactive=bool(_avail_gpus),
        info="Spread videos across GPUs (one video per GPU); a single video is frame-split. "
             "Close other GPU/heavy apps before a big batch.",
        visible=False,
    )
    with gr.Row():
        ui['extract_btn'] = gr.Button("Extract", visible=False, variant="primary", scale=4)
        ui['extract_cancel_btn'] = gr.Button("Cancel", visible=False, interactive=False, scale=1)
        # TODO(2-D): implement extract_crop_video handler before re-enabling this
        # button.  Kept declared and permanently invisible so any external
        # reference (CLI, MCP) does not break.  Removed from
        # all_ui_elements_to_control update list below.
        ui['extract_crop_video_btn'] = gr.Button(
            "Extract Crop Video", visible=False, interactive=False
        )
    # Live frame-granular bar in its own component (never overlaps the log).
    ui['extract_status'] = gr.Markdown(value="", visible=False)
    ui['cancel_event'] = gr.State(None)
    ui['latent_file_list'] = gr.Textbox(
        label="Log Output",
        visible=False,
        lines=10,
        max_lines=20,
    )

    # Elements to control visibility on tab select.
    # NOTE: extract_crop_video_btn intentionally NOT in this list (2-D); it has
    # no handler and stays permanently hidden.  Indices below match the updates
    # produced by init_select_video_list.
    all_ui_elements_to_control = [
        ui['session_selector'],          # 0
        ui['session_status'],            # 1
        ui['select_model'],              # 2
        ui['select_roi_id'],             # 3
        ui['batch_size'],                # 4
        ui['video_select']['group'],     # 5
        ui['video_count'],               # 6
        ui['skip_existing'],             # 7
        ui['remove_background_switch'],  # 8
        ui['adv_accordion'],             # 9
        ui['extract_btn'],               # 10
        ui['latent_file_list'],          # 11
        ui['auto_batch_btn'],            # 12
        ui['mem_warning'],               # 13
        ui['extract_cancel_btn'],        # 14
        ui['hdr_model'],                 # 15
        ui['hdr_source'],                # 16
        ui['video_select']['btn_row'],   # 17
        ui['video_select']['all_state'], # 18
        ui['video_select']['accordion'], # 19
        ui['extract_status'],            # 20
        ui['multi_gpu_toggle'],          # 21
    ]

    # ------------------------------------------------------------------
    # Event bindings
    # ------------------------------------------------------------------

    extract_tab.select(
        init_select_video_list,
        inputs=[storage_path, project_name],
        outputs=all_ui_elements_to_control,
    )

    # Per-video selection quick buttons (All / None / Invert / halves).
    wire_video_selector(ui['video_select'])

    # Session selector: update status + ERA warning
    ui["session_selector"].change(
        fn=_on_session_selected,
        inputs=[storage_path, project_name, ui["session_selector"]],
        outputs=[ui["session_status"], ui["era_kit_warning"]],
    )

    # ERA warning also updates when ERA checkbox changes
    ui["eliminate_rotation_asymmetry"].change(
        fn=_on_era_kit_warning,
        inputs=[ui["session_selector"], ui["eliminate_rotation_asymmetry"]],
        outputs=[ui["era_kit_warning"]],
    )

    # Extract button — disable during run, enable Cancel
    # Generator owns the button states (running → reset); a fresh cancel flag is
    # created first. Mirrors the tracking/pre-process tabs.
    ui['extract_btn'].click(
        fn=init_cancel_event,
        outputs=ui['cancel_event'],
        queue=False,
    ).then(
        fn=ui_extract_roi_latent,
        inputs=[
            storage_path, project_name, ui['select_model'], ui['select_roi_id'],
            ui['video_select']['group'], ui['batch_size'], ui['skip_existing'],
            ui['remove_background_switch'],
            ui['eliminate_rotation_asymmetry'], ui['era_roi_id'],
            ui['pooling_method'], ui['pooling_scales'], ui['feature_layers'],
            ui['latent_precision'], ui['multi_gpu_toggle'],
            ui['session_selector'], ui['cancel_event'],
        ],
        outputs=[ui['latent_file_list'], ui['extract_btn'], ui['extract_cancel_btn'],
                 ui['extract_status']],
        show_progress="hidden",  # we render our own bar in extract_status
    )

    # Cancel: set the flag + immediate relabel; the generator's final yield resets
    # buttons. No cancels=[…] — the event aborts the in-flight video at its next batch.
    ui['extract_cancel_btn'].click(
        fn=request_cancel,
        inputs=ui['cancel_event'],
        outputs=ui['extract_cancel_btn'],
        queue=False,
    )

    # Memory guard
    _mem_inputs = [
        ui['select_model'],
        ui['batch_size'],
        ui['eliminate_rotation_asymmetry'],
        ui['pooling_method'],
        ui['pooling_scales'],
    ]

    def _mem_update(model_type, batch_size_str, era_switch, pooling_method, pooling_scales_list):
        import torch
        from castle.core.memory_guard import check as _check
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            batch_size = int(batch_size_str)
        except (ValueError, TypeError):
            return gr.update(value="", visible=False)
        risky, msg = _check(model_type, batch_size, device, rotate=bool(era_switch))
        if risky:
            return gr.update(
                value=f'<p style="color:#c05000;background:#fff4e6;padding:6px 10px;border-radius:4px;margin:4px 0">{msg}</p>',
                visible=True,
            )
        return gr.update(value="", visible=False)

    for _comp in _mem_inputs:
        _comp.change(_mem_update, inputs=_mem_inputs, outputs=ui['mem_warning'])

    def _auto_batch(model_type, era_switch, pooling_method, pooling_scales_list):
        import torch
        from castle.core.memory_guard import suggest_batch_size as _suggest
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return str(_suggest(model_type, device, rotate=bool(era_switch)))

    ui['auto_batch_btn'].click(
        _auto_batch,
        inputs=[ui['select_model'], ui['eliminate_rotation_asymmetry'], ui['pooling_method'], ui['pooling_scales']],
        outputs=ui['batch_size'],
    )

    return ui
