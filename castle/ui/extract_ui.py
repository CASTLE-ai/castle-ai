"""
castle/ui/extract_ui.py
UI Layer for Extraction.
Delegates all logic to castle.service.extraction_service and castle.core.extractor.
"""

import logging
import os

import gradio as gr
from tqdm import tqdm

from castle.core.data import Preprocess
from castle.core.extractor import extract_roi_latent_from_video, extract_roi_rotation_latent_from_video
from castle.utils.video_manager import get_project_config
from castle.utils.h5_io import H5IO

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
            "ℹ️ Using raw source video + tracking mask.",
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
                    "ℹ️ KIT 已對齊身體軸，Eliminate Rotation Asymmetry 效益有限，"
                    "但不影響正確性。"
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
        except Exception:
            pass
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
        gr.update(choices=[], value=None, visible=False),  # 5  select_video
        gr.update(value=0, visible=False),  # 6  video_count
        gr.update(visible=False),  # 7  skip_existing
        gr.update(visible=False),  # 8  remove_background_switch
        gr.update(visible=False),  # 9  adv_accordion
        gr.update(visible=False),  # 10 extract_btn
        gr.update(visible=False),  # 11 latent_file_list
        gr.update(visible=False),  # 12 auto_batch_btn
        gr.update(value="", visible=False),  # 13 mem_warning
        gr.update(visible=False, interactive=False),  # 14 extract_cancel_btn
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
            choices_with_all = list(choices)
            choices_with_all.append("All")

            # Session selector choices
            session_choices = ["(None — use raw source)"]
            try:
                from castle.core.preprocess_session import list_sessions
                metas = list_sessions(storage_path, project_name)
                session_choices += [
                    f"{m['session_name']} | {m['method']} | {len(m.get('videos', []))} videos"
                    for m in metas
                ]
            except Exception:
                pass

            updates[0] = gr.update(choices=session_choices, value=session_choices[0], visible=True)  # session_selector
            updates[1] = gr.update(value="ℹ️ Using raw source video + tracking mask.", visible=True)  # session_status
            updates[2] = gr.update(visible=True)   # select_model
            updates[3] = gr.update(visible=True)   # select_roi_id
            updates[4] = gr.update(visible=True)   # batch_size
            updates[5] = gr.update(choices=choices_with_all, value="All", visible=True)  # select_video
            updates[6] = gr.update(value=video_count_val, visible=True)  # video_count
            updates[7] = gr.update(visible=True)   # skip_existing
            updates[8] = gr.update(visible=True)   # remove_background_switch
            updates[9] = gr.update(visible=True)   # adv_accordion
            updates[10] = gr.update(visible=True)  # extract_btn
            updates[11] = gr.update(visible=True)  # latent_file_list
            updates[12] = gr.update(visible=True)  # auto_batch_btn
            # mem_warning (13) stays hidden until reactive check triggers
            updates[14] = gr.update(visible=True, interactive=False)  # extract_cancel_btn
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

@handle_assertion_error
def ui_extract_roi_latent(
    storage_path: str,
    project_name: str,
    select_model: str,
    select_roi: str,
    select_video: str,
    batch_size: str,
    skip_existing: bool,
    remove_background_switch: bool = False,
    era_switch: bool = False,
    era_roi_id: int = 2,
    pooling_method: str = 'weighted_average',
    pooling_scales_list: list = None,
    feature_layers_str: str = '',
    session_display: str = "(None — use raw source)",
    progress=gr.Progress(),
) -> str:

    messages = []
    preprocess_args = Preprocess(remove_background_switch=bool(remove_background_switch))

    parsed_scales = [int(s) for s in pooling_scales_list] if pooling_scales_list else [1, 2, 4]
    parsed_layers = None
    if feature_layers_str and feature_layers_str.strip():
        try:
            parsed_layers = [int(x.strip()) for x in feature_layers_str.split(',') if x.strip()]
        except ValueError:
            raise ValueError(f"Invalid feature layers format: '{feature_layers_str}'.")

    session_id = _parse_session_id_from_display(session_display)

    _, config = get_project_config(storage_path, project_name)
    video_list = sorted(config['source']) if select_video == "All" else [select_video]

    if session_id:
        messages.append(f"Using pre-process session: {session_display.split(' | ')[0]}")

    videos_to_process = []
    messages.append(f"Starting pre-flight check for {len(video_list)} videos...")
    for video_name in video_list:
        tags = []
        if preprocess_args.remove_background_switch:
            tags.append("rmbg")
        if pooling_method == 'multiscale' and parsed_scales:
            scales_str = "x".join(str(s) for s in sorted(parsed_scales))
            tags.append(f"spp{scales_str}")
        if parsed_layers:
            layers_str = "x".join(str(lay) for lay in sorted(parsed_layers))
            tags.append(f"L{layers_str}")

        suffix = "_".join([select_model] + tags)
        pre_tag = f"_pre-{session_id}" if session_id else ""
        latent_filename = f'{os.path.splitext(video_name)[0]}_ROI_{select_roi}_{suffix}{pre_tag}.npz'
        latent_dir = os.path.join(storage_path, project_name, 'latent', select_model)
        output_path = os.path.join(latent_dir, latent_filename)

        if skip_existing and os.path.exists(output_path):
            messages.append(f"  ⏩ Skipping existing: {video_name}")
            continue
        videos_to_process.append(video_name)

    if not videos_to_process:
        messages.append("\n✅ All videos already have latent files. Nothing to extract.")
        return "\n".join(messages)

    messages.append(f"\nFound {len(videos_to_process)} new videos to process.")

    def update_progress(p, desc=None):
        progress(p, desc=desc)

    success_count = 0
    failed_videos = []

    for video_name in tqdm(videos_to_process, desc="Extracting Latents"):
        # Resolve paths from session
        source_video_path = None
        mask_path_override = None
        if session_id:
            from castle.core.preprocess_session import get_preprocessed_paths
            try:
                vpath, mpath = get_preprocessed_paths(storage_path, project_name, session_id, video_name)
                source_video_path = str(vpath)
                mask_path_override = str(mpath)
            except FileNotFoundError as exc:
                messages.append(f"  ❌ {video_name}: not preprocessed in session — {exc}")
                failed_videos.append(video_name)
                continue

        try:
            messages.append(f"\nProcessing {video_name}...")
            path = extract_roi_latent_from_video(
                storage_path=storage_path,
                project_name=project_name,
                video_name=video_name,
                roi_id=int(select_roi),
                model_name=select_model,
                batch_size=int(batch_size),
                preprocess_config=preprocess_args,
                skip_existing=skip_existing,
                progress_callback=update_progress,
                pooling_method=pooling_method,
                pooling_scales=parsed_scales if pooling_method == 'multiscale' else None,
                feature_layers=parsed_layers,
                source_video_path=source_video_path,
                mask_path_override=mask_path_override,
                session_id=session_id,
            )
            if path:
                messages.append(f"  ✅ Saved: {os.path.basename(path)}")
                success_count += 1
            else:
                messages.append(f"  ⚠️ No path returned for {video_name}.")
        except Exception as e:
            failed_videos.append(video_name)
            messages.append(f"  ❌ Error processing {video_name}: {e}")

    summary_msg = f"\n\n🎉 Extraction Complete!\nSuccessfully processed {success_count}/{len(videos_to_process)} videos."
    if failed_videos:
        summary_msg += f"\n⚠️ Failed videos: {', '.join(failed_videos)}"
    messages.append(summary_msg)

    # --- Eliminate Rotation Asymmetry (ERA) ---
    if era_switch:
        import numpy as _np
        _era_roi_id = int(era_roi_id)
        _first_video = videos_to_process[0] if videos_to_process else video_list[0]
        _mask_path_check = os.path.join(storage_path, project_name, 'track', _first_video, 'mask_list.h5')
        _tail_roi_ok = True
        try:
            with H5IO(_mask_path_check, read_only=True) as _h5:
                _guard_mask = _h5.read_mask(0)
            if _guard_mask is not None:
                _roi_ids = set(_np.unique(_guard_mask[_guard_mask > 0]).tolist())
                if _era_roi_id not in _roi_ids or len(_roi_ids) < 2:
                    _tail_roi_ok = False
                    messages.append(
                        f"\n⚠️ ERA skipped: Reference ROI (ID {_era_roi_id}) not found "
                        f"(detected: {sorted(_roi_ids) or 'none'})."
                    )
        except Exception as _e:
            messages.append(f"\n⚠️ Could not validate ERA ROI ({_e}); proceeding.")

        if _tail_roi_ok:
            messages.append("\n\n--- Eliminate Rotation Asymmetry ---")
            rot_success = 0
            rot_failed = []
            for video_name in tqdm(videos_to_process, desc="ERA"):
                # Resolve paths for ERA (use session video if selected)
                era_source = None
                era_mask = None
                if session_id:
                    from castle.core.preprocess_session import get_preprocessed_paths
                    try:
                        vpath, mpath = get_preprocessed_paths(storage_path, project_name, session_id, video_name)
                        era_source = str(vpath)
                        era_mask = str(mpath)
                    except FileNotFoundError:
                        pass

                try:
                    messages.append(f"\nERA: {video_name}...")
                    rpath = extract_roi_rotation_latent_from_video(
                        storage_path=storage_path,
                        project_name=project_name,
                        video_name=video_name,
                        roi_id=int(select_roi),
                        model_name=select_model,
                        batch_size=int(batch_size),
                        preprocess_config=preprocess_args,
                        skip_existing=skip_existing,
                        progress_callback=update_progress,
                        source_video_path=era_source,
                        mask_path_override=era_mask,
                        session_id=session_id,
                    )
                    if rpath:
                        messages.append(f"  ✅ {os.path.basename(rpath)}")
                        rot_success += 1
                    else:
                        messages.append(f"  ⚠️ ERA returned no path for {video_name}.")
                except Exception as e:
                    rot_failed.append(video_name)
                    messages.append(f"  ❌ ERA error for {video_name}: {e}")

            messages.append(
                f"\n🎉 ERA Complete: {rot_success}/{len(videos_to_process)} succeeded."
                + (f"\n⚠️ Failed: {', '.join(rot_failed)}" if rot_failed else "")
            )

    return "\n".join(messages)


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
        info="Select a Pre-process session to use its stabilised video and aligned masks for extraction.",
    )
    ui["session_status"] = gr.Markdown(
        value="ℹ️ Using raw source video + tracking mask.",
        visible=False,
    )

    with gr.Row():
        with gr.Column(scale=3):
            with gr.Row():
                ui['select_model'] = gr.Dropdown(
                    label="Visual Model",
                    choices=["dinov2_vitb14_reg4_pretrain", "dinov3_vitb16", "dinov3_vitl16"],
                    value="dinov3_vitb16",
                    visible=False,
                    info="DINOv2/v3 backbone used for feature extraction.",
                    scale=2,
                )
                ui['select_roi_id'] = gr.Textbox(
                    label="ROI ID",
                    value="1",
                    visible=False,
                    info="Region of Interest ID to extract features from.",
                    scale=1,
                )
            with gr.Row():
                ui['batch_size'] = gr.Textbox(
                    label="Batch Size",
                    value="32",
                    visible=False,
                    info="Frames processed per GPU batch. Use 'Auto Batch Size' to find a safe value.",
                    scale=2,
                )
                ui['auto_batch_btn'] = gr.Button("Auto Batch Size", size="sm", visible=False, scale=1)
            with gr.Row():
                ui['select_video'] = gr.Dropdown(
                    label="Target Video",
                    value=None,
                    visible=False,
                    info="Select a specific video or 'All' to process the entire project.",
                    scale=2,
                )
                ui['video_count'] = gr.Number(
                    label="Project Video Count",
                    value=0,
                    interactive=False,
                    visible=False,
                    scale=1,
                )
            with gr.Row():
                ui['skip_existing'] = gr.Checkbox(
                    label="Skip existing files",
                    value=True,
                    visible=False,
                    info="Skip videos that already have a latent file saved to disk.",
                )
                ui['remove_background_switch'] = gr.Checkbox(
                    label="Remove Background",
                    value=False,
                    visible=False,
                    info="Zero out pixels outside the ROI mask before extracting features (on-the-fly).",
                )

        with gr.Column(scale=2):
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
            ui['adv_accordion'] = adv_accordion

    ui['mem_warning'] = gr.HTML(value="", visible=False)
    with gr.Row():
        ui['extract_btn'] = gr.Button("Extract", visible=False, variant="primary")
        ui['extract_cancel_btn'] = gr.Button("Cancel", visible=False, interactive=False)
        # TODO(2-D): implement extract_crop_video handler before re-enabling this
        # button.  Kept declared and permanently invisible so any external
        # reference (CLI, MCP) does not break.  Removed from
        # all_ui_elements_to_control update list below.
        ui['extract_crop_video_btn'] = gr.Button(
            "Extract Crop Video", visible=False, interactive=False
        )
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
        ui['select_video'],              # 5
        ui['video_count'],               # 6
        ui['skip_existing'],             # 7
        ui['remove_background_switch'],  # 8
        ui['adv_accordion'],             # 9
        ui['extract_btn'],               # 10
        ui['latent_file_list'],          # 11
        ui['auto_batch_btn'],            # 12
        ui['mem_warning'],               # 13
        ui['extract_cancel_btn'],        # 14
    ]

    # ------------------------------------------------------------------
    # Event bindings
    # ------------------------------------------------------------------

    extract_tab.select(
        init_select_video_list,
        inputs=[storage_path, project_name],
        outputs=all_ui_elements_to_control,
    )

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
    def _before_extract():
        return gr.update(interactive=False), gr.update(interactive=True)

    def _after_extract():
        return gr.update(interactive=True), gr.update(interactive=False)

    _extract_click = ui['extract_btn'].click(
        fn=_before_extract,
        outputs=[ui['extract_btn'], ui['extract_cancel_btn']],
        queue=False,
    )
    (
        _extract_click
        .then(
            fn=ui_extract_roi_latent,
            inputs=[
                storage_path, project_name, ui['select_model'], ui['select_roi_id'],
                ui['select_video'], ui['batch_size'], ui['skip_existing'],
                ui['remove_background_switch'],
                ui['eliminate_rotation_asymmetry'], ui['era_roi_id'],
                ui['pooling_method'], ui['pooling_scales'], ui['feature_layers'],
                ui['session_selector'],
            ],
            outputs=ui['latent_file_list'],
        )
        .then(
            fn=_after_extract,
            outputs=[ui['extract_btn'], ui['extract_cancel_btn']],
            queue=False,
        )
    )

    ui['extract_cancel_btn'].click(
        fn=_after_extract,
        outputs=[ui['extract_btn'], ui['extract_cancel_btn']],
        cancels=[_extract_click],
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
