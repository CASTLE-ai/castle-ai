"""Post-tracking analysis and mask editing UI."""

import os

import gradio as gr

from .plot_mask_info import Plotter
from castle.utils.plot import generate_mix_image, generate_mask_image
from castle.utils.h5_io import H5IO
from castle.utils.video_io import WriteArray
from castle.utils.analysis_utils import compute_roi_info, save_kinematic_csv


def plot_basic_mask_info(storage_path, project_name, source_video, progress=gr.Progress()):
    if source_video is None:
        gr.Warning("Please select a video first.")
        return None, None, None, None, None
    project_path = os.path.join(storage_path, project_name)
    video_name = source_video.video_name
    track_dir_path = os.path.join(project_path, 'track', video_name)

    rois_results_path = os.path.join(track_dir_path, 'mask_list.h5')
    if not os.path.exists(rois_results_path):
        gr.Warning(f"Mask file not found: {rois_results_path}")
        return None, None, None, None, None
    rois_results = H5IO(rois_results_path, read_only=True)
    try:
        n_rois = rois_results.get_n_rois()
        total_frames = len(rois_results)

        roi_info_list = compute_roi_info(rois_results, n_rois, total_frames, progress_fn=progress.tqdm)

        mask_kinematic_csv_path = save_kinematic_csv(track_dir_path, video_name, roi_info_list)
    finally:
        rois_results.close()
    return Plotter.plot_position(roi_info_list), Plotter.plot_speed(roi_info_list), Plotter.plot_area(roi_info_list), roi_info_list, mask_kinematic_csv_path


def generate_mask_video(storage_path, project_name, source_video, progress=gr.Progress()):
    if source_video is None:
        gr.Warning("Please select a video first.")
        return None
    project_path = os.path.join(storage_path, project_name)
    video_name = source_video.video_name
    track_dir_path = os.path.join(project_path, 'track', video_name)
    rois_results_path = os.path.join(track_dir_path, 'mask_list.h5')
    video_name_wo_extension = video_name.split('.')[0]
    output_path = os.path.join(track_dir_path, f'{video_name_wo_extension}-rois.mp4')
    if not os.path.exists(rois_results_path):
        gr.Warning(f"Mask file not found: {rois_results_path}")
        return None
    output = WriteArray(output_path, fps=source_video.fps, crf=15)
    rois_results = H5IO(rois_results_path, read_only=True)
    try:
        n_frames = len(rois_results)
        progress(0.0, desc=f"Rendering ROI video (0/{n_frames})")

        for i in range(n_frames):
            rois = rois_results[i]
            out_frame = generate_mask_image(rois)
            output.append(out_frame)
            if i % 30 == 0 or i == n_frames - 1:
                progress((i + 1) / n_frames, desc=f"Rendering ROI video ({i + 1}/{n_frames})")
    finally:
        rois_results.close()
        output.close()
    return output_path


def generate_mix_video(storage_path, project_name, source_video, progress=gr.Progress()):
    if source_video is None:
        gr.Warning("Please select a video first.")
        return None
    project_path = os.path.join(storage_path, project_name)
    video_name = source_video.video_name
    track_dir_path = os.path.join(project_path, 'track', video_name)
    rois_results_path = os.path.join(track_dir_path, 'mask_list.h5')
    video_name_wo_extension = video_name.split('.')[0]
    output_path = os.path.join(track_dir_path, f'{video_name_wo_extension}-mix.mp4')
    if not os.path.exists(rois_results_path):
        gr.Warning(f"Mask file not found: {rois_results_path}")
        return None
    from castle.utils.video_io import encode_overlay_video
    source_path = os.path.join(project_path, 'sources', video_name)
    progress(0.0, desc="Rendering mix video…")

    def _cb(frac, desc=""):
        progress(frac, desc=desc or "Rendering mix video…")

    return encode_overlay_video(
        source_path, rois_results_path, output_path, source_video.fps,
        generate_mix_image, progress_callback=_cb,
    )


def create_post_track_ui(storage_path, project_name, source_video):
    ui = dict()
    roi_info_list = gr.State(None)
    # with gr.Accordion('Basic Kinematic Infomation', open=True, visible=False) as ui['basic_mask_info_accordion']:
    ui['analysis_mask'] = gr.Button("Analysis Mask", interactive=True, visible=False)
    with gr.Row(visible=True):
        ui['position_plot'] = gr.Plot(label="Position", visible=False)
    with gr.Row(visible=True):
        ui['velocity_plot'] = gr.Plot(label="Speed", visible=False)
    with gr.Row(visible=True):
        ui['area_plot'] = gr.Plot(label="Area", visible=False)  
    with gr.Row(visible=True):
        # with gr.Column(scale=2):
        #     ui['generate_mask_kinematic_btn'] = gr.Button("Generate Basic Kinematic CSV", interactive=False, visible=False)
        # with gr.Column(scale=8):
        ui['mask_kinematic_file'] = gr.File(label="Basic Kinematic CSV", interactive=False, visible=False)
    with gr.Row(visible=True):
        with gr.Column(scale=2):
            ui['generate_mask_video_btn'] = gr.Button("Generate ROIs Video", interactive=True, visible=False)
        with gr.Column(scale=8):
            ui['mask_video'] = gr.File(label="ROIs Video", interactive=False, visible=False)
    with gr.Row(visible=True):
        with gr.Column(scale=2):
            ui['generate_mix_video_btn'] = gr.Button("Generate Mix Video", interactive=True, visible=False)
        with gr.Column(scale=8):
            ui['mix_video'] = gr.File(label="Mix Video", interactive=False, visible=False)

    ui['analysis_mask'].click(
        fn=plot_basic_mask_info,
        inputs=[storage_path, project_name, source_video],
        outputs=[ui['position_plot'], ui['velocity_plot'], ui['area_plot'], roi_info_list, ui['mask_kinematic_file']]
    )

    # ui['generate_mask_kinematic_btn'].click(
    #     fn=generate_mask_kinematic_csv,
    #     inputs=[storage_path, project_name, source_video, roi_info_list],
    #     outputs=ui['mask_kinematic_file']
    # )

    ui['generate_mask_video_btn'].click(
        fn=generate_mask_video,
        inputs=[storage_path, project_name, source_video],
        outputs=ui['mask_video'],
        show_progress=True,
    )

    ui['generate_mix_video_btn'].click(
        fn=generate_mix_video,
        inputs=[storage_path, project_name, source_video],
        outputs=ui['mix_video'],
        show_progress=True,
    )
      
    return ui
