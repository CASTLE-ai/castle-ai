import os
import cv2
import numpy as np

import gradio as gr

from .plot_mask_info import Plotter
from castle.utils.plot import generate_mix_image, generate_mask_image
from castle.utils.h5_io import H5IO
from castle.utils.video_io import ReadArray, WriteArray

def plot_basic_mask_info(storage_path, project_name, source_video, progress=gr.Progress()):
    project_path = os.path.join(storage_path, project_name)
    video_name = source_video.video_name
    track_dir_path = os.path.join(project_path, 'track', video_name)

    rois_results_path = os.path.join(track_dir_path, f'mask_list.h5')
    rois_results = H5IO(rois_results_path) # TODO: check file is exist first
    n_rois = rois_results.get_n_rois()

    total_frames = len(rois_results)
    roi_info_list = [{"x":[], "y":[], "area":[]} for i in range(n_rois)]
    for i_frame in progress.tqdm(range(total_frames)):
        for i in range(n_rois):
            if not rois_results.has_mask(i_frame):
                roi_info_list[i]['x'].append(np.nan)
                roi_info_list[i]['y'].append(np.nan)
                roi_info_list[i]['area'].append(0)
                continue


            mask = rois_results[i_frame][:]
            mask = cv2.inRange(mask, i+1, i+1)
            output = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_32S)
            num_labels, _, stats, centroids = output
            if num_labels <= 1:
                roi_info_list[i]['x'].append(np.nan)
                roi_info_list[i]['y'].append(np.nan)
                roi_info_list[i]['area'].append(0)
                continue


            areas = [stats[j, cv2.CC_STAT_AREA] for j in range(1, num_labels)]
            max_label = np.argmax(areas)
            roi_info_list[i]['x'].append(round(centroids[max_label + 1][0]))
            roi_info_list[i]['y'].append(round(centroids[max_label + 1][1]))
            roi_info_list[i]['area'].append(areas[max_label])
            
    for i in range(n_rois):
        roi_info_list[i]['x'] = np.array(roi_info_list[i]['x'])
        roi_info_list[i]['y'] = np.array(roi_info_list[i]['y'])
        roi_info_list[i]['area'] = np.array(roi_info_list[i]['area']).astype(int)

    del rois_results
    return Plotter.plot_position(roi_info_list), Plotter.plot_speed(roi_info_list), Plotter.plot_area(roi_info_list), roi_info_list


def generate_mask_kinematic_csv(storage_path, project_name, source_video, roi_info_list):
    project_path = os.path.join(storage_path, project_name)
    video_name = source_video.video_name
    track_dir_path = os.path.join(project_path, 'track', video_name)
    video_name_wo_extension = video_name.split('.')[0]
    mask_kinematic_csv_path = os.path.join(track_dir_path, f'{video_name_wo_extension}-basic-infomation.csv')
    df = Plotter.create_pandas(roi_info_list)
    df.to_csv(mask_kinematic_csv_path)
    return mask_kinematic_csv_path


def generate_mask_video(storage_path, project_name, source_video):
    project_path = os.path.join(storage_path, project_name)
    video_name = source_video.video_name
    track_dir_path = os.path.join(project_path, 'track', video_name)
    rois_results_path = os.path.join(track_dir_path, f'mask_list.h5')
    video_name_wo_extension = video_name.split('.')[0]
    output_path = os.path.join(track_dir_path, f'{video_name_wo_extension}-rois.mp4')
    output = WriteArray(output_path, fps=source_video.fps, crf=15)
    rois_results = H5IO(rois_results_path) # TODO: check file is exist first
    n_frames = len(rois_results)

    for i in range(n_frames):
        rois = rois_results[i]
        out_frame = generate_mask_image(rois)
        output.append(out_frame)

    del rois_results, output
    return output_path


def generate_mix_video(storage_path, project_name, source_video):
    project_path = os.path.join(storage_path, project_name)
    video_name = source_video.video_name
    track_dir_path = os.path.join(project_path, 'track', video_name)
    rois_results_path = os.path.join(track_dir_path, f'mask_list.h5')
    video_name_wo_extension = video_name.split('.')[0]
    output_path = os.path.join(track_dir_path, f'{video_name_wo_extension}-mix.mp4')
    output = WriteArray(output_path, fps=source_video.fps, crf=15)
    rois_results = H5IO(rois_results_path) # TODO: check file is exist first
    n_frames = len(rois_results)

    for i in range(n_frames):
        rois = rois_results[i]
        frame = source_video[i]
        out_frame = generate_mix_image(frame, rois)
        output.append(out_frame)

    del rois_results, output
    return output_path


def create_post_track_ui(storage_path, project_name, source_video):
    ui = dict()
    roi_info_list = gr.State(None)
    # with gr.Accordion('Basic Kinematic Infomation', open=True, visible=False) as ui['basic_mask_info_accordion']:
    ui['analysis_mask'] = gr.Button("Analysis Mask", interactive=True, visible=False)
    with gr.Row(visible=True):
        with gr.Tab(label='position'):
            ui['position_plot'] = gr.Plot(label="Position", visible=False)
        with gr.Tab(label='speed'):
            ui['velocity_plot'] = gr.Plot(label="Speed", visible=False)
        with gr.Tab(label='area'):
            ui['area_plot'] = gr.Plot(label="Area", visible=False)  
    with gr.Row(visible=True):
        with gr.Column(scale=2):
            ui['generate_mask_kinematic_btn'] = gr.Button("Generate Basic Kinematic CSV", interactive=False, visible=False)
        with gr.Column(scale=8):
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
        outputs=[ui['position_plot'], ui['velocity_plot'], ui['area_plot'], roi_info_list]
    )

    ui['generate_mask_kinematic_btn'].click(
        fn=generate_mask_kinematic_csv,
        inputs=[storage_path, project_name, source_video, roi_info_list],
        outputs=ui['mask_kinematic_file']
    )

    ui['generate_mask_video_btn'].click(
        fn=generate_mask_video,
        inputs=[storage_path, project_name, source_video],
        outputs=ui['mask_video']
    )

    ui['generate_mix_video_btn'].click(
        fn=generate_mix_video,
        inputs=[storage_path, project_name, source_video],
        outputs=ui['mix_video']
    )
      
    return ui