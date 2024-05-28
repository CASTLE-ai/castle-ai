import gradio as gr
import os
import cv2
import h5py
import numpy as np
import av

from .plot_mask_info import Plotter
from castle.utils.plot import generate_mix_image, generate_mask_image
# from api.segmentor import merge_frame_and_mask, mask2img
# from media.tracking_io import TrackingIO
from castle.utils.h5_io import H5IO

def plot_basic_mask_info(storage_path, project_name, source_video, progress=gr.Progress()):
    project_path = os.path.join(storage_path, project_name)
    video_name = source_video.video_name
    track_dir_path = os.path.join(project_path, 'track', video_name)
    mask_list_path = os.path.join(track_dir_path, f'mask_list.h5')
    tracker = H5IO(mask_list_path)

    with h5py.File(mask_list_path, 'r') as f:
        roi_count = tracker.read_config('roi_count')
        total_frames = tracker.read_config('total_frames')
        roi_info_list = [{"x":[], "y":[], "area":[]} for i in range(roi_count)]
        for i_frame in progress.tqdm(range(total_frames)):
            for i in range(roi_count):
                if f"{i_frame}" in f:
                    mask = f[str(i_frame)][:]
                    # print('mask', mask.shape)
                    mask = cv2.inRange(mask, i+1, i+1)
                    output = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_32S)
                    num_labels, _, stats, centroids = output
                    if num_labels > 1:
                        areas = [stats[j, cv2.CC_STAT_AREA] for j in range(1, num_labels)]
                        max_label = np.argmax(areas)
                        roi_info_list[i]['x'].append(round(centroids[max_label + 1][0]))
                        roi_info_list[i]['y'].append(round(centroids[max_label + 1][1]))
                        roi_info_list[i]['area'].append(areas[max_label])
                        continue
                roi_info_list[i]['x'].append(np.nan)
                roi_info_list[i]['y'].append(np.nan)
                roi_info_list[i]['area'].append(0)


    # print(roi_info_list)
    for i in range(roi_count):
        roi_info_list[i]['x'] = np.array(roi_info_list[i]['x'])
        roi_info_list[i]['y'] = np.array(roi_info_list[i]['y'])
        roi_info_list[i]['area'] = np.array(roi_info_list[i]['area']).astype(int)

    del tracker
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
    mask_list_path = os.path.join(track_dir_path, f'mask_list.h5')
    video_name_wo_extension = video_name.split('.')[0]
    mask_video_path = os.path.join(track_dir_path, f'{video_name_wo_extension}-mask.mp4')
    output = av.open(mask_video_path, 'w')

    # Add a stream, specify codec and settings
    stream = output.add_stream('libx264', rate=source_video.fps)
    stream.options = {'crf': '22'}
    stream.pix_fmt = 'yuv420p'  # Most common pixel format
    
    tracker = H5IO(mask_list_path)
    stream.height = tracker.read_config('height')
    stream.width = tracker.read_config('width')
    total_frames = tracker.read_config('total_frames')

    for i_frame in range(total_frames):
        it = tracker.read_mask(i_frame)
        frame = av.VideoFrame.from_ndarray(generate_mask_image(it), format='rgb24')
        for packet in stream.encode(frame):
            output.mux(packet)

    for packet in stream.encode():
        output.mux(packet)

    output.close()
    del tracker
    return mask_video_path

def generate_mix_video(storage_path, project_name, source_video):
    project_path = os.path.join(storage_path, project_name)
    video_name = source_video.video_name
    track_dir_path = os.path.join(project_path, 'track', video_name)
    mask_list_path = os.path.join(track_dir_path, f'mask_list.h5')
    tracker = H5IO(mask_list_path)

    video_name_wo_extension = video_name.split('.')[0]
    mix_video_path = os.path.join(track_dir_path, f'{video_name_wo_extension}-mix.mp4')
    output = av.open(mix_video_path, 'w')

    # Add a stream, specify codec and settings
    stream = output.add_stream('libx264', rate=source_video.fps)
    stream.options = {'crf': '22'}

    stream.height = tracker.read_config('height')
    stream.width = tracker.read_config('width')
    total_frames = tracker.read_config('total_frames')
    stream.pix_fmt = 'yuv420p'  # Most common pixel format
    for i in range(total_frames):
        it = tracker.read_mask(i)
        frame = source_video[i]
        mix = generate_mix_image(frame, it)
        mix = av.VideoFrame.from_ndarray(mix, format='rgb24')
        for packet in stream.encode(mix):
            output.mux(packet)

    for packet in stream.encode():
        output.mux(packet)

    output.close()
    del tracker
    return mix_video_path

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
            ui['generate_mask_kinematic_btn'] = gr.Button("Generate Basic Kinematic CSV", interactive=True, visible=False)
        with gr.Column(scale=8):
            ui['mask_kinematic_file'] = gr.File(label="Basic Kinematic CSV", interactive=False, visible=False)
    with gr.Row(visible=True):
        with gr.Column(scale=2):
            ui['generate_mask_video_btn'] = gr.Button("Generate Mask Video", interactive=True, visible=False)
        with gr.Column(scale=8):
            ui['mask_video'] = gr.File(label="Mask Video", interactive=False, visible=False)
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