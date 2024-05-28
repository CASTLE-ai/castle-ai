import os
import json
import gradio as gr
import numpy as np
from castle import generate_sa
from castle.utils.plot import generate_mix_image, generate_image_with_dots
# from api.segmentor import MultiObjectSegmentor, merge_frame_and_mask

# model_config = json.load(open('config/model_config.json', 'r'))

def keep_click_mode_switch_only(mode):
    if mode == 'Add':
        return 'Remove'
    return 'Add'

def index_slide_event(segmentor, source_video, index):
    del segmentor
    frame = source_video[index]
    return None, frame, frame, gr.update(interactive=True)


def label_click_fn(segmentor, frame, mode, evt: gr.SelectData):
    if segmentor == None:
        # segmentor = MultiObjectSegmentor(model_config['sam_args'])
        segmentor = generate_sa(model_type='vit_b')
        segmentor.set_frame(frame)


    mode = 1 if mode == 'Add' else 0
    point = (evt.index[0], evt.index[1])  # (y, x)
    gr.Info(f'Click! Point: {evt.index[0]}, {evt.index[1]}')

    
    mask = segmentor.segment_with_click(point, mode)
    mix_img = generate_mix_image(frame, mask)
    mix_img_with_dots = generate_image_with_dots(mix_img, segmentor.click_points, segmentor.click_modes)
    
    return segmentor, mix_img_with_dots
    # return segmentor, merge_frame_and_mask(frame, mask, segmentor.click_points, segmentor.click_modes)

def reset_click_mode():
    return 'Add'

def next_roi(segmentor):
    segmentor.next_roi()
    return segmentor

def save_rois(storage_path, project_name, segmentor, index, source_video):
    project_path = os.path.join(storage_path, project_name)
    video_name = source_video.video_name
    label_dir_path = os.path.join(project_path, 'label', video_name)
    os.makedirs(label_dir_path, exist_ok=True)

    label_path = os.path.join(label_dir_path, f'{index}')
    mask = segmentor.temp_mask
    frame = segmentor.frame
    np.savez_compressed(label_path, frame=frame, mask=mask)
    gr.Info(f"Save ROI at frame {index}.")
    return

def save_rois_event(storage_path, project_name, segmentor, index, source_video):
    save_rois(storage_path, project_name, segmentor, index, source_video)
    return None

def clean_rois_event():
    return None


def create_label_ui(storage_path, project_name, source_video):
    ui = dict()
    segmentor = gr.State(None)
    ui['select_frame'] = gr.State(None)
    with gr.Row(visible=True):
        with gr.Column(scale=2):
            ui['click_mode'] = gr.Textbox(label='click mode', value='Add', interactive=False, visible=False)
            ui['click_mode_switch'] = gr.Button('Change mode', interactive=True, visible=False)
            ui['next_roi_btn'] = gr.Button('Label Next ROI', interactive=True, visible=False)
            ui['save_rois_btn'] = gr.Button('Save ROIs', interactive=True, visible=False)
            ui['clean_rois_btn'] = gr.Button('Clean ROIs', interactive=True, visible=False)
            
        with gr.Column(scale=8):
            ui['display_view'] = gr.Image(label='Display', interactive=False, visible=False)
            ui['index_slide'] = gr.Slider(label='Frame', minimum=0, step=1, maximum=1, value=0, interactive=True, visible=False)


    ui['click_mode_switch'].click(
        fn=keep_click_mode_switch_only,
        inputs=ui['click_mode'],
        outputs=ui['click_mode']
    )

    ui['index_slide'].change(
        fn=index_slide_event,
        inputs=[segmentor, source_video, ui['index_slide']],
        outputs=[segmentor, ui['select_frame'], ui['display_view'], ui['display_view']]
    )

    ui['display_view'].select(
        fn=label_click_fn,
        inputs=[segmentor, ui['select_frame'], ui['click_mode']],
        outputs=[segmentor, ui['display_view']],
    )

    ui['next_roi_btn'].click(
        fn=reset_click_mode,
        outputs=ui['click_mode']
    )

    ui['next_roi_btn'].click(
        fn=next_roi,
        inputs=segmentor,
        outputs=segmentor
    )

    ui['save_rois_btn'].click(
        fn=save_rois_event,
        inputs=[storage_path, project_name, segmentor, ui['index_slide'], source_video],
        outputs=segmentor,
    )

    ui['clean_rois_btn'].click(
        fn=clean_rois_event,
        # inputs=[storage_path, project_name, segmentor, ui['index_slide'], source_video],
        outputs=segmentor,
    )

    ui['clean_rois_btn'].click(
        fn=index_slide_event,
        inputs=[segmentor, source_video, ui['index_slide']],
        outputs=[segmentor, ui['select_frame'], ui['display_view'], ui['display_view']]
    )

    ui['index_slide'].change(
        fn=reset_click_mode,
        outputs=ui['click_mode']
    )

    return ui