import gradio as gr
import os
import numpy as np
from api.segmentor import merge_frame_and_mask, mask2img

from media.tracking_io import TrackingIO

def index_slide_apply(storage_path, project_name, source_video, index, mode):
    if mode == 'Image':
        return source_video.read_by_index(index).to_rgb().to_ndarray()
    
    project_path = os.path.join(storage_path, project_name)
    video_name = source_video.video_name
    track_dir_path = os.path.join(project_path, 'track', video_name)
    mask_list_path = os.path.join(track_dir_path, f'mask_list.h5')

    tracker = TrackingIO(mask_list_path)

    if mode == 'Image & Mask':
        frame = source_video.read_by_index(index).to_rgb().to_ndarray()
        mix = merge_frame_and_mask(frame, tracker.read_mask(index))
        return mix
    if mode == 'Mask':
        return mask2img(tracker.read_mask(index))
    
    del tracker



def create_view_ui(storage_path, project_name, source_video):
    ui = dict()
    with gr.Row(visible=True):
        with gr.Column(scale=2):
            ui['display_mode_drop'] = gr.Dropdown(
                    choices=["Image", "Image & Mask", "Mask"],
                    label="Display Mode", value="Image",
                    interactive=True, visible=False)

            ui['step_frame'] = gr.Dropdown(
                    choices=["1", "5", "10", "100", "1000", "10000"],
                    label="Frame step number", value="1",
                    interactive=True, visible=False)
            
        with gr.Column(scale=8):
            ui['display_view'] = gr.Image(label='Display', interactive=False, visible=False)
            ui['index_slide'] = gr.Slider(label="Frame", minimum=0, step=1, maximum=1, value=0, interactive=True, visible=False)
          
    ui['index_slide'].change(
        fn=index_slide_apply,
        inputs=[storage_path, project_name, source_video, ui['index_slide'], ui['display_mode_drop']],
        outputs=ui['display_view']
    )
    step_frame_apply = lambda step_frame: gr.update(step=step_frame)
    ui['step_frame'].change(
        fn=step_frame_apply,
        inputs=ui['step_frame'],
        outputs=ui['index_slide']
    )
    return ui
    