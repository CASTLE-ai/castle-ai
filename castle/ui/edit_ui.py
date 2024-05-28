import os
import gradio as gr
import json
# from media.source_video import SourceVideo
from castle.utils.video_io import ReadArray

from .view_ui import create_view_ui
from .label_ui import create_label_ui
from .track_ui import create_track_ui
from .post_track_ui import create_post_track_ui

def list_project_video(storage_path, project_name):
    project_path = os.path.join(storage_path, project_name)
    project_config_path = os.path.join(project_path, 'config.json')
    project_config = json.load(open(project_config_path, 'r'))
    if not 'source' in project_config:
         return gr.update(choices=[])
    
    subdirectories = sorted(project_config['source'])
    return gr.update(choices=subdirectories)

def unlock_select_video_edit_btn():
    return gr.update(interactive=True)


def unlock_ui(object_count):
    return [gr.update(visible=True) for _ in range(object_count)]

def edit_btn_event(storage_path, project_name, video_name):
    source_video = ReadArray(os.path.join(storage_path, project_name, 'sources', video_name))
    frame = source_video[0]
    maxi = len(source_video)-1
    return source_video, video_name, gr.update(maximum=maxi), gr.update(maximum=maxi), gr.update(maximum=maxi), gr.update(maximum=maxi, value=maxi), frame, frame, frame

def collapse_source_detial():
    return gr.update(open=False)


def create_edit_ui(storage_path, project_name):
    ui = dict()
    source_video = gr.State(None)
    ui['select_video'] = gr.State(None)

    with gr.Accordion('Select Source Video', open=True, visible=False) as ui['source_accordion']:
        ui['select_video_drop'] = gr.Dropdown(label="Select Video", interactive=True, visible=False)
        ui['select_video_edit_btn'] = gr.Button('Edit', interactive=False, visible=False)

    with gr.Tab(label='View'):
        view_ui = create_view_ui(storage_path, project_name, source_video)
    
    with gr.Tab(label='Label ROI'):
        label_ui = create_label_ui(storage_path, project_name, source_video)

    with gr.Tab(label='Tracking') as track_tab:
        track_ui = create_track_ui(storage_path, project_name, source_video, track_tab)

    with gr.Tab(label='Analysis'):
        post_track_ui = create_post_track_ui(storage_path, project_name, source_video)
        pass
    


    view_ui_object_count = gr.State(len(view_ui))
    label_ui_object_count = gr.State(len(label_ui))
    track_ui_object_count = gr.State(len(track_ui))
    post_track_ui_object_count = gr.State(len(post_track_ui))

    ui['select_video_drop'].focus(
         fn=list_project_video,
         inputs=[storage_path, project_name],
         outputs=ui['select_video_drop']
    )
    ui['select_video_drop'].select(
        fn=unlock_select_video_edit_btn,
        outputs=ui['select_video_edit_btn']
    )
    ui['select_video_edit_btn'].click(
        fn=unlock_ui,
        inputs=view_ui_object_count,
        outputs=[v for k, v in view_ui.items()]
    )
    ui['select_video_edit_btn'].click(
        fn=unlock_ui,
        inputs=label_ui_object_count,
        outputs=[v for k, v in label_ui.items()]
    )
    ui['select_video_edit_btn'].click(
        fn=unlock_ui,
        inputs=track_ui_object_count,
        outputs=[v for k, v in track_ui.items()]
    )
    ui['select_video_edit_btn'].click(
        fn=unlock_ui,
        inputs=post_track_ui_object_count,
        outputs=[v for k, v in post_track_ui.items()]
    )
    
    ui['select_video_edit_btn'].click(
        fn=edit_btn_event, 
        inputs=[storage_path, project_name, ui['select_video_drop']],
        outputs=[source_video, ui['select_video'], view_ui['index_slide'], label_ui['index_slide'], track_ui['start_frame'], track_ui['stop_frame'], view_ui['display_view'], label_ui['display_view'], label_ui['select_frame']]
    )

    ui['select_video_edit_btn'].click(
        fn=collapse_source_detial,
        outputs=ui['source_accordion']
    )

    return ui