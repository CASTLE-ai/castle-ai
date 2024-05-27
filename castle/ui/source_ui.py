import os
import gradio as gr
import shutil
import json


def upload_local_videos(storage_path, project_name, upload_video_path):
    project_path = os.path.join(storage_path, project_name)
    project_config_path = os.path.join(project_path, 'config.json')
    source_dir_path = os.path.join(project_path, 'sources/')
    os.makedirs(source_dir_path, exist_ok=True)
    project_config = json.load(open(project_config_path, 'r'))
    if not 'source' in project_config:
            project_config['source'] = []

    for it in upload_video_path:
        if os.path.basename(it.name) in project_config['source']:
             gr.Info("Already in this project")
             continue
        shutil.copy(it.name, source_dir_path)
        project_config['source'].append(os.path.basename(it.name))
    json.dump(project_config, open(project_config_path,'w'))

def update_project_video_info(storage_path, project_name):
    project_path = os.path.join(storage_path, project_name)
    project_config_path = os.path.join(project_path, 'config.json')
    project_config = json.load(open(project_config_path, 'r'))
    if not 'source' in project_config:
        return
    return
     
#          return ""
#     return str(project_config['source'])


video_extensions = ['.mp4', '.avi', '.mov', '.wmv', '.flv', '.mkv']

# Function to check if a file is a video
def is_video(file_name):
    _, ext = os.path.splitext(file_name)
    return ext.lower() in video_extensions

def list_server_video(storage_path):
    subdirectories = sorted([d for d in os.listdir(storage_path) if is_video(os.path.join(storage_path, d))])
    return gr.update(choices=subdirectories)

def upload_server_video(storage_path, project_name, video_storage_path, select_server_video):
     project_path = os.path.join(storage_path, project_name)
     project_config_path = os.path.join(project_path, 'config.json')
     source_dir_path = os.path.join(project_path, 'sources/')
     os.makedirs(source_dir_path, exist_ok=True)
     project_config = json.load(open(project_config_path, 'r'))
     if not 'source' in project_config:
          project_config['source'] = []

     if os.path.basename(select_server_video) in project_config['source']:
          gr.Info("Already in this project")
          return
     
     shutil.copy(os.path.join(video_storage_path, select_server_video), source_dir_path)
     project_config['source'].append(os.path.basename(select_server_video))
     json.dump(project_config, open(project_config_path,'w'))


def create_source_ui(storage_path, project_name):
    ui = dict()
    
#     ui['project_video_info'] = gr.Textbox(
#          label='Project Video Information', 
#          interactive=False,visible=False)
#     gr.HTML(value="<br>")
    ui['upload_local_videos'] = gr.File(label="Add Local Videos", file_types=["video"], interactive=True, file_count="multiple", visible=False)
    gr.HTML(value="<br>")
    # ui['upload_another'] = gr.Button('Upload Another Video', interactive=True, visible=False)
    ui['storage_path'] = gr.Textbox(
         label='Add Server Videos: Storage Location',
         info='The location which storage videos',
         value='demo/',
         interactive=True, visible=False)
    ui['select_server_video'] = gr.Dropdown(
         label="Add Server Video", 
         interactive=True, visible=False)
    
    ui['upload_local_videos'].upload(
         fn=upload_local_videos,
         inputs=[
              storage_path, project_name, 
              ui['upload_local_videos']]
    )

    ui['upload_local_videos'].upload(
         fn=update_project_video_info,
         inputs=[storage_path, project_name],
     #     outputs=ui['project_video_info']
    )

    ui['select_server_video'].focus(
         fn=list_server_video,
         inputs=ui['storage_path'],
         outputs=ui['select_server_video']
    )
    ui['select_server_video'].select(
        fn=upload_server_video,
        inputs=[storage_path, project_name, 
              ui['storage_path'], ui['select_server_video']]
    )

    return ui