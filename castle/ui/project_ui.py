import os
import json
import shutil
import datetime
import gradio as gr

def lock_project_page(object_count):
    return [gr.update(interactive=False) for i in range(object_count)]

def unlock_old_project_btn(object_count):
    return [gr.update(interactive=True) for i in range(object_count)]

def list_project(storage_path):
    li = [d for d in os.listdir(storage_path) if os.path.isdir(os.path.join(storage_path, d))]
    if len(li) > 1:
        subdirectories = sorted(li)
    else:
        subdirectories = li
    return gr.update(choices=subdirectories)

def create_new_project(storage_path, project_name):
    project_path = os.path.join(storage_path, project_name)
    project_config_path = os.path.join(project_path, 'config.json')
    os.makedirs(project_path)
    project_config = dict()
    project_config['project_name'] = project_name
    json.dump(project_config, open(project_config_path,'w'))

def default_new_project_name(new_project_name):
    if len(new_project_name) > 0:
        return gr.update(value=new_project_name)
    
    create_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    return gr.update(value=f'{create_time}-Project')

def set_project_name(project_name):
    return project_name

def delete_old_project(storage_path, project_name):
    folder_path = os.path.join(storage_path, project_name)
    if os.path.exists(folder_path):
        print('folder_path', folder_path)
        shutil.rmtree(folder_path)
        gr.Info(f"Deleted {project_name}")
        print("Deleted")
    else:
        print("No such folder")



def create_project_ui(OS_SYS, root=''):
    if root == None:
        root = os.path.join('projects','')
    sep = os.sep
    if not root.endswith(sep):
        root += sep
    if not os.path.exists(root):
        os.makedirs(root)
        
    ui = dict()
    DEFAULT_DEVICE = 'mps' if OS_SYS == 'Darwin' else 'cuda'
    ui['device'] = gr.Textbox(label='Device',
                              info='For example: mps, cpu, cuda, or cuda:x',
                              value=DEFAULT_DEVICE,
                              interactive=False, visible=False)

    ui['storage_path'] = gr.Textbox(label='Storage Location',
                        info='The location which storage all project',
                        value=root,
                        interactive=True)
    
    with gr.Tab(label='Open Old Project'):
        ui['old_project_drop'] = gr.Dropdown(label="Open Project", interactive=True)
        with gr.Row(visible=True):
            with gr.Column(scale=9):
                ui['old_project_open_btn'] = gr.Button('Open', interactive=False)
            with gr.Column(scale=1):
                ui['old_project_delete_btn'] = gr.Button('Delete', interactive=False)
    with gr.Tab(label='New Project'):
        ui['new_project_name'] = gr.Textbox(label='New Project Name', interactive=True)
        ui['new_project_create_btn'] = gr.Button('Create', interactive=True)

    
    ui['project_name'] = gr.State(None)
    
    object_count = gr.State(len(ui))
    old_project_btn_list = [ui['old_project_open_btn'], ui['old_project_delete_btn']]
    old_project_btn_count = gr.State(len(old_project_btn_list))


    ui['new_project_create_btn'].click(
        fn=lock_project_page,
        inputs=object_count,
        outputs=[v for k, v in ui.items()]
    )
    ui['old_project_open_btn'].click(
        fn=lock_project_page,
        inputs=object_count,
        outputs=[v for k, v in ui.items()]
    )
    # ui['old_project_open_btn'].click(
    #     fn=lock_project_page,
    #     inputs=object_count,
    #     outputs=[v for k, v in ui.items()]
    # )
    ui['old_project_drop'].focus(
        fn=list_project,
        inputs=ui['storage_path'],
        outputs=ui['old_project_drop']
    )
    ui['old_project_drop'].select(
        fn=unlock_old_project_btn,
        inputs=old_project_btn_count,
        outputs=old_project_btn_list,
    )
    ui['new_project_create_btn'].click(
        fn=create_new_project,
        inputs=[ui[it] for it in ['storage_path', 'new_project_name']]
    )
    
    ui['new_project_name'].focus(
        fn=default_new_project_name,
        inputs=ui['new_project_name'],
        outputs=ui['new_project_name']
    )
    ui['new_project_create_btn'].click(
        fn=set_project_name,
        inputs=ui['new_project_name'],
        outputs=ui['project_name']
    )
    ui['old_project_open_btn'].click(
        fn=set_project_name,
        inputs=ui['old_project_drop'],
        outputs=ui['project_name']
    )
    ui['old_project_delete_btn'].click(
        fn=delete_old_project,
        inputs=[ui['storage_path'], ui['old_project_drop']],
    )

    return ui
