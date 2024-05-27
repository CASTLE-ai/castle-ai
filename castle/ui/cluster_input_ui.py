import gradio as gr
import os
import json
import glob
import numpy as np

def init_drop(storage_path, project_name):
    if project_name == None:
        return gr.update(choices=[])

    project_path = os.path.join(storage_path, project_name)
    project_config_path = os.path.join(project_path, 'config.json')
    project_config = json.load(open(project_config_path, 'r'))
    latent_list = [k for k, v in project_config['latent'].items()]
    latent_list.append("All")


    return gr.update(choices=latent_list), gr.update(choices=[]), latent_list, []



def select_latent(target_latent,  unselect_list, select_list):
    if target_latent == 'All':
        unselect_list.remove('All')
        select_list.extend(unselect_list)
        unselect_list = ['All']
    else:
        unselect_list.remove(target_latent)
        select_list.append(target_latent)
    return gr.update(choices=unselect_list), gr.update(choices=select_list), unselect_list, select_list

def unselect_latent(target_latent,  unselect_list, select_list):
    select_list.remove(target_latent)
    unselect_list.append(target_latent)
    return gr.update(choices=unselect_list), gr.update(choices=select_list, value=""), unselect_list, select_list


def build_umap_intput(storage_path, project_name, umap_input_name, select_list):
    if len(umap_input_name) == 0:
        gr.Info("The name is not allow empty.")
        return
    project_path = os.path.join(storage_path, project_name)
    project_config_path = os.path.join(project_path, 'config.json')
    project_config = json.load(open(project_config_path, 'r'))
    latent_dir_path = os.path.join(storage_path, project_name, 'latent')
    if not 'umap_input' in project_config:
        project_config['umap_input'] = dict()

    umap_input_dir_path = os.path.join(project_path, 'umap_input')
    os.makedirs(umap_input_dir_path, exist_ok=True)
    umap_input_path = os.path.join(umap_input_dir_path, f'{umap_input_name}_umap_input.npz')

    umap_input_mask = []
    for it in select_list:
        latent = np.load(f'{latent_dir_path}/{it}')['latent']
        mask = (~np.isnan(latent[:, 0]))
        umap_input_mask.extend(mask)

    umap_input_mask = np.array(umap_input_mask)
    # print('umap_input_mask', umap_input_mask.shape)

    np.savez_compressed(umap_input_path, umap_mask=umap_input_mask)
    project_config['umap_input'][f'{umap_input_name}_umap_input.npz'] = {
        'latent': select_list,
        'source': [project_config['latent'][it] for it in select_list],

    }
    json.dump(project_config, open(project_config_path,'w'))
    gr.Info("Successful build the list as a UMAP Input")
    return



def create_cluster_input_ui(storage_path, project_name, cluster_page_tab):
    ui = dict()

    select_list = gr.State([])
    unselect_list = gr.State([])
    with gr.Row(visible=True):
        with gr.Column(scale=5):
            ui['unselect_drop'] = gr.Dropdown(label='Unselect List', interactive=True)
            ui['select_latent'] = gr.Button('Select')
        with gr.Column(scale=5):
            ui['select_drop'] = gr.Dropdown(label='Select List', interactive=True)
            ui['unselect_latent'] = gr.Button('Unselect')
    ui['name'] = gr.Textbox(label='Name')
    ui['build'] = gr.Button('Build List as UMAP Input')



    cluster_page_tab.select(
        fn=init_drop,
        inputs=[storage_path, project_name],
        outputs=[ui['unselect_drop'], ui['select_drop'], unselect_list, select_list]
    )


    ui['select_latent'].click(
        fn=select_latent,
        inputs=[ui['unselect_drop'], unselect_list, select_list],
        outputs=[ui['unselect_drop'], ui['select_drop'], unselect_list, select_list],
    )

    ui['unselect_latent'].click(
        fn=unselect_latent,
        inputs=[ui['select_drop'], unselect_list, select_list],
        outputs=[ui['unselect_drop'], ui['select_drop'], unselect_list, select_list],
    )

    ui['build'].click(
        fn=build_umap_intput,
        inputs=[storage_path, project_name, ui['name'], select_list],
    )


    return ui