import json
import os
import gradio as gr
import numpy as np
import umap
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image


def init_umap_input_list(storage_path, project_name):
    if project_name == None:
        return gr.update(choices=[])

    project_path = os.path.join(storage_path, project_name)
    project_config_path = os.path.join(project_path, 'config.json')
    project_config = json.load(open(project_config_path, 'r'))
    umap_input_list = [k for k, v in project_config['umap_input'].items()]

    return gr.update(choices=umap_input_list)


def umap_dimension_reduction(X, layer_1_neighber, layer_2_neighber):
    
    layer_1_neighber, layer_2_neighber = int(layer_1_neighber), int(layer_2_neighber)
    print('layer_1_neighber, layer_2_neighber',layer_1_neighber, layer_2_neighber)
    if layer_2_neighber == 0:
        reducer = umap.UMAP(n_neighbors=layer_1_neighber, min_dist=0.1, n_components=2)
        data_2d = reducer.fit_transform(X)
        return data_2d, [reducer]

    if len(X) < layer_1_neighber or len(X) < layer_2_neighber:
        gr.Info("neighber should smaller than len(X)")
        return [], []
    reducer_1 = umap.UMAP(n_neighbors=layer_1_neighber, min_dist=0.3, n_components=100)
    reducer_2 = umap.UMAP(n_neighbors=layer_2_neighber, min_dist=0.1, n_components=2)
    print('X.shape', X.shape)
    # print(X[0])
    Z = reducer_1.fit_transform(X)
    Z = reducer_2.fit_transform(Z)
    return Z, [reducer_1, reducer_2]


def plot_on_img(X_2d):
    if len(X_2d) == 0:
        X_2d = np.array([[0, 0]])

    # 在散點圖中添加 X 和 Y 軸的比例尺
    fig, ax = plt.subplots(figsize=(7.2, 7.2), dpi=100)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # 繪製散點圖
    x = X_2d[:, 0]
    y = X_2d[:, 1]
    c = np.arange(len(X_2d))
    ax.scatter(x, y, c=c)

    # 隱藏坐標軸標籤
    ax.axis('off')
    # 將圖像轉換為 PIL.Image 對象
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    plt.close()
    return Image.open(buf)



def generate_umap_gallery(storage_path, project_name, umap_input):
    project_path = os.path.join(storage_path, project_name)
    project_config_path = os.path.join(project_path, 'config.json')
    project_config = json.load(open(project_config_path, 'r'))
    latent_dir_path = os.path.join(storage_path, project_name, 'latent')
    latent_list = project_config['umap_input'][umap_input]['latent']
    umap_input_dir_path = os.path.join(project_path, 'umap_input')
    umap_input_path = os.path.join(umap_input_dir_path, umap_input)
    umap_input_mask = np.load(umap_input_path)['umap_mask']
    print("umap_input_mask",umap_input_mask.shape)
    # print(umap_input_mask)
    multi_latent = np.zeros((0, project_config['observer_dim']))
    print('latent_list', len(latent_list))
    for it in latent_list:
        latent_path = os.path.join(latent_dir_path, it)
        latent = np.load(latent_path)['latent']
        multi_latent = np.append(multi_latent, latent, axis=0)


    print('multi_latent',multi_latent.shape)
    umap_list = []
    img_list = []
    for i in [15, 30, 50, 200, 500]:
        for j in [0, 10, 20]:
            X_2d = np.zeros((len(multi_latent), 2)) + np.nan
            try:
                X_2d[umap_input_mask], umap = umap_dimension_reduction(multi_latent[umap_input_mask], i, j)
                img_list.append(plot_on_img(X_2d))
                umap_list.append(umap)
            except:
                pass
    return img_list, umap_list



def create_dimension_reduction_ui(storage_path, project_name, init_tab):
    ui = dict()

    umap_list = gr.State(None)
    with gr.Row(visible=True):
        with gr.Column(scale=2):
            ui['input_list'] = gr.Dropdown(label='Input List')
            ui['umap_library'] = gr.Dropdown(label='UMAP Library', choices=['umap_learn (on CPU)'], value="umap_learn (on CPU)")
            ui['umap_windown_size'] = gr.Number(label='UMAP Window size', precision=0)
            ui['generate_umap_gallery'] = gr.Button('Generate Some UMAP Result')
        with gr.Column(scale=8):
            ui['umap_gallery'] = gr.Gallery(label='UMAP Gallery', columns=5)
    



    init_tab.select(
        fn=init_umap_input_list,
        inputs=[storage_path, project_name],
        outputs=ui['input_list']
    )

    ui['generate_umap_gallery'].click(
        fn=generate_umap_gallery,
        inputs=[storage_path, project_name, ui['input_list']],
        outputs=[ui['umap_gallery'], umap_list]
    )

    return ui