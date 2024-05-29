import os
import json
from re import L
import gradio as gr
import numpy as np
from castle.utils.latent_explorer import Latent
from PIL import Image
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import io
from castle.utils.video_io import ReadArray


umap_config_template = '''[
    {
        "n_neighbors": 30,
        "min_dist": 0.0,
        "n_components": 2
    }
]'''

hdbscan_config_template ='''{
    "n_neighbors": 50,
    "min_dist": 0.0
}'''


dbscan_config_template='''{
    "eps": 2.0
}'''

def padding(mi, mx, scale=1.05):
    mid = (mi + mx) / 2
    d = mx - mi
    return (mid - (d / 2) * scale), (mid + (d / 2) * scale)

class MultiVideos:
    def __init__(self, storage_path, project_name, select_roi_id, bin_size):
        self.source_path = os.path.join(storage_path, project_name, 'sources')
        project_path = os.path.join(storage_path, project_name)
        project_config_path = os.path.join(project_path, 'config.json')
        project_config = json.load(open(project_config_path, 'r'))
        latent_list = [(k,v) for k, v in project_config['latent'].items() if f'ROI_{select_roi_id}' in k]
        latent_dir_path = os.path.join(storage_path, project_name, 'latent')

        self.latents = []
        self.videos_meta = []
        self.bin_size = bin_size


        for it, v in latent_list:
            latent = np.load(os.path.join(latent_dir_path, it))['latent']
            n = (len(latent) // bin_size)* bin_size
            self.latents.append(latent[:n])
            self.videos_meta.append(((len(latent) // bin_size), v))

    
    def bin_index2frame(self, index):

        for vn, v in self.videos_meta:

            if index >= vn:
                index -= vn
                continue
            video_path = os.path.join(self.source_path, v)
            return ReadArray(video_path)[index*self.bin_size + self.bin_size // 2]
        gr.Info('bin_index2frame error')
        return None

    def get_latents(self):
        return Latent(np.array(self.latents), self.bin_size)




class EmbeddingScatterPlot:
    def __init__(self, local_latents):
        data = local_latents.embedding
        self.local_latents = local_latents
        self.data = data
        self.xlim = padding(data[:,0].min(), data[:,0].max())
        self.ylim = padding(data[:,1].min(), data[:,1].max())
        self.selected_point = (np.nan, np.nan)
        self.selected_index = -1
        self.tree = KDTree(data)
    
    def pixel_2_embedding(self, px, py):
        px, py = float(px), float(py)
        ex = (px / self.width) * (self.xlim[1] - self.xlim[0]) + self.xlim[0]
        ey = (py / self.height) * (self.ylim[1] - self.ylim[0]) + self.ylim[0]
        return ex, ey

    def plot(self):
        plt.figure()
        self.local_latents.plot_embedding()
        # plt.scatter(self.data[:,0], self.data[:,1], color='blue')
        plt.scatter(self.selected_point[0], self.selected_point[1], color='red')
        plt.axis('off')
        plt.xlim(self.xlim)
        plt.ylim(self.ylim[1], self.ylim[0])

        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg', bbox_inches='tight', pad_inches=0)
        plt.close()
        buf.seek(0)
        img = Image.open(buf)

        self.width, self.height = img.size

        return img
    
    def click(self, x, y):
        x, y = self.pixel_2_embedding(x, y)
        index = self.near_point(x, y)
        
        self.selected_point = self.data[index]

        self.selected_index = index

        self.selected_index = np.arange(len(self.local_latents.index_mask))[self.local_latents.index_mask][index]
        gr.Info(f'select frame {index}')
        return self.plot()
        
    
    def near_point(self, x, y):
        distance, index = self.tree.query((x, y))
        return index



def embedding_plot_click(mulvideo, Z_plt, evt: gr.SelectData):
    emb_plot = Z_plt.click(evt.index[0], evt.index[1])
    index = Z_plt.selected_index
    # gr.Info(f'index: {index}')
    frame = mulvideo.bin_index2frame(index)
    return emb_plot, frame


def collapse_accordion():
    return gr.update(open=False)

# def init_latent_explorer(storage_path, project_name, select_roi_id, bin_size):
#     if project_name == None:
#         return None
    
#     project_path = os.path.join(storage_path, project_name)
#     project_config_path = os.path.join(project_path, 'config.json')
#     project_config = json.load(open(project_config_path, 'r'))
#     latent_list = [k for k, v in project_config['latent'].items() if f'ROI_{select_roi_id}' in k]
#     latent_dir_path = os.path.join(storage_path, project_name, 'latent')
#     latents = []
#     for it in latent_list:
#         # latent = np.load(f'{latent_dir_path}/{it}')['latent']
#         latent = np.load(os.path.join(latent_dir_path, it))['latent']
#         latents.append(latent)

#     return Latent(np.array(latents), bin_size)


def update_select_cluster_list(latents):
    li = [k for k,v in latents.behavior_name2cluster_id.items()]

    return gr.update(choices=li)



# def select_cluster(latents, cluster_name):
#     return latents.select(selected_cluster=cluster_name)


def generate_embedding(latents, cluster_name, cfg):
    
    try:
        cfg = json.loads(cfg)
    except:
        cfg = dict()
        gr.Info('Json format error')
        return None, None
    local_latents = latents.select(selected_cluster=cluster_name)
    local_latents.build_embedding(cfg)
    Z_plt = EmbeddingScatterPlot(local_latents)
    return local_latents, Z_plt, Z_plt.plot()


def generate_local_cluster(local_latents, method, cfg):
    try:
        cfg = json.loads(cfg)
    except:
        gr.Info('Json format error')
        return None, None
    local_latents.build_cluster(method=method, configs=cfg)
    Z_plt = EmbeddingScatterPlot(local_latents)
    return Z_plt, Z_plt.plot()


def change_cluster_method_template(method):
    if method == 'dbscan':
        return dbscan_config_template
    elif method == 'hdbscan':
        return hdbscan_config_template
    else:
        gr.Info(f'method error got{method}, expect dbscan or hdbscan')



# x.label_cluster(0, 'behavior_bb')
def label_local_cluster(local_latents, cluster_id, cluster_name):
    if len(cluster_name) == 0:
        gr.Info('Name is empty')
        return 
        
    local_latents.label_cluster(cluster_id, cluster_name)  
    gr.Info(f'Name {cluster_id} as {cluster_name}')

def import_info_from_local_latent(latents, local_latents):
    latents.import_local_latent(local_latents)

    fig = plt.figure(figsize=(12, 2))
    latents.plot_syllables()
    plt.tight_layout()

    return fig, update_select_cluster_list(latents)



def init_mulvideo(storage_path, project_name, select_roi_id, bin_size):
    if project_name == None:
        return None
    
    mulv = MultiVideos(storage_path, project_name, select_roi_id, bin_size)
    return mulv, mulv.get_latents()



def create_cluster_page_ui(storage_path, project_name, cluster_page_tab):
    ui = dict()
    with gr.Accordion('Input setting', visible=False) as ui['cluster_input_accordion']:
        ui['select_roi_id'] = gr.Textbox(label="Enter ROI ID", value="1", info="ex: 1,2,3.", visible=False)
        ui['bin_size'] = gr.Number(label='Time window (frame)', value=1, interactive=True, visible=False)
        ui['reset'] = gr.Button("Initialize", interactive=True, visible=False)
    
    latents = gr.State(None)
    local_latents = gr.State(None)
    local_embedding_plot = gr.State(None)
    mulvideo = gr.State(None)
    with gr.Row(visible=True):
        with gr.Column(scale=5):
            ui['embedding_plot'] = gr.Image(label='Embedding', interactive=False, visible=False)
        with gr.Column(scale=5):
            ui['display'] = gr.Image(label='Display', interactive=False, visible=False)  

    with gr.Row(visible=True):
        with gr.Column(scale=2):
            ui['select_cluster'] = gr.Dropdown(label="Select Cluster",  visible=False,interactive=True)
            ui['umap_config_text'] = gr.Textbox(label='UMAP configs', value=umap_config_template, lines=5, max_lines=30, interactive=True, visible=False)
            ui['umap_run'] = gr.Button("Generate Embedding", interactive=True, visible=False)
        with gr.Column(scale=2):
            ui['cluster_method'] = gr.Dropdown(['dbscan', 'hdbscan'], label='Cluster method',
                              value='dbscan',interactive=True)
            ui['cluster_config_text'] = gr.Textbox(label='Cluster configs', lines=5, max_lines=30, interactive=True, visible=False)
            ui['cluster_run'] = gr.Button("Generate Cluster", interactive=True, visible=False)
        with gr.Column(scale=2):
            ui['label_cluster_id'] = gr.Number(label='Cluster id', interactive=True, visible=False)
            ui['label_cluster_name'] = gr.Textbox(label='Cluster name', interactive=True, visible=False)
            ui['label_cluster_btn'] = gr.Button("Enter", interactive=True, visible=False)
            ui['label_cluster_submit_btn'] = gr.Button("Submit", interactive=True, visible=False)


            
    ui['syllables_plot'] = gr.Plot(label='Syllable', visible=False)
    

    ui['reset'].click(
        fn=init_mulvideo,
        inputs=[storage_path, project_name, ui['select_roi_id'], ui['bin_size']],
        outputs=[mulvideo, latents]
    )

    ui['select_cluster'].focus(
        fn=update_select_cluster_list,
        inputs=latents,
        outputs=ui['select_cluster']

    )
    # ui['select_cluster'].select(
    #     fn=select_cluster,
    #     inputs=[latents, ui['select_cluster']],
    #     outputs=local_latents,

    # )
    ui['umap_run'].click(
        fn=generate_embedding,
        inputs=[latents, ui['select_cluster'], ui['umap_config_text']],
        outputs=[local_latents, local_embedding_plot, ui['embedding_plot']]
    )

    ui['cluster_method'].select(
        fn=change_cluster_method_template,
        inputs=ui['cluster_method'],
        outputs=ui['cluster_config_text']
    )
    ui['reset'].click(
        fn=change_cluster_method_template,
        inputs=ui['cluster_method'],
        outputs=ui['cluster_config_text']
    )
    ui['reset'].click(
        fn=collapse_accordion,
        outputs=ui['cluster_input_accordion']
    )

    ui['embedding_plot'].select(
        fn=embedding_plot_click,
        inputs=[mulvideo, local_embedding_plot],
        outputs= [ui['embedding_plot'], ui['display']]
    )
    ui['cluster_run'].click(
        fn=generate_local_cluster,
        inputs=[local_latents, ui['cluster_method'], ui['cluster_config_text']],
        outputs=[local_embedding_plot, ui['embedding_plot'] ],
    )
    ui['label_cluster_btn'].click(
        fn=label_local_cluster,
        inputs=[local_latents, ui['label_cluster_id'], ui['label_cluster_name']],
    )
    ui['label_cluster_submit_btn'].click(
        fn=import_info_from_local_latent,
        inputs=[latents, local_latents],
        outputs=[ui['syllables_plot'], ui['select_cluster']]

    )

    return ui