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
import pandas as pd

umap_config_template = '''[
    {
        "n_neighbors": 30,
        "min_dist": 0.0,
        "n_components": 2
    }
]'''




dbscan_config_template='''{
    "eps": 1.0
}'''

def padding(mi, mx, scale=1.05):
    mid = (mi + mx) / 2
    d = mx - mi
    return (mid - (d / 2) * scale), (mid + (d / 2) * scale)


# Define a function to convert frame number to timestamp
def frame_to_timestamp(frame_number, fps):
    seconds = frame_number / fps
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    milliseconds = (seconds % 1) * 1000
    return f"{hours:02}:{minutes:02}:{int(seconds):02},{int(milliseconds):03}"

class MultiVideos:
    def __init__(self, storage_path, project_name, select_roi_id, bin_size):
        self.source_path = os.path.join(storage_path, project_name, 'sources')
        project_path = os.path.join(storage_path, project_name)
        self.project_path = project_path
        project_config_path = os.path.join(project_path, 'config.json')
        project_config = json.load(open(project_config_path, 'r'))
        latent_list = [(k,v) for k, v in project_config['latent'].items() if f'ROI_{select_roi_id}' in k]
        latent_dir_path = os.path.join(storage_path, project_name, 'latent')

        # self.latents = None
        self.videos_meta = []
        self.bin_size = bin_size


        for it, v in latent_list:
            latent = np.load(os.path.join(latent_dir_path, it))['latent']
            n = (len(latent) // bin_size) * bin_size
            if n == 0:
                continue
            
            if not hasattr(self, "latents"):
                self.latents = np.zeros((0, latent.shape[-1]))
            
            if not hasattr(self, "fps"):
                self.fps = ReadArray(os.path.join(self.source_path, v)).fps

            self.latents = np.append(self.latents, latent[:n], axis=0)
            self.videos_meta.append(((len(latent) // bin_size), v))
            print(it, self.latents[-1].shape)

            print(len(self.latents))
    
    def bin_index2frame(self, index):

        for vn, v in self.videos_meta:

            if index >= vn:
                index -= vn
                continue
            video_path = os.path.join(self.source_path, v)
            gr.Info(f'select frame {index*self.bin_size + self.bin_size // 2} from video {v}')
            return ReadArray(video_path)[index*self.bin_size + self.bin_size // 2]
        gr.Info('bin_index2frame error')
        return None

    def get_latents(self):
        return Latent(self.latents, self.bin_size)
    

    def generate_subtitle(self, syllabels, meta):
        subtitle_path = os.path.join(self.project_path, 'subtitles')
        os.makedirs(subtitle_path, exist_ok=True)
        subtitle_list = []

        cum = 0
        for vn, v in self.videos_meta: # what is the define of vn?
            bin_length = vn
            this_video_syllabels = syllabels[cum:cum+bin_length]
            data = np.repeat(this_video_syllabels, self.bin_size)
            srt_entries = []
            n = len(data)
            delta_index = np.arange(n-1)[(data[:-1] != data[1:])]
            delta_index = np.concatenate([[-1],delta_index,[n-1]])
            for i in range(len(delta_index)-1):

                start_frame = delta_index[i]+1
                end_frame = delta_index[i+1]
                start_time = frame_to_timestamp(start_frame, self.fps)
                end_time = frame_to_timestamp(end_frame, self.fps)
                behavior = data[start_frame+1]
                srt_entries.append(f"{i + 1}\n{start_time} --> {end_time}\n{meta[behavior]['name']}\n")
                
            srt_content = "\n".join(srt_entries)

            video_basename = os.path.basename(v).split('.')[0]
            output_path = os.path.join(subtitle_path, video_basename) + '.srt'
            with open(output_path, 'w', encoding='utf-8') as file:
                file.write(srt_content)
                
            subtitle_list.append(output_path)
            # Join all entries into a single SRT content string
            


            cum += bin_length



        return subtitle_list




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
        # gr.Info(f'select frame {self.selected_index}')
        return self.plot()
        
    
    def near_point(self, x, y):
        distance, index = self.tree.query((x, y))
        return index



def embedding_plot_click(mulvideo, Z_plt, evt: gr.SelectData):
    if hasattr(evt, 'index'):
        emb_plot = Z_plt.click(evt.index[0], evt.index[1])
    else:
        gr.Info('click event error')
    index = Z_plt.selected_index
    # gr.Info(f'index: {index}')
    frame = mulvideo.bin_index2frame(index)
    return emb_plot, frame


def collapse_accordion():
    return gr.update(open=False)




def update_select_cluster_list(latents):
    if hasattr(latents, 'behavior_name2cluster_id'):
        li = [k for k,v in latents.behavior_name2cluster_id.items()]
    else:
        li = []
        gr.Info('latent init error, please wait 1s and try one times.')

    return gr.update(choices=li)



# def select_cluster(latents, cluster_name):
#     return latents.select(selected_cluster=cluster_name)


def generate_embedding(latents, cluster_name, cfg):
    
    try:
        cfg = json.loads(cfg)
    except:
        cfg = dict()
        gr.Info('UMAP config Json format error')
        return None, None
    local_latents = latents.select(selected_cluster=cluster_name)
    local_latents.build_embedding(cfg)
    Z_plt = EmbeddingScatterPlot(local_latents)
    return local_latents, Z_plt, Z_plt.plot()


def generate_local_cluster(local_latents, method, cfg):
    try:
        cfg = json.loads(cfg)
    except:
        gr.Info('Cluster Json format error')
        return None, None
    local_latents.build_cluster(method=method, configs=cfg)
    Z_plt = EmbeddingScatterPlot(local_latents)
    return Z_plt, Z_plt.plot()


def change_cluster_method_template(method):
    if method == 'dbscan':
        return dbscan_config_template
    else:
        gr.Info(f'method error got{method}, expect dbscan')



# x.label_cluster(0, 'behavior_bb')
def label_local_cluster(local_latents, cluster_id, cluster_name):
    if len(cluster_name) == 0:
        gr.Info('Name is empty')
        return 
        
    local_latents.label_cluster(cluster_id, cluster_name)  
    gr.Info(f'Name {cluster_id} as {cluster_name}')

def convert_latent_cluster_to_subtitle(storage_path, project_name, latents, mulvideo):
    # project_path = os.path.join(storage_path, project_name)
    # source_path =  os.path.join(storage_path, 'sources')
    # project_config_path = os.path.join(project_path, 'config.json')
    # project_config = json.load(open(project_config_path, 'r'))
    # video_path = os.path.join(source_path, project_config["source"][0])
    return mulvideo.generate_subtitle(latents.cluster, latents.cluster_meta)


def import_info_from_local_latent(storage_path, project_name, latents, local_latents, mulvideo):
    try:
        latents.import_local_latent(local_latents)
    except:
        gr.Info('Do not use same cluster name')
        return None, update_select_cluster_list(latents), None, None, None


    fig = plt.figure(figsize=(12, 2))
    latents.plot_syllables()
    plt.tight_layout()

    df1 = {
        'Id': [k for k, v in latents.cluster_meta.items()],
        'Name': [v['name'] for k, v in latents.cluster_meta.items()],
    }
    df1 = pd.DataFrame(df1)
    df2 = {
        'behavior': np.repeat(latents.cluster, latents.time_window)
    }
    df2 = pd.DataFrame(df2)

    cluster_path = os.path.join(storage_path, project_name, 'cluster')
    os.makedirs(cluster_path, exist_ok=True)


    df1_path = os.path.join(cluster_path, 'id.csv')
    df2_path = os.path.join(cluster_path, 'time_series.csv')
    df1.to_csv(df1_path, index=False)  
    df2.to_csv(df2_path)

    subtitle_path = convert_latent_cluster_to_subtitle(storage_path, project_name, latents, mulvideo)
    

    return fig, update_select_cluster_list(latents), df1_path, df2_path, subtitle_path



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
            ui['cluster_method'] = gr.Dropdown(['dbscan'], label='Cluster method',
                              value='dbscan',interactive=True)
            ui['cluster_config_text'] = gr.Textbox(label='Cluster configs', lines=5, max_lines=30, interactive=True, visible=False)
            ui['cluster_run'] = gr.Button("Generate Cluster", interactive=True, visible=False)
        with gr.Column(scale=2):
            ui['label_cluster_id'] = gr.Number(label='Cluster id', interactive=True, visible=False)
            ui['label_cluster_name'] = gr.Textbox(label='Cluster name', interactive=True, visible=False)
            ui['label_cluster_btn'] = gr.Button("Enter", interactive=True, visible=False)
            ui['label_cluster_submit_btn'] = gr.Button("Submit", interactive=True, visible=False)


            
    ui['syllables_plot'] = gr.Plot(label='Syllable', visible=False)
    with gr.Row(visible=True):
        with gr.Column(scale=2):
            ui['behavior_id_csv'] = gr.File(label="Behavior ID", interactive=False, visible=False)
        with gr.Column(scale=2):
            ui['behavior_time_series_csv'] = gr.File(label="Behavior time series", interactive=False, visible=False)
        with gr.Column(scale=2):
            ui['behavior_time_series_srt'] = gr.File(label="Behavior time series (SRT)", interactive=False, visible=False)



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
        inputs=[storage_path, project_name, latents, local_latents, mulvideo],
        outputs=[ui['syllables_plot'], ui['select_cluster'], ui['behavior_id_csv'], ui['behavior_time_series_csv'], ui['behavior_time_series_srt']]
    )

    return ui