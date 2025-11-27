import os
import json
from re import L
import gradio as gr
import numpy as np
from castle.utils.latent_explorer import Latent
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
from scipy.spatial import KDTree
import io
from castle.utils.video_io import ReadArray
import pandas as pd
from sklearn.decomposition import PCA

# Configure matplotlib to reduce warnings about open figures
# Increase the warning threshold to 50 (default is 20)
matplotlib.rcParams['figure.max_open_warning'] = 50

umap_config_template = '''[
    {
        "n_neighbors": 100,
        "min_dist": 0.0,
        "n_components": 2
    }
]'''

umap_config_low_magnification_template = '''[
    {
        "n_neighbors": 30,
        "min_dist": 0.0,
        "n_components": 2
    }
]'''
umap_config_intermediate_magnification_template = '''[
    {
        "n_neighbors": 30,
        "min_dist": 0.0,
        "n_components": 5
    },
    {
        "n_neighbors": 30,
        "min_dist": 0.0,
        "n_components": 2
    }
]'''
umap_config_high_magnification_template = '''[
    {
        "n_neighbors": 30,
        "min_dist": 0.0,
        "n_components": 10
    },
    {
        "n_neighbors": 30,
        "min_dist": 0.0,
        "n_components": 2
    }
]'''


dbscan_config_template='''{
    "eps": 1.0
}'''


preset_dropdown_list = ['Low-magnification objective 1000', 'Low-magnification objective 500', 'Low-magnification objective 300', 'Low-magnification objective 100', 'Low-magnification objective 50', 'Low-magnification objective 25']
preset_dropdown_list += ['Intermediate-magnification objective (1000, 500)', 'Intermediate-magnification objective (500, 300)', 'Intermediate-magnification objective (300, 100)', 'Intermediate-magnification objective (100, 50)', 'Intermediate-magnification objective (50, 25)']
preset_dropdown_list += ['High-magnification objective (1000, 500)', 'High-magnification objective (500, 300)', 'High-magnification objective (300, 100)', 'High-magnification objective (100, 50)', 'High-magnification objective (50, 25)']

def update_umap_config_text_with_preset(preset_dropdown):
    """根據使用者選擇的預設來生成對應的 UMAP 配置字串並調整 n_neighbors 數值"""
    if preset_dropdown is None:
        return umap_config_template
    
    import re
    import json
    
    # 解析預設選項
    if 'Low-magnification objective' in preset_dropdown:
        # 提取數字 ex: "Low-magnification objective 1000" -> 1000
        numbers = re.findall(r'\d+', preset_dropdown)
        if numbers:
            n_neighbors = int(numbers[0])
            config = [
                {
                    "n_neighbors": n_neighbors,
                    "min_dist": 0.0,
                    "n_components": 2
                }
            ]
            return json.dumps(config, indent=4)
    
    elif 'Intermediate-magnification objective' in preset_dropdown:
        # 提取數字 ex: "Intermediate-magnification objective (1000, 500)" -> [1000, 500]
        numbers = re.findall(r'\d+', preset_dropdown)
        if len(numbers) >= 2:
            n_neighbors_1 = int(numbers[0])
            n_neighbors_2 = int(numbers[1])
            config = [
                {
                    "n_neighbors": n_neighbors_1,
                    "min_dist": 0.0,
                    "n_components": 5
                },
                {
                    "n_neighbors": n_neighbors_2,
                    "min_dist": 0.0,
                    "n_components": 2
                }
            ]
            return json.dumps(config, indent=4)
    
    elif 'High-magnification objective' in preset_dropdown:
        # 提取數字 ex: "High-magnification objective (1000, 500)" -> [1000, 500]
        numbers = re.findall(r'\d+', preset_dropdown)
        if len(numbers) >= 2:
            n_neighbors_1 = int(numbers[0])
            n_neighbors_2 = int(numbers[1])
            config = [
                {
                    "n_neighbors": n_neighbors_1,
                    "min_dist": 0.0,
                    "n_components": 10
                },
                {
                    "n_neighbors": n_neighbors_2,
                    "min_dist": 0.0,
                    "n_components": 2
                }
            ]
            return json.dumps(config, indent=4)
    
    # 如果沒有匹配到任何預設，返回預設模板
    return umap_config_template

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
            gr.Info(f'Loading: {v}')
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
        gr.Info(f'Finish')
    
    def bin_index2frame(self, index):

        for vn, v in self.videos_meta:

            if index >= vn:
                index -= vn
                continue
            video_path = os.path.join(self.source_path, v)
            gr.Info(f'{index*self.bin_size + self.bin_size // 2}\n{v}')
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
                end_frame = delta_index[i+1]+1
                start_time = frame_to_timestamp(start_frame, self.fps)
                end_time = frame_to_timestamp(end_frame, self.fps)
                behavior = data[start_frame]
                if behavior == -1:
                    behavior_name = "Unclustered"
                else:
                    try:
                        behavior_name = meta[behavior]['name']
                    except KeyError:
                        # Fallback if behavior ID is somehow missing from meta, though the primary issue is -1
                        behavior_name = f"Unknown Cluster {behavior}" 
                srt_entries.append(f"{i + 1}\n{start_time} --> {end_time}\n{behavior_name}\n")
                
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
        fig = plt.figure()
        self.local_latents.plot_embedding()
        # plt.scatter(self.data[:,0], self.data[:,1], color='blue')
        plt.scatter(self.selected_point[0], self.selected_point[1], color='red')
        plt.axis('off')
        plt.xlim(self.xlim)
        plt.ylim(self.ylim[1], self.ylim[0])

        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf)

        self.width, self.height = img.size

        return img
    
    def plot_named_embedding(self):
        fig = plt.figure()
        self.local_latents.plot_name_embedding()
        plt.scatter(self.selected_point[0], self.selected_point[1], color='red')
        plt.axis('off')
        plt.xlim(self.xlim)
        plt.ylim(self.ylim[1], self.ylim[0])

        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf)

        self.width, self.height = img.size

        return img

        

    def save_named_embedding(self, save_path):
        index_mask = self.local_latents.index_mask
        masked_emb = self.local_latents.embedding
        masked_cls = self.local_latents.cluster
        config = self.local_latents.configs
        n_samples = len(index_mask)
        n_features = masked_emb.shape[-1]

        emb = np.zeros((n_samples, n_features)) + np.nan
        emb[index_mask] = masked_emb

        cls = np.zeros(n_samples, dtype=np.int16) - 1
        cls[index_mask] = masked_cls



        np.savez_compressed(save_path, emb=emb, cls=cls, config=config)
        # plt.figure()
        # self.local_latents.plot_name_embedding()
        # plt.scatter(self.selected_point[0], self.selected_point[1], color='red')
        # plt.axis('off')
        # plt.xlim(self.xlim)
        # plt.ylim(self.ylim[1], self.ylim[0])
        # plt.savefig(save_path)#, bbox_inches='tight', pad_inches=0)


    
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


def generate_pca_cumulative_curve(data):
    """Generate PCA cumulative explained variance curve"""
    # Handle NaN values by removing rows with NaN
    valid_mask = ~np.isnan(data).any(axis=1)
    clean_data = data[valid_mask]
    
    if len(clean_data) == 0:
        return None
    
    # Determine number of components (min of samples and features)
    n_components = min(clean_data.shape[0], clean_data.shape[1], 50)  # Cap at 50 for visualization
    
    # Fit PCA
    pca = PCA(n_components=n_components)
    pca.fit(clean_data)
    
    # Calculate cumulative explained variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    
    # Calculate "1 - (AUC / Total Area)" metric
    # AUC using trapezoidal rule (cumulative variance starts from 0 at x=0)
    # We prepend 0 to cumulative_variance for the area calculation
    cumulative_with_zero = np.concatenate([[0], cumulative_variance])
    x_values = np.arange(n_components + 1)  # 0 to n_components
    auc = np.trapz(cumulative_with_zero, x_values)
    total_area = n_components * 1.0  # Rectangle: width=n_components, height=1.0
    complexity_metric = 1 - (auc / total_area)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Plot individual explained variance as bars
    ax.bar(range(1, n_components + 1), pca.explained_variance_ratio_, 
           alpha=0.6, color='#6EE368', label='Individual')
    
    # Plot cumulative explained variance as line
    ax.plot(range(1, n_components + 1), cumulative_variance, 
            'o-', color='#636EFA', linewidth=2, markersize=4, label='Cumulative')
    
    # Add horizontal lines at key thresholds
    for threshold in [0.8, 0.9, 0.95]:
        ax.axhline(y=threshold, color='#FF6692', linestyle='--', alpha=0.5, linewidth=1)
        # Find the component that reaches this threshold
        idx = np.argmax(cumulative_variance >= threshold)
        if cumulative_variance[idx] >= threshold:
            ax.annotate(f'{threshold*100:.0f}% @ PC{idx+1}', 
                       xy=(idx + 1, threshold), 
                       xytext=(idx + 3, threshold + 0.02),
                       fontsize=9, color='#FF6692')
    
    ax.set_xlabel('Principal Component', fontsize=11)
    ax.set_ylabel('Explained Variance Ratio', fontsize=11)
    ax.set_title('PCA Cumulative Explained Variance', fontsize=13, fontweight='bold')
    ax.legend(loc='center right')
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0.5, n_components + 0.5)
    ax.grid(True, alpha=0.3)
    
    # Add the complexity metric text box
    textstr = f'Complexity Index: {complexity_metric:.4f}\n(1 - AUC/TotalArea)'
    props = dict(boxstyle='round', facecolor='#FFFACD', alpha=0.8, edgecolor='#DAA520')
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='left', bbox=props)
    
    plt.tight_layout()
    
    # Convert to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf)
    
    return img


def generate_embedding(latents, cluster_name, cfg):
    
    try:
        cfg = json.loads(cfg)
    except:
        cfg = dict()
        gr.Info('UMAP config Json format error')
        return None, None, None, None
    
    local_latents = latents.select(selected_cluster=cluster_name)
    if len(local_latents.data) == 0:
        gr.Info('This Cluster is empty.')
        return None, None, None, None
    
    # Generate PCA cumulative curve before UMAP
    pca_curve_img = generate_pca_cumulative_curve(local_latents.data.copy())
    
    local_latents.build_embedding(cfg)
    Z_plt = EmbeddingScatterPlot(local_latents)
    return local_latents, Z_plt, Z_plt.plot(), pca_curve_img


def generate_local_cluster(local_latents, eps):
    try:
        cfg = json.loads(dbscan_config_template)
    except:
        gr.Info('Cluster Json format error')
        return None, None

    cfg['eps'] = eps
    local_latents.build_cluster(method='dbscan', configs=cfg)
    Z_plt = EmbeddingScatterPlot(local_latents)
    return Z_plt, Z_plt.plot()



def label_local_cluster(local_latents, cluster_id, cluster_name):
    if len(cluster_name) == 0:
        gr.Info('Name is empty')
        return 
        
    local_latents.label_cluster(cluster_id, cluster_name)  
    gr.Info(f'Name {cluster_id} as {cluster_name}')

def convert_latent_cluster_to_subtitle(storage_path, project_name, latents, mulvideo):
    
    return mulvideo.generate_subtitle(latents.cluster, latents.cluster_meta)


def import_info_from_local_latent(storage_path, project_name, latents, local_latents, mulvideo):
    try:
        start_cluster_id = latents.num_cluster
        latents.import_local_latent(local_latents)
    except:
        gr.Info('Do not use same cluster name')
        return None, update_select_cluster_list(latents), None, None, None


    fig = plt.figure(figsize=(12, 2))
    latents.plot_syllables()
    plt.tight_layout()
    # Note: fig is returned and will be displayed by Gradio, so we don't close it here
    # Gradio will handle the figure lifecycle

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
    
    Z_plt = EmbeddingScatterPlot(local_latents)
    cluster_name = ""
    for _, it in local_latents.export.items():
        cluster_name += it['name'] + '_'

    local_embedding_path = os.path.join(cluster_path, f'cluster_{cluster_name}.npz')
    Z_plt.save_named_embedding(save_path = local_embedding_path)
    return fig, update_select_cluster_list(latents), df1_path, df2_path, subtitle_path, Z_plt, Z_plt.plot_named_embedding(), local_embedding_path



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
        with gr.Column(scale=2):
            ui['select_cluster'] = gr.Dropdown(label="Select Cluster",  visible=False,interactive=True)
            ui['preset_dropdown'] = gr.Dropdown(preset_dropdown_list, value='Low-magnification objective 100', label="UMAP preset",  visible=False,interactive=True)
            ui['umap_config_text'] = gr.Textbox(label='UMAP configs', value=umap_config_template, lines=8, max_lines=8, interactive=True, visible=False)
            ui['umap_run'] = gr.Button("Generate Embedding", interactive=True, visible=False)
            ui['eps'] = gr.Number(label='epsilon-neighborhood radius', interactive=True, visible=False, value=1, step=0.1, minimum=0.1, maximum=10)
            ui['cluster_run'] = gr.Button("Generate Cluster", interactive=True, visible=False)
            ui['label_cluster_id'] = gr.Number(label='Cluster id', interactive=True, visible=False)
            ui['label_cluster_name'] = gr.Textbox(label='Cluster name', interactive=True, visible=False)
            ui['label_cluster_btn'] = gr.Button("Enter", interactive=True, visible=False)
            ui['label_cluster_submit_btn'] = gr.Button("Submit", interactive=True, visible=False)
        with gr.Column(scale=8):
            ui['pca_curve'] = gr.Image(label='PCA Cumulative Explained Variance', interactive=False, visible=False)
            ui['embedding_plot'] = gr.Image(label='Embedding', interactive=False, visible=False)
            ui['display'] = gr.Image(label='Display', interactive=False, visible=False)  
            ui['display_eps'] = gr.File(label="Display EPS", interactive=False, visible=False)
            

            
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
    ui['preset_dropdown'].select(
        fn=update_umap_config_text_with_preset,
        inputs=ui['preset_dropdown'],
        outputs=ui['umap_config_text']
    )
    ui['umap_run'].click(
        fn=generate_embedding,
        inputs=[latents, ui['select_cluster'], ui['umap_config_text']],
        outputs=[local_latents, local_embedding_plot, ui['embedding_plot'], ui['pca_curve']]
    )

    # ui['cluster_method'].select(
    #     fn=change_cluster_method_template,
    #     inputs=ui['cluster_method'],
    #     outputs=ui['eps']
    # )
    # ui['reset'].click(
    #     fn=change_cluster_method_template,
    #     inputs=ui['cluster_method'],
    #     outputs=ui['eps']
    # )
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
        inputs=[local_latents, ui['eps']],
        outputs=[local_embedding_plot, ui['embedding_plot'] ],
    )
    ui['label_cluster_btn'].click(
        fn=label_local_cluster,
        inputs=[local_latents, ui['label_cluster_id'], ui['label_cluster_name']],
    )
    ui['label_cluster_submit_btn'].click(
        fn=import_info_from_local_latent,
        inputs=[storage_path, project_name, latents, local_latents, mulvideo],
        outputs=[ui['syllables_plot'], ui['select_cluster'], ui['behavior_id_csv'], ui['behavior_time_series_csv'], ui['behavior_time_series_srt'], local_embedding_plot, ui['embedding_plot'], ui['display_eps']],
    )

    return ui