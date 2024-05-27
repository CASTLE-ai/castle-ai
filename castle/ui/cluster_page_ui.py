import gradio as gr
from .cluster_input_ui import create_cluster_input_ui
from .dimension_reduction_ui import create_dimension_reduction_ui
from .dbscan_ui import create_dbscan_ui

def create_cluster_page_ui(storage_path, project_name, cluster_page_tab):
    ui = dict()

    with gr.Tab(label='Build UMAP input from video'):
        cluster_input_ui = create_cluster_input_ui(storage_path, project_name, cluster_page_tab)
        pass

    with gr.Tab(label='UMAP') as umap_tab:
        dimension_reduction_ui = create_dimension_reduction_ui(storage_path, project_name, umap_tab)
        pass

    with gr.Tab(label='DBSCAN'):
        dbscan_ui = create_dbscan_ui(storage_path, project_name)
        pass

    with gr.Tab(label='Export Cluster Information'):
        pass


    
    return ui