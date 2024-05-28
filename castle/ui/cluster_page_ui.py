import gradio as gr


def create_cluster_page_ui(storage_path, project_name, cluster_page_tab):
    ui = dict()
    ui['reset'] = gr.Button("Initialize", interactive=True, visible=False)
    with gr.Row(visible=True):
        with gr.Column(scale=5):
            ui['display'] = gr.Image(label='Display', interactive=False, visible=False)
        with gr.Column(scale=5):
            ui['embedding_plot'] = gr.Image(label='Embedding', interactive=False, visible=False)
    with gr.Row(visible=True):
        with gr.Column(scale=5):
            ui['umap_config_text'] = gr.Textbox(label='UMAP configs', interactive=False, visible=False)
            ui['umap_run'] = gr.Button("Generate Embedding", interactive=True, visible=False)
            ui['cluster_method'] = gr.Dropdown(['hdbscan', 'dbscan'], label='Cluster method',
                              value='hdbscan',interactive=True)
            ui['cluster_config_text'] = gr.Textbox(label='Cluster configs', interactive=False, visible=False)
            ui['cluster_run'] = gr.Button("Cluster", interactive=True, visible=False)
        with gr.Column(scale=5):
            ui['label_cluster_id'] = gr.Number(label='Cluster id', interactive=True, visible=False)
            ui['label_cluster_name'] = gr.Textbox(label='Cluster name', interactive=True, visible=False)
            ui['label_cluster_btn'] = gr.Button("Enter", interactive=True, visible=False)
            ui['label_cluster_submit_btn'] = gr.Button("Submit", interactive=True, visible=False)
    ui['syllables_plot'] = gr.Image(label='Syllable', interactive=False, visible=False)
    
    return ui