import gradio as gr

def create_dbscan_ui(storage_path, project_name):
    ui = dict()

    with gr.Row(visible=True):
        with gr.Column(scale=2):
            gr.Markdown('# DBSCAN')
            ui['eps'] = gr.Textbox(label='eps')
            ui['min_sample'] = gr.Textbox(label='min_sample')
            ui['create_cluster'] = gr.Button('Cluster')
            
            gr.Markdown('# Merge')
            ui['merge_source_id'] = gr.Dropdown(label='merge_source_id')
            ui['merge_target_id'] = gr.Dropdown(label='merge_target_id')
            ui['merge_cluster'] = gr.Button('Merge')

            gr.Markdown('# Save')
            ui['save_source_id'] = gr.Dropdown(label='save_source_id')
            ui['save_cluster_name'] = gr.Textbox(label='cluster name')
            ui['save_cluster'] = gr.Button('Save')

        with gr.Column(scale=4):
            ui['info_dr_img'] = gr.Plot(label='info_dr_img')
            ui['cluster_hist'] = gr.Plot(label='cluster_hist')
            ui['click_dr_img'] = gr.Image(label='click_dr_img')

        with gr.Column(scale=4):
            ui['frame'] = gr.Image(label='Frame')
            ui['bar_plot_color_with_cluster'] = gr.Plot(label='bar_plot_color_with_cluster')
            ui['select_video'] = gr.Dropdown(label='select_video')
            ui['select_frame_slider'] = gr.Slider(label='select_frame_slider')
            


    return ui