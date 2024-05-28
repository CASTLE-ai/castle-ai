import gradio as gr


from .project_ui import create_project_ui
from .source_ui import create_source_ui
from .edit_ui import create_edit_ui
from .extract_ui import create_extract_ui
from .cluster_page_ui import create_cluster_page_ui

def show_ui(project_name, object_count):
    if not project_name == None:
        return [gr.update(visible=True) for i in range(object_count)]
    else:
        return [gr.update(visible=False) for i in range(object_count)]


def create_ui(OS_SYS, root=''):
    
    with gr.Blocks(theme=gr.themes.Soft()) as app:
        with gr.Tab(label='Project'):
            project_ui = create_project_ui(OS_SYS, root)
            project_name = project_ui['project_name']
            storage_path = project_ui['storage_path']

        
        with gr.Tab(label='Upload Videos') as source_tab:
            source_ui = create_source_ui(storage_path, project_name)
            source_ui_object_count = gr.State(len(source_ui))
            source_tab.select(
                fn=show_ui, 
                inputs=[project_ui['project_name'], source_ui_object_count], 
                outputs=[v for k, v in source_ui.items()]
            )

        with gr.Tab(label='Tracking ROIs') as edit_tab:
            edit_ui = create_edit_ui(storage_path, project_name)
            edit_ui_object_count = gr.State(len(edit_ui))
            edit_tab.select(
                fn=show_ui, 
                inputs=[project_ui['project_name'], edit_ui_object_count], 
                outputs=[v for k, v in edit_ui.items()]
            )

        with gr.Tab(label='Extract Latent') as extract_tab:
            extract_ui = create_extract_ui(storage_path, project_name, extract_tab)
            extract_ui_object_count = gr.State(len(extract_ui))
            extract_tab.select(
                fn=show_ui, 
                inputs=[project_ui['project_name'], extract_ui_object_count], 
                outputs=[v for k, v in extract_ui.items()]
            )
            pass

        with gr.Tab(label='Latent Explorer') as cluster_page_tab:
            cluster_ui = create_cluster_page_ui(storage_path, project_name, cluster_page_tab)
            cluster_ui_object_count = gr.State(len(cluster_ui))
            cluster_page_tab.select(
                fn=show_ui, 
                inputs=[project_ui['project_name'], cluster_ui_object_count], 
                outputs=[v for k, v in cluster_ui.items()]
            )
            pass

        # with gr.Tab(label='Export'):
        #     pass

    return app

