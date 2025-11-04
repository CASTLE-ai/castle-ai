"""Main UI for Castle AI - Behavior analysis tool."""

import gradio as gr

from .project_ui import create_project_ui
from .source_ui import create_source_ui
from .edit_ui import create_edit_ui
from .extract_ui import create_extract_ui
from .cluster_page_ui import create_cluster_page_ui


def toggle_tab_visibility(project_name, object_count):
    """Toggle tab UI visibility based on project selection.
    
    Args:
        project_name: Name of the selected project (None if no project)
        object_count: Number of UI elements to toggle
        
    Returns:
        list: List of Gradio updates to show/hide UI elements
    """
    is_visible = project_name is not None
    return [gr.update(visible=is_visible) for _ in range(object_count)]


def create_ui(OS_SYS, root=''):
    """Create the main Gradio UI with multiple tabs."""
    with gr.Blocks(theme=gr.themes.Soft()) as app:      
        # Project configuration tab
        with gr.Tab(label='0. Project'):
            project_ui = create_project_ui(OS_SYS, root)
            project_name = project_ui['project_name']
            storage_path = project_ui['storage_path']

        # Upload videos tab
        with gr.Tab(label='1. Upload Videos') as source_tab:
            source_ui = create_source_ui(storage_path, project_name)
            source_ui_object_count = gr.State(len(source_ui))
            source_tab.select(
                fn=toggle_tab_visibility,
                inputs=[project_ui['project_name'], source_ui_object_count],
                outputs=[v for k, v in source_ui.items()]
            )

        # Tracking ROIs tab
        with gr.Tab(label='2. Tracking ROIs') as edit_tab:
            edit_ui = create_edit_ui(storage_path, project_name)
            edit_ui_object_count = gr.State(len(edit_ui))
            edit_tab.select(
                fn=toggle_tab_visibility,
                inputs=[project_ui['project_name'], edit_ui_object_count],
                outputs=[v for k, v in edit_ui.items()]
            )

        # Extract latent features tab
        with gr.Tab(label='3. Extract Latent') as extract_tab:
            extract_ui = create_extract_ui(storage_path, project_name, extract_tab)
            extract_ui_object_count = gr.State(len(extract_ui))
            extract_tab.select(
                fn=toggle_tab_visibility,
                inputs=[project_ui['project_name'], extract_ui_object_count],
                outputs=[v for k, v in extract_ui.items()]
            )

        # Behavior analysis tab
        with gr.Tab(label='4. Behavior Microscope') as cluster_page_tab:
            cluster_ui = create_cluster_page_ui(storage_path, project_name, cluster_page_tab)
            cluster_ui_object_count = gr.State(len(cluster_ui))
            cluster_page_tab.select(
                fn=toggle_tab_visibility,
                inputs=[project_ui['project_name'], cluster_ui_object_count],
                outputs=[v for k, v in cluster_ui.items()]
            )

    return app
