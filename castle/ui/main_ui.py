"""Main UI for Castle AI - Behavior analysis tool."""

import gradio as gr

from .project_ui import create_project_ui
from .source_ui import create_source_ui
from .edit_ui import create_edit_ui
from .extract_ui import create_extract_ui
from .cluster_page_ui import create_cluster_page_ui
from .annotator_ui import create_annotator_ui
from .analysis_ui import create_analysis_ui
from .export_ui import create_export_ui


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


_CASTLE_JS = """
function castleTreeClick(el, name) {
    document.querySelectorAll('.cct-node').forEach(function(n) {
        n.classList.remove('cct-selected');
    });
    el.classList.add('cct-selected');
    var wrap = document.getElementById('castle-tree-select');
    if (!wrap) return;
    var tb = wrap.querySelector('textarea') || wrap.querySelector('input');
    if (!tb) return;
    var proto = Object.getPrototypeOf(tb);
    var desc = Object.getOwnPropertyDescriptor(proto, 'value');
    if (desc && desc.set) {
        desc.set.call(tb, name);
    } else {
        tb.value = name;
    }
    tb.dispatchEvent(new Event('input', { bubbles: true }));
}
"""


def create_ui(OS_SYS, root=''):
    """Create the main Gradio UI with multiple tabs."""
    with gr.Blocks(js=_CASTLE_JS) as app:

        # Project configuration tab
        with gr.Tab(label='0. Project'):
            project_ui = create_project_ui(OS_SYS, root)
            project_name = project_ui['project_name']
            storage_path = project_ui['storage_path']

        # Application load event to populate the project dropdown automatically
        app.load(
            fn=project_ui['list_project_dropdown'],
            inputs=storage_path,
            outputs=project_ui['project_drop']
        )

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
            # The UI update logic is now handled within create_edit_ui
            _edit_ui = create_edit_ui(storage_path, project_name, edit_tab)

        # Extract latent features tab
        with gr.Tab(label='3. Extract Latent') as extract_tab:
            # The UI update logic is handled within create_extract_ui
            _extract_ui = create_extract_ui(storage_path, project_name, extract_tab)

        # Behavior analysis tab (Stage 4)
        with gr.Tab(label='4. Behavior Microscope') as cluster_page_tab:
            with gr.Tabs():
                # Sub-tab: Clustering workspace
                with gr.Tab(label='Clustering'):
                    cluster_ui, shared_states = create_cluster_page_ui(
                        storage_path, project_name, cluster_page_tab
                    )

                # Sub-tab: Cluster Annotator (A-04)
                with gr.Tab(label='Cluster Annotator') as annotator_tab:
                    _annotator_ui = create_annotator_ui(
                        storage_path, project_name, annotator_tab,
                    )

            cluster_ui_object_count = gr.State(len(cluster_ui))
            cluster_page_tab.select(
                fn=toggle_tab_visibility,
                inputs=[project_ui['project_name'], cluster_ui_object_count],
                outputs=[v for k, v in cluster_ui.items()]
            )

        # Analysis tab (Stage 5)
        with gr.Tab(label='5. Analysis') as analysis_tab:
            _analysis_ui = create_analysis_ui(
                storage_path, project_name, analysis_tab,
            )

        # Export tab (Stage 6)
        with gr.Tab(label='6. Export'):
            create_export_ui(storage_path, project_name)

    return app
