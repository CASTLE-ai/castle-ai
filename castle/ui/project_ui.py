"""Project UI components for Castle AI."""

import logging

import gradio as gr

from ..utils.project_manager import (
    list_projects,
    create_project,
    delete_project,
    generate_default_project_name,
    initialize_storage
)

logger = logging.getLogger(__name__)


# UI callback functions
def lock_project_page(object_count):
    """Lock all UI elements on the project page."""
    return [gr.update(interactive=False) for _ in range(object_count)]


def unlock_project_btn(object_count):
    """Unlock old project buttons."""
    return [gr.update(interactive=True) for _ in range(object_count)]


def list_project_dropdown(storage_path):
    """Update dropdown with available projects."""
    if not storage_path:
        return gr.update(choices=[])
    projects = list_projects(storage_path)
    return gr.update(choices=projects)


def create_new_project_wrapper(storage_path, project_name):
    """Wrapper for creating a new project with error handling."""
    if not storage_path or not project_name:
        gr.Warning("Storage path and project name are required.")
        return
    try:
        create_project(storage_path, project_name)
        gr.Info(f"Created project: {project_name}")
    except FileExistsError as e:
        gr.Warning(str(e))
    except Exception as e:
        logger.exception("Failed to create project %r", project_name)
        gr.Error(f"Failed to create project: {str(e)}")


def default_new_project_name_ui(new_project_name):
    """Generate default project name for UI."""
    default_name = generate_default_project_name(new_project_name)
    return gr.update(value=default_name)


def set_project_name(project_name):
    """Set the current project name."""
    return project_name


def delete_project_wrapper(storage_path, project_name):
    """Wrapper for deleting a project with UI feedback."""
    if not storage_path or not project_name:
        gr.Warning("Storage path and project name are required.")
        return
    if delete_project(storage_path, project_name):
        gr.Info(f"Deleted project: {project_name}")
    else:
        gr.Warning(f"Project not found: {project_name}")



def create_project_ui(OS_SYS, root=''):
    """Create the project management UI.
    
    Args:
        OS_SYS: Operating system type
        root: Root storage path for projects
        
    Returns:
        dict: Dictionary containing all UI components
    """
    # Welcome and user guidance (collapsible)
    with gr.Accordion("🏰 Welcome to Castle AI - Getting Started Guide", open=True, visible=True):
        gr.Markdown("""
        **Castle AI** helps you analyze animal behavior in videos using AI-powered Region of Interest (ROI) tracking 
        and clustering. Follow this workflow to get the best results:
        
        ### 🔄 Complete Workflow
        
        **Step 0: Create or Open a Project**  
        Start by creating a new project or opening an existing one below. Each project maintains 
        its own set of ROI prompts, videos, and analysis results.
        
        **Step 1: Upload Videos** → *Tab: Upload Videos*  
        Add video files to your project. You can upload local videos or import videos from a server directory.
        
        **Step 2: Track ROIs** → *Tab: Tracking ROIs*  
        Create ROI diverse prompts, run tracking across frames, and refine iteratively.
        
        **Step 3: Extract Latent Features** → *Tab: Extract Latent*  
        Extract latent feature representations from tracked ROIs. These features capture the essential characteristics 
        of behaviors for clustering analysis.
        
        **Step 4: Analyze Behaviors** → *Tab: Behavior Microscope*  
        Explore and cluster behaviors using the extracted features. Discover behavior patterns and visualize 
        behavioral diversity in your videos.
        
        ---
        
        💡 **Tip**: All videos within a project share the same ROI prompts, making it easy to apply consistent 
        tracking across multiple recordings.
        """)
    # Initialize storage directory
    storage_path = initialize_storage(root)
    
    ui = {}
    
    # Storage path input
    with gr.Accordion('Change Storage Location (Optional)', open=False, visible=True):
        ui['storage_path'] = gr.Textbox(
            label='Storage Location',
            info='The location which stores all projects',
            value=storage_path,
            interactive=True
        )
    
    # Open existing project tab
    with gr.Tab(label='Open Project'):
        ui['project_drop'] = gr.Dropdown(
            label="Open Project",
            interactive=True
        )
        with gr.Row(visible=True):
            with gr.Column(scale=9):
                ui['project_open_btn'] = gr.Button(
                    'Open',
                    interactive=False
                )
            with gr.Column(scale=1):
                ui['project_delete_btn'] = gr.Button(
                    'Delete',
                    interactive=False
                )
    
    # Create new project tab
    with gr.Tab(label='New Project'):
        ui['new_project_name'] = gr.Textbox(
            label='New Project Name',
            interactive=True
        )
        ui['new_project_create_btn'] = gr.Button(
            'Create',
            interactive=True
        )
    
    # State variables
    ui['project_name'] = gr.State(None)
    
    # Count UI elements for bulk updates
    object_count = gr.State(len(ui))
    project_btn_list = [ui['project_open_btn'], ui['project_delete_btn']]
    project_btn_count = gr.State(len(project_btn_list))
    
    # Event handlers - Lock UI when creating/opening projects
    ui['new_project_create_btn'].click(
        fn=lock_project_page,
        inputs=object_count,
        outputs=[v for k, v in ui.items()]
    )
    ui['project_open_btn'].click(
        fn=lock_project_page,
        inputs=object_count,
        outputs=[v for k, v in ui.items()]
    )
    
    # Enable buttons when project is selected
    ui['project_drop'].select(
        fn=unlock_project_btn,
        inputs=project_btn_count,
        outputs=project_btn_list
    )
    
    # Create new project
    ui['new_project_create_btn'].click(
        fn=create_new_project_wrapper,
        inputs=[ui['storage_path'], ui['new_project_name']]
    )
    
    # Generate default project name on focus
    ui['new_project_name'].focus(
        fn=default_new_project_name_ui,
        inputs=ui['new_project_name'],
        outputs=ui['new_project_name']
    )
    
    # Set project name when creating or opening
    ui['new_project_create_btn'].click(
        fn=set_project_name,
        inputs=ui['new_project_name'],
        outputs=ui['project_name']
    )
    ui['project_open_btn'].click(
        fn=set_project_name,
        inputs=ui['project_drop'],
        outputs=ui['project_name']
    )
    
    # Delete project
    ui['project_delete_btn'].click(
        fn=delete_project_wrapper,
        inputs=[ui['storage_path'], ui['project_drop']]
    )
    
    # Add list_project_dropdown to the returned ui dict
    ui['list_project_dropdown'] = list_project_dropdown
    
    return ui
