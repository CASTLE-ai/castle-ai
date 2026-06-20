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
        raise gr.Error(f"Failed to create project: {str(e)}")


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


_DELETE_PROJECT_IDLE_LABEL = "Delete"
_DELETE_PROJECT_ARMED_LABEL = "⚠️ Confirm DELETE?"


def _on_open_project(project_name):
    """Persistent feedback after Open (the click also locks widgets / sets state)."""
    if not project_name:
        return gr.update()
    return gr.update(
        value=f"✅ Opened project: **{project_name}** — continue at **“1. Upload Videos”**.",
        visible=True,
    )


# Delete handlers return 6 outputs, in this order:
#   delete_btn, cancel_btn, warning, confirm_state, project_drop, status
def _on_delete_project_click(storage_path, project_name, confirmed):
    """Two-step delete project handler.

    First click (confirmed=False): arm the button — show Cancel + warning,
    change label to "⚠️ Confirm DELETE?", set state to True. Nothing is deleted.

    Second click (confirmed=True): execute deletion, refresh the project dropdown
    (so the deleted project disappears) and show a persistent status line.
    """
    if not project_name:
        gr.Warning("Please select a project to delete.")
        return (
            gr.update(value=_DELETE_PROJECT_IDLE_LABEL),
            gr.update(visible=False),
            gr.update(visible=False),
            False,
            gr.update(),                 # project_drop unchanged
            gr.update(),                 # status unchanged
        )

    if not confirmed:
        # First click — arm.
        return (
            gr.update(value=_DELETE_PROJECT_ARMED_LABEL),
            gr.update(visible=True),
            gr.update(visible=True),
            True,
            gr.update(),                 # project_drop unchanged
            gr.update(visible=False),    # clear any stale status while arming
        )

    # Second click — execute, reset, refresh dropdown, report.
    delete_project_wrapper(storage_path, project_name)
    return (
        gr.update(value=_DELETE_PROJECT_IDLE_LABEL),
        gr.update(visible=False),
        gr.update(visible=False),
        False,
        gr.update(choices=list_projects(storage_path), value=None),  # drop the deleted project
        gr.update(value=f"🗑️ Deleted project: **{project_name}**", visible=True),
    )


def _on_delete_project_cancel():
    """Cancel armed delete: reset label, hide cancel + warning, reset state."""
    return (
        gr.update(value=_DELETE_PROJECT_IDLE_LABEL),
        gr.update(visible=False),
        gr.update(visible=False),
        False,
        gr.update(),                     # project_drop unchanged
        gr.update(),                     # status unchanged
    )



def create_project_ui(OS_SYS, root=''):
    """Create the project management UI.
    
    Args:
        OS_SYS: Operating system type
        root: Root storage path for projects
        
    Returns:
        dict: Dictionary containing all UI components
    """
    # Welcome and user guidance (collapsible)
    with gr.Accordion("🏰 Welcome to Castle AI - Getting Started Guide", open=False, visible=True):
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
        # Open is the primary action (wider); Delete is a narrower secondary
        # button, aligned on the same row. Cancel appears below only when armed.
        with gr.Row():
            ui['project_open_btn'] = gr.Button(
                'Open',
                variant='primary',
                interactive=False,
                scale=4,
            )
            ui['project_delete_btn'] = gr.Button(
                'Delete',
                variant='secondary',
                interactive=False,
                scale=1,
            )
        ui['project_delete_warning'] = gr.Markdown(
            "⚠️ **This will permanently delete the project, including all videos, "
            "tracking masks, latents, and clustering results. This cannot be undone.**",
            visible=False,
        )
        with gr.Row():
            ui['project_delete_cancel_btn'] = gr.Button(
                'Cancel',
                interactive=True,
                visible=False,
            )
        # Persistent feedback after Open / Delete (gr.Info toasts are transient).
        ui['project_status'] = gr.Markdown("", visible=False)
        # Tracks whether the user has already clicked Delete once (armed → confirm).
        ui['project_delete_confirm_state'] = gr.State(False)
    
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

    # Only genuinely interactive widgets accept gr.update(interactive=...).
    # Markdown and State raise TypeError if passed that kwarg (Gradio 6.x).
    _lockable = [
        ui['storage_path'],
        ui['project_drop'],
        ui['project_open_btn'],
        ui['project_delete_btn'],
        ui['project_delete_cancel_btn'],
        ui['new_project_name'],
        ui['new_project_create_btn'],
    ]
    object_count = gr.State(len(_lockable))
    project_btn_list = [ui['project_open_btn'], ui['project_delete_btn']]
    project_btn_count = gr.State(len(project_btn_list))

    # Event handlers - Lock UI when creating/opening projects
    ui['new_project_create_btn'].click(
        fn=lock_project_page,
        inputs=object_count,
        outputs=_lockable,
    )
    ui['project_open_btn'].click(
        fn=lock_project_page,
        inputs=object_count,
        outputs=_lockable,
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
    # Persistent "Opened …" feedback (the clicks above only lock widgets / set state).
    ui['project_open_btn'].click(
        fn=_on_open_project,
        inputs=ui['project_drop'],
        outputs=ui['project_status'],
    )

    # Delete project — two-step confirmation. First click arms the button
    # (label flips to "⚠️ Confirm DELETE?" and Cancel becomes visible); second
    # click executes the delete, refreshes the dropdown, and reports status.
    _delete_outputs = [
        ui['project_delete_btn'],
        ui['project_delete_cancel_btn'],
        ui['project_delete_warning'],
        ui['project_delete_confirm_state'],
        ui['project_drop'],
        ui['project_status'],
    ]
    ui['project_delete_btn'].click(
        fn=_on_delete_project_click,
        inputs=[ui['storage_path'], ui['project_drop'], ui['project_delete_confirm_state']],
        outputs=_delete_outputs,
        queue=False,
    )
    ui['project_delete_cancel_btn'].click(
        fn=_on_delete_project_cancel,
        inputs=None,
        outputs=_delete_outputs,
        queue=False,
    )
    # Selecting a different project resets the armed state (the user may have
    # changed their mind about which project to delete).
    ui['project_drop'].change(
        fn=_on_delete_project_cancel,
        inputs=None,
        outputs=_delete_outputs,
        queue=False,
    )
    
    # Add list_project_dropdown to the returned ui dict
    ui['list_project_dropdown'] = list_project_dropdown
    
    return ui
