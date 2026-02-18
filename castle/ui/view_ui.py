"""View UI for displaying video frames and ROI masks."""

import logging

import gradio as gr

from ..utils.roi_manager import get_frame_display, save_frame_to_knowledge

logger = logging.getLogger(__name__)


# UI callback functions
def update_frame_display(storage_path, project_name, source_video, frame_index, display_mode):
    """Update frame display based on selected mode.
    
    Args:
        storage_path: Path to the storage directory
        project_name: Name of the project
        source_video: Video array object
        frame_index: Index of the frame to display
        display_mode: Display mode ('Image', 'Image & Mask', 'Mask')
        
    Returns:
        numpy.ndarray: Frame image to display
    """
    if source_video is None or not storage_path or not project_name:
        return None
    return get_frame_display(storage_path, project_name, source_video, frame_index, display_mode)


def add_frame_to_knowledge_wrapper(storage_path, project_name, source_video, frame_index):
    """Wrapper for adding frame to knowledge base with UI feedback.
    
    Args:
        storage_path: Path to the storage directory
        project_name: Name of the project
        source_video: Video array object
        frame_index: Index of the frame to save
    """
    success, message = save_frame_to_knowledge(storage_path, project_name, source_video, frame_index)
    
    if success:
        gr.Info(message)
    else:
        gr.Warning(message)


def update_frame_step(step_value):
    """Update frame slider step size.
    
    Args:
        step_value: Step size for the frame slider
        
    Returns:
        Gradio update object
    """
    if step_value is None:
        return gr.update()
    return gr.update(step=int(step_value))


def create_view_ui(storage_path, project_name, source_video):
    """Create the view UI for displaying video frames.
    
    Args:
        storage_path: Gradio State for storage path
        project_name: Gradio State for project name
        source_video: Gradio State for video source
        
    Returns:
        dict: Dictionary containing all UI components
    """
    ui = {}
    
    with gr.Row(visible=True):
        # Controls column
        with gr.Column(scale=2):
            ui['display_mode_drop'] = gr.Dropdown(
                choices=["Image", "Image & Mask", "Mask"],
                label="Display Mode",
                value="Image",
                interactive=True,
                visible=False
            )
            
            ui['step_frame'] = gr.Dropdown(
                choices=["1", "5", "10", "100", "1000", "10000"],
                label="Frame Step Size",
                value="1",
                interactive=True,
                visible=False
            )
            
            ui['adding_to_knowledge_btn'] = gr.Button(
                'Add to ROI Prompts',
                interactive=True,
                visible=False
            )
        
        # Display column
        with gr.Column(scale=8):
            ui['display_view'] = gr.Image(
                label='Display',
                interactive=False,
                visible=False
            )
            ui['index_slide'] = gr.Slider(
                label="Frame",
                minimum=0,
                step=1,
                maximum=1,
                value=0,
                interactive=True,
                visible=False
            )
    
    # Event handlers - Update display when frame index or mode changes
    ui['index_slide'].change(
        fn=update_frame_display,
        inputs=[storage_path, project_name, source_video, ui['index_slide'], ui['display_mode_drop']],
        outputs=ui['display_view']
    )
    
    ui['display_mode_drop'].change(
        fn=update_frame_display,
        inputs=[storage_path, project_name, source_video, ui['index_slide'], ui['display_mode_drop']],
        outputs=ui['display_view']
    )
    
    # Event handlers - Update slider step size
    ui['step_frame'].change(
        fn=update_frame_step,
        inputs=ui['step_frame'],
        outputs=ui['index_slide']
    )
    
    # Event handlers - Add frame to knowledge base
    ui['adding_to_knowledge_btn'].click(
        fn=add_frame_to_knowledge_wrapper,
        inputs=[storage_path, project_name, source_video, ui['index_slide']]
    )
    
    return ui
