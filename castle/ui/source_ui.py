"""Source video management UI for Castle AI."""

import logging
import os
import gradio as gr

from ..utils.video_manager import (
    add_video_to_project,
    scan_video_directory,
    add_videos_batch
)

logger = logging.getLogger(__name__)


# UI callback functions
def upload_local_videos_wrapper(storage_path, project_name, upload_video_files):
    """Wrapper for uploading local videos with UI feedback.
    
    Args:
        storage_path: Path to the storage directory
        project_name: Name of the project
        upload_video_files: List of uploaded video files
    """
    if not upload_video_files:
        return
    if not storage_path or not project_name:
        gr.Warning("Please select a project first.")
        return
    
    for video_file in upload_video_files:
        video_name = os.path.basename(video_file.name)
        success, message = add_video_to_project(
            storage_path, 
            project_name, 
            video_file.name, 
            video_name
        )
        
        if success:
            gr.Info(message)
        else:
            gr.Warning(message)


def scan_server_videos_wrapper(video_directory):
    """Wrapper for scanning server video directory.
    
    Args:
        video_directory: Path to the video directory
        
    Returns:
        tuple: (video_list, summary_text, button_interactive)
    """
    try:
        video_list, summary = scan_video_directory(video_directory)
    except Exception as exc:  # noqa: BLE001 - surface, don't swallow
        logger.exception("scan_video_directory failed for %r", video_directory)
        gr.Warning(f"Could not read directory '{video_directory}': {exc}")
        return [], "", gr.update(interactive=False)

    if not video_list:
        # Don't leave the user staring at an empty box + disabled button with no
        # explanation (bad path / no videos / permissions).
        gr.Info(f"No videos found in '{video_directory}'. Check the folder path.")

    # Enable button only if videos are found
    button_interactive = len(video_list) > 0

    return video_list, summary, gr.update(interactive=button_interactive)


def add_server_videos_batch_wrapper(storage_path, project_name, video_directory, video_list):
    """Wrapper for adding all server videos in batch.
    
    Args:
        storage_path: Path to the storage directory
        project_name: Name of the project
        video_directory: Path to the video directory
        video_list: List of video file names to add
        
    Returns:
        tuple: (empty_video_list, reset_summary, disabled_button)
    """
    if not storage_path or not project_name:
        gr.Warning("Please select a project first.")
        return video_list, "", gr.update(interactive=False)
    if not video_list:
        gr.Warning("No videos to add")
        return video_list, "", gr.update(interactive=False)
    
    success_count, fail_count, messages = add_videos_batch(
        storage_path,
        project_name,
        video_directory,
        video_list
    )
    
    # Show results
    if success_count > 0:
        gr.Info(messages[0])
    
    # Show failed messages as warnings
    for msg in messages[1:]:
        gr.Warning(msg)
    
    # Reset UI after adding
    return [], "", gr.update(interactive=False)


def create_source_ui(storage_path, project_name):
    """Create the source video management UI.
    
    Args:
        storage_path: Gradio State for storage path
        project_name: Gradio State for project name
        
    Returns:
        dict: Dictionary containing all UI components
    """
    ui = {}
    
    # User guidance (collapsible)
    with gr.Accordion("📹 How to Upload Videos", open=False, visible=False) as ui['guidance_accordion']:
        gr.Markdown("""
        Add video files to your project using one of the two methods below:
        
        **Method 1: Upload Videos**  
        Select and upload video files directly from your desktop. Supports multiple file selection.
        
        **Method 2: Add Videos from Folder**  
        Enter a directory path on the server to batch import all videos from that folder. 
        The system will scan the directory, show you a preview of found videos, and let you add them all at once.
        
        💡 **Tip**: All videos added to a project will share the same ROI prompts during tracking.
        """)
    
    # Local video upload
    ui['upload_local_videos'] = gr.File(
        label="Upload Videos",
        file_types=["video"],
        interactive=True,
        file_count="multiple",
        visible=False
    )
    
    gr.HTML(value="<br>")
    
    # Server video batch import
    ui['video_directory'] = gr.Textbox(
        label='Add Videos from Folder',
        info='Enter the directory path containing video files',
        value='demo/',
        interactive=True,
        visible=False
    )
    
    ui['video_scan_info'] = gr.Textbox(
        label="Scan Results",
        info='Videos found in the directory',
        lines=8,
        interactive=False,
        visible=False,
        value=""
    )
    
    ui['add_videos_btn'] = gr.Button(
        'Add All Videos',
        interactive=False,
        visible=False
    )
    
    # Hidden state to store video list
    video_list_state = gr.State([])
    
    # Event handlers - Upload local videos
    ui['upload_local_videos'].upload(
        fn=upload_local_videos_wrapper,
        inputs=[storage_path, project_name, ui['upload_local_videos']]
    )
    
    # Event handlers - Scan server video directory
    ui['video_directory'].change(
        fn=scan_server_videos_wrapper,
        inputs=ui['video_directory'],
        outputs=[video_list_state, ui['video_scan_info'], ui['add_videos_btn']]
    )
    
    # Event handlers - Add all server videos
    ui['add_videos_btn'].click(
        fn=add_server_videos_batch_wrapper,
        inputs=[storage_path, project_name, ui['video_directory'], video_list_state],
        outputs=[video_list_state, ui['video_scan_info'], ui['add_videos_btn']]
    )
    
    return ui