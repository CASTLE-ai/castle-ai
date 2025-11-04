"""Edit UI for tracking ROIs in videos."""

import gradio as gr

from ..utils.video_io import ReadArray
from ..utils.video_manager import get_project_videos
from .view_ui import create_view_ui
from .label_ui import create_label_ui
from .knowledge_ui import create_knowledge_ui
from .track_ui import create_track_ui
from .post_track_ui import create_post_track_ui
from .batch_track_ui import create_batch_track_ui

# UI callback functions
def list_project_video_dropdown(storage_path, project_name):
    """List all videos in the project for dropdown selection.
    
    Args:
        storage_path: Path to the storage directory
        project_name: Name of the project
        
    Returns:
        Gradio update object with video choices
    """
    videos = get_project_videos(storage_path, project_name)
    return gr.update(choices=videos)


def unlock_select_video_edit_btn():
    """Unlock the edit button when a video is selected."""
    return gr.update(interactive=True)


def unlock_ui(object_count):
    """Unlock all UI elements."""
    return [gr.update(visible=True) for _ in range(object_count)]


def load_video_for_editing(storage_path, project_name, video_name):
    """Load video and initialize UI when edit button is clicked.
    
    Args:
        storage_path: Path to the storage directory
        project_name: Name of the project
        video_name: Name of the video file
        
    Returns:
        tuple: Video source, video name, and UI updates for sliders and displays
    """
    import os
    
    # Load video
    video_path = os.path.join(storage_path, project_name, 'sources', video_name)
    source_video = ReadArray(video_path)
    
    # Get first frame and max frame index
    first_frame = source_video[0]
    max_frame = len(source_video) - 1
    
    # Return video data and UI updates
    return (
        source_video,                               # source_video state
        video_name,                                 # select_video state
        gr.update(maximum=max_frame),              # view index_slide
        gr.update(maximum=max_frame),              # label index_slide
        gr.update(maximum=max_frame),              # track start_frame
        gr.update(maximum=max_frame, value=max_frame),  # track stop_frame
        first_frame,                               # view display_view
        first_frame,                               # label display_view
        first_frame                                # label select_frame
    )


def collapse_source_detail():
    """Collapse the source video selection accordion."""
    return gr.update(open=False)


def create_edit_ui(storage_path, project_name):
    """Create the edit UI for ROI tracking.
    
    Args:
        storage_path: Gradio State for storage path
        project_name: Gradio State for project name
        
    Returns:
        dict: Dictionary containing all UI components
    """
    ui = {}
    source_video = gr.State(None)
    ui['select_video'] = gr.State(None)
    
    # User guidance (collapsible)
    with gr.Accordion("📋 ROI Tracking Workflow Guide", open=False, visible=False) as ui['guidance_accordion']:
        gr.Markdown("""
        ### 🎯 Phase 1: Single Video Tracking (Build Your ROI Prompts)
        
        **Step 1: Label Initial ROI**  
        Start with the **first frame** of your **first video**. In the **Label ROI** tab, annotate the ROI you want to track.
        
        **Step 2: Track and Monitor**  
        Go to the **Tracking** tab and start tracking. **Watch the real-time progress carefully!**  
        - If tracking looks good → Let it continue
        - If you see errors → **Cancel immediately** (don't waste time on bad tracking)
        
        **Step 3: Fix Errors Iteratively**  
        - Annotate the frames where tracking failed (in **Label ROI** tab)
        - Adjust the **start frame index** in Tracking settings
        - Re-run tracking from that point until completion
        - Repeat until the entire video is tracked successfully
        
        ### 🚀 Phase 2: Batch Video Tracking (Scale Up)
        
        Once your ROI prompts successfully cover the tracking requirements for one video, you can use **Batch Video Tracking** 
        to process multiple videos efficiently with the same prompts.
        
        ### 💡 Best Practices
        
        **ROI Prompts Guidelines:**
        - **Diversity = Stability**: Varied examples improve tracking robustness across different scenarios
        - **Fewer = Faster**: Fewer prompts mean faster execution speed
        - **Sweet Spot**: Recommended range is **5-30 prompts** for optimal balance
        
        **Pro Tips:**
        - Monitor tracking in real-time to catch issues early
        - Quality matters more than quantity - well-chosen prompts work better
        
        **⚠️ Note on Real-Time Display:**  
        When viewing intermediate results during tracking, you may notice the frame and mask appear to be off by one frame. 
        This is **normal** - the system displays the latest frame and ROI simultaneously, but they may not always share 
        the exact same frame index due to processing timing. If you want to verify tracking accuracy, check the results 
        in the **View** tab after tracking completes.
        """)

    with gr.Tab(label='Single Video Tracking') as single_tracking_tab:
        # Video selection
        with gr.Accordion('Select Source Video', open=True, visible=False) as ui['source_accordion']:
            ui['select_video_drop'] = gr.Dropdown(
                label="Select Video",
                interactive=True,
                visible=False
            )
            ui['select_video_edit_btn'] = gr.Button(
                'Edit',
                interactive=False,
                visible=False
            )

        # Label ROI tab - Create and label ROI prompts
        with gr.Tab(label='Label ROI'):
            label_ui = create_label_ui(storage_path, project_name, source_video)
        
        with gr.Tab(label='ROI Prompts') as knowledge_tab:
            knowledge_ui = create_knowledge_ui(storage_path, project_name, knowledge_tab)
        
        # Tracking tab - Run ROI tracking across frames
        with gr.Tab(label='Tracking') as track_tab:
            track_ui = create_track_ui(storage_path, project_name, source_video, track_tab)
        
        # View tab - Preview video frames
        with gr.Tab(label='View'):
            view_ui = create_view_ui(storage_path, project_name, source_video)
        
        # Analysis tab - Review tracking results
        with gr.Tab(label='Analysis'):
            post_track_ui = create_post_track_ui(storage_path, project_name, source_video)
    
    with gr.Tab(label='Batch Videos Tracking') as batch_tracking_tab:
        batch_tracking_ui = create_batch_track_ui(storage_path, project_name)

    # Count UI elements for visibility management
    view_ui_object_count = gr.State(len(view_ui))
    label_ui_object_count = gr.State(len(label_ui))
    knowledge_ui_object_count = gr.State(len(knowledge_ui))
    track_ui_object_count = gr.State(len(track_ui))
    post_track_ui_object_count = gr.State(len(post_track_ui))
    batch_tracking_ui_object_count = gr.State(len(batch_tracking_ui))

    # Event handlers - Load video list when dropdown is focused
    ui['select_video_drop'].focus(
        fn=list_project_video_dropdown,
        inputs=[storage_path, project_name],
        outputs=ui['select_video_drop']
    )
    
    # Event handlers - Enable edit button when video is selected
    ui['select_video_drop'].select(
        fn=unlock_select_video_edit_btn,
        outputs=ui['select_video_edit_btn']
    )
    
    # Event handlers - Show all tabs when edit button is clicked
    ui['select_video_edit_btn'].click(
        fn=unlock_ui,
        inputs=view_ui_object_count,
        outputs=[v for k, v in view_ui.items()]
    )
    ui['select_video_edit_btn'].click(
        fn=unlock_ui,
        inputs=label_ui_object_count,
        outputs=[v for k, v in label_ui.items()]
    )
    ui['select_video_edit_btn'].click(
        fn=unlock_ui,
        inputs=knowledge_ui_object_count,
        outputs=[v for k, v in knowledge_ui.items()]
    )
    ui['select_video_edit_btn'].click(
        fn=unlock_ui,
        inputs=track_ui_object_count,
        outputs=[v for k, v in track_ui.items()]
    )
    ui['select_video_edit_btn'].click(
        fn=unlock_ui,
        inputs=post_track_ui_object_count,
        outputs=[v for k, v in post_track_ui.items()]
    )
    ui['select_video_edit_btn'].click(
        fn=unlock_ui,
        inputs=batch_tracking_ui_object_count,
        outputs=[v for k, v in batch_tracking_ui.items()]
    )

    # Event handlers - Load video and initialize UI
    ui['select_video_edit_btn'].click(
        fn=load_video_for_editing,
        inputs=[storage_path, project_name, ui['select_video_drop']],
        outputs=[
            source_video,
            ui['select_video'],
            view_ui['index_slide'],
            label_ui['index_slide'],
            track_ui['start_frame'],
            track_ui['stop_frame'],
            view_ui['display_view'],
            label_ui['display_view'],
            label_ui['select_frame']
        ]
    )
    
    # Event handlers - Collapse video selection after editing starts
    ui['select_video_edit_btn'].click(
        fn=collapse_source_detail,
        outputs=ui['source_accordion']
    )
    
    return ui