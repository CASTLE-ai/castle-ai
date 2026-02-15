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
    """List all videos in the project for dropdown selection."""
    videos = get_project_videos(storage_path, project_name)
    return gr.update(choices=videos)


def unlock_select_video_edit_btn():
    """Unlock the edit button when a video is selected."""
    return gr.update(interactive=True)


def handle_edit_click(storage_path, project_name, video_name, view_ui_count, label_ui_count, knowledge_ui_count, track_ui_count, post_track_ui_count):
    """
    Handles all actions when the 'Edit' button is clicked.
    Merges logic from load_video_for_editing, unlock_ui, and collapse_source_detail
    to prevent deadlocks from multiple concurrent Gradio events.
    """
    import os
    
    # 1. Logic from load_video_for_editing
    video_path = os.path.join(storage_path, project_name, 'sources', video_name)
    source_video = ReadArray(video_path)
    first_frame = source_video[0]
    max_frame = len(source_video) - 1

    load_video_outputs = [
        source_video,
        video_name,
        gr.update(maximum=max_frame),
        gr.update(maximum=max_frame),
        gr.update(maximum=max_frame),
        gr.update(maximum=max_frame, value=max_frame),
        first_frame,
        first_frame,
        first_frame
    ]

    # 2. Logic from collapse_source_detail
    collapse_output = [gr.update(open=False)]

    # 3. Logic from unlock_ui
    view_ui_updates = [gr.update(visible=True) for _ in range(view_ui_count)]
    label_ui_updates = [gr.update(visible=True) for _ in range(label_ui_count)]
    knowledge_ui_updates = [gr.update(visible=True) for _ in range(knowledge_ui_count)]
    track_ui_updates = [gr.update(visible=True) for _ in range(track_ui_count)]
    post_track_ui_updates = [gr.update(visible=True) for _ in range(post_track_ui_count)]
    
    unlock_outputs = (
        view_ui_updates +
        label_ui_updates +
        knowledge_ui_updates +
        track_ui_updates +
        post_track_ui_updates
    )
    
    return tuple(load_video_outputs + collapse_output + unlock_outputs)


def create_edit_ui(storage_path, project_name, edit_tab):
    """Create the edit UI for ROI tracking."""
    ui = {}
    source_video = gr.State(None)
    ui['select_video'] = gr.State(None)
    
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

    with gr.Tab(label='Single Video Tracking') as _single_tracking_tab:
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

        with gr.Tab(label='Label ROI'):
            label_ui = create_label_ui(storage_path, project_name, source_video)
        
        with gr.Tab(label='ROI Prompts') as knowledge_tab:
            knowledge_ui = create_knowledge_ui(storage_path, project_name, knowledge_tab)
        
        with gr.Tab(label='Tracking') as track_tab:
            track_ui = create_track_ui(storage_path, project_name, source_video, track_tab)
        
        with gr.Tab(label='View'):
            view_ui = create_view_ui(storage_path, project_name, source_video)
        
        with gr.Tab(label='Analysis'):
            post_track_ui = create_post_track_ui(storage_path, project_name, source_video)
    
    with gr.Tab(label='Batch Videos Tracking') as batch_tracking_tab:
        batch_tracking_ui, batch_tracking_states = create_batch_track_ui(storage_path, project_name, batch_tracking_tab)

    view_ui_object_count = gr.State(len(view_ui))
    label_ui_object_count = gr.State(len(label_ui))
    knowledge_ui_object_count = gr.State(len(knowledge_ui))
    track_ui_object_count = gr.State(len(track_ui))
    post_track_ui_object_count = gr.State(len(post_track_ui))
    
    all_ui_to_show_on_select = [
        ui['guidance_accordion'],
        ui['source_accordion'],
        ui['select_video_drop'],
        ui['select_video_edit_btn']
    ]

    def show_edit_ui(project_name):
        is_visible = project_name is not None
        return [gr.update(visible=is_visible)] * len(all_ui_to_show_on_select)

    (
        edit_tab.select(
            fn=show_edit_ui,
            inputs=[project_name],
            outputs=all_ui_to_show_on_select
        )
        .then(
            fn=list_project_video_dropdown,
            inputs=[storage_path, project_name],
            outputs=ui['select_video_drop']
        )
    )

    ui['select_video_drop'].select(
        fn=unlock_select_video_edit_btn,
        outputs=ui['select_video_edit_btn']
    )

    edit_button_inputs = [
        storage_path,
        project_name,
        ui['select_video_drop'],
        view_ui_object_count,
        label_ui_object_count,
        knowledge_ui_object_count,
        track_ui_object_count,
        post_track_ui_object_count
    ]
    
    edit_button_outputs = [
        source_video,
        ui['select_video'],
        view_ui['index_slide'],
        label_ui['index_slide'],
        track_ui['start_frame'],
        track_ui['stop_frame'],
        view_ui['display_view'],
        label_ui['display_view'],
        label_ui['select_frame'],
        ui['source_accordion'],
    ] + list(view_ui.values()) + list(label_ui.values()) + list(knowledge_ui.values()) + list(track_ui.values()) + list(post_track_ui.values())

    ui['select_video_edit_btn'].click(
        fn=handle_edit_click,
        inputs=edit_button_inputs,
        outputs=edit_button_outputs
    )
    
    return ui
