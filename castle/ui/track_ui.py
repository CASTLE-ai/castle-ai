"""Tracking UI for ROI tracking in videos."""

import time
from typing import Any, Dict, Generator, Tuple

import gradio as gr

from ..utils.tracking_manager import read_roi_labels, ROITracker
from ..utils.plot import generate_mix_image


# UI callback functions
def load_label_list(storage_path: str, project_name: str, source_video: Any) -> list:
    """Load ROI labels when track tab is selected.
    
    Args:
        storage_path: Path to the storage directory
        project_name: Name of the project
        source_video: Video source object
        
    Returns:
        List of label dictionaries
    """
    if source_video is None:
        return []
    
    return read_roi_labels(storage_path, project_name)


def initialize_tracker(
    storage_path: str,
    project_name: str,
    source_video: Any,
    start_frame: int,
    stop_frame: int,
    model_type: str,
) -> ROITracker:
    """Initialize ROI tracker with configuration.
    
    Args:
        storage_path: Path to the storage directory
        project_name: Name of the project
        source_video: Video source object
        start_frame: Starting frame index
        stop_frame: Stopping frame index
        model_type: Tracking model type
        
    Returns:
        Initialized ROITracker instance
    """
    print(f"Initializing ROITracker: frames {start_frame} to {stop_frame}, model: {model_type}")
    return ROITracker(storage_path, project_name, source_video, start_frame, stop_frame, model_type)


def run_tracking(tracker: ROITracker, progress=gr.Progress(track_tqdm=True)) -> str:
    """Run the ROI tracking process.
    
    Args:
        tracker: The ROITracker instance
        progress: Gradio Progress instance for displaying progress (auto-injected)
        
    Returns:
        Status message with frame range
    """
    status = tracker.track(progress)
    return f"{status}. Tracked from frame {tracker.start_frame} to {tracker.stop_frame}"


def toggle_intermediate_display(tracker: ROITracker) -> Generator[Tuple[Any, str], None, None]:
    """Toggle and stream intermediate tracking results.
    
    Args:
        tracker: The ROITracker instance
        
    Yields:
        Tuple of (mixed_image, display_mode_text)
    """
    print(f"Toggling intermediate display. Current: {tracker.show_middle_result}")
    tracker.toggle_display_mode()
    
    while tracker.show_middle_result:
        time.sleep(1)
        frame, mask = tracker.get_current_result()
        if frame is not None and mask is not None:
            mixed_image = generate_mix_image(frame, mask)
            display_mode = "Show" if tracker.show_middle_result else "Close"
            yield mixed_image, display_mode


def cancel_tracking(tracker: ROITracker) -> None:
    """Cancel the ongoing tracking process.
    
    Args:
        tracker: The ROITracker instance
    """
    tracker.cancel_tracking()


def create_track_ui(
    storage_path: str, project_name: str, source_video: Any, track_tab: gr.Tab
) -> Dict[str, Any]:
    """Create the tracking UI components.
    
    Args:
        storage_path: Gradio State for storage path
        project_name: Gradio State for project name
        source_video: Gradio State for video source
        track_tab: The Gradio Tab component
        
    Returns:
        Dictionary containing all UI components
    """
    ui = {}
    
    # State variables
    label_list_state = gr.State(None)
    tracker_state = gr.State(None)
    
    # Tracking configuration and controls
    with gr.Accordion("ROI Tracking Settings", open=True, visible=False) as inference_accordion:
        with gr.Row(visible=True):
            # Controls column
            with gr.Column(scale=2):
                start_frame = gr.Slider(
                    label="Start Frame (inclusive)",
                    minimum=0,
                    step=1,
                    maximum=1,
                    value=0,
                    interactive=True,
                    visible=False
                )
                stop_frame = gr.Slider(
                    label="Stop Frame (inclusive)",
                    minimum=0,
                    step=1,
                    maximum=1,
                    value=1,
                    interactive=True,
                    visible=False
                )
                model_dropdown = gr.Dropdown(
                    choices=["r50_deaotl", "sam2"],
                    label="Tracking Model",
                    info="ResNet-50",
                    value="r50_deaotl",
                    interactive=True
                )
                init_tracker_btn = gr.Button(
                    "Apply parameters",
                    interactive=True,
                    visible=False
                )
                tracking_btn = gr.Button(
                    "Start Tracking",
                    interactive=True,
                    visible=False
                )
                progress_text = gr.Textbox(
                    label="Progress",
                    visible=False
                )
                display_mode_text = gr.Textbox(
                    value="Close",
                    label="Display Mode",
                    interactive=False,
                    visible=False
                )
                display_middle_result_btn = gr.Button(
                    "Show Intermediate Results",
                    interactive=True,
                    visible=False
                )
                cancel_btn = gr.Button(
                    "Cancel",
                    interactive=True,
                    visible=False
                )
            
            # Display column
            with gr.Column(scale=8):
                display = gr.Image(
                    label="Tracking Display",
                    interactive=False,
                    visible=False
                )

    # Store UI elements in dictionary
    ui.update({
        "inference_accordion": inference_accordion,
        "start_frame": start_frame,
        "stop_frame": stop_frame,
        "model_dropdown": model_dropdown,
        "init_tracker_btn": init_tracker_btn,
        "tracking_btn": tracking_btn,
        "progress_text": progress_text,
        "display_mode_text": display_mode_text,
        "display_middle_result_btn": display_middle_result_btn,
        "cancel_btn": cancel_btn,
        "display": display,
    })
    
    # Event handlers - Load labels when tab is selected
    track_tab.select(
        fn=load_label_list,
        inputs=[storage_path, project_name, source_video],
        outputs=[label_list_state]
    )
    
    # Event handlers - Initialize tracker
    tracking_config = [start_frame, stop_frame, model_dropdown]
    init_tracker_inputs = [storage_path, project_name, source_video] + tracking_config
    init_tracker_btn.click(
        fn=initialize_tracker,
        inputs=init_tracker_inputs,
        outputs=tracker_state
    )
    
    # Event handlers - Run tracking
    tracking_btn.click(
        fn=run_tracking,
        inputs=tracker_state,
        outputs=progress_text
    )
    
    # Event handlers - Toggle intermediate results display
    display_middle_result_btn.click(
        fn=toggle_intermediate_display,
        inputs=tracker_state,
        outputs=[display, display_mode_text]
    )
    
    # Event handlers - Cancel tracking
    cancel_btn.click(
        fn=cancel_tracking,
        inputs=tracker_state
    )
    
    return ui
