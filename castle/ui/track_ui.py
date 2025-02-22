import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Generator

import numpy as np
import gradio as gr
from natsort import natsorted

from castle import generate_aot
from castle.utils.h5_io import H5IO
from castle.utils.plot import generate_mix_image


def read_label(
    storage_path: str, project_name: str, source_video: Optional[Any]
) -> List[Dict[str, Any]]:
    """
    Read all label files for the given project and return a list of labels.

    Each label is a dict with keys:
        - index: a string identifier combining file index and video basename.
        - frame: the frame data.
        - mask: the corresponding mask.

    Args:
        storage_path: Base storage directory.
        project_name: Name of the project.
        source_video: Video source object; if None, an empty list is returned.

    Returns:
        A list of dictionaries containing label information.
    """
    if source_video is None:
        return []

    project_path = Path(storage_path) / project_name
    label_dir = project_path / "label"

    label_list = []
    # Iterate through all subdirectories in natural sorted order
    for label_folder in natsorted([p for p in label_dir.iterdir() if p.is_dir()]):
        video_basename = label_folder.name
        # Iterate through all .npz files in the folder
        for npz_file in natsorted(list(label_folder.glob("*.npz"))):
            index = npz_file.stem
            data = np.load(npz_file)
            # Expect keys 'frame' and 'mask'
            if "frame" not in data or "mask" not in data:
                continue
            frame = data["frame"]
            mask = data["mask"]
            label_list.append(
                {
                    "index": f"{index}, {video_basename}",
                    "frame": frame,
                    "mask": mask,
                }
            )
    return label_list


def read_label_to_gallery(
    storage_path: str, project_name: str, source_video: Optional[Any]
) -> Tuple[List[Dict[str, Any]]]:
    """
    Generate a gallery list based on the label data.

    Each gallery entry is a tuple (mixed_image, label_index).

    Args:
        storage_path: Base storage directory.
        project_name: Name of the project.
        source_video: Video source object.

    Returns:
        A tuple containing the original label list and a gallery list.
    """
    label_list = read_label(storage_path, project_name, source_video)
    gallery_list = [
        (generate_mix_image(label["frame"], label["mask"]), label["index"])
        for label in label_list
    ]
    return label_list


class InferenceTracker:
    """
    This class sets up a tracker for performing ROI tracking on a video,
    using previously known frames and masks as references.
    """

    def __init__(
        self,
        storage_path: str,
        project_name: str,
        source_video: Any,
        start: int,
        stop: int,
        model_aot: str,
    ) -> None:
        self.cancel: bool = False
        self.show_middle_result: bool = False
        self.model_aot: str = model_aot

        project_path = Path(storage_path) / project_name
        video_name = source_video.video_name
        self.track_dir_path: Path = project_path / "track" / video_name
        self.track_dir_path.mkdir(parents=True, exist_ok=True)

        self.source_video = source_video
        self.start = int(start)
        self.stop = int(stop)
        self.max_len = 30

        # Prepare reference knowledge from labels
        self.knowledges: List[Tuple[Any, Any]] = []
        label_list = read_label(storage_path, project_name, source_video)
        self.roi_count: int = 0
        for label in label_list:
            frame, mask = label["frame"], label["mask"]
            self.knowledges.append((frame, mask))
            # Update roi_count to be the maximum value found in masks
            self.roi_count = max(self.roi_count, int(np.max(mask)))

    def tracking(self, progress: gr.Progress) -> str:
        """
        Track ROIs over the specified frames and write each frame's result into an H5 file.

        Args:
            progress: A gradio Progress object for displaying progress.

        Returns:
            A status message ("Done" or "Cancel").
        """
        time.sleep(0.5)
        tracker = generate_aot(model_type=self.model_aot)
        mask_list_path = self.track_dir_path / "mask_list.h5"
        mask_seq = H5IO(str(mask_list_path))

        # Write video and ROI configuration settings
        mask_seq.write_config("roi_count", self.roi_count)
        mask_seq.write_config("total_frames", len(self.source_video))
        mask_seq.write_config("height", self.source_video.video_stream.height)
        mask_seq.write_config("width", self.source_video.video_stream.width)

        # Add all reference ROI frames to the tracker’s memory
        for frame, mask in self.knowledges:
            tracker.add_reference_frame(frame, mask, self.roi_count, -1)

        delta = 1 if self.start < self.stop else -1
        for i in progress.tqdm(range(self.start, self.stop + delta, delta)):
            if self.cancel:
                self.show_middle_result = False
                self.cancel = False
                del mask_seq
                return "Cancel"

            frame = self.source_video[i]
            mask = tracker.track(frame)
            tracker.update_memory(mask)
            # Save latest frame and processed mask for potential display
            self.frame = frame
            self.mask = mask.squeeze().detach().cpu().numpy().astype(np.uint8)
            mask_seq.write_mask(i, self.mask)

        self.show_middle_result = False
        del mask_seq
        return "Done"

    def set_cancel(self) -> None:
        """Set the flag to cancel tracking."""
        self.cancel = True

    def flip_show_middle_result(self) -> None:
        """Toggle the display of intermediate results."""
        self.show_middle_result = not self.show_middle_result


def init_inference_tracker(
    storage_path: str,
    project_name: str,
    source_video: Any,
    start: int,
    stop: int,
    model_aot: str,
) -> InferenceTracker:
    """
    Initialize and return an instance of InferenceTracker.
    """
    print("Initializing InferenceTracker with start:", start, "and stop:", stop)
    return InferenceTracker(storage_path, project_name, source_video, start, stop, model_aot)


def run_inference_tracker(
    tracker: InferenceTracker, progress: gr.Progress = gr.Progress()
) -> str:
    """
    Run the tracking process.

    Args:
        tracker: The InferenceTracker instance.
        progress: A gradio Progress instance.

    Returns:
        A status message.
    """
    status = tracker.tracking(progress)
    return f"{status}. From {tracker.start} to {tracker.stop}"


def click_middle_result(
    tracker: InferenceTracker,
) -> Generator[Tuple[Any, str], None, None]:
    """
    Toggle the display of intermediate results and yield updates.

    This generator repeatedly yields a tuple (mixed_image, display_mode) every second.
    """
    print("Toggling middle result display. Current state:", tracker.show_middle_result)
    tracker.flip_show_middle_result()
    while tracker.show_middle_result:
        time.sleep(1)
        mixed_image = generate_mix_image(tracker.frame, tracker.mask)
        yield mixed_image, display_middle_result_mode(tracker.show_middle_result)


def display_middle_result_mode(is_showing: bool) -> str:
    """
    Return the display mode string based on a boolean flag.

    Args:
        is_showing: True if the intermediate result is being shown.

    Returns:
        "Show" if is_showing is True, otherwise "Close".
    """
    return "Show" if is_showing else "Close"


def set_cancel(tracker: InferenceTracker) -> None:
    """Trigger cancellation of the tracking process."""
    tracker.set_cancel()


def create_track_ui(
    storage_path: str, project_name: str, source_video: Any, track_tab: gr.Tab
) -> Dict[str, Any]:
    """
    Create and return the Gradio UI components for tracking.

    Args:
        storage_path: Base storage directory.
        project_name: Name of the project.
        source_video: Video source object.
        track_tab: The Gradio Tab component where UI elements are added.

    Returns:
        A dictionary of UI elements.
    """
    ui: Dict[str, Any] = {}

    label_list_state = gr.State(None)
    tracker_state = gr.State(None)

    # with gr.Accordion("ROIs Knowledge", visible=False) as gallery_accordion:
    #     gallery = gr.Gallery(
    #         label="Label Frame",
    #         show_label=True,
    #         allow_preview=False,
    #         object_fit="contain",
    #         columns=3,
    #     )
    #     ui["gallery"] = gallery

    with gr.Accordion("Inference", open=True, visible=False) as inference_accordion:
        with gr.Row(visible=True):
            with gr.Column(scale=2):
                start_frame = gr.Slider(
                    label="Start Frame (include)",
                    minimum=0,
                    step=1,
                    maximum=1,
                    value=0,
                    interactive=True,
                    visible=False,
                )
                stop_frame = gr.Slider(
                    label="Stop Frame (include)",
                    minimum=0,
                    step=1,
                    maximum=1,
                    value=1,
                    interactive=True,
                    visible=False,
                )
                model_dropdown = gr.Dropdown(
                    choices=["r50_deaotl", "swinb_deaotl"],
                    label="Tracking Model",
                    info="ResNet-50 or Swin-transformer",
                    value="r50_deaotl",
                    interactive=True,
                )
                init_tracker_btn = gr.Button("Init Tracker", interactive=True, visible=False)
                tracking_btn = gr.Button("Tracking ROIs", interactive=True, visible=False)
                progress_text = gr.Textbox(label="Progress", visible=False)
                display_mode_text = gr.Textbox(
                    value="Close", label="Display Mode", interactive=False, visible=False
                )
                display_middle_result_btn = gr.Button(
                    "Display middle result", interactive=True, visible=False
                )
                cancel_btn = gr.Button("Cancel", interactive=True, visible=False)

            with gr.Column(scale=8):
                display = gr.Image(label="Display", interactive=False, visible=False)

    # Store UI elements into the ui dict for external access if needed.
    ui.update(
        {
            # "gallery_accordion": gallery_accordion,
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
        }
    )

    tracking_config = [start_frame, stop_frame, model_dropdown]
    # Set up the gallery from the label data.
    track_tab.select(
        fn=read_label_to_gallery,
        inputs=[storage_path, project_name, source_video],
        outputs=[label_list_state],
    )

    init_tracker_inputs = [storage_path, project_name, source_video] + tracking_config
    init_tracker_btn.click(
        fn=init_inference_tracker,
        inputs=init_tracker_inputs,
        outputs=tracker_state,
    )

    tracking_btn.click(
        fn=run_inference_tracker,
        inputs=tracker_state,
        outputs=progress_text,
    )

    display_middle_result_btn.click(
        fn=click_middle_result,
        inputs=tracker_state,
        outputs=[display, display_mode_text],
    )

    cancel_btn.click(fn=set_cancel, inputs=tracker_state)

    return ui
