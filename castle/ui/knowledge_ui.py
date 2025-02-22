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
                    "index": f"{index}, {video_basename}", # this is display name, not index
                    "frame": frame,
                    "mask": mask,
                }
            )
    return label_list


def read_label_to_gallery(
    storage_path: str, project_name: str, source_video: Optional[Any]
) -> Tuple[List[Dict[str, Any]], List[Tuple[Any, str]]]:
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
    return label_list, gallery_list

import os

def delete_file_if_exists(file_path):
    """如果檔案存在則刪除"""
    if os.path.exists(file_path):
        os.remove(file_path)
        gr.Info(f"檔案 {file_path} 已刪除")
    else:
        gr.Info(f"檔案 {file_path} 不存在")


def delete_selected(storage_path, project_name, source_video, label_list, index):
    # print(label_list[index], index)
    target_file = label_list[index][1] # this index is display name, not index
    project_path = Path(storage_path) / project_name
    label_dir = os.path.join(project_path, "label")
    frame_index, video_name = target_file.split(', ')
    delete_file_if_exists(os.path.join(label_dir, video_name, frame_index) + '.npz')
    return read_label_to_gallery(storage_path, project_name, source_video)[1]


def get_select_index(evt: gr.SelectData):
        return evt.index

def create_knowledge_ui(
    storage_path: str, project_name: str, source_video: Any, knowledge_tab: gr.Tab
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

    selected_image = gr.State(None)
    label_list_state = gr.State(None)

    gallery = gr.Gallery(
        label="Label Frame",
        show_label=True,
        allow_preview=False,
        object_fit="contain",
        interactive=False,
        columns=3,
    )
    ui["gallery"] = gallery
    delete_selected_btn = gr.Button("Delete", interactive=True, visible=False)

    ui.update(
        {
            "delete_selected_btn": delete_selected_btn,

        }
    )


    # Set up the gallery from the label data.
    knowledge_tab.select(
        fn=read_label_to_gallery,
        inputs=[storage_path, project_name, source_video],
        outputs=[label_list_state, gallery],
    )
    

    gallery.select(get_select_index, None, selected_image)

    delete_selected_btn.click(
        fn=delete_selected,
        inputs=[storage_path, project_name, source_video, gallery, selected_image],
        outputs=gallery,
    )





    return ui
