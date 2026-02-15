import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gradio as gr

from castle.utils.plot import generate_mix_image
from castle.utils.tracking_manager import read_roi_labels


def read_label_to_gallery(
    storage_path: str, project_name: str
) -> Tuple[List[Dict[str, Any]], List[Tuple[Any, str]]]:
    """
    Generate a gallery list based on the label data.

    Each gallery entry is a tuple (mixed_image, label_index).

    Args:
        storage_path: Base storage directory.
        project_name: Name of the project.

    Returns:
        A tuple containing the original label list and a gallery list.
    """
    label_list = read_roi_labels(storage_path, project_name)
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


def delete_selected(storage_path, project_name, label_list, index):
    # print(label_list[index], index)
    if index >= len(label_list): return []
    target_file = label_list[index]["index"] # this index is display name, not index
    project_path = Path(storage_path) / project_name
    label_dir = os.path.join(project_path, "label")
    frame_index, video_name = target_file.split(', ')
    delete_file_if_exists(os.path.join(label_dir, video_name, frame_index) + '.npz')
    return read_label_to_gallery(storage_path, project_name)[1]


def get_select_index(evt: gr.SelectData):
        return evt.index

def create_knowledge_ui(
    storage_path: str, project_name: str, knowledge_tab: gr.Tab
) -> Dict[str, Any]:
    """
    Create and return the Gradio UI components for knowledge base.

    Args:
        storage_path: Base storage directory.
        project_name: Name of the project.
        knowledge_tab: The Gradio Tab component where UI elements are added.

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
        inputs=[storage_path, project_name],
        outputs=[label_list_state, gallery],
    )
    

    gallery.select(get_select_index, None, selected_image)

    delete_selected_btn.click(
        fn=delete_selected,
        inputs=[storage_path, project_name, label_list_state, selected_image],
        outputs=gallery,
    )





    return ui
