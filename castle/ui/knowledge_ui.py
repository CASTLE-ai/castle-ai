"""Knowledge base UI for viewing tracked labels."""

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

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


_DELETE_IDLE = "Delete"


def _armed_text(target: str) -> str:
    return f"⚠️ Delete '{target}'? Click again to confirm"


def _reset_delete_armed_state():
    """Disarm a pending delete (e.g. when the gallery selection changes)."""
    return gr.update(value=_DELETE_IDLE), None


def delete_selected(storage_path, project_name, label_list, index, armed_index):
    """Two-click delete: first click arms (button shows the target), second confirms.

    Returns ``(label_list, gallery, delete_btn_update, new_armed_index)``.
    """
    if index is None or not label_list or index >= len(label_list):
        gr.Warning("No item selected. Please select a gallery item before deleting.")
        new_list, new_gallery = read_label_to_gallery(storage_path, project_name)
        return new_list, new_gallery, gr.update(value=_DELETE_IDLE), None

    target_file = label_list[index]["index"]  # display name "frame_index, video_name"

    # First click on this item (or selection changed) — arm, don't delete yet.
    if armed_index != index:
        return label_list, gr.update(), gr.update(value=_armed_text(target_file)), index

    # Same item clicked twice — execute the delete.
    project_path = Path(storage_path) / project_name
    label_dir = os.path.join(project_path, "label")
    frame_index, video_name = target_file.split(', ')
    file_path = os.path.join(label_dir, video_name, frame_index) + '.npz'
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            gr.Info(f"Deleted label: {target_file}")
        else:
            gr.Warning(f"File does not exist: {file_path}")
    except OSError as exc:
        gr.Warning(f"Error deleting file: {exc}")
    new_list, new_gallery = read_label_to_gallery(storage_path, project_name)
    return new_list, new_gallery, gr.update(value=_DELETE_IDLE), None


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
    # Local state (not in the ui dict, so it doesn't affect knowledge_ui_count):
    # which gallery index is armed for the two-click delete confirmation.
    delete_armed_index_state = gr.State(None)

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
    

    # Selecting a different frame disarms any pending delete confirmation.
    gallery.select(get_select_index, None, selected_image).then(
        fn=_reset_delete_armed_state,
        inputs=None,
        outputs=[delete_selected_btn, delete_armed_index_state],
        queue=False,
    )

    # Two-click delete: first click arms (button relabels with the target),
    # second click on the same item confirms.
    delete_selected_btn.click(
        fn=delete_selected,
        inputs=[storage_path, project_name, label_list_state, selected_image,
                delete_armed_index_state],
        outputs=[label_list_state, gallery, delete_selected_btn, delete_armed_index_state],
        queue=False,
    )





    return ui
