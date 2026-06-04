"""Reusable "Videos to process" multi-select widget for the batch tabs.

Batch tracking, pre-process and extract each let the user pick which videos to
process so one project can be split across machines (process a disjoint subset on
each, then merge the per-video output files). This module builds the shared
widget — a ``gr.CheckboxGroup`` (default: all checked) plus a row of quick
buttons (All / None / Invert / First half / Second half) — so the three tabs
don't each reimplement it.

The First/Second-half split is deterministic across machines: both load the same
``sorted(config['source'])``, so "First half" on machine A and "Second half" on
machine B cover the project with no overlap.
"""

from typing import Any, Dict, List, Tuple

import gradio as gr


def build_video_selector(label: str = "Videos to process",
                         visible: bool = False) -> Dict[str, Any]:
    """Create the selector components.

    Layout: the quick-button row sits **above** a collapsible ``gr.Accordion``
    that holds the (possibly long) checkbox list, so it can be folded away.

    Returns a dict with: ``group`` (CheckboxGroup), ``accordion`` (the collapsible
    container wrapping the list — toggle this for show/hide), ``btn_row`` (Row
    holding the five quick buttons), ``btn_all/btn_none/btn_invert/btn_first/
    btn_second``, and ``all_state`` (gr.State holding the full sorted video list,
    set on tab-select).
    """
    sel: Dict[str, Any] = {}
    # Quick buttons go ABOVE the list (act on whatever the list shows).
    with gr.Row(visible=visible) as btn_row:
        sel["btn_all"] = gr.Button("All", size="sm")
        sel["btn_none"] = gr.Button("None", size="sm")
        sel["btn_invert"] = gr.Button("Invert", size="sm")
        sel["btn_first"] = gr.Button("First half", size="sm")
        sel["btn_second"] = gr.Button("Second half", size="sm")
    sel["btn_row"] = btn_row
    # The list itself lives in a collapsible accordion (default open).
    with gr.Accordion(label, open=True, visible=visible) as accordion:
        sel["group"] = gr.CheckboxGroup(
            choices=[],
            value=[],
            show_label=False,  # the accordion header already names it
            info="Tick the videos to process on this machine. Split a project "
                 "across machines with First/Second half (deterministic, no overlap).",
            interactive=True,
        )
    sel["accordion"] = accordion
    sel["all_state"] = gr.State([])
    return sel


def wire_video_selector(sel: Dict[str, Any]) -> None:
    """Wire the five quick buttons to update the CheckboxGroup value.

    Reads the full list from ``all_state`` and the current selection from the
    group. All handlers are ``queue=False`` (instant, no GPU work).
    """
    group = sel["group"]
    all_state = sel["all_state"]

    def _half(all_videos, second: bool):
        n = len(all_videos)
        mid = (n + 1) // 2  # first half gets the extra one on odd counts
        chosen = list(all_videos)[mid:] if second else list(all_videos)[:mid]
        return gr.update(value=chosen)

    sel["btn_all"].click(lambda a: gr.update(value=list(a)),
                         inputs=all_state, outputs=group, queue=False)
    sel["btn_none"].click(lambda: gr.update(value=[]),
                          outputs=group, queue=False)
    sel["btn_invert"].click(
        lambda a, cur: gr.update(value=[v for v in a if v not in set(cur or [])]),
        inputs=[all_state, group], outputs=group, queue=False)
    sel["btn_first"].click(lambda a: _half(a, second=False),
                           inputs=all_state, outputs=group, queue=False)
    sel["btn_second"].click(lambda a: _half(a, second=True),
                            inputs=all_state, outputs=group, queue=False)


def populate_selector(videos) -> Tuple[Any, List[str]]:
    """For a tab's ``.select()`` handler: returns ``(group_update, all_list)``.

    Sets the group's choices to ``videos`` with **all checked by default**, and
    the list to stash into ``all_state``.
    """
    vids = list(videos)
    return gr.update(choices=vids, value=vids), vids


def resolve_selected(config_sources, selected) -> List[str]:
    """Videos to actually process: ``config`` order, intersected with the checked
    set (drops any stale names no longer in the project)."""
    chosen = set(selected or [])
    return [v for v in sorted(config_sources) if v in chosen]
