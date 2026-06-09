"""Regression: the Submit overwrite-confirm gate must not blank the cluster tree.

Field bug: clicking Submit on a node that already has children showed the
"Confirm Overwrite" button but ALSO cleared the cluster-tree selection and hid
the tree (the gate branch returned ``None`` for the 9 display outputs, which
Gradio renders as empty). The follow-up "Confirm Overwrite" click then saw an
empty parent and aborted with "No cluster selected". The gate must leave the
tree + selection untouched (``gr.update()``).
"""

import json
import os

import gradio as gr

import castle.service.clustering_service as cs
from castle.ui.cluster_handlers import label_all_and_submit, on_tree_node_select


class _Latents:
    """Minimal Latent stand-in: only the attrs the gate branch touches."""
    behavior_name2cluster_id = {
        "init_a2_b0_c0": 1, "init_a2_b0_c1": 2,
        "init_a2_b0_c2": 3, "init_a2_b0_c3": 4,
    }


def _noop(update) -> bool:
    """True if `update` is a no-op gr.update() (preserves the component value)."""
    return isinstance(update, dict) and "value" not in update


def test_overwrite_gate_preserves_tree_and_selection(monkeypatch, tmp_path):
    # Sidecar meta exists -> first Submit click arms the overwrite gate.
    monkeypatch.setattr(cs, "load_node_meta", lambda *a, **k: {"embedding_npz": "x.npz"})

    out = label_all_and_submit(
        str(tmp_path), "proj", _Latents(), object(), object(),
        "init_a2_b0", overwrite_confirmed=False,
    )

    assert len(out) == 13
    # outputs[1]=cluster_tree_html, outputs[2]=cluster_tree_select must be
    # PRESERVED (no-op), NOT None — else the tree blanks and selection clears.
    assert _noop(out[1]), f"tree_html clobbered: {out[1]!r}"
    assert _noop(out[2]), f"tree_select clobbered: {out[2]!r}"
    # the other 7 display outputs are preserved too
    for i in (0, 3, 4, 5, 6, 7, 8):
        assert _noop(out[i]), f"display output {i} clobbered: {out[i]!r}"
    # the gate is armed: overwrite_state True + confirm button shown
    assert out[9] is True
    assert isinstance(out[10], dict) and out[10].get("visible") is True


def test_no_cluster_selected_does_not_blank_tree(monkeypatch, tmp_path):
    # Submitting with no node selected warns but must not wipe the tree.
    out = label_all_and_submit(
        str(tmp_path), "proj", _Latents(), object(), object(),
        "", overwrite_confirmed=False,
    )
    assert _noop(out[1]) and _noop(out[2])
    assert out[9] is False


def test_node_sidecar_restores_eps_and_min_samples(tmp_path):
    """min_samples persists in the node sidecar and is restored on node-select,
    exactly like eps (regression for the new DBSCAN-min-points persistence)."""
    cluster = tmp_path / "proj" / "cluster"
    cluster.mkdir(parents=True)
    (cluster / "node_initX_meta.json").write_text(json.dumps({
        "parent_cluster_name": "initX", "umap_config": None,
        "eps": 0.7, "min_samples": 12, "preset": None,
        "umap_seed": None, "embedding_npz": None,
    }))
    out = on_tree_node_select("initX", None, str(tmp_path), "proj")
    assert len(out) == 10                       # added min_samples output
    assert dict(out[1]).get("value") == 0.7     # eps restored
    assert dict(out[9]).get("value") == 12      # min_samples restored
