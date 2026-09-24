"""Regression: tree-node onclick handlers need a *global* castleTreeClick.

Gradio 6 runs ``launch(js=...)`` as a function, so a bare ``function`` declaration
there is local and the cluster tree cannot be clicked.
"""


def test_castle_tree_click_is_assigned_to_window():
    from castle.ui.main_ui import CASTLE_JS

    assert "window.castleTreeClick" in CASTLE_JS
    assert CASTLE_JS.strip().startswith("() =>")
