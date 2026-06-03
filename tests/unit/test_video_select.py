"""Tests for the shared per-video selector helpers (castle/ui/video_select.py).

These cover the pure logic (resolve_selected, the half/invert math used by the
quick buttons). The Gradio wiring itself is exercised via create_ui() building
in the integration/build checks.
"""

from castle.ui.video_select import resolve_selected


def _first_half(allv):
    n = len(allv)
    return list(allv)[: (n + 1) // 2]


def _second_half(allv):
    n = len(allv)
    return list(allv)[(n + 1) // 2:]


def _invert(allv, current):
    return [v for v in allv if v not in set(current)]


def test_halves_partition_even_and_odd():
    # First/Second half must be disjoint and union to the whole list (so two
    # machines each take one half with no overlap and no gap).
    for n in (0, 1, 2, 25, 26):
        allv = [f"v{i}.mp4" for i in range(n)]
        first, second = _first_half(allv), _second_half(allv)
        assert set(first) & set(second) == set()          # disjoint
        assert set(first) | set(second) == set(allv)       # cover everything
        assert len(first) + len(second) == n
        if n:
            assert len(first) >= len(second)               # odd → extra goes to first


def test_invert_selection():
    allv = ["a.mp4", "b.mp4", "c.mp4", "d.mp4"]
    assert _invert(allv, ["a.mp4", "c.mp4"]) == ["b.mp4", "d.mp4"]
    assert _invert(allv, []) == allv
    assert _invert(allv, allv) == []


def test_resolve_selected_orders_by_config_and_drops_stale():
    sources = ["b.mp4", "a.mp4", "c.mp4"]
    # Returns sorted-by-config order, intersected with the checked set.
    assert resolve_selected(sources, ["c.mp4", "a.mp4"]) == ["a.mp4", "c.mp4"]
    # Stale names (no longer in the project) are dropped.
    assert resolve_selected(sources, ["a.mp4", "ghost.mp4"]) == ["a.mp4"]
    # Empty / None selection → nothing to process.
    assert resolve_selected(sources, []) == []
    assert resolve_selected(sources, None) == []
