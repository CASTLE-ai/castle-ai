"""Colorblind-safe shared figure palette (project decision 2026-06-20).

All analysis figures colour clusters via one Okabe-Ito-based palette, so cluster
*i* is the same colour across figures and legible under colour-vision deficiency.
"""

import re

from castle.core.config import OKABE_ITO, color_for_cluster


def test_first_eight_are_canonical_okabe_ito():
    cols = [color_for_cluster(i) for i in range(8)]
    assert cols == OKABE_ITO
    assert len(set(cols)) == 8  # all distinct


def test_deterministic_and_valid_hex():
    for i in range(40):
        c = color_for_cluster(i)
        assert re.fullmatch(r"#[0-9a-fA-F]{6}", c), c
        assert color_for_cluster(i) == color_for_cluster(i)  # deterministic


def test_all_colours_unique_across_realistic_cluster_counts():
    # No two clusters may share an identical colour, or two distinct behaviours
    # become visually indistinguishable. 48 covers any realistic count; black at
    # index 7 (a lightness extreme) is the regression-prone case — guard it.
    cols = [color_for_cluster(i) for i in range(48)]
    assert len(set(cols)) == 48, "duplicate cluster colours"
    blacks = {color_for_cluster(i) for i in (7, 15, 23, 31)}
    assert len(blacks) == 4, f"black-hue wraps collapsed: {blacks}"


def test_figure_modules_use_the_shared_palette():
    # the ethogram figure helper routes through the same colorblind-safe source
    from castle.visualization.ethogram_plots import _get_cluster_colors
    assert _get_cluster_colors({}, 5) == [color_for_cluster(i) for i in range(5)]
