"""Unified dual-mode cluster palette (project decision 2026-06-20).

ONE engine colours clusters everywhere (figures + interactive tree/scatter) with
a normal/colorblind toggle. Default is colorblind (Okabe-Ito) so publication
figures stay colour-vision-safe; 'normal' (VIVID) is the opt-in vibrant mode.
"""

import re

import pytest

from castle.core import config
from castle.core.config import (
    OKABE_ITO, VIVID, color_for_cluster, color_for_name,
    get_color_mode, palette_color, set_color_mode,
)


@pytest.fixture(autouse=True)
def _reset_color_mode():
    """Each test starts from the default (env/override cleared)."""
    config._COLOR_MODE = None
    yield
    config._COLOR_MODE = None


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


# --- dual-mode engine ------------------------------------------------------

def test_default_mode_is_colorblind_and_keeps_figures_okabe_ito():
    assert get_color_mode() == 'colorblind'
    assert [color_for_cluster(i) for i in range(8)] == OKABE_ITO


def test_mode_switches_ladder_and_is_validated():
    set_color_mode('normal')
    assert get_color_mode() == 'normal'
    assert [color_for_cluster(i) for i in range(len(VIVID))] == VIVID
    set_color_mode('colorblind')
    assert color_for_cluster(0) == OKABE_ITO[0]
    with pytest.raises(ValueError):
        set_color_mode('rainbow')


def test_env_default_and_bad_value_fallback(monkeypatch):
    config._COLOR_MODE = None
    monkeypatch.setenv('CASTLE_COLOR_MODE', 'normal')
    assert get_color_mode() == 'normal'
    monkeypatch.setenv('CASTLE_COLOR_MODE', 'nonsense')
    assert get_color_mode() == 'colorblind'  # bad value ignored


def test_both_modes_unique_across_full_cycle():
    for mode in ('colorblind', 'normal'):
        set_color_mode(mode)
        n = len(config._PALETTE_LADDERS[mode]) * (len(config._WRAP_LIGHTNESS) + 1)
        cols = [color_for_cluster(i) for i in range(n)]
        assert len(set(cols)) == n, f"{mode}: {n} slots not all distinct"


def test_color_for_name_sentinels_and_large_md5_index():
    # unlabeled / container sentinels -> grey
    assert color_for_name('') == 'grey'
    assert color_for_name('init') == 'grey'
    # md5 of a name is a huge int; must still yield a valid, deterministic hex
    for mode in ('colorblind', 'normal'):
        set_color_mode(mode)
        c = color_for_name('init_a0_b3')
        assert re.fullmatch(r"#[0-9a-fA-F]{6}", c), c
        assert color_for_name('init_a0_b3') == c
    # name-hashing into a finite colorblind-safe palette has birthday-paradox
    # collisions (the inherent cost of unifying onto a fixed palette), so we
    # guarantee VALID + DETERMINISTIC colours and good-but-not-perfect spread,
    # not zero collisions. Positional figure colours (tested above) ARE exact.
    set_color_mode('colorblind')
    names = [f"init_a{i}" for i in range(8)]
    cols = [color_for_name(n) for n in names]
    assert all(re.fullmatch(r"#[0-9a-fA-F]{6}", c) for c in cols)
    assert len(set(cols)) >= 6  # most distinct over the 56-slot ladder


def test_palette_color_explicit_mode_overrides_global():
    set_color_mode('colorblind')
    assert palette_color(0, mode='normal') == VIVID[0]
    assert palette_color(0, mode='colorblind') == OKABE_ITO[0]
