"""Regression test for the ethogram frequency-bar name/height pairing.

Previously the report built bar *names* sorted by integer cluster id but bar
*heights* with a no-op sort key (``lambda _: 0``), which kept dict-insertion
order — so every bar in the published figure could be mislabeled. This pins the
correct pairing (and the integer-vs-string sort).
"""

from castle.analysis.report import _ethogram_bar_data, _provenance_html


def test_names_and_freqs_paired_by_integer_cid():
    # Insertion order is deliberately NOT integer-cid order, and includes "10"
    # to catch lexical-vs-integer sorting ("10" must come after "2").
    bout_stats = {
        "2": {"cluster_name": "groom", "frequency": 0.20},
        "0": {"cluster_name": "walk", "frequency": 0.50},
        "10": {"cluster_name": "rear", "frequency": 0.10},
        "1": {"cluster_name": "sniff", "frequency": 0.30},
    }
    names, freqs = _ethogram_bar_data(bout_stats)
    # sorted by int cid → 0, 1, 2, 10
    assert names == ["walk", "sniff", "groom", "rear"]
    assert freqs == [50.0, 30.0, 20.0, 10.0]
    # the pairing invariant, stated directly
    assert dict(zip(names, freqs)) == {
        "walk": 50.0, "sniff": 30.0, "groom": 20.0, "rear": 10.0,
    }


def test_falls_back_to_cid_when_name_missing():
    bout_stats = {"0": {"frequency": 0.4}, "1": {"cluster_name": "x", "frequency": 0.1}}
    names, freqs = _ethogram_bar_data(bout_stats)
    assert names == ["0", "x"]
    assert freqs == [40.0, 10.0]


def test_provenance_html_states_version_and_stack():
    env = {
        "castle_version": "0.0.18",
        "python": "3.10.19",
        "device": "cuda",
        "gpus": ["NVIDIA GeForce RTX 3060"],
        "packages": {"torch": "2.9.1", "numpy": "1.26.4", "cuml": "25.8.0"},
    }
    html = _provenance_html(env)
    assert "CASTLE 0.0.18" in html
    assert "device=cuda" in html
    assert "RTX 3060" in html
    assert "torch 2.9.1" in html and "cuml 25.8.0" in html


def test_provenance_html_tolerates_empty_env():
    # No packages / no gpus → still produces a valid one-line div, no crash.
    html = _provenance_html({})
    assert html.startswith("<div") and html.endswith("</div>")
    assert "CASTLE ?" in html
