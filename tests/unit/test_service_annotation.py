"""Unit tests for castle.service.annotation_service."""

import os
import tempfile

from castle.service.annotation_service import (
    DEFAULT_SCHEMES,
    list_schemes,
    get_scheme_labels,
    save_scheme,
    save_annotations,
    load_annotations,
)


def test_default_schemes():
    assert '5-class' in DEFAULT_SCHEMES
    assert '10-class' in DEFAULT_SCHEMES
    assert len(DEFAULT_SCHEMES['5-class']) == 5
    assert len(DEFAULT_SCHEMES['10-class']) == 10


def test_default_scheme_labels():
    assert 'Running' in DEFAULT_SCHEMES['5-class']
    assert 'Sniffing' in DEFAULT_SCHEMES['10-class']


def test_list_schemes_defaults_only():
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, 'test', 'cluster'), exist_ok=True)
        schemes = list_schemes(tmp, 'test')
        assert '5-class' in schemes
        assert '10-class' in schemes


def test_save_load_scheme():
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, 'test', 'cluster'), exist_ok=True)
        save_scheme(tmp, 'test', 'custom-3', ['A', 'B', 'C'])
        schemes = list_schemes(tmp, 'test')
        assert 'custom-3' in schemes
        assert schemes['custom-3'] == ['A', 'B', 'C']


def test_save_scheme_overwrites():
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, 'test', 'cluster'), exist_ok=True)
        save_scheme(tmp, 'test', 'my-scheme', ['X'])
        save_scheme(tmp, 'test', 'my-scheme', ['X', 'Y'])
        schemes = list_schemes(tmp, 'test')
        assert schemes['my-scheme'] == ['X', 'Y']


def test_get_scheme_labels():
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, 'test', 'cluster'), exist_ok=True)
        labels = get_scheme_labels(tmp, 'test', '5-class')
        assert len(labels) == 5


def test_get_scheme_labels_missing():
    with tempfile.TemporaryDirectory() as tmp:
        labels = get_scheme_labels(tmp, 'test', 'nonexistent')
        assert labels == []


def test_save_load_annotations():
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, 'test', 'cluster'), exist_ok=True)
        annotations = {
            'root_a0': {
                'behavior_label': 'running',
                'scheme': '5-class',
                'annotator': 'tester',
                'timestamp': '2025-01-01',
            },
            'root_a1': {
                'behavior_label': 'walking',
                'scheme': '5-class',
                'annotator': 'tester',
                'timestamp': '2025-01-01',
            },
        }
        save_annotations(tmp, 'test', annotations)
        loaded = load_annotations(tmp, 'test')
        assert len(loaded) == 2
        assert loaded['root_a0']['behavior_label'] == 'running'
        assert loaded['root_a1']['behavior_label'] == 'walking'


def test_load_annotations_empty():
    with tempfile.TemporaryDirectory() as tmp:
        loaded = load_annotations(tmp, 'nonexistent')
        assert loaded == {}
