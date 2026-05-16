"""Unit tests for pure-logic helpers in castle.core.cluster."""

import numpy as np
from unittest.mock import MagicMock, patch
from castle.core.cluster import (
    auto_generate_cluster_name,
    find_nearest_embedding,
    frame_to_timestamp,
    LatentAggregator,
)


# ---- auto_generate_cluster_name ----

def test_auto_generate_cluster_name_root():
    assert auto_generate_cluster_name("root", 0) == "root_a0"
    assert auto_generate_cluster_name("root", 5) == "root_a5"


def test_auto_generate_cluster_name_nested():
    assert auto_generate_cluster_name("root_a0", 1) == "root_a0_b1"
    assert auto_generate_cluster_name("root_a0_b1", 2) == "root_a0_b1_c2"


def test_auto_generate_cluster_name_none():
    result = auto_generate_cluster_name(None, 0)
    assert result == "root_a0"


def test_auto_generate_cluster_name_deep():
    name = auto_generate_cluster_name("root_a0_b1_c2", 3)
    assert name == "root_a0_b1_c2_d3"


# ---- find_nearest_embedding ----

def test_find_nearest_embedding():
    data = np.array([[0, 0], [1, 1], [2, 2], [10, 10]], dtype=float)
    idx, dist = find_nearest_embedding(data, 0.9, 0.9)
    assert idx == 1  # Closest to (1,1)
    assert dist < 0.2


def test_find_nearest_embedding_exact():
    data = np.array([[0, 0], [5, 5], [10, 10]], dtype=float)
    idx, dist = find_nearest_embedding(data, 5.0, 5.0)
    assert idx == 1
    assert dist < 1e-10


def test_find_nearest_embedding_with_tree():
    from scipy.spatial import KDTree
    data = np.array([[0, 0], [3, 4], [6, 8]], dtype=float)
    tree = KDTree(data)
    idx, dist = find_nearest_embedding(data, 3.1, 4.1, tree=tree)
    assert idx == 1


# ---- frame_to_timestamp ----

def test_frame_to_timestamp_zero():
    ts = frame_to_timestamp(0, 30.0)
    assert ts == "00:00:00,000"


def test_frame_to_timestamp_one_second():
    ts = frame_to_timestamp(30, 30.0)
    assert ts == "00:00:01,000"


def test_frame_to_timestamp_complex():
    # 3661 seconds = 1h 1m 1s at 1 fps
    ts = frame_to_timestamp(3661, 1.0)
    assert ts.startswith("01:01:01")


# ---- LatentAggregator VideoReader cache (C-02) ----

class TestVideoReaderCache:
    """Test the LRU VideoReader cache on LatentAggregator."""

    def _make_aggregator(self):
        """Create a minimal LatentAggregator without triggering __init__ file I/O."""
        import threading
        from collections import OrderedDict
        agg = object.__new__(LatentAggregator)
        # PERF-03: cache became OrderedDict to support move_to_end + popitem.
        agg._video_reader_cache = OrderedDict()
        agg._cache_max_size = 3
        agg._frame_cache = OrderedDict()
        agg._frame_cache_max = 256
        agg._cache_lock = threading.Lock()
        agg.source_path = '/fake'
        agg.bin_size = 1
        agg.notify = lambda *a, **kw: None
        agg.videos_meta = []
        return agg

    def test_cache_creates_reader(self):
        agg = self._make_aggregator()
        mock_reader = MagicMock()
        with patch('castle.core.cluster.VideoReader', return_value=mock_reader) as MockVR:
            reader = agg._get_cached_reader('/fake/video.mp4')
            MockVR.assert_called_once_with('/fake/video.mp4')
            assert reader is mock_reader
            assert '/fake/video.mp4' in agg._video_reader_cache

    def test_cache_returns_existing(self):
        agg = self._make_aggregator()
        mock_reader = MagicMock()
        agg._video_reader_cache['/fake/video.mp4'] = mock_reader
        with patch('castle.core.cluster.VideoReader') as MockVR:
            reader = agg._get_cached_reader('/fake/video.mp4')
            MockVR.assert_not_called()
            assert reader is mock_reader

    def test_cache_evicts_oldest_when_full(self):
        agg = self._make_aggregator()
        agg._cache_max_size = 2
        r1 = MagicMock(name='reader1')
        r2 = MagicMock(name='reader2')
        agg._video_reader_cache['a.mp4'] = r1
        agg._video_reader_cache['b.mp4'] = r2

        r3 = MagicMock(name='reader3')
        with patch('castle.core.cluster.VideoReader', return_value=r3):
            reader = agg._get_cached_reader('c.mp4')
        
        assert reader is r3
        r1.close.assert_called_once()  # oldest evicted
        assert 'a.mp4' not in agg._video_reader_cache
        assert 'b.mp4' in agg._video_reader_cache
        assert 'c.mp4' in agg._video_reader_cache

    def test_close_clears_all(self):
        agg = self._make_aggregator()
        r1 = MagicMock()
        r2 = MagicMock()
        agg._video_reader_cache['a.mp4'] = r1
        agg._video_reader_cache['b.mp4'] = r2
        agg.close()
        r1.close.assert_called_once()
        r2.close.assert_called_once()
        assert len(agg._video_reader_cache) == 0

    def test_lru_reorder_on_access(self):
        """Accessing an existing key should move it to the end (most recent)."""
        agg = self._make_aggregator()
        agg._cache_max_size = 2
        r1 = MagicMock(name='reader1')
        r2 = MagicMock(name='reader2')
        agg._video_reader_cache['a.mp4'] = r1
        agg._video_reader_cache['b.mp4'] = r2

        # Access a.mp4 to make it most recent
        agg._get_cached_reader('a.mp4')

        # Now adding c.mp4 should evict b.mp4 (oldest), not a.mp4
        r3 = MagicMock(name='reader3')
        with patch('castle.core.cluster.VideoReader', return_value=r3):
            agg._get_cached_reader('c.mp4')

        r2.close.assert_called_once()
        assert 'b.mp4' not in agg._video_reader_cache
        assert 'a.mp4' in agg._video_reader_cache
        assert 'c.mp4' in agg._video_reader_cache
