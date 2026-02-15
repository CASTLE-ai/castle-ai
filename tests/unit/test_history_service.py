"""Tests for castle.service.history_service — undo/redo mechanism."""
import numpy as np
import pytest

from castle.service.history_service import HistoryManager, ClusterSnapshot


class MockLatent:
    """Minimal mock that mirrors the fields HistoryManager touches."""

    def __init__(self):
        self.cluster = np.array([0, 0, 1, 1, 2, 2])
        self.cluster_meta = {0: {'name': 'a', 'color': '#FF0000'}}
        self.embedding = np.random.randn(6, 2)


# ------------------------------------------------------------------
# Basic undo / redo
# ------------------------------------------------------------------

def test_save_and_undo():
    mgr = HistoryManager()
    lat = MockLatent()
    original_cluster = lat.cluster.copy()

    mgr.save_state(lat, "initial")
    lat.cluster = np.array([1, 1, 1, 1, 1, 1])  # mutation

    desc = mgr.undo(lat)
    assert desc == "initial"
    assert np.array_equal(lat.cluster, original_cluster)


def test_redo():
    mgr = HistoryManager()
    lat = MockLatent()

    mgr.save_state(lat, "before change")
    new_cluster = np.array([2, 2, 2, 2, 2, 2])
    lat.cluster = new_cluster.copy()

    mgr.undo(lat)   # back to original
    mgr.redo(lat)    # forward to changed
    assert np.array_equal(lat.cluster, new_cluster)


def test_undo_restores_cluster_meta():
    mgr = HistoryManager()
    lat = MockLatent()
    original_meta = {0: {'name': 'a', 'color': '#FF0000'}}

    mgr.save_state(lat, "before meta change")
    lat.cluster_meta = {0: {'name': 'changed', 'color': '#00FF00'}}

    mgr.undo(lat)
    assert lat.cluster_meta == original_meta


def test_undo_restores_embedding():
    mgr = HistoryManager()
    lat = MockLatent()
    original_emb = lat.embedding.copy()

    mgr.save_state(lat, "before emb change")
    lat.embedding = np.zeros((6, 2))

    mgr.undo(lat)
    assert np.array_equal(lat.embedding, original_emb)


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------

def test_undo_empty():
    mgr = HistoryManager()
    lat = MockLatent()
    assert mgr.undo(lat) is None


def test_redo_empty():
    mgr = HistoryManager()
    lat = MockLatent()
    assert mgr.redo(lat) is None


def test_max_history():
    mgr = HistoryManager(max_history=3)
    lat = MockLatent()
    for i in range(5):
        mgr.save_state(lat, f"action {i}")
    assert len(mgr._undo_stack) == 3
    # Oldest should be "action 2" (0 and 1 evicted)
    assert mgr._undo_stack[0].description == "action 2"


def test_new_action_clears_redo():
    mgr = HistoryManager()
    lat = MockLatent()
    mgr.save_state(lat, "a1")
    lat.cluster = np.ones(6, dtype=int)
    mgr.undo(lat)
    assert mgr.can_redo
    mgr.save_state(lat, "a2")  # new action
    assert not mgr.can_redo


# ------------------------------------------------------------------
# Properties
# ------------------------------------------------------------------

def test_can_undo_redo_flags():
    mgr = HistoryManager()
    lat = MockLatent()
    assert not mgr.can_undo
    assert not mgr.can_redo

    mgr.save_state(lat, "x")
    assert mgr.can_undo
    assert not mgr.can_redo

    mgr.undo(lat)
    assert not mgr.can_undo
    assert mgr.can_redo


def test_descriptions():
    mgr = HistoryManager()
    lat = MockLatent()
    assert mgr.undo_description == ""
    assert mgr.redo_description == ""

    mgr.save_state(lat, "first")
    assert mgr.undo_description == "first"

    mgr.undo(lat)
    # After undo, the redo stack has the current state (no description)
    assert mgr.can_redo
    assert mgr.undo_description == ""


def test_clear():
    mgr = HistoryManager()
    lat = MockLatent()
    mgr.save_state(lat, "x")
    mgr.clear()
    assert not mgr.can_undo
    assert not mgr.can_redo


# ------------------------------------------------------------------
# Snapshot independence (no aliasing)
# ------------------------------------------------------------------

def test_snapshot_independence():
    """Ensure saved snapshot is independent of the original latent."""
    mgr = HistoryManager()
    lat = MockLatent()
    mgr.save_state(lat, "before")

    # Mutate in-place
    lat.cluster[0] = 99
    lat.cluster_meta[0]['name'] = 'mutated'

    mgr.undo(lat)
    assert lat.cluster[0] != 99
    assert lat.cluster_meta[0]['name'] == 'a'


def test_multiple_undo_redo_roundtrip():
    """Undo/redo multiple times should be stable."""
    mgr = HistoryManager()
    lat = MockLatent()

    states = []
    for i in range(4):
        mgr.save_state(lat, f"step {i}")
        lat.cluster = np.full(6, i, dtype=int)
        states.append(lat.cluster.copy())

    # Undo all 4
    for i in range(4):
        mgr.undo(lat)

    # Redo all 4
    for i in range(4):
        mgr.redo(lat)
        assert np.array_equal(lat.cluster, states[i])


def test_no_embedding_attribute():
    """HistoryManager should work even if latent has no embedding."""
    mgr = HistoryManager()

    class BareLat:
        def __init__(self):
            self.cluster = np.array([0, 1, 2])
            self.cluster_meta = {}

    lat = BareLat()
    mgr.save_state(lat, "bare")
    lat.cluster = np.array([9, 9, 9])
    mgr.undo(lat)
    assert np.array_equal(lat.cluster, np.array([0, 1, 2]))
