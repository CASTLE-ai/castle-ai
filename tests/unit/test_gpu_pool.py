"""Unit tests for the video-level multi-GPU work-queue primitive.

No real GPUs are used — ``worker`` is a plain callable and the device string is
just a label, so these run anywhere.
"""

import threading
import time

import pytest

from castle.core import gpu_pool
from castle.core.gpu_pool import (
    CANCELLED,
    multi_gpu_enabled,
    resolve_device_ids,
    run_on_device_pool,
)


def test_drains_all_items_results_aligned_in_order():
    items = ["a", "b", "c", "d", "e"]
    out = run_on_device_pool(items, lambda it, dev: it.upper(), device_ids=[0, 1])
    assert out == ["A", "B", "C", "D", "E"]


def test_devices_are_passed_and_distinct_per_worker():
    seen = set()
    lock = threading.Lock()

    def worker(it, dev):
        with lock:
            seen.add(dev)
        time.sleep(0.01)  # hold the slot so both workers are used
        return dev

    out = run_on_device_pool(list(range(8)), worker, device_ids=[0, 2])
    assert seen == {"cuda:0", "cuda:2"}
    assert all(r in ("cuda:0", "cuda:2") for r in out)


def test_per_item_error_isolation():
    def worker(it, dev):
        if it == 2:
            raise ValueError("boom on 2")
        return it * 10

    out = run_on_device_pool([0, 1, 2, 3], worker, device_ids=[0, 1])
    assert out[0] == 0 and out[1] == 10 and out[3] == 30
    assert isinstance(out[2], ValueError)
    assert "boom on 2" in str(out[2])


def test_on_done_fired_for_every_item():
    done = []
    lock = threading.Lock()

    def on_done(item, res):
        with lock:
            done.append((item, res))

    items = ["x", "y", "z"]
    run_on_device_pool(items, lambda it, dev: it + "!", device_ids=[0, 1], on_done=on_done)
    assert sorted(done) == [("x", "x!"), ("y", "y!"), ("z", "z!")]


def test_cancel_before_start_marks_all_cancelled():
    ev = threading.Event()
    ev.set()
    out = run_on_device_pool([1, 2, 3], lambda it, dev: it, device_ids=[0], cancel_event=ev)
    assert out == [CANCELLED, CANCELLED, CANCELLED]


def test_cancel_midway_leaves_later_items_cancelled():
    ev = threading.Event()
    processed = []
    lock = threading.Lock()

    def worker(it, dev):
        with lock:
            processed.append(it)
        if it == 0:
            ev.set()  # request cancel after the first item
        return it

    out = run_on_device_pool([0, 1, 2, 3, 4], worker, device_ids=[1], cancel_event=ev)
    # Item 0 ran; at least the tail is cancelled (single worker stops pulling).
    assert out[0] == 0
    assert out[-1] is CANCELLED
    assert 4 not in processed


def test_empty_device_ids_raises():
    with pytest.raises(ValueError):
        run_on_device_pool([1, 2], lambda it, dev: it, device_ids=[])


def test_returns_none_results_not_confused_with_cancel():
    # A worker legitimately returning None must not be reported as CANCELLED.
    out = run_on_device_pool([1, 2], lambda it, dev: None, device_ids=[0])
    assert out == [None, None]
    assert out[0] is not CANCELLED


# ---- gate: multi_gpu_enabled / resolve_device_ids -------------------------

def _patch_cuda(monkeypatch, available, count):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: count)


def test_gate_off_by_default(monkeypatch):
    monkeypatch.delenv("CASTLE_MULTI_GPU", raising=False)
    _patch_cuda(monkeypatch, True, 4)
    assert multi_gpu_enabled() is False
    assert resolve_device_ids() == []


@pytest.mark.parametrize("val", ["", "0", "false", "no", "off"])
def test_gate_falsey_values(monkeypatch, val):
    monkeypatch.setenv("CASTLE_MULTI_GPU", val)
    _patch_cuda(monkeypatch, True, 4)
    assert multi_gpu_enabled() is False


def test_gate_on_with_multiple_gpus(monkeypatch):
    monkeypatch.setenv("CASTLE_MULTI_GPU", "1")
    _patch_cuda(monkeypatch, True, 2)
    assert multi_gpu_enabled() is True
    assert resolve_device_ids() == [0, 1]


def test_gate_on_but_single_gpu(monkeypatch):
    monkeypatch.setenv("CASTLE_MULTI_GPU", "1")
    _patch_cuda(monkeypatch, True, 1)
    assert multi_gpu_enabled() is False
    assert resolve_device_ids() == []


def test_gate_handles_torch_errors(monkeypatch):
    monkeypatch.setenv("CASTLE_MULTI_GPU", "1")
    import torch

    def boom():
        raise RuntimeError("no driver")

    monkeypatch.setattr(torch.cuda, "is_available", boom)
    assert multi_gpu_enabled() is False
    assert resolve_device_ids() == []
