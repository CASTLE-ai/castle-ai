"""Tests for castle.core.batch BatchRunner concurrency guardrail (PR3 Stage 7.2).

parallel=True shares one process-wide ModelRegistry + one CUDA device across
threads, so concurrent projects unload each other's models / OOM. On CUDA the
runner must force sequential execution; CPU-only parallelism is unaffected.
"""

import threading

import pytest

from castle.service.batch import BatchConfig, BatchRunner


def _two_project_runner():
    cfg = BatchConfig(
        projects=[{"name": "a", "project": "pa"}, {"name": "b", "project": "pb"}],
        parallel=True,
        max_workers=2,
    )
    return BatchRunner(cfg)


def test_parallel_forced_sequential_on_cuda(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    runner = _two_project_runner()
    seen_threads = []

    def fake_process(idx, n, spec, cb):
        seen_threads.append(threading.current_thread().name)
        return {"name": spec["name"], "status": "success"}

    monkeypatch.setattr(runner, "_process_project", fake_process)
    results = runner.run()

    assert len(results) == 2
    # Forced sequential → every project ran on the main thread.
    main = threading.current_thread().name
    assert all(t == main for t in seen_threads)


def test_parallel_allowed_without_cuda(monkeypatch):
    import time
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    runner = _two_project_runner()
    seen_threads = []

    def fake_process(idx, n, spec, cb):
        time.sleep(0.02)  # keep both in flight so the pool uses 2 threads
        seen_threads.append(threading.current_thread().name)
        return {"name": spec["name"], "status": "success"}

    monkeypatch.setattr(runner, "_process_project", fake_process)
    runner.run()

    # CPU-only parallelism is preserved → at least one worker thread (non-main).
    main = threading.current_thread().name
    assert any(t != main for t in seen_threads)
