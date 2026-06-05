"""Tests for batch-size auto-sizing wiring (PR Stage E).

The existing memory_guard.suggest_batch_size (VRAM-scaled, rotation-aware,
0.75 margin) is now the single source of truth, wired into the extraction
service and auto_config instead of a hard-coded 32 / a duplicate capped table.
"""

import pytest

from castle.service import extraction_service as es
from castle.service import auto_config as ac
from castle.service.extraction_service import make_preprocess_config


def test_auto_batch_size_uses_suggest_with_rotate(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    seen = {}

    def fake_suggest(model, device, *, rotate=False, **k):
        seen["model"], seen["device"], seen["rotate"] = model, device, rotate
        return 48

    monkeypatch.setattr("castle.core.memory_guard.suggest_batch_size", fake_suggest)
    cfg = make_preprocess_config(rotate_roi_tail_switch=True)
    bs = es._auto_batch_size("dinov3_vitb16", cfg)
    assert bs == 48
    assert seen == {"model": "dinov3_vitb16", "device": "cuda", "rotate": True}


def test_auto_batch_size_falls_back_on_error(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("no mem info")
    monkeypatch.setattr("castle.core.memory_guard.suggest_batch_size", boom)
    bs = es._auto_batch_size("dinov3_vitb16", make_preprocess_config())
    assert bs == es.EXTRACTION_BATCH_SIZE  # static default fallback


def test_recommend_config_delegates_batch_to_suggest(monkeypatch):
    monkeypatch.setattr("castle.core.memory_guard.suggest_batch_size",
                        lambda model, device, **k: 57)
    out = ac.recommend_config(
        "/no/such/video.mp4",
        gpu_info={"available": True, "name": "RTX 4090", "vram_mb": 24000, "vram_free_mb": 23000},
        model_name="dinov3_vitb16",
    )
    assert out["extraction"]["batch_size"] == 57


def test_recommend_config_cpu_only_is_batch_one(monkeypatch):
    out = ac.recommend_config(
        "/no/such/video.mp4",
        gpu_info={"available": False, "name": "CPU only", "vram_mb": 0, "vram_free_mb": 0},
    )
    assert out["extraction"]["batch_size"] == 1
