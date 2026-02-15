# CASTLE QC Audit Report

> **Date**: 2026-02-15 16:40 (Asia/Taipei)  
> **Branch**: `dev`  
> **Method**: code-qc skill (7-phase structured audit)  
> **Auditor**: Claude (READ-ONLY, no code changes)

---

## Executive Summary

| Phase | Status | Key Metric |
|-------|--------|------------|
| 1. Test Suite | ✅ PASS | 134/134 passed (10.83s) |
| 2. Import Integrity | ⚠️ 1 known | 70/71 — myumap (cuML required) |
| 3. Static Analysis | ⚠️ 94 findings | 0 bare except, 0 print; 78 F401, 9 F841, 7 B006 |
| 4. Smoke Tests | ✅ PASS | 14/14 all green |
| 5. UI/Frontend | ✅ PASS | Gradio + Desktop + CLI OK |
| 6. File Consistency | ✅ PASS | 83/83 syntax OK, git clean |
| 7. Documentation | ⚠️ 20 classes | 0 module docstrings missing, 20 class docstrings missing |

**Overall Health**: 🟢 **Good** — Core functionality solid, no blockers. Hygiene issues (unused imports, class docstrings) are non-critical.

---

## Phase 1: Test Suite

```
PYTHONPATH=. python -m pytest tests/unit/ -v --tb=short
```

| Metric | Value |
|--------|-------|
| Total | 134 |
| Passed | 134 |
| Failed | 0 |
| Errors | 0 |
| Skipped | 0 |
| Duration | 10.83s |

All unit tests pass cleanly. Test count increased from 102 → 134 (+32 new tests) since the previous audit.

---

## Phase 2: Import Integrity

```
python import_check.py castle --exclude aot,sam,dinov2,dinov3 --json
```

| Metric | Value |
|--------|-------|
| Total modules | 71 |
| Passed | 70 |
| Failed | 1 |
| Skipped (vendored) | 84 |
| Duration | 11.96s |

**Failed module:**
- `castle.utils.myumap` — requires RAPIDS cuML (GPU-only dependency). This is expected and handled gracefully with CPU fallback.

---

## Phase 3: Static Analysis (ruff 0.15.1)

```
ruff check --select E722,T201,B006,F401,F841 --statistics castle/
```

| Rule | Description | Count | Status |
|------|-------------|-------|--------|
| E722 | Bare except | **0** | ✅ Fixed (was 2) |
| T201 | Print statements | **0** | ✅ Fixed (was 40) |
| B006 | Mutable argument defaults | 7 | ⚠️ New finding |
| F401 | Unused imports | 78 | ⚠️ Hygiene |
| F841 | Unused variables | 9 | ⚠️ Hygiene |
| **Total** | | **94** | 67 auto-fixable |

### B006 Locations (Mutable defaults):
- `castle/core/models.py:103` — `_multiscale_pooling(scales=[1, 2, 4])`
- `castle/utils/explorer.py:187`
- `castle/utils/latent_explorer.py:202, 209`
- `castle/visualization/embedding_plots.py:21, 69, 210`

### F401 Breakdown (78 unused imports):
- **Re-exports** (intentional, need `__all__`): `service/__init__.py` (5), `ui/__init__.py` (1), `utils/__init__.py` (2), `visualization/__init__.py` (5) — 13 total
- **Genuine unused**: typing imports (Optional, List, Dict, Tuple, Any) — ~25
- **Unused library imports**: pandas, torch, cv2, json, os, numpy, shutil — ~20
- **Unused specific imports**: VideoIOError, ReadArray, etc. — ~20

### F841 Locations (9 unused variables):
- `castle/core/extractor.py:450` — `e` in except clause
- `castle/core/project_config.py:111` — `field_types`
- `castle/ui/cluster_handlers.py:137` — `time_window`
- `castle/ui/cluster_handlers.py:296` — `e` in except clause
- `castle/ui/cluster_tree.py:41` — `color`
- `castle/ui/edit_ui.py:120` — `single_tracking_tab`
- `castle/ui/main_ui.py:56, 61, 74` — UI component variables

---

## Phase 4: Smoke Tests

14 dynamically-generated smoke tests covering all service layer and core functionality:

| # | Test | Status |
|---|------|--------|
| 1 | Project service (create, list, info) | ✅ PASS |
| 2 | Annotation service (schemes, save/load) | ✅ PASS |
| 3 | History service (undo/redo) | ✅ PASS |
| 4 | ProjectConfig (round-trip + A-06 fields) | ✅ PASS |
| 5 | Multi-scale pooling backward compat (scales=[1]) | ✅ PASS |
| 6 | Multi-scale output shapes ([1]→768, [1,2]→3840, [1,2,4]→16128) | ✅ PASS |
| 7 | CLI (all subcommand --help) | ✅ PASS |
| 8 | Bout service (find_bouts) | ✅ PASS |
| 9 | Mask filter (filter_largest_component) | ✅ PASS |
| 10 | Environment (get_device, get_num_workers) | ✅ PASS |
| 11 | Cluster tree markdown | ✅ PASS |
| 12 | Lazy import speed (< 0.5s) | ✅ PASS |
| 13 | EmbeddingScatterPlot import | ✅ PASS |
| 14 | H5IO context manager | ✅ PASS |

---

## Phase 5: UI/Frontend Verification

| Component | Status | Details |
|-----------|--------|---------|
| Gradio UI | ✅ PASS | `from castle.ui import create_ui` |
| Desktop (PyQt6) | ✅ PASS | MainWindow, ProjectPanel, ClusterPanel, ExtractPanel, TrackingPanel |
| CLI (typer) | ✅ PASS | project, cluster, track, extract, info — all `--help` exit 0 |

---

## Phase 6: File Consistency

### Syntax Check
```
python syntax_check.py castle/ --exclude aot,sam,dinov2,dinov3 --json
```
- **83/83 files**: All pass Python syntax validation ✅

### Git State
- `git status --short`: Clean (no uncommitted changes)
- `git diff --check`: Clean (no whitespace issues)

---

## Phase 7: Documentation

### Docstring Coverage
- **Module docstrings**: 60/60 — all present ✅
- **Class docstrings**: 20 missing (mostly in utils/ and core/ layers)

Missing class docstrings:
| File | Class |
|------|-------|
| core/data.py | Preprocess, VideoDataset |
| core/environment.py | Environment |
| core/extractor.py | ProgressCallback |
| core/models.py | DINOv2Encoder, DINOv3Encoder |
| ui/plot_mask_info.py | Plotter |
| utils/explorer.py | Latent, FocusLatent |
| utils/image_segment.py | Segmentor, MultiObjectSegmentor |
| utils/latent_explorer.py | Latent, LocalLatent |
| utils/myumap.py | UMAP |
| utils/profiler.py | Profiler, TimeBlock, SystemMonitor |
| utils/video_object_segment.py | AOTTracker, AOTTrackerInferEngine, DeAOTTrackerInferEngine |

### README
- ✅ Exists (`README.md`, 5871 bytes)

### IMPROVEMENT_PLAN
- ✅ Exists (`docs/IMPROVEMENT_PLAN.md`)
- ⚠️ **Slightly outdated**: B-04 (Undo/Redo) and A-05 (Desktop) show 🔄 but are implemented per git log (`574a47f`, `161221c`)

---

## Delta from Previous Audit

| Metric | Previous | Current | Δ |
|--------|----------|---------|---|
| Unit tests | 102 passed | 134 passed | **+32** ✅ |
| Import failures | 1 (myumap) | 1 (myumap) | 0 (unchanged, expected) |
| E722 bare except | 2 → 0 | 0 | Maintained ✅ |
| T201 print stmts | 40 → 0 | 0 | Maintained ✅ |
| Missing module docstrings | 17 → 0 | 0 | Maintained ✅ |
| Missing class docstrings | N/A | 20 | New check (not in previous) |
| F401 unused imports | N/A | 78 | New finding (ruff) |
| F841 unused variables | N/A | 9 | New finding (ruff) |
| B006 mutable defaults | N/A | 7 | New finding (ruff) |

### What improved:
- **+32 new unit tests** covering A-06 multi-scale pooling, B-04 undo/redo, and visualization
- **All previous fixes maintained** (bare except, print statements, module docstrings)
- **New features verified**: Multi-scale pooling, Desktop GUI, CLI, ProjectConfig A-06 fields

### What needs attention:
- **78 unused imports** — mostly typing imports and re-exports missing `__all__`
- **7 mutable default arguments** — should use `None` + default pattern
- **20 class docstrings** — secondary priority (utils/low-level classes)
- **IMPROVEMENT_PLAN** status for B-04/A-05 outdated

---

## Recommendations

### Priority 1 (Quick wins):
1. Run `ruff check --fix --select F401` to auto-clean 67 unused imports (add `__all__` to `__init__.py` re-exports first)
2. Fix 7 B006 mutable defaults (change `def f(x=[])` to `def f(x=None); x = x or []`)

### Priority 2 (Housekeeping):
3. Add class docstrings to the 20 missing classes
4. Update IMPROVEMENT_PLAN.md: mark B-04 and A-05 as ✅

### Priority 3 (Nice to have):
5. Remove `castle/utils/video_io_old.py` (appears to be dead code importing from video_io)
6. Clean up 9 unused variable assignments

---

*Baseline saved to `.qc-baseline.json`*
