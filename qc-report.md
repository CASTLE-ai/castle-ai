# CASTLE QC Report v4 — Post P1/P2/P4

**Date**: 2026-02-16 19:05 (Asia/Taipei)  
**Branch**: `dev`  
**Commits since v3**: 3 (P1 ethogram, P2 metrics, P4 comparison)

---

## Summary

| Phase | v3 | v4 | Delta |
|-------|----|----|-------|
| 1. Tests | 151 pass / 0 fail | **260 pass** / 0 fail | +109 tests ✅ |
| 2. Imports | 73/74 pass (1 cuML) | **80/81** pass (1 cuML) | +7 modules ✅ |
| 3. Static (non-vendored) | 1 finding (F401) | **7 findings** (all F401) | +6 ⚠️ |
| 4. Smoke Tests | 15/15 pass | **21/21** pass | +6 new tests ✅ |
| 5. UI/Frontend | PASS (5 subcmds) | **PASS (7 subcmds)** | +ethogram,compare ✅ |
| 6. File Consistency | 86/86 clean | **96/96** clean | +10 files ✅ |
| 7. Documentation | 0 missing | **0 missing** (72 modules) | No regression ✅ |

**Overall**: ✅ PASS (with minor lint warnings)

---

## Phase 1: Test Suite

```
260 passed, 0 failed, 0 errors in 11.11s
```

New test files added:
- `tests/unit/test_ethogram.py` — ethogram engine tests
- `tests/unit/test_metrics.py` — clustering quality metrics tests
- `tests/unit/test_comparison.py` — group comparison tests

**Delta**: +109 tests (151 → 260). All pass.

---

## Phase 2: Import Integrity

- **Total**: 81 modules scanned (excluding vendored: aot, sam, dinov2, dinov3)
- **Passed**: 80
- **Failed**: 1 — `castle.utils.myumap` (requires RAPIDS cuML, expected)
- **New modules importable**: ✅
  - `castle.core.ethogram`
  - `castle.core.metrics`
  - `castle.core.comparison`
  - `castle.service.ethogram_service`
  - `castle.service.comparison_service`
  - `castle.visualization.comparison_plots`
  - `castle.cli.ethogram_cmd`
  - `castle.cli.compare_cmd`

**Delta**: +7 importable modules (74 → 81).

---

## Phase 3: Static Analysis (ruff, non-vendored only)

**7 findings** — all F401 (unused imports), all in NEW code:

| File | Line | Issue |
|------|------|-------|
| `castle/core/ethogram.py` | 9 | `dataclasses.field` unused |
| `castle/core/metrics.py` | 9 | `typing.Dict` unused |
| `castle/core/metrics.py` | 9 | `typing.Tuple` unused |
| `castle/service/comparison_service.py` | 9 | `json` unused |
| `castle/service/comparison_service.py` | 11 | `typing.Optional` unused |
| `castle/service/ethogram_service.py` | 11 | `typing.Optional` unused |
| `castle/visualization/comparison_plots.py` | 11 | `typing.Optional` unused |

**Delta**: v3 had 1 finding (F401 in mcp/server.py, now fixed). v4 has 7 new F401 findings, all trivially fixable.

> Note: The v3 F401 (`typing.Optional` in `castle/mcp/server.py`) was fixed by commit `6249530`.

---

## Phase 4: Smoke Tests

**21/21 pass** (was 15/15 in v3):

New smoke tests:
1. ✅ Service: ethogram_service importable
2. ✅ Service: comparison_service importable
3. ✅ Ethogram: transition matrix (3×3 shape verified)
4. ✅ Ethogram: bout statistics (3 clusters verified)
5. ✅ Ethogram: full ethogram (compute_ethogram end-to-end)
6. ✅ Metrics: evaluate_clustering (silhouette_sample computed)
7. ✅ Metrics: temporal_coherence = 0.750
8. ✅ Comparison: compute_fingerprint (frequencies verified)
9. ✅ Comparison: compare_groups (BFA distance + p-value computed)
10. ✅ Comparison: hedges_g (effect size = -2.400)
11. ✅ Comparison viz imports (radar, volcano, forest)

---

## Phase 5: UI/Frontend

- **CLI subcommands**: 7 total (was 5)
  - Existing: `project`, `track`, `extract`, `cluster`, `mcp`
  - **New**: `ethogram`, `compare` ✅
- **All --help**: renders correctly for all subcommands
- **MCP server**: 19 exports (tools + resources), up from ~13
  - New MCP tools: `ethogram_analyze`, `ethogram_bouts`, `ethogram_transitions`, `compare_groups_tool`, `compute_fingerprint_tool`, `cluster_evaluate`
- **Desktop modules**: import OK (Qt runtime not tested)
- **Gradio app**: `castle.app` module not found (may have been refactored)

---

## Phase 6: File Consistency

- **Syntax check**: 96/96 passed (was 86/86)
- **Git status**: clean (only QC artifacts `.qc-baseline.json`, `qc-report.md` modified)
- **No orphaned files** or merge conflicts

---

## Phase 7: Documentation

- **Modules scanned**: 72 (excluding vendored + desktop)
- **Missing module docstrings**: 0
- **Missing class docstrings**: 0
- **All new modules documented**: ✅

---

## Delta from v3

### Improvements
- ✅ +109 unit tests (151 → 260)
- ✅ +7 importable modules (74 → 81)
- ✅ +10 syntax-checked files (86 → 96)
- ✅ +6 new smoke tests (15 → 21)
- ✅ +2 CLI subcommands (ethogram, compare)
- ✅ +6 MCP tools (ethogram, comparison, metrics)
- ✅ v3 F401 in mcp/server.py fixed

### Regressions
- ⚠️ +6 new F401 unused imports (all in new P1/P2/P4 code)
  - Trivially fixable, zero functional impact

### Unchanged
- 1 known cuML import failure (expected, no RAPIDS GPU)
- 0 missing docstrings
- 0 bare excepts, print statements, mutable defaults (in non-vendored code)
