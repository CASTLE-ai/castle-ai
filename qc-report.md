# QC Report v5 — CASTLE Behavioral Analysis Framework

**Date**: 2026-02-16 20:10 (Asia/Taipei)  
**Commit**: `139d649` (branch: `dev`)  
**Verdict**: ✅ **PASS**

---

## Summary

| Phase | Status | Details |
|-------|--------|---------|
| 1. Test Suite | ✅ PASS | 335 passed, 0 failed (14.3s) |
| 2. Import Integrity | ✅ PASS | 81/82 ok, 1 optional (cuML) |
| 3. Static Analysis | ✅ PASS | 0 findings (ruff E722/T201/B006/F401/F841) |
| 4. Smoke Tests | ✅ PASS | 26/26 passed |
| 5. UI/Frontend | ✅ PASS | 7 CLI + 5 new subcommands + MCP |
| 6. File Consistency | ✅ PASS | 0 placeholders, clean git |
| 7. Documentation | ✅ PASS | 23/23 docstrings, concepts page, GPU benchmarks |

---

## Delta from v4

| Metric | v4 | v5 | Change |
|--------|-----|-----|--------|
| Unit tests | 260 | 335 | **+75** ✅ |
| Lint findings (non-vendored) | 7 F401 | 0 | **-7** ✅ |
| Smoke tests | 21 | 26 | **+5** ✅ |
| `[HUMAN TO CONFIRM]` | 5 | 0 | **-5** ✅ |
| New modules | 0 | 6 | **+6** |
| New CLI commands | 0 | 5 | **+5** |
| New MCP tools | 0 | ~5 | **+5** |
| Docstring coverage (new) | — | 23/23 | **100%** |

---

## Phase 1: Test Suite

```
335 passed, 1 warning in 14.34s
```

New test files (+75):
- `tests/unit/test_temporal_smooth.py` — 22 tests
- `tests/unit/test_cluster_transfer.py` — 16 tests
- `tests/unit/test_paired.py` — 24 tests
- `tests/unit/test_nwb_export.py` — 13 tests

All tests run in under 15 seconds. No timeouts, no flaky tests.

## Phase 2: Import Integrity

- 82 non-vendored modules scanned
- 81 import successfully
- 1 optional failure: `castle.utils.myumap` (requires cuML — GPU-only, auto-fallback to umap-learn)
- 0 critical failures

## Phase 3: Static Analysis

```
ruff check castle/ --exclude thirdparty,aot,sam,dinov2 --select E722,T201,B006,F401,F841
All checks passed!
```

Previous v4 had 7 F401 (unused imports) — all fixed in `c269dba` and `139d649`.

## Phase 4: Smoke Tests (26/26)

### New Feature Smoke Tests
| Test | Result |
|------|--------|
| `temporal_smooth` round-trip (median + min_bout) | ✅ |
| `cluster_transfer` save → load → apply | ✅ |
| `auto_cluster` scoring + MICROSCOPE_PRESETS (4 presets) | ✅ |
| `CASTLE_STORAGE` environment variable | ✅ |
| `NWB export` round-trip (write → verify exists, 196KB) | ✅ |

### CLI Subcommand Tests
| Command | Result |
|---------|--------|
| `castle project --help` | ✅ |
| `castle track --help` | ✅ |
| `castle extract --help` | ✅ |
| `castle cluster --help` | ✅ |
| `castle ethogram --help` | ✅ |
| `castle compare --help` | ✅ |
| `castle info --help` | ✅ |
| `castle cluster auto --help` | ✅ |
| `castle cluster save-model --help` | ✅ |
| `castle cluster apply-model --help` | ✅ |
| `castle ethogram export-nwb --help` | ✅ |
| `castle compare run` has `--paired` flag | ✅ |

### Module Import Tests
All 6 new core modules + 2 new service modules import cleanly.

## Phase 5: UI/Frontend

- **CLI**: 7 top-level commands, 5 new subcommands — all `--help` works
- **MCP Server**: 21+ tools/resources (9 original + ethogram + metrics + comparison + cluster transfer + auto + NWB)
- **Gradio UI**: Not tested (requires GPU/display)
- **Desktop**: Not tested (requires Qt runtime)

## Phase 6: File Consistency

- `[HUMAN TO CONFIRM]` placeholders: **0** (was 5 in docs)
- Git status: clean working tree
- No syntax errors in any non-vendored Python file

## Phase 7: Documentation

### New Documentation
- `docs/getting-started/concepts.md` — biology-friendly explanation of CASTLE
- `docs/getting-started/gpu-requirements.md` — filled with RTX 4090 benchmark data + comparison table
- `docs/tutorials/step3-extract.md` — timing benchmarks added
- `docs/tutorials/step4-analysis.md` — practical tips added
- `mkdocs.yml` — Core Concepts added to nav

### Docstring Coverage (New Modules)
- 23 docstrings checked across 6 new files: **23/23 present (100%)**

---

## New Features Added This Session

1. **Temporal Smoothing** (`castle/core/temporal_smooth.py`)
   - Median filter + minimum bout duration filter
   - Integrated into ethogram pipeline (`--smooth`)
   
2. **Cluster Transfer** (`castle/core/cluster_transfer.py`)
   - Save/load clustering models (.npz)
   - k-NN classification in feature space or UMAP space
   - Enables longitudinal studies

3. **Paired Statistical Tests** (`castle/core/comparison.py`)
   - Sign-flip paired permutation test
   - Per-feature paired tests with BH-FDR
   - Paired Hedges' g effect sizes

4. **NWB Export** (`castle/core/nwb_export.py`)
   - Export to Neurodata Without Borders format
   - BehavioralTimeSeries + TimeIntervals + bout stats
   - Optional dependency (pynwb)

5. **CASTLE_STORAGE Environment Variable** (`castle/cli/storage_util.py`)
   - No more `--storage` on every command
   - Priority: arg → env var → current dir

6. **Automated Behavior Microscope** (`castle/core/auto_cluster.py`)
   - Parameter sweep using Raiso-optimized presets
   - Quality-based selection (temporal coherence + CH + bout quality)
   - `castle cluster auto` CLI

---

*QC v5 completed. PASS.*
