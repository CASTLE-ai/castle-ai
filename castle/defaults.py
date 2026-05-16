"""Centralized defaults for CASTLE pipeline (UX-03).

Every constant here is the **single source of truth** for a value that
historically appeared in two or more call sites (CLI / project config /
service layer / MCP server). When a default needs to change, change it
here — no more "I updated batch_size in project_config but forgot
extraction_service".

Every entry is paired with a docstring explaining **why** that specific
number was chosen, so reviewers can challenge the rationale instead of
guessing intent.

Scope rule (kept narrow on purpose):
- Only values used at ≥2 call sites are centralized.
- Single-use defaults stay at their definition site to avoid premature
  abstraction.

Not the right home for:
- Per-project user-tunable config (lives in
  :class:`castle.core.project_config.ProjectConfig`).
- Internal cache sizes that the user never sees and that only one module
  cares about (e.g. ``_VIDEO_READER_CACHE_MAX`` stays in ``cluster.py``).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

EXTRACTION_BATCH_SIZE: int = 32
"""Batch size for DINOv3 feature extraction.

Rationale: On a 12 GB GPU running ViT-B/16 at 592×592, 32 frames fit in
roughly 9 GB of VRAM with the SDPA attention path, leaving headroom for
variance in mask size. Reduce to 16 on 8 GB cards (the
:class:`~castle.core.project_config.ExtractionConfig` field lets the
user override per-project).

Used by: ``ExtractionConfig.batch_size``, ``extraction_service`` defaults,
``worker_threads`` defaults, ``auto_config`` heuristics.
"""

TRACKING_BATCH_SIZE: int = 16
"""Batch size for ROI tracking (SAM2 + DeAOT).

Rationale: Tracking holds the full mask volume + visual encoder activations
in VRAM, so the practical batch is half that of pure feature extraction.
16 is the lowest "still comfortable" batch on a 12 GB card.

Used by: ``TrackingConfig.batch_size``, ``pipeline``,
``tracking_manager``, ``models.SAM2Model``.
"""

# ---------------------------------------------------------------------------
# Temporal binning
# ---------------------------------------------------------------------------

BIN_SIZE: int = 1
"""Temporal binning factor: aggregate N consecutive frames into one latent.

Rationale: ``bin_size=1`` preserves frame-level resolution, which is what
researchers default to when exploring a new dataset for the first time.
Increase to 3–5 for 30+ fps videos where consecutive frames are highly
correlated (saves 3–5× storage and clustering time at minimal cost to
behavioral resolution).

Used by: ``ExtractionConfig.bin_size``, ``ClusteringSession`` default,
``mcp.server`` tool default, desktop ``SyllableBar`` initial state.
"""

# ---------------------------------------------------------------------------
# UMAP defaults
# ---------------------------------------------------------------------------

UMAP_N_NEIGHBORS: int = 100
"""Default UMAP ``n_neighbors`` for the Behavior Microscope.

Rationale: At ~30 fps, 100 neighbors corresponds to roughly 3 seconds of
behavior context — large enough to capture short bouts (grooming, rearing)
without smearing across longer locomotion sequences. The exact value isn't
load-bearing; UMAP is fairly insensitive in the 50–300 range for the
datasets CASTLE targets.

Used by: ``UMAPConfig.n_neighbors``, ``mcp.server`` tool default.
"""

UMAP_MIN_DIST: float = 0.0
"""Default UMAP ``min_dist``.

Rationale: ``min_dist=0`` produces tightly-packed clusters, which is
exactly what DBSCAN downstream needs to find sharp density boundaries.
Larger ``min_dist`` spreads the embedding for visual clarity but blurs
DBSCAN's edges — only useful when the embedding is the *final* product
(rare in CASTLE workflows).

Used by: ``UMAPConfig.min_dist``, ``mcp.server`` tool default.
"""

# ---------------------------------------------------------------------------
# DBSCAN defaults
# ---------------------------------------------------------------------------

DBSCAN_EPS: float = 1.0
"""Default DBSCAN ``eps`` (neighborhood radius in UMAP space).

Rationale: With ``UMAP_MIN_DIST=0`` and ``UMAP_N_NEIGHBORS=100``, the
intra-cluster nearest-neighbor distance in 2D UMAP output sits around
0.3–0.7 for typical CASTLE datasets, and inter-cluster gaps land near
1.0–2.0. ``eps=1.0`` is the "click run and see what falls out" starting
point — the Behavior Microscope is designed so the user sweeps this value
interactively rather than relying on the default for science.

Used by: ``ClusterConfig.eps``, ``mcp.server`` tool default,
``cli.cluster_cmd`` ``--eps`` option default,
``clustering_service`` docstring example.
"""

# ---------------------------------------------------------------------------
# Memory budgets
# ---------------------------------------------------------------------------

MEMMAP_THRESHOLD_GB: float = 2.0
"""Latent-aggregation memmap switch threshold, in **GiB**.

Rationale: A 1 h video at 30 fps with DINOv3-vitb16 (768-dim float32)
produces about 320 MB of latent; 10 such videos = 3.2 GB. A typical
desktop has 16–32 GB RAM, so a 2 GiB threshold leaves comfortable
headroom for downstream UMAP/DBSCAN scratch space without prematurely
forcing disk I/O on small projects.

Override at runtime via the ``CASTLE_MEMMAP_THRESHOLD_GB`` environment
variable (see :func:`castle.core.cluster._memmap_threshold_bytes`).

Used by: ``cluster._DEFAULT_MEMMAP_THRESHOLD_GB`` (env-var fallback).
"""
