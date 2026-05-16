"""Shared types and exception hierarchy for CASTLE (P0-C / ARCH-04).

This module centralizes:

- ``CastleError`` — root exception type all CASTLE-raised errors inherit from.
  Splitting into three sub-bases lets callers catch coarsely
  (``except CastleIOError``) or finely (``except VideoReadError``):

  - ``CastleIOError`` — I/O / file / network / HDF5 problems.
  - ``CastleDataError`` — data shape / quality / domain problems.
  - ``CastleAlgorithmError`` — algorithm-level failures.

- ``ExtractionResult`` — frozen dataclass describing the outcome of a
  successful latent-extraction run (path, frame count, batches failed,
  feature dim). Currently used by callers that want richer info than a
  bare path string; the core ``extract_*`` functions continue to return
  a path string and raise on failure for backwards compatibility.

Design note: the hierarchy is intentionally narrow (three coarse bases +
a small number of leaves) — see plan section [D11].
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class CastleError(Exception):
    """Root of every error CASTLE deliberately raises.

    Catch this to handle any expected CASTLE failure without also swallowing
    unrelated ``Exception`` subclasses.
    """


class CastleIOError(CastleError):
    """I/O / file / network / HDF5 problems."""


class CastleDataError(CastleError):
    """Data shape / quality / domain problems."""


class CastleAlgorithmError(CastleError):
    """Algorithm-level failures (model load, clustering convergence, ...)."""


# --- I/O leaves --------------------------------------------------------------


class VideoReadError(CastleIOError):
    """Raised when a video file cannot be opened or its metadata is unusable."""


class MaskNotFoundError(CastleIOError):
    """Raised when a tracker mask file required for extraction is missing."""


class LatentCorruptError(CastleIOError):
    """Raised when an existing latent ``.npz`` is unreadable or malformed."""


# --- Data leaves -------------------------------------------------------------


class ROINotFoundError(CastleDataError):
    """Raised when a required ROI id is not present in a frame's mask."""


class InsufficientDataError(CastleDataError):
    """Raised when an algorithm needs more samples than were supplied.

    Example: UMAP with ``n_neighbors`` greater than the number of points.
    """


class PreprocessingError(CastleDataError):
    """Raised when a frame's preprocessing pipeline fails."""


# --- Algorithm leaves --------------------------------------------------------


class ExtractionError(CastleAlgorithmError):
    """Raised when latent extraction cannot produce a usable result."""


class NoClustersFound(CastleAlgorithmError):
    """Raised when clustering converges with fewer than two clusters."""


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExtractionResult:
    """Summary of one successful ``extract_roi_latent_from_video`` run.

    Attributes:
        latent_path: Absolute path to the saved ``.npz`` file.
        n_frames: Total frames represented in the latent (post-binning).
        n_batches_failed: How many DataLoader batches failed during the
            run (below the abort threshold; see [BUG-04]).
        feature_dim: Last-axis dimensionality of the latent tensor.
    """

    latent_path: Path
    n_frames: int
    n_batches_failed: int
    feature_dim: int
