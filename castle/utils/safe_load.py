"""Safe loaders for CASTLE latent ``.npz`` files (P1 / BUG-10).

A truncated or malformed ``.npz`` previously surfaced as a cryptic
``BadZipFile`` or ``KeyError: 'latent'`` deep inside ``LatentAggregator``,
breaking the whole clustering session. ``load_latent_safe`` converts these
into a :class:`LatentCorruptError` with an actionable hint so the user can
delete the offending file and re-run ``castle extract``.

This helper is intentionally tiny — it does not try to repair the file; the
correct response is always "delete and re-extract."
"""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path
from typing import Union

import numpy as np

from castle.core.types import LatentCorruptError

logger = logging.getLogger(__name__)


def load_latent_safe(path: Union[str, Path]) -> np.ndarray:
    """Load the ``latent`` array from a CASTLE-format ``.npz`` file.

    Args:
        path: Absolute or relative path to the ``.npz``.

    Returns:
        The 2D latent array, shape ``(N, F)``. Dtype is preserved.

    Raises:
        LatentCorruptError: The file is missing, unreadable, malformed, or
            does not contain a 2D ``latent`` entry. The exception message
            ends with a recovery hint pointing the user at ``castle extract``.
    """
    path = Path(path)
    if not path.exists():
        raise LatentCorruptError(
            f"Latent file not found: {path}. "
            f"Hint: re-run `castle extract` for the affected video."
        )

    try:
        with np.load(path, allow_pickle=False) as data:
            if 'latent' not in data.files:
                raise LatentCorruptError(
                    f"{path} is missing the 'latent' key. "
                    f"Keys present: {list(data.files)}. "
                    f"Hint: delete and re-run `castle extract`."
                )
            arr = np.asarray(data['latent'])
    except (zipfile.BadZipFile, EOFError, OSError, ValueError) as exc:
        # numpy raises plain ValueError for "Cannot load file containing
        # pickled data when allow_pickle=False" when the file is truncated
        # / not actually a zip — treat the same as a corrupt npz.
        raise LatentCorruptError(
            f"{path} is corrupted ({type(exc).__name__}: {exc}). "
            f"Hint: delete the file and re-run `castle extract` for the "
            f"affected video."
        ) from exc

    if arr.ndim != 2:
        raise LatentCorruptError(
            f"{path} has latent of shape {arr.shape}; expected 2D (N, F)."
        )

    if not np.isfinite(arr).all():
        n_bad = int((~np.isfinite(arr)).sum())
        # Actually handle non-finite values instead of only warning: normalise
        # +/-Inf to NaN so EVERY downstream non-finite check catches them
        # consistently. Previously Inf slipped past the NaN-only row exclusion
        # (np.isnan(inf) is False) and only blew up later at the embedding's
        # isfinite assertion, taking down the whole clustering session for one
        # bad video instead of excluding its bad rows (contract C-4).
        logger.warning(
            "%s contains %d non-finite values; converting to NaN so the "
            "affected rows are excluded downstream instead of crashing the "
            "clustering session.", path, n_bad,
        )
        arr = np.where(np.isfinite(arr), arr, np.nan)

    return arr
