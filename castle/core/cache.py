"""
castle/core/cache.py
Content-addressed cache for pipeline outputs.

Cache key = SHA-256(video_path + file_mtime + preprocess_config + model_name)
Manifest stored as JSON in: {cache_dir}/.cache_manifest.json
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_MANIFEST_FILENAME = ".cache_manifest.json"


class PipelineCache:
    """Content-addressed cache for pipeline outputs.

    Maps a deterministic hash key to the path of a previously computed
    output file (e.g. a ``*.npz`` latent array).  The manifest is persisted
    as JSON inside *cache_dir* so it survives process restarts.

    Parameters
    ----------
    cache_dir : str
        Directory where the manifest JSON file is stored (e.g.
        ``{project}/latent/``).  Created automatically if absent.

    Example
    -------
    >>> cache = PipelineCache("/data/project/latent")
    >>> key = cache.compute_key("/data/project/sources/vid.mp4",
    ...                         {"center_roi": True}, "dinov2_vitb14")
    >>> if cache.is_cached(key):
    ...     path = cache.get(key)
    ... else:
    ...     path = run_extraction(...)
    ...     cache.put(key, path)
    """

    def __init__(self, cache_dir: str) -> None:
        self.cache_dir = cache_dir
        self._manifest_path = os.path.join(cache_dir, _MANIFEST_FILENAME)
        os.makedirs(cache_dir, exist_ok=True)
        self._manifest: dict = self._load_manifest()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _load_manifest(self) -> dict:
        """Load the manifest from disk, returning an empty dict on failure."""
        if not os.path.exists(self._manifest_path):
            return {}
        try:
            with open(self._manifest_path, encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, dict):
                logger.warning("Cache manifest corrupted (not a dict); resetting.")
                return {}
            return data
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load cache manifest (%s); starting fresh.", exc)
            return {}

    def _save_manifest(self) -> None:
        """Persist the in-memory manifest to disk atomically."""
        tmp_path = self._manifest_path + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(self._manifest, fh, indent=2)
            os.replace(tmp_path, self._manifest_path)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to save cache manifest: %s", exc)
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_key(
        self,
        video_path: str,
        config: dict,
        model_name: str,
    ) -> str:
        """Compute a deterministic cache key from the extraction inputs.

        The key is derived from:
        - Absolute video path
        - File modification time (seconds since epoch, as a string)
        - Canonical JSON representation of *config*
        - *model_name*

        If the file does not exist the mtime is treated as ``"missing"``.

        Parameters
        ----------
        video_path : str
            Path to the source video file.
        config : dict
            Preprocessing configuration (must be JSON-serialisable).
        model_name : str
            Name of the visual encoder model.

        Returns
        -------
        str
            64-character hex SHA-256 digest.
        """
        abs_path = os.path.abspath(video_path)

        try:
            mtime = str(os.path.getmtime(abs_path))
        except OSError:
            mtime = "missing"

        config_json = json.dumps(config, sort_keys=True, ensure_ascii=False)

        payload = "\n".join([abs_path, mtime, config_json, model_name])
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def is_cached(self, key: str) -> bool:
        """Return *True* if *key* is registered **and** its output file exists.

        Stale entries (file deleted externally) are silently purged from the
        manifest.

        Parameters
        ----------
        key : str
            Cache key produced by :meth:`compute_key`.
        """
        if key not in self._manifest:
            return False

        output_path = self._manifest[key]
        if not os.path.exists(output_path):
            logger.debug(
                "Cache entry %s points to missing file %s; invalidating.",
                key[:12],
                output_path,
            )
            self.invalidate(key)
            return False

        return True

    def get(self, key: str) -> Optional[str]:
        """Return the cached output path for *key*, or *None* if not found.

        Parameters
        ----------
        key : str
            Cache key produced by :meth:`compute_key`.

        Returns
        -------
        str or None
            Absolute path to the cached output file, or ``None``.
        """
        if not self.is_cached(key):
            return None
        return self._manifest[key]

    def put(self, key: str, output_path: str) -> None:
        """Register *output_path* under *key* and persist the manifest.

        Parameters
        ----------
        key : str
            Cache key produced by :meth:`compute_key`.
        output_path : str
            Absolute path to the output file that was just created.
        """
        self._manifest[key] = os.path.abspath(output_path)
        self._save_manifest()
        logger.debug("Cache: stored key %s → %s", key[:12], output_path)

    def invalidate(self, key: str) -> None:
        """Remove *key* from the manifest (does **not** delete the file).

        Parameters
        ----------
        key : str
            Cache key to remove.  A no-op if the key is not present.
        """
        if key in self._manifest:
            del self._manifest[key]
            self._save_manifest()
            logger.debug("Cache: invalidated key %s", key[:12])

    def clear(self) -> None:
        """Remove all entries from the manifest (does **not** delete files)."""
        self._manifest.clear()
        self._save_manifest()
        logger.info("Cache: manifest cleared (%s)", self._manifest_path)

    def __len__(self) -> int:
        """Return the number of entries in the manifest."""
        return len(self._manifest)

    def __repr__(self) -> str:
        return (
            f"PipelineCache(cache_dir={self.cache_dir!r}, "
            f"entries={len(self._manifest)})"
        )
