"""
castle/core/cluster_data.py
Unified cluster data structure.

Consolidates the various cluster artefacts (cluster_*.npz, time_series_*.csv,
id.csv, annotations.csv) into a single, typed container.
"""

from __future__ import annotations

import glob
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_UNSET_LABELS: np.ndarray = np.array([], dtype=np.int64)


def _parse_color(raw: object) -> tuple[int, int, int]:
    """Convert a stored color value to an ``(R, G, B)`` integer tuple.

    Supports:

    * Named CSS colors (string) — mapped via :mod:`matplotlib.colors` when
      available, otherwise defaults to ``(128, 128, 128)``.
    * ``"#RRGGBB"`` hex strings.
    * Already-tuple/list values.

    Args:
        raw: Raw color value as read from ``id.csv``.

    Returns:
        ``(R, G, B)`` tuple with values in ``[0, 255]``.
    """
    if isinstance(raw, (tuple, list)) and len(raw) >= 3:  # noqa: UP006
        return (int(raw[0]), int(raw[1]), int(raw[2]))

    if isinstance(raw, str):
        s = raw.strip()
        if s.startswith("#") and len(s) == 7:
            return (int(s[1:3], 16), int(s[3:5], 16), int(s[5:7], 16))
        # Try matplotlib named colours.
        try:
            import matplotlib.colors as mcolors  # noqa: PLC0415

            rgba = mcolors.to_rgba(s)
            return (int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255))
        except Exception:  # noqa: BLE001
            pass

    return (128, 128, 128)  # fallback: grey


# ---------------------------------------------------------------------------
# ClusterData
# ---------------------------------------------------------------------------


@dataclass
class ClusterData:
    """Unified cluster data structure.

    Consolidates ``cluster_*.npz``, ``time_series_*.csv``, ``id.csv``, and
    ``annotations.csv`` into a single, typed container.

    Attributes:
        labels:      Flat leaf cluster assignments for every temporal bin,
                     shape ``(N,)``.  ``-1`` means unassigned.
        hierarchy:   Optional tree structure produced by hierarchical
                     clustering (pass-through; not interpreted here).
        names:       Mapping ``cluster_id → human-readable name``.
        colors:      Mapping ``cluster_id → (R, G, B)`` tuple.
        annotations: Mapping ``cluster_id → annotation label string``.

    Example::

        cd = ClusterData.load("/data/projects/my_project/cluster")
        print(cd.n_clusters())
        frames = cd.get_cluster_frames(0)
    """

    labels: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    hierarchy: dict = field(default_factory=dict)
    names: dict[int, str] = field(default_factory=dict)
    colors: dict[int, tuple] = field(default_factory=dict)
    annotations: dict[int, str] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Class-method constructors
    # ------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        cluster_dir: str | Path,
        session_id: Optional[str] = None,
    ) -> "ClusterData":
        """Load cluster data from a project cluster directory.

        Reads (in order of precedence):

        1. ``id.csv`` — cluster IDs, names, colors.
        2. ``cluster_*.npz`` — flat label array (``cluster`` key expected).
        3. ``time_series_*.csv`` — concatenated as fallback label source when
           no ``.npz`` label array is present.
        4. ``annotations.csv`` (or ``sessions/<session_id>/annotations.csv``)
           — per-cluster annotation strings.

        Args:
            cluster_dir: Path to the project's ``cluster/`` directory.
            session_id:  Optional session identifier for per-session files.

        Returns:
            A populated :class:`ClusterData` instance.

        Raises:
            FileNotFoundError: If *cluster_dir* does not exist.
        """
        import pandas as pd  # noqa: PLC0415

        cluster_dir = Path(cluster_dir)
        if not cluster_dir.is_dir():
            raise FileNotFoundError(f"Cluster directory not found: {cluster_dir}")

        # ---- 1. id.csv → names + colors --------------------------------
        names: dict[int, str] = {}
        colors: dict[int, tuple] = {}
        id_csv = cluster_dir / "id.csv"
        if id_csv.exists():
            try:
                id_df = pd.read_csv(id_csv)
                for _, row in id_df.iterrows():
                    cid = int(row["Id"])
                    names[cid] = str(row.get("Name", f"cluster_{cid}"))
                    colors[cid] = _parse_color(row.get("Color", "grey"))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to read id.csv: %s", exc)

        # ---- 2. cluster_*.npz → labels + hierarchy ---------------------
        labels: np.ndarray = _UNSET_LABELS.copy()
        hierarchy: dict = {}
        npz_files = sorted(glob.glob(str(cluster_dir / "cluster_*.npz")))
        if npz_files:
            # Use the first (or only) non-model npz that contains a 'cluster' key.
            for npz_path in npz_files:
                try:
                    data = np.load(npz_path, allow_pickle=True)
                    if "cluster" in data:
                        labels = data["cluster"].astype(np.int64)
                        if "hierarchy" in data:
                            hierarchy = data["hierarchy"].item()
                        break
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Could not load %s: %s", npz_path, exc)

        # ---- 3. time_series_*.csv → fallback label source --------------
        if labels.size == 0:
            ts_files = sorted(glob.glob(str(cluster_dir / "time_series_*.csv")))
            if ts_files:
                chunks: list[np.ndarray] = []
                for ts_path in ts_files:
                    try:
                        ts_df = pd.read_csv(ts_path)
                        col = "cluster_id" if "cluster_id" in ts_df.columns else ts_df.columns[0]
                        chunks.append(ts_df[col].to_numpy(dtype=np.int64))
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Could not load %s: %s", ts_path, exc)
                if chunks:
                    labels = np.concatenate(chunks)

        # ---- 4. annotations.csv → annotations --------------------------
        annotations: dict[int, str] = {}
        if session_id:
            ann_path = cluster_dir / "sessions" / session_id / "annotations.csv"
        else:
            ann_path = cluster_dir / "annotations.csv"

        if ann_path.exists():
            try:
                ann_df = pd.read_csv(ann_path)
                # Support either cluster_id or cluster_name as key column.
                if "cluster_id" in ann_df.columns:
                    for _, row in ann_df.iterrows():
                        cid = int(row["cluster_id"])
                        annotations[cid] = str(row.get("annotation", ""))
                elif "cluster_name" in ann_df.columns:
                    # Resolve name → id via names mapping.
                    name2id = {v: k for k, v in names.items()}
                    for _, row in ann_df.iterrows():
                        cname = str(row["cluster_name"])
                        cid = name2id.get(cname, -1)
                        if cid >= 0:
                            annotations[cid] = str(row.get("annotation", ""))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to read annotations: %s", exc)

        return cls(
            labels=labels,
            hierarchy=hierarchy,
            names=names,
            colors=colors,
            annotations=annotations,
        )

    @classmethod
    def from_arrays(
        cls,
        embeddings: np.ndarray,
        cluster_ids: np.ndarray,
        hierarchy: Optional[dict] = None,
    ) -> "ClusterData":
        """Create a :class:`ClusterData` from raw arrays.

        Useful when constructing cluster data from freshly computed clustering
        results, before any files have been written.

        Args:
            embeddings:  Raw embedding array, shape ``(N, D)``.  Not stored
                         directly; used here only for shape validation.
            cluster_ids: Integer cluster assignment per embedding, shape ``(N,)``.
            hierarchy:   Optional hierarchical tree structure.

        Returns:
            A :class:`ClusterData` with auto-generated names and grey colors.

        Raises:
            ValueError: If *embeddings* and *cluster_ids* have mismatched lengths.
        """
        if len(embeddings) != len(cluster_ids):
            raise ValueError(
                f"embeddings ({len(embeddings)}) and cluster_ids ({len(cluster_ids)}) "
                "must have the same length."
            )

        labels = np.asarray(cluster_ids, dtype=np.int64)
        unique_ids = sorted(set(int(c) for c in labels if c >= 0))
        names = {cid: f"cluster_{cid}" for cid in unique_ids}
        colors: dict[int, tuple] = {cid: (128, 128, 128) for cid in unique_ids}

        return cls(
            labels=labels,
            hierarchy=hierarchy or {},
            names=names,
            colors=colors,
            annotations={},
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(
        self,
        cluster_dir: str | Path,
        session_id: Optional[str] = None,
    ) -> None:
        """Save cluster data to a project cluster directory.

        Writes:

        * ``id.csv`` — cluster IDs, names, colors.
        * ``cluster_data.npz`` — labels and hierarchy arrays.
        * ``annotations.csv`` (or ``sessions/<session_id>/annotations.csv``).

        Args:
            cluster_dir: Path to the project's ``cluster/`` directory.
            session_id:  Optional session identifier for per-session files.
        """
        import pandas as pd  # noqa: PLC0415

        cluster_dir = Path(cluster_dir)
        cluster_dir.mkdir(parents=True, exist_ok=True)

        # ---- id.csv ---------------------------------------------------
        id_rows = sorted(self.names.items())
        id_df = pd.DataFrame(
            [
                {
                    "Id": cid,
                    "Name": self.names.get(cid, f"cluster_{cid}"),
                    "Color": "#{:02x}{:02x}{:02x}".format(*self.colors.get(cid, (128, 128, 128))),
                }
                for cid, _ in id_rows
            ]
        )
        id_df.to_csv(cluster_dir / "id.csv", index=False)

        # ---- cluster_data.npz ----------------------------------------
        save_kwargs: dict = {"cluster": self.labels}
        if self.hierarchy:
            save_kwargs["hierarchy"] = np.array(self.hierarchy, dtype=object)
        np.savez(cluster_dir / "cluster_data.npz", **save_kwargs)

        # ---- annotations.csv -----------------------------------------
        if self.annotations:
            if session_id:
                ann_dir = cluster_dir / "sessions" / session_id
                ann_dir.mkdir(parents=True, exist_ok=True)
                ann_path = ann_dir / "annotations.csv"
            else:
                ann_path = cluster_dir / "annotations.csv"

            ann_df = pd.DataFrame(
                [
                    {"cluster_id": cid, "annotation": ann}
                    for cid, ann in sorted(self.annotations.items())
                ]
            )
            ann_df.to_csv(ann_path, index=False)

        logger.info("ClusterData saved to %s", cluster_dir)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_cluster_frames(self, cluster_id: int) -> np.ndarray:
        """Return the frame (bin) indices belonging to *cluster_id*.

        Args:
            cluster_id: Integer cluster identifier.

        Returns:
            1-D integer array of indices into :attr:`labels` where the
            assignment equals *cluster_id*.
        """
        return np.where(self.labels == cluster_id)[0]

    def n_clusters(self) -> int:
        """Return the number of unique cluster IDs (excluding ``-1``).

        Returns:
            Count of distinct non-negative cluster IDs.
        """
        if self.labels.size == 0:
            return len(self.names)
        return int(np.sum(np.unique(self.labels) >= 0))
