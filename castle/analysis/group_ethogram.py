"""Group ethogram construction and visualization for multi-subject recordings.

Builds a synchronized, multi-subject ethogram from a list of
:class:`~castle.core.multi_subject.SubjectTrack` objects (each with ``labels``
assigned after clustering) and generates publication-quality visualizations
where each horizontal row represents one subject.

Usage example::

    from castle.analysis.group_ethogram import build_group_ethogram, plot_group_ethogram

    ethogram = build_group_ethogram(tracks, fps=30.0)
    path = plot_group_ethogram(ethogram, output_path="/tmp/group_ethogram.png")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from castle.core.multi_subject import SubjectTrack

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build_group_ethogram(
    tracks: list[SubjectTrack],
    fps: float,
    cluster_names: Optional[dict[int, str]] = None,
    distance_threshold: float = 50.0,
    duration_threshold: int = 15,
) -> dict:
    """Build a synchronized ethogram for all subjects in a multi-subject recording.

    For each subject a per-subject ethogram is computed via
    :func:`~castle.core.ethogram.compute_ethogram` (requires ``labels`` to be
    set on the track).  Social events are detected via
    :func:`~castle.analysis.social_features.detect_social_events` and included
    as a shared timeline.

    Parameters
    ----------
    tracks : list[SubjectTrack]
        Processed subject tracks.  Each track must have ``labels`` assigned.
    fps : float
        Frames per second (used to convert frame indices to time).
    cluster_names : dict[int, str] or None
        Optional mapping from cluster id to human-readable name shared across
        all subjects.
    distance_threshold : float
        Proximity threshold (px) forwarded to
        :func:`~castle.analysis.social_features.detect_social_events`.
    duration_threshold : int
        Minimum event duration (frames) forwarded to
        :func:`~castle.analysis.social_features.detect_social_events`.

    Returns
    -------
    dict
        Dictionary with the following structure::

            {
                "fps": float,
                "n_frames": int,
                "n_subjects": int,
                "subject_ids": list[int],
                "per_subject": {
                    subject_id: {
                        "ethogram": Ethogram,     # castle.core.ethogram.Ethogram
                        "labels": np.ndarray,     # (N,)
                        "cluster_names": dict,    # id → name
                    },
                    ...
                },
                "social_events": list[dict],  # from detect_social_events
                "time_axis": np.ndarray,      # (N,) seconds
            }

    Raises
    ------
    ValueError
        If *tracks* is empty, frame counts differ, or any track has no
        ``labels`` assigned.
    """
    from castle.analysis.social_features import detect_social_events
    from castle.core.ethogram import compute_ethogram

    if not tracks:
        raise ValueError("tracks must be a non-empty list.")

    n_frames = tracks[0].n_frames
    for t in tracks[1:]:
        if t.n_frames != n_frames:
            raise ValueError(
                f"Track {t.subject_id} has {t.n_frames} frames; "
                f"expected {n_frames}."
            )

    cluster_names = cluster_names or {}
    subject_ids = [t.subject_id for t in tracks]

    per_subject: dict[int, dict] = {}
    for track in tracks:
        if track.labels is None:
            raise ValueError(
                f"Track {track.subject_id} has no labels. "
                "Run clustering before building the group ethogram."
            )
        ethogram = compute_ethogram(
            cluster_labels=track.labels,
            fps=fps,
            cluster_names=cluster_names,
        )
        per_subject[track.subject_id] = {
            "ethogram": ethogram,
            "labels": track.labels.copy(),
            "cluster_names": dict(ethogram.cluster_names),
        }
        logger.debug(
            "build_group_ethogram: subject %d — %d clusters, %d bouts",
            track.subject_id,
            ethogram.n_clusters,
            len(ethogram.bouts),
        )

    social_events = detect_social_events(
        tracks,
        distance_threshold=distance_threshold,
        duration_threshold=duration_threshold,
    )

    time_axis = np.arange(n_frames, dtype=np.float64) / fps

    result: dict = {
        "fps": fps,
        "n_frames": n_frames,
        "n_subjects": len(tracks),
        "subject_ids": subject_ids,
        "per_subject": per_subject,
        "social_events": social_events,
        "time_axis": time_axis,
    }

    logger.info(
        "build_group_ethogram: %d subjects, %.1f s (%.0f frames), "
        "%d social events",
        len(tracks),
        n_frames / fps,
        n_frames,
        len(social_events),
    )
    return result


# ---------------------------------------------------------------------------
# Visualize
# ---------------------------------------------------------------------------

# Default qualitative colour palette (up to 20 clusters)
_DEFAULT_COLORS = [
    "#4E79A7",
    "#F28E2B",
    "#E15759",
    "#76B7B2",
    "#59A14F",
    "#EDC948",
    "#B07AA1",
    "#FF9DA7",
    "#9C755F",
    "#BAB0AC",
    "#1F77B4",
    "#FF7F0E",
    "#2CA02C",
    "#D62728",
    "#9467BD",
    "#8C564B",
    "#E377C2",
    "#7F7F7F",
    "#BCBD22",
    "#17BECF",
]


def plot_group_ethogram(
    ethogram: dict,
    output_path: str,
    figsize: Optional[tuple[float, float]] = None,
    bar_height: float = 0.8,
    social_event_color: str = "#CC0000",
    dpi: int = 150,
) -> str:
    """Generate a multi-subject ethogram visualization.

    Each horizontal row represents one subject.  Frame-level cluster
    assignments are drawn as a colour-coded raster (similar to a spike raster
    but continuous).  Detected social interaction events are drawn as vertical
    shaded spans below the per-subject rows.

    Parameters
    ----------
    ethogram : dict
        Output of :func:`build_group_ethogram`.
    output_path : str
        Destination path for the saved figure (PNG or SVG depending on
        extension).
    figsize : tuple(width, height) or None
        Matplotlib figure size in inches.  Defaults to ``(14, n_subjects * 1.5 + 1.5)``.
    bar_height : float
        Fractional height of each subject's colour bar (0–1).  Default 0.8.
    social_event_color : str
        Colour for social event shading.  Default dark red.
    dpi : int
        Dots per inch for raster output.  Default 150.

    Returns
    -------
    str
        Absolute path to the saved figure file.

    Raises
    ------
    ImportError
        If ``matplotlib`` is not installed.
    """
    try:
        import matplotlib  # noqa: PLC0415

        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.patches as mpatches  # noqa: PLC0415
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plot_group_ethogram. "
            "Install it with: pip install matplotlib"
        ) from exc

    n_subjects: int = ethogram["n_subjects"]
    subject_ids: list[int] = ethogram["subject_ids"]
    per_subject: dict = ethogram["per_subject"]
    time_axis: np.ndarray = ethogram["time_axis"]
    social_events: list[dict] = ethogram["social_events"]
    fps: float = ethogram["fps"]

    if figsize is None:
        figsize = (14.0, float(n_subjects) * 1.5 + 1.5)

    # Collect all cluster ids across all subjects to build a global colour map
    all_cluster_ids: set[int] = set()
    for sid in subject_ids:
        all_cluster_ids.update(int(c) for c in per_subject[sid]["cluster_names"])
    sorted_cids = sorted(all_cluster_ids)
    color_map: dict[int, str] = {
        cid: _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)]
        for i, cid in enumerate(sorted_cids)
    }

    fig, axes = plt.subplots(
        nrows=n_subjects + 1,  # extra row for social events
        ncols=1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={"height_ratios": [1.0] * n_subjects + [0.4]},
    )
    if n_subjects == 1:
        axes = list(axes) if hasattr(axes, "__iter__") else [axes, axes]

    axes = list(axes)

    # --- Per-subject rows ---
    for row_idx, sid in enumerate(subject_ids):
        ax = axes[row_idx]
        labels: np.ndarray = per_subject[sid]["labels"]

        # Build broken-bar collection: one span per consecutive run
        xranges_by_cluster: dict[int, list[tuple[float, float]]] = {
            cid: [] for cid in sorted_cids
        }

        n = len(labels)
        t_idx = 0
        while t_idx < n:
            cid = int(labels[t_idx])
            run_start = t_idx
            while t_idx < n and int(labels[t_idx]) == cid:
                t_idx += 1
            # -1 is an unlabeled gap (DBSCAN noise / dropped frame), not a
            # behavioral state: leave it blank instead of crashing on a missing
            # colour-map key. Also skip any stray label absent from the global
            # map defensively.
            if cid == -1 or cid not in xranges_by_cluster:
                continue
            t_start_s = time_axis[run_start]
            t_end_s = time_axis[t_idx - 1] + (1.0 / fps)
            xranges_by_cluster[cid].append((t_start_s, t_end_s - t_start_s))

        yrange = (0.5 - bar_height / 2.0, bar_height)
        for cid, xranges in xranges_by_cluster.items():
            if xranges:
                # ax.broken_barh replaces the removed BrokenBarHCollection
                # (matplotlib >= 3.10) and is stable across versions.
                ax.broken_barh(
                    xranges,
                    yrange,
                    facecolors=color_map[cid],
                    edgecolors="none",
                )

        ax.set_xlim(time_axis[0], time_axis[-1])
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks([0.5])
        ax.set_yticklabels([f"S{sid}"], fontsize=9)
        ax.tick_params(left=False)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)

    # --- Social events row ---
    ax_social = axes[n_subjects]
    if social_events:
        for event in social_events:
            t_start_s = event["start_frame"] / fps
            t_end_s = (event["end_frame"] + 1) / fps
            ax_social.axvspan(
                t_start_s,
                t_end_s,
                color=social_event_color,
                alpha=0.5,
                linewidth=0,
            )

    ax_social.set_xlim(time_axis[0], time_axis[-1])
    ax_social.set_ylim(0.0, 1.0)
    ax_social.set_yticks([0.5])
    ax_social.set_yticklabels(["Social"], fontsize=9)
    ax_social.tick_params(left=False)
    for spine in ("top", "right", "left"):
        ax_social.spines[spine].set_visible(False)
    ax_social.set_xlabel("Time (s)", fontsize=10)

    # --- Legend ---
    # Gather unique cluster names for the legend
    legend_patches = []
    seen_cids: set[int] = set()
    for sid in subject_ids:
        for cid, name in per_subject[sid]["cluster_names"].items():
            cid_int = int(cid)
            if cid_int not in seen_cids:
                seen_cids.add(cid_int)
                legend_patches.append(
                    mpatches.Patch(
                        color=color_map.get(cid_int, "#999999"),
                        label=name,
                    )
                )
    if legend_patches:
        axes[0].legend(
            handles=legend_patches,
            loc="upper right",
            fontsize=8,
            ncol=min(len(legend_patches), 6),
            framealpha=0.7,
        )

    fig.suptitle("Group Ethogram", fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    logger.info("plot_group_ethogram: saved '%s'", out)
    return str(out.resolve())
