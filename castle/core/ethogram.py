"""Ethogram analysis: transition matrices, bout statistics, and temporal dynamics.

Provides data structures and computation functions for behavioral ethogram
analysis including bout extraction, transition probability matrices,
and temporal coherence metrics.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Output schema version (see docs/behavior_data_contract.md). Bumped to 2.0
# when coverage / exclude_reason fields and the frequency_valid_only /
# stationarity_jsd metrics were added alongside the deprecated legacy keys.
ETHOGRAM_SCHEMA_VERSION = "2.0"

# Per-frame exclusion reasons (docs/behavior_data_contract.md C-1). The value
# 0 ("valid") is a real label; any other value means the frame is excluded
# from ethogram statistics (cluster_id == -1).
EXCLUDE_REASON_NAMES: Dict[int, str] = {
    0: "valid",
    1: "dbscan_noise",
    2: "nonfinite_latent",
    3: "tracking_loss",
    4: "manual_exclude",
}


def excluded_reason_counts(
    cluster_labels: np.ndarray,
    exclude_reason: Optional[np.ndarray] = None,
) -> Dict[str, int]:
    """Count excluded (-1) frames by their reason.

    Args:
        cluster_labels: per-frame labels (-1 == excluded).
        exclude_reason: optional per-frame reason enum (same length). When
            ``None`` (no reason column persisted) every excluded frame is
            bucketed as ``"unknown"``.

    Returns:
        ``{reason_name: count}`` over excluded frames, summing to the number of
        ``-1`` frames.
    """
    labels = np.asarray(cluster_labels)
    excluded = labels == -1
    n_excluded = int(np.sum(excluded))
    if n_excluded == 0:
        return {}
    if exclude_reason is None:
        return {"unknown": n_excluded}

    reason = np.asarray(exclude_reason)
    counts: Dict[str, int] = {}
    for value in reason[excluded]:
        v = int(value)
        # An excluded frame tagged 0 ("valid") is an inconsistency → unknown.
        name = EXCLUDE_REASON_NAMES.get(v, "unknown") if v != 0 else "unknown"
        counts[name] = counts.get(name, 0) + 1
    return counts


def derive_exclude_reason(
    cluster_labels: np.ndarray,
    latent_data: np.ndarray,
) -> np.ndarray:
    """Derive a per-row ``exclude_reason`` enum from labels + latent data.

    A ``-1`` row is bucketed by inspecting the latent it came from: a
    non-finite (NaN/Inf) latent row → ``nonfinite_latent`` (2), otherwise it is
    DBSCAN/HDBSCAN ``dbscan_noise`` (1). Valid rows (``cluster_id >= 0``) → 0.
    ``tracking_loss`` (3) and ``manual_exclude`` (4) are set elsewhere.

    Args:
        cluster_labels: 1-D per-row labels (-1 == excluded).
        latent_data: 2-D latent matrix aligned row-for-row with the labels.

    Returns:
        ``int8`` array of reason enums, same length as ``cluster_labels``.
    """
    labels = np.asarray(cluster_labels)
    finite = np.isfinite(np.asarray(latent_data)).all(axis=1)
    reason = np.zeros(len(labels), dtype=np.int8)
    noise = labels == -1
    reason[noise & ~finite] = 2  # nonfinite_latent
    reason[noise & finite] = 1   # dbscan_noise
    return reason


@dataclass
class BoutInfo:
    """Single behavioral bout (consecutive frames of same cluster)."""
    cluster_id: int
    start_frame: int
    end_frame: int          # exclusive
    duration_frames: int
    duration_seconds: float


@dataclass
class BoutStatistics:
    """Per-cluster bout statistics."""
    cluster_id: int
    cluster_name: str
    n_bouts: int
    total_frames: int
    frequency: float            # DEPRECATED: cluster frames / ALL frames (incl
                                # -1 gaps). Does not sum to 1 under exclusions;
                                # kept for backward compatibility. Prefer
                                # frequency_valid_only.
    mean_duration_s: float
    median_duration_s: float
    std_duration_s: float
    cv_duration: float          # coefficient of variation
    min_duration_s: float
    max_duration_s: float
    mean_inter_bout_interval_s: float  # mean time between bouts of same type
    # cluster frames / valid (non -1) frames; per-cluster values sum to 1.
    frequency_valid_only: float = 0.0


@dataclass
class TransitionMatrix:
    """Behavioral transition probability matrix."""
    matrix: np.ndarray          # K x K probability matrix
    counts: np.ndarray          # K x K raw count matrix
    cluster_ids: List[int]
    cluster_names: List[str]
    n_transitions: int
    entropy: float              # transition entropy (behavioral complexity)
    stationarity: float         # DEPRECATED: cosine similarity (not a valid
                                # distribution distance); kept for backward
                                # compatibility. Prefer stationarity_jsd.
    # 1 - Jensen-Shannon divergence^2 between the stationary distribution and
    # the observed distribution. NaN when the chain is reducible / the
    # stationary distribution is not uniquely identifiable.
    stationarity_jsd: float = float("nan")
    # "ok" | "not_identifiable_reducible_chain" | "trivial"
    stationarity_status: str = "ok"


@dataclass
class Ethogram:
    """Complete ethogram analysis result.

    Unclustered/noise frames (cluster id ``-1`` — DBSCAN noise plus any
    extraction-dropped frames carried as NaN→-1) are treated as *unlabeled
    gaps*, not as a behavioral state: they are excluded from ``n_clusters``,
    bouts, the transition matrix and temporal coherence, and reported
    separately via ``n_unlabeled`` / ``unlabeled_fraction``.
    """
    cluster_labels: np.ndarray  # per-frame cluster assignments
    fps: float
    n_frames: int
    n_clusters: int
    cluster_names: Dict[int, str]
    bouts: List[BoutInfo]
    bout_stats: Dict[int, BoutStatistics]
    transition_matrix: TransitionMatrix
    temporal_coherence: float
    n_unlabeled: int = 0            # frames with cluster id -1 (noise/dropped)
    unlabeled_fraction: float = 0.0  # n_unlabeled / n_frames
    # Coverage (docs/behavior_data_contract.md C-1). n_valid_frames ==
    # n_frames - n_unlabeled; valid_frame_fraction is its complement of
    # unlabeled_fraction. excluded_reason_counts buckets the -1 frames by
    # exclude_reason (or {"unknown": n} when no reason data was supplied).
    n_valid_frames: int = 0
    valid_frame_fraction: float = 1.0
    excluded_reason_counts: Dict[str, int] = field(default_factory=dict)
    schema_version: str = ETHOGRAM_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Computation functions
# ---------------------------------------------------------------------------

def extract_bouts(cluster_labels: np.ndarray, fps: float = 30.0) -> List[BoutInfo]:
    """Extract all behavioral bouts from frame-level cluster labels.

    A bout is a maximal consecutive run of the same cluster ID.  Noise /
    unclustered frames (``cluster_id == -1``) are **not** emitted as bouts;
    they still segment the real bouts on either side (a run of ``-1`` between
    two ``0`` runs leaves two separate ``0`` bouts, never one merged bout).

    Args:
        cluster_labels: 1-D integer array of per-frame cluster assignments.
        fps: Frames per second (used to convert durations).

    Returns:
        List of :class:`BoutInfo` in temporal order.
    """
    if len(cluster_labels) == 0:
        return []

    from castle.service.bout_service import find_bouts as _find_bouts

    unique_ids = np.unique(cluster_labels)
    # Collect (start, end, cluster_id) tuples then sort by start.
    # Skip -1 (noise/unlabeled): it is a gap, not a behavioral state.
    raw: List[Tuple[int, int, int]] = []
    for cid in unique_ids:
        if int(cid) == -1:
            continue
        for start, end in _find_bouts(cluster_labels, int(cid)):
            raw.append((start, end, int(cid)))

    raw.sort(key=lambda t: t[0])

    bouts: List[BoutInfo] = []
    for start, end, cid in raw:
        dur_frames = end - start
        bouts.append(BoutInfo(
            cluster_id=cid,
            start_frame=start,
            end_frame=end,
            duration_frames=dur_frames,
            duration_seconds=dur_frames / fps,
        ))
    return bouts


def compute_bout_statistics(
    bouts: List[BoutInfo],
    cluster_labels: np.ndarray,
    fps: float,
    cluster_names: Optional[Dict[int, str]] = None,
) -> Dict[int, BoutStatistics]:
    """Compute per-cluster bout statistics.

    Args:
        bouts: List of :class:`BoutInfo` (as returned by :func:`extract_bouts`).
        cluster_labels: Full per-frame label array (used for frequency).
        fps: Frames per second.
        cluster_names: Optional mapping of cluster_id → human name.

    Returns:
        Dict mapping cluster_id to :class:`BoutStatistics`.
    """
    if cluster_names is None:
        cluster_names = {}

    total_frames = len(cluster_labels)
    # Valid frames exclude -1 gaps; used for frequency_valid_only so that
    # per-cluster frequencies sum to 1 (behavioral-budget convention).
    n_valid_frames = int(np.sum(np.asarray(cluster_labels) != -1))
    # Group bouts by cluster
    from collections import defaultdict
    grouped: Dict[int, List[BoutInfo]] = defaultdict(list)
    for b in bouts:
        grouped[b.cluster_id].append(b)

    stats: Dict[int, BoutStatistics] = {}
    for cid, cid_bouts in grouped.items():
        durations = np.array([b.duration_seconds for b in cid_bouts])
        n_bouts = len(cid_bouts)
        total_cluster_frames = sum(b.duration_frames for b in cid_bouts)
        mean_d = float(np.mean(durations))
        std_d = float(np.std(durations, ddof=0))
        # BUG-16: guard against NaN-tainted std propagating into output CV.
        cv = std_d / mean_d if (mean_d > 0 and np.isfinite(std_d)) else 0.0

        # Inter-bout interval: time between end of one bout and start of next
        # for the same cluster (sorted by start_frame)
        sorted_bouts = sorted(cid_bouts, key=lambda b: b.start_frame)
        if len(sorted_bouts) > 1:
            ibis = []
            for i in range(len(sorted_bouts) - 1):
                gap_frames = sorted_bouts[i + 1].start_frame - sorted_bouts[i].end_frame
                ibis.append(gap_frames / fps)
            mean_ibi = float(np.mean(ibis))
        else:
            mean_ibi = 0.0

        stats[cid] = BoutStatistics(
            cluster_id=cid,
            cluster_name=cluster_names.get(cid, f"cluster_{cid}"),
            n_bouts=n_bouts,
            total_frames=total_cluster_frames,
            frequency=total_cluster_frames / total_frames if total_frames > 0 else 0.0,
            frequency_valid_only=(
                total_cluster_frames / n_valid_frames if n_valid_frames > 0 else 0.0
            ),
            mean_duration_s=mean_d,
            median_duration_s=float(np.median(durations)),
            std_duration_s=std_d,
            cv_duration=cv,
            min_duration_s=float(np.min(durations)),
            max_duration_s=float(np.max(durations)),
            mean_inter_bout_interval_s=mean_ibi,
        )
    return stats


def compute_transition_matrix(
    cluster_labels: np.ndarray,
    cluster_names: Optional[Dict[int, str]] = None,
) -> TransitionMatrix:
    """Compute transition probability matrix from frame-level labels.

    ``P[i, j] = P(cluster_j at t+1 | cluster_i at t)``

    Self-transitions (i == j) are excluded: the matrix only counts actual
    state changes so that rows sum to 1 over off-diagonal entries.  Noise /
    unclustered frames (``-1``) are excluded from the axes, and any transition
    with ``-1`` on either side is not counted (a behavior → unlabeled gap →
    behavior is not a real behavioral transition).

    Args:
        cluster_labels: 1-D integer array.
        cluster_names: Optional id→name mapping.

    Returns:
        :class:`TransitionMatrix` with probability and count matrices.
    """
    if cluster_names is None:
        cluster_names = {}

    unique_ids = sorted(int(x) for x in np.unique(cluster_labels) if int(x) != -1)
    K = len(unique_ids)
    id_to_idx = {cid: i for i, cid in enumerate(unique_ids)}
    names = [cluster_names.get(cid, f"cluster_{cid}") for cid in unique_ids]

    counts = np.zeros((K, K), dtype=np.float64)

    if len(cluster_labels) > 1:
        prev = cluster_labels[:-1]
        curr = cluster_labels[1:]
        # Count actual transitions only, excluding any pair touching -1.
        mask = (prev != curr) & (prev != -1) & (curr != -1)
        for p, c in zip(prev[mask], curr[mask]):
            counts[id_to_idx[int(p)], id_to_idx[int(c)]] += 1

    n_transitions = int(counts.sum())

    # Row-normalise to probabilities
    prob = np.zeros_like(counts)
    row_sums = counts.sum(axis=1)
    nonzero = row_sums > 0
    prob[nonzero] = counts[nonzero] / row_sums[nonzero, np.newaxis]

    # Transition entropy: H = -sum p*log(p) over all non-zero entries
    flat_p = prob.ravel()
    flat_p = flat_p[flat_p > 0]
    entropy = float(-np.sum(flat_p * np.log2(flat_p))) if len(flat_p) > 0 else 0.0

    # Stationarity: legacy cosine score (deprecated) + JSD-based score with an
    # explicit identifiability status (see docs/behavior_data_contract.md C-5).
    stationarity, stationarity_jsd, stationarity_status = _compute_stationarity(
        prob, cluster_labels, unique_ids, id_to_idx
    )

    return TransitionMatrix(
        matrix=prob,
        counts=counts,
        cluster_ids=unique_ids,
        cluster_names=names,
        n_transitions=n_transitions,
        entropy=entropy,
        stationarity=stationarity,
        stationarity_jsd=stationarity_jsd,
        stationarity_status=stationarity_status,
    )


def _stationary_distribution(full_prob: np.ndarray) -> Tuple[Optional[np.ndarray], str]:
    """Find the unique stationary distribution π of a row-stochastic matrix.

    Returns ``(pi, status)``. ``pi`` is ``None`` (never fabricated) when the
    chain is reducible / π is not uniquely identifiable: a zero-sum row
    (absorbing or never-sourced state), a non-simple eigenvalue 1
    (reducible/periodic), or a complex / non-single-signed dominant eigenvector.
    """
    row_sums = full_prob.sum(axis=1)
    # A zero-sum row means that state is never a transition source → the matrix
    # is not row-stochastic and π is undefined (reducible / absorbing).
    if np.any(row_sums <= 0):
        return None, "not_identifiable_reducible_chain"
    try:
        eigenvalues, eigenvectors = np.linalg.eig(full_prob.T)
    except np.linalg.LinAlgError:
        return None, "not_identifiable_reducible_chain"
    near_one = np.where(np.abs(eigenvalues - 1.0) < 1e-8)[0]
    # Reducible / periodic chains carry eigenvalue 1 with multiplicity > 1.
    if len(near_one) != 1:
        return None, "not_identifiable_reducible_chain"
    vec = eigenvectors[:, near_one[0]]
    if np.max(np.abs(vec.imag)) > 1e-8:
        return None, "not_identifiable_reducible_chain"
    pi = vec.real
    # A valid stationary vector is single-signed; flip an all-negative one.
    if np.all(pi <= 1e-12):
        pi = -pi
    if np.any(pi < -1e-8):
        return None, "not_identifiable_reducible_chain"
    pi = np.clip(pi, 0.0, None)
    total = pi.sum()
    if total <= 0:
        return None, "not_identifiable_reducible_chain"
    return pi / total, "ok"


def _legacy_cosine_stationarity(full_prob: np.ndarray, observed: np.ndarray) -> float:
    """DEPRECATED cosine-similarity stationarity.

    Preserved bit-for-bit from the original implementation so legacy outputs
    stay reproducible. Cosine is not a valid distribution distance — prefer the
    JSD-based score.
    """
    try:
        eigenvalues, eigenvectors = np.linalg.eig(full_prob.T)
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        pi = np.abs(np.real(eigenvectors[:, idx]))
        if pi.sum() > 0:
            pi = pi / pi.sum()
        else:
            return 0.0
    except np.linalg.LinAlgError:
        return 0.0
    norm_pi = float(np.linalg.norm(pi))
    norm_obs = float(np.linalg.norm(observed))
    if norm_pi > 0 and norm_obs > 0:
        return float(np.dot(pi, observed) / (norm_pi * norm_obs))
    return 0.0


def _compute_stationarity(
    prob: np.ndarray,
    cluster_labels: np.ndarray,
    unique_ids: List[int],
    id_to_idx: Dict[int, int],
) -> Tuple[float, float, str]:
    """Stationarity of the behavioral Markov chain.

    Returns ``(stationarity_cosine, stationarity_jsd, status)``:

    - ``stationarity_cosine`` — DEPRECATED cosine similarity (kept for backward
      compatibility; not a valid distribution distance).
    - ``stationarity_jsd`` — ``1 - JSD(π, observed)^2`` in ``[0, 1]``; ``NaN``
      when the chain is reducible / π is not uniquely identifiable.
    - ``status`` — ``"trivial"`` (≤1 state), ``"ok"``, or
      ``"not_identifiable_reducible_chain"``.

    A proxy for how close the sequence is to a stationary Markov process; not a
    guarantee of true stationarity.
    """
    K = prob.shape[0]
    if K <= 1:
        return 1.0, 1.0, "trivial"

    # Full transition matrix (with self-transitions), excluding any -1 pair.
    full_counts = np.zeros((K, K), dtype=np.float64)
    if len(cluster_labels) > 1:
        for p, c in zip(cluster_labels[:-1], cluster_labels[1:]):
            pi_, ci_ = int(p), int(c)
            if pi_ == -1 or ci_ == -1:
                continue
            full_counts[id_to_idx[pi_], id_to_idx[ci_]] += 1
    row_sums = full_counts.sum(axis=1)
    full_prob = np.zeros_like(full_counts)
    nonzero = row_sums > 0
    full_prob[nonzero] = full_counts[nonzero] / row_sums[nonzero, np.newaxis]

    # Observed empirical distribution over labeled frames.
    observed = np.zeros(K)
    for cid in unique_ids:
        observed[id_to_idx[cid]] = np.sum(cluster_labels == cid)
    if observed.sum() > 0:
        observed /= observed.sum()

    cosine = _legacy_cosine_stationarity(full_prob, observed)

    pi, status = _stationary_distribution(full_prob)
    if pi is None:
        return cosine, float("nan"), status

    from scipy.spatial.distance import jensenshannon

    jsd_distance = float(jensenshannon(pi, observed, base=2))
    if not np.isfinite(jsd_distance):
        return cosine, float("nan"), "not_identifiable_reducible_chain"
    return cosine, 1.0 - jsd_distance ** 2, status


def compute_temporal_coherence(cluster_labels: np.ndarray, window: int = 1) -> float:
    """Compute temporal coherence: fraction of frames matching their neighbors.

    High coherence → stable, long bouts (good segmentation).
    Low coherence → flickering labels (noisy segmentation).

    Noise / unclustered frames (``-1``) are excluded: a neighbor pair where
    either side is ``-1`` is not counted (an unlabeled gap is neither a match
    nor a flicker).  Computed only over pairs of labeled frames.

    Args:
        cluster_labels: 1-D integer array.
        window: Neighbor distance (default 1 = adjacent frames).

    Returns:
        Float in [0, 1].  Returns 1.0 for arrays of length ≤ window or with
        no labeled neighbor pairs.
    """
    n = len(cluster_labels)
    if n <= window:
        return 1.0
    left = cluster_labels[:-window]
    right = cluster_labels[window:]
    valid = (left != -1) & (right != -1)
    if not np.any(valid):
        return 1.0
    matches = left[valid] == right[valid]
    return float(np.mean(matches))


def compute_ethogram(
    cluster_labels: np.ndarray,
    fps: float = 30.0,
    cluster_names: Optional[Dict[int, str]] = None,
    exclude_reason: Optional[np.ndarray] = None,
) -> Ethogram:
    """Compute a complete ethogram from cluster labels.

    Orchestrates bout extraction, bout statistics, transition matrix,
    and temporal coherence into a single :class:`Ethogram` result.

    Args:
        cluster_labels: 1-D integer array of per-frame cluster assignments.
        fps: Frames per second.
        cluster_names: Optional id→name mapping.
        exclude_reason: Optional per-frame exclusion-reason enum (same length;
            see :data:`EXCLUDE_REASON_NAMES`). When ``None`` every ``-1`` frame
            is bucketed as ``"unknown"`` in ``excluded_reason_counts``.

    Returns:
        :class:`Ethogram` dataclass.
    """
    if cluster_names is None:
        cluster_names = {}

    labels = np.asarray(cluster_labels)
    n_frames = len(labels)
    # -1 (noise/unclustered/dropped) is an unlabeled gap, not a behavioral
    # state: exclude it from the cluster set and report it separately.
    unique_ids = sorted(int(x) for x in np.unique(labels) if int(x) != -1)
    n_clusters = len(unique_ids)
    n_unlabeled = int(np.sum(labels == -1))
    unlabeled_fraction = (n_unlabeled / n_frames) if n_frames > 0 else 0.0
    n_valid_frames = n_frames - n_unlabeled
    valid_frame_fraction = (n_valid_frames / n_frames) if n_frames > 0 else 1.0
    reason_counts = excluded_reason_counts(labels, exclude_reason)

    # Fill in any missing names
    names = {cid: cluster_names.get(cid, f"cluster_{cid}") for cid in unique_ids}

    bouts = extract_bouts(labels, fps)
    bout_stats = compute_bout_statistics(bouts, labels, fps, names)
    tm = compute_transition_matrix(labels, names)
    tc = compute_temporal_coherence(labels)

    return Ethogram(
        cluster_labels=labels,
        fps=fps,
        n_frames=n_frames,
        n_clusters=n_clusters,
        cluster_names=names,
        bouts=bouts,
        bout_stats=bout_stats,
        transition_matrix=tm,
        temporal_coherence=tc,
        n_unlabeled=n_unlabeled,
        unlabeled_fraction=unlabeled_fraction,
        n_valid_frames=n_valid_frames,
        valid_frame_fraction=valid_frame_fraction,
        excluded_reason_counts=reason_counts,
        schema_version=ETHOGRAM_SCHEMA_VERSION,
    )
