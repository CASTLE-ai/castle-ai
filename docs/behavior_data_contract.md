# Behavior Data Contract

This document is the authoritative semantic contract for behavioral labels,
frame rate, per-video aggregation, missing data, and Markov-chain stationarity
in CASTLE's ethogram / comparison pipelines. It exists so that scientific
outputs are unambiguous and reproducible, and so that future changes are
checked against a fixed reference (`tests/unit/test_behavior_contract.py`).

Status: **C-1…C-7 below are introduced by the scientific-correctness PR (PR1).**
Each clause notes which fields are added, which legacy fields are kept
(deprecated) for backward compatibility, and which pieces are deferred.

---

## C-1 Label semantics & exclusion reason

`cluster_id == -1` means **invalid / excluded from ethogram statistics** — it is
*not* a behavioral state. It is excluded from `n_clusters`, bouts, the transition
matrix, and temporal coherence (authoritative reference:
`castle/core/ethogram.py` `extract_bouts`, `compute_transition_matrix`,
`compute_temporal_coherence`, `compute_ethogram`).

A `-1` can arise from several distinct causes. We record the cause with a
per-frame `exclude_reason` enum and persist it next to the label:

| value | name              | meaning                                              | available |
|-------|-------------------|------------------------------------------------------|-----------|
| 0     | `valid`           | a real label (cluster_id ≥ 0)                        | PR1       |
| 1     | `dbscan_noise`    | DBSCAN/HDBSCAN assigned noise                        | PR1       |
| 2     | `nonfinite_latent`| latent row was NaN/Inf → marked -1                   | PR1       |
| 3     | `tracking_loss`   | empty ROI mask / detection lost (no usable data)     | **PR2**   |
| 4     | `manual_exclude`  | excluded by a human in the UI                        | PR1       |

- **Persistence**: written as an extra `exclude_reason` integer column in each
  `cluster/time_series_{basename}.csv`.
- **`tracking_loss` (value 3) is deferred to PR2**: today an empty mask becomes a
  near-zero latent (a silent fake), not a flagged gap. After PR2 fixes
  "empty mask → NaN", those frames are distinguished from `nonfinite_latent`.
- **Backward compatibility**: a CSV without the `exclude_reason` column is read
  with every `-1` counted under an `unknown` bucket. No error.

**Required ethogram output coverage fields** (in addition to the legacy
`n_unlabeled` / `unlabeled_fraction`):

- `n_valid_frames`, `n_excluded_frames`
- `valid_frame_fraction`
- `excluded_reason_counts`: `{reason_name: count}`, summing to `n_excluded_frames`.

## C-2 fps policy

fps is **per-video** and never silently defaults to 30.0:

- `fps is None` → read the video's own fps from `sources/` via `_video_fps`.
- `fps <= 0` (including `0.0`) → raise `CastleDataError` (not a legal override).
- finite `fps > 0` → use it.

The legacy `effective_fps = fps or data["fps"] or 30.0` truthiness pattern is
removed (it both hid the per-video fps and silently swallowed `fps == 0.0`).

## C-3 Per-video aggregation rules

When a project holds several videos, the public ethogram APIs compute **one
ethogram per video** (each with its own fps) and then aggregate. They never
concatenate frames across videos into one sequence.

- **Transition matrix**: sum raw counts across videos, then normalise **once**
  (do not average per-video probability matrices — that wrongly equal-weights
  short and long videos).
- **Bouts**: extract bouts per video with that video's fps (durations in
  seconds), then merge the per-video bout lists. No bout crosses a video
  boundary.
- **Temporal coherence**: weighted by each video's number of valid adjacent
  frame pairs.
- **frequency** (added, not replaced):
  - `frequency_valid_only` = cluster valid frames / total valid frames
    (per-cluster values sum to 1 when `n_valid_frames > 0`). **Preferred.**
  - `frequency` (legacy, **deprecated**) = cluster frames / all frames
    (fraction of total recording time; does not sum to 1 under exclusions).
  - `valid_frame_fraction` is always reported so a 30% under 95%-valid is not
    confused with a 30% under 40%-valid.
- **Mixed fps** is reported explicitly: `fps_policy="per_video"`,
  `video_fps={basename: fps}`, `mixed_fps: bool`.

## C-4 Missing-data representation

The canonical signal for "this latent row is unusable" is a boolean
`valid_latent_mask` (plus an optional `invalid_reason`). NaN may be carried in
the latent array as a defensive marker, but **downstream filters on the mask**,
not on NaN (UMAP / HDBSCAN / np.save / distance metrics dislike NaN). This
reuses the existing `index_mask` + `cls = int16(-1)` mechanism
(`castle/ui/embedding_scatter.py`). *(latent-layer work lands in PR2 Stage 4.)*

## C-5 Stationarity

The Markov-chain stationarity proxy is reported with an explicit status:

- `stationarity_jsd` = `1 - jensenshannon(pi, observed, base=2) ** 2` (added).
- `stationarity_status`:
  - `"ok"` — a unique stationary distribution was identified.
  - `"not_identifiable_reducible_chain"` — the chain is reducible / the
    stationary distribution is non-unique (multiple absorbing components, a
    zero-sum row, or a complex/duplicated eigenvalue near 1). In this case
    `stationarity_jsd = NaN` (we do **not** fabricate an answer via self-loops).
- `stationarity` (legacy, **deprecated**) = cosine similarity between `pi` and
  the observed distribution. Kept for backward compatibility only; cosine is
  not a valid distribution distance.

## C-6 Fingerprint schema

Behavioral fingerprints (group comparison) are dimensioned by the **global**
cluster-id set shared across all animals, so every animal's feature vector has
identical length even when an animal never exhibits some behavior. Output adds:

- `fingerprint_schema_version`
- `cluster_id_order`: the global ordered cluster-id list used for all vectors.
- `frequency_definition`: `"valid_frames_only"`.
- `missing_duration_policy`: `"NaN"` — an absent behavior's duration / IBI is
  undefined (NaN), not 0; structural counts (frequency, bout_count) are 0.

## C-7 Ethogram schema_version

Every public ethogram output carries `schema_version` so consumers can detect
the new coverage / reason fields and the deprecated `frequency` / `stationarity`
keys.

---

## Scientific invariants (enforced by tests)

For any project (see `tests/unit/test_behavior_contract.py`):

1. `sum(frequency_valid_only over clusters) == 1` when `n_valid_frames > 0`.
2. No bout has `cluster_id == -1`.
3. No transition crosses a video boundary (per-video bout counts, not merged).
4. fps used is finite and `> 0`; `fps <= 0` raises `CastleDataError`.
5. Comparison fingerprints have identical length across all animals.
6. `sum(excluded_reason_counts.values()) == n_excluded_frames`.
