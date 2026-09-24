# Step 5: Export and Use Results

After completing the behavior analysis, CASTLE provides a dedicated **7. Export** tab for packaging project data, plus several output files generated automatically during analysis.

---

## Export Tab (7. Export)

The Export tab lets you select which data components to include in a downloadable ZIP archive:

| Component | Contents |
|-----------|----------|
| **Tracking Masks** | `track/<video>/mask_list.h5` — per-frame HDF5 masks |
| **Latent Features** | `latent/` — extracted feature vectors |
| **Cluster Results** | `cluster/id.csv`, `cluster_*.npz`, `time_series_*.csv` |
| **Annotations** | `cluster/sessions/<id>/annotations.csv` — labels + comments |
| **Grid Videos** | `cluster/grid_videos/*.mp4` — pre-rendered cluster mosaic clips |
| **Analysis Outputs** | `analysis/` + session analysis folders |
| **Source Videos** | `sources/` — original videos (off by default) |

1. Switch to the **7. Export** tab
2. Select a session (for annotations)
3. Check the components you want
4. Click **📦 Export** — the archive is built and a download link appears

---

## Available Outputs

Outputs are saved in the `cluster/` directory within your project (SRT subtitles go to `subtitles/`):

```
projects/my-project/
├── cluster/
│   ├── id.csv                          # Cluster ID → name/colour mapping
│   ├── time_series_<video>.csv         # Frame-by-frame assignments, one per video
│   ├── time_series_<video>.meta.json   # fps + cluster-name map for that CSV
│   ├── cluster_init_a0_init_a1_.npz    # Embedding + cluster data
│   ├── grid_videos/                    # Pre-rendered cluster grid videos
│   └── sessions/
│       └── <session_id>/
│           ├── annotations.csv         # Cluster labels + comments
│           ├── umap_log.jsonl          # UMAP seed + config per stage
│           └── time_series_<video>.csv # Snapshot of this session's assignments
└── subtitles/
    └── <video>.srt                     # Behavioral labels as subtitles
```

### Behavior ID CSV (`id.csv`)

Maps cluster IDs to their auto-generated hierarchical names (behavior labels from the Cluster Annotator are stored in `annotations.csv`). `Color` is empty unless a custom colour was set.

```csv
Id,Name,Color
0,init,grey
1,init_a0,
2,init_a1,
3,init_a2,
```

### Time Series CSV (`time_series_<video>.csv`)

Frame-by-frame behavioral state assignments, one file per video. Each row corresponds to one original video frame.

```csv
behavior,exclude_reason
1,0
1,0
1,0
3,0
-1,1
...
```

- Values correspond to cluster IDs from `id.csv`
- `-1` indicates unclassified frames (e.g., frames with missing tracking data)
- `exclude_reason` codes why a frame is `-1`: `0` = not excluded, `1` = DBSCAN noise, `2` = non-finite latent
- When using a time window > 1, values are repeated for each frame in the window

### SRT Subtitles

Standard subtitle files (`.srt` format) that can be overlaid on the original videos. Each subtitle entry shows the behavioral label for that time segment. Useful for:

- Quick visual verification of results
- Presentations and lab meetings
- Sharing with collaborators who don't use CASTLE

### Embedding NPZ

Contains the UMAP coordinates and cluster assignments:

```python
import numpy as np

data = np.load('cluster_init_a0_init_a1_.npz')
embeddings = data['emb']    # Shape: (n_samples, 2) — UMAP coordinates
clusters = data['cls']      # Shape: (n_samples,) — cluster assignments
config = data['config']     # UMAP configuration used
```

- NaN values in embeddings indicate frames that were excluded from analysis
- Cluster value of `-1` indicates unclassified frames

!!! tip "Reproducing an embedding"
    Each clustering session writes a `umap_log.jsonl` file, with one JSON line per UMAP stage recording the resolved random seed and the config used. To reproduce an embedding exactly, reuse the logged seed. For bit-identical results, run the CPU/deterministic UMAP path.

---

## Using Results in Your Research

### Loading Data in Python

```python
import pandas as pd
import numpy as np

# Load behavioral time series
ts = pd.read_csv('projects/my-project/cluster/time_series_<video>.csv')
behaviors = ts['behavior'].values

# Load cluster names
ids = pd.read_csv('projects/my-project/cluster/id.csv')
id_to_name = dict(zip(ids['Id'], ids['Name']))

# Convert to named behaviors
named_behaviors = [id_to_name.get(b, 'unknown') for b in behaviors]

# Basic statistics
from collections import Counter
print(Counter(named_behaviors))
```

### Loading Data in R

```r
# Load behavioral time series
ts <- read.csv("projects/my-project/cluster/time_series_<video>.csv")
behaviors <- ts$behavior

# Load cluster names
ids <- read.csv("projects/my-project/cluster/id.csv")

# Merge
ts$name <- ids$Name[match(ts$behavior, ids$Id)]

# Summary
table(ts$name)
```

### Common Analyses

#### Behavior Duration Distribution

```python
import numpy as np

def get_bout_durations(behaviors, target_cluster, fps=30):
    """Get durations of consecutive bouts of a behavior."""
    is_target = (behaviors == target_cluster)
    changes = np.diff(is_target.astype(int))
    starts = np.where(changes == 1)[0] + 1
    ends = np.where(changes == -1)[0] + 1
    
    if is_target[0]:
        starts = np.concatenate([[0], starts])
    if is_target[-1]:
        ends = np.concatenate([ends, [len(behaviors)]])
    
    durations = (ends - starts) / fps  # Convert to seconds
    return durations
```

#### Comparing Groups

```python
# Compare behavior proportions between two conditions
from scipy import stats

group_a_time = behaviors_a  # From condition A
group_b_time = behaviors_b  # From condition B

for cluster_id, name in id_to_name.items():
    prop_a = np.mean(group_a_time == cluster_id)
    prop_b = np.mean(group_b_time == cluster_id)
    print(f"{name}: Group A = {prop_a:.3f}, Group B = {prop_b:.3f}")
```

### For Publication

!!! tip "Citing CASTLE"
    See the [Citation](../citation.md) page for BibTeX entries.

Recommended figures for publications:

- **UMAP embedding** colored by cluster — shows behavioral space structure
- **Ethogram** — timeline visualization of behavioral states
- **Cluster representative frames** — example frames from each behavioral category
- **Duration/proportion bar charts** — quantitative comparison between groups

!!! note "Ethograms are per video"
    Ethogram results are generated **one per video** (i.e. one per animal/subject), not pooled across videos. When exporting for publication, pick the video/subject for each ethogram figure rather than expecting a single combined timeline.

---

## Loading Latent Features

For advanced analysis, you can also work directly with the latent features extracted in Step 3:

```python
import numpy as np

# Load latent features (default encoder: dinov3_vitb16, 768-dim)
data = np.load('projects/my-project/latent/dinov3_vitb16/video_ROI_1_dinov3_vitb16.npz')
latent = data['latent']  # Shape: (n_frames, feature_dim)

# Use with your own dimensionality reduction or clustering
from sklearn.decomposition import PCA
pca = PCA(n_components=50)
reduced = pca.fit_transform(latent[~np.isnan(latent).any(axis=1)])
```

---

## Integration with Other Tools

CASTLE's outputs are in standard formats (CSV, NPZ) compatible with most analysis pipelines:

- **DeepLabCut / SLEAP**: compare pose estimation with CASTLE's training-free approach
- **SimBA / B-SOiD**: use CASTLE's behavioral labels as input
- **Custom pipelines**: load CSV time series directly

---

## Summary

You've completed the full CASTLE workflow:

1. ✅ Created a project and uploaded videos
2. ✅ Tracked ROIs with SAM + DeAOT
3. ✅ Extracted latent features with DINOv3 (DINOv2 still available as an option)
4. ✅ Discovered behavioral clusters with UMAP + DBSCAN
5. ✅ Annotated clusters with behavior labels and comments
6. ✅ Reviewed Ethogram, Quality Metrics, and Group Comparison in the Analysis tab
7. ✅ Exported results as a ZIP archive for analysis and publication

For questions or issues, check the [FAQ](../faq.md) or open an issue on [GitHub](https://github.com/CASTLE-ai/castle-ai/issues).
