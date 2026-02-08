# Step 5: Export and Use Results

After completing the behavior analysis in Step 4, CASTLE automatically generates several output files. This page explains the available outputs, their formats, and how to use them in your research.

---

## Available Outputs

All outputs are saved in the `cluster/` directory within your project:

```
projects/my-project/
└── cluster/
    ├── id.csv                          # Cluster ID → name mapping
    ├── time_series.csv                 # Frame-by-frame assignments
    ├── cluster_behavior1_behavior2_.npz # Embedding + cluster data
    └── (SRT subtitle files)
```

### Behavior ID CSV (`id.csv`)

Maps cluster IDs to their human-assigned names.

```csv
Id,Name
0,init
1,grooming
2,rearing
3,locomotion
```

### Time Series CSV (`time_series.csv`)

Frame-by-frame behavioral state assignments. Each row corresponds to one frame.

```csv
,behavior
0,1
1,1
2,1
3,3
4,3
...
```

- Values correspond to cluster IDs from `id.csv`
- `-1` indicates unclassified frames (e.g., frames with missing tracking data)
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

data = np.load('cluster_grooming_rearing_.npz')
embeddings = data['emb']    # Shape: (n_samples, 2) — UMAP coordinates
clusters = data['cls']      # Shape: (n_samples,) — cluster assignments
config = data['config']     # UMAP configuration used
```

- NaN values in embeddings indicate frames that were excluded from analysis
- Cluster value of `-1` indicates unclassified frames

---

## Using Results in Your Research

### Loading Data in Python

```python
import pandas as pd
import numpy as np

# Load behavioral time series
ts = pd.read_csv('projects/my-project/cluster/time_series.csv', index_col=0)
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
ts <- read.csv("projects/my-project/cluster/time_series.csv")
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

---

## Loading Latent Features

For advanced analysis, you can also work directly with the latent features extracted in Step 3:

```python
import numpy as np

# Load latent features
data = np.load('projects/my-project/latent/dinov2_vitb14_reg4_pretrain/video_ROI_1_dinov2_vitb14_reg4_pretrain.npz')
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
3. ✅ Extracted latent features with DINOv2/v3
4. ✅ Discovered behavioral clusters with UMAP + DBSCAN
5. ✅ Exported results for analysis and publication

For questions or issues, check the [FAQ](../faq.md) or open an issue on [GitHub](https://github.com/CASTLE-ai/castle-ai/issues).
