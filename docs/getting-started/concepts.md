# Core Concepts

## What CASTLE Does (In Plain English)

CASTLE watches your video and creates a **"visual fingerprint"** for each frame — a set of 768 numbers that capture what the animal looks like at that moment. Similar-looking poses get similar fingerprints.

Then it groups these fingerprints into **clusters**. Each cluster represents a distinct behavioral pattern: frames where the mouse is grooming look different from frames where it's running, so they end up in different clusters.

The key insight: **CASTLE never needs you to label body parts or train a classifier.** It uses a pre-trained vision model (DINOv3, `dinov3_vitb16` by default) that already "understands" visual similarity from having seen millions of images. You just tell it what each cluster *means* — and that's the only human input required.

---

## Key Terms

### Visual Features (Latent Vectors)

Think of each video frame as having a **barcode** — except instead of thin and thick lines, it's a sequence of 768 numbers. This barcode captures the essential visual information about the animal's posture, position, and appearance in that frame.

Two frames where the mouse is in a similar pose will have similar barcodes, even if the mouse is in a slightly different position in the arena. Frames showing very different behaviors (say, grooming vs. rearing) will have very different barcodes.

These barcodes are formally called **latent vectors**, and they're produced by a visual foundation model (DINOv3, `dinov3_vitb16` by default) that has been pre-trained on millions of images. You don't need to train anything — the model already knows how to extract meaningful visual features. Larger encoders are also selectable (`dinov3_vitl16`, 1024 numbers per frame), as is the previous-generation `dinov2_vitb14_reg4_pretrain` (also 768 numbers).

### Embedding (UMAP Plot)

Imagine taking all your frame-barcodes and **pinning them on a 2D corkboard**, where similar barcodes get pinned close together and dissimilar ones end up far apart. That's what UMAP does.

UMAP (Uniform Manifold Approximation and Projection) is a dimensionality reduction technique. It takes the 768-dimensional barcodes and projects them down to 2 dimensions so you can actually *see* the structure. On the resulting scatter plot:

- **Nearby points** = frames that look similar (probably the same behavior)
- **Distant points** = frames that look different (probably different behaviors)
- **Dense clumps** = behaviors that the animal performs consistently
- **Scattered points** = transitional or ambiguous frames

!!! note "Standardization and reproducibility"
    By default, the first (raw-feature) UMAP stage now **standardizes** its input (per-feature z-score) before projecting, which sharpens cluster separation. This changes the embedding compared to older runs, so your DBSCAN `eps` may need re-tuning. Every UMAP run also records the random seed it resolved, logged one line per stage to a per-session `umap_log.jsonl`; reuse that seed (on the deterministic CPU path) to reproduce an embedding exactly. Both standardization and the seed are configurable in the UMAP config JSON.

### Clustering (DBSCAN)

Once you can see the clumps on the UMAP plot, the next step is to **draw boundaries around them** — that's clustering.

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) finds groups of points that are packed closely together. It has one key parameter: **epsilon (eps)**, which controls how close points need to be to belong to the same cluster.

- **Smaller eps** → more clusters (finer behavioral categories)
- **Larger eps** → fewer clusters (broader categories)
- Points that don't fit any cluster are marked as **noise** (labeled -1)

You choose the granularity that makes biological sense for your question.

### Bout

A **bout** is a continuous episode of one behavior. If the mouse grooms for 3 seconds, pauses, then grooms again for 2 seconds, that's two grooming bouts.

CASTLE identifies bouts by looking at the sequence of cluster labels over time. Key bout statistics include:

- **Bout duration** — how long each episode lasts
- **Bout frequency** — how often a behavior occurs per unit time
- **Inter-bout interval** — time between consecutive bouts of the same behavior

These statistics are often more informative than simple time budgets. Two animals might spend the same total time grooming, but one does it in many short bouts while the other does fewer long bouts — that's a meaningful difference.

### Transition Matrix

A transition matrix is a table showing **"after behavior A, what usually happens next?"**

Each cell shows the probability of switching from one behavior (row) to another (column). For example, if the mouse almost always rears after grooming but rarely rears after running, that structure shows up in the transition matrix.

This reveals the **grammar of behavior** — the rules governing how behaviors flow into each other. Drug treatments, genetic modifications, or environmental changes often alter these transition probabilities even when overall time budgets look similar.

### Behavioral Fingerprint

A behavioral fingerprint is a **complete summary of one animal's behavior**, combining:

1. **Time budget** — how much time in each behavior (e.g., 40% resting, 25% grooming, 20% locomotion, 15% rearing)
2. **Bout statistics** — how long each bout lasts and how variable they are
3. **Transition probabilities** — how behaviors flow into each other

Think of it like a personality profile for the animal's behavior during that recording session. You can then compare fingerprints between individuals, groups, or conditions to ask: "Does the treatment change behavior?"

!!! note
    Ethogram results are written **per video** — one ethogram per animal/video — so each recording's fingerprint stays separate and directly comparable.

---

## The Behavior Microscope

CASTLE's analysis interface is called the **Behavior Microscope** because it works like a real microscope:

1. **Low magnification** — start with a broad view to identify major behavioral categories (e.g., active vs. inactive)
2. **Zoom in** — select one cluster and re-analyze it at higher magnification to discover sub-categories (e.g., within "active": grooming, rearing, locomotion)
3. **Zoom in further** — continue subdividing until you reach the granularity you need (e.g., within "grooming": face-grooming, body-grooming, paw-licking)

This hierarchical approach means you don't need to decide the number of behaviors in advance. You discover the structure in your data, layer by layer.

---

## How CASTLE Differs from Other Tools

| | CASTLE | Keypoint-based (DeepLabCut, SLEAP) | Supervised (B-SOiD, KPMS) |
|---|---|---|---|
| **Input** | Raw video + ROI masks | Labeled body-part positions | Keypoints + behavior labels |
| **Training required** | None | Keypoint labeling (~100s of frames) | Keypoint labeling + behavior annotation |
| **What you label** | Cluster names (after analysis) | Body parts (before analysis) | Body parts + behaviors (before analysis) |
| **New experiment** | Start immediately | Re-label keypoints if anatomy differs | Re-label everything |
| **Approach** | Appearance-based (what it looks like) | Pose-based (where joints are) | Pose + classifier |

CASTLE's main advantage is **speed to first result** — you can go from raw video to behavioral categories without any manual annotation of body parts or training data. The trade-off is that routine re-analysis of similar experiments is faster with supervised methods that can reuse trained models.
