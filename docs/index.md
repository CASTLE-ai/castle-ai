# CASTLE

**Combined Approach for Segmentation and Tracking with Latent Extraction**

CASTLE is a **training-free** framework for unsupervised animal behavior analysis. It combines foundation models — SAM for segmentation, DeAOT for tracking, and DINOv2/v3 for feature extraction — to automatically discover behavioral patterns from video, without any manual labeling or model training. It works across species and experimental setups out of the box.

![CASTLE Pipeline](assets/Flowchart.png)

---

## Features

- 🎯 **Training-free** — no manual labeling or model training required
- 🐭 **Cross-species** — works on mice, rats, flies, C. elegans, and more
- 🧠 **Foundation model powered** — SAM + DeAOT + DINOv2/v3
- 📹 **Stabilized Camera Preprocessing** — zero-phase Butterworth filtering of centroid trajectories with dynamic crop extraction, producing head-fixed 518×518 video optimal for DINOv2
- 🖥️ **Interactive GUI** — Gradio web UI and native PyQt6 desktop app
- 🔬 **Hierarchical analysis** — explore behavior at multiple magnification scales
- 🏷️ **Cluster Annotator** — grid video browser with per-session labels, comment field, and auto-save
- 📊 **Analysis tab** — Ethogram, Quality Metrics (silhouette, CH, DB), Group Comparison
- 📦 **Export tab** — ZIP download with selectable data components
- 📁 **Publication-ready outputs** — CSV labels, UMAP plots, ethograms, SRT subtitles, NWB export

---

## Quick Links

<div class="grid cards" markdown>

- :material-download: **[Installation](getting-started/installation.md)** — Get CASTLE running
- :material-rocket-launch: **[Quick Start](getting-started/quickstart.md)** — First analysis in 5 minutes
- :material-school: **[Tutorials](tutorials/overview.md)** — Step-by-step guides
- :material-cog: **[Algorithm](technical/algorithm.md)** — How it works
- :material-file-document: **[API Reference](reference/api.md)** — Module documentation
- :material-format-quote-close: **[Citation](citation.md)** — Cite CASTLE in your paper

</div>

---

## The Pipeline

```
Raw Video → SAM (segment) → DeAOT (track) → Preprocess (stabilize) → DINOv2/v3 (features) → UMAP + DBSCAN (cluster)
              ↓                                      ↓                                                   ↓
         Annotator ←──────────────────────── label / comment ─────────────────────────────── Analysis / Export
```

1. **Segment** regions of interest with point-and-click (SAM)
2. **Track** ROIs across all video frames (DeAOT)
3. **Preprocess** *(optional)* — stabilize the virtual camera with zero-phase Butterworth filtering and dynamic crop extraction (`castle preprocess`)
4. **Extract** visual features from tracked/preprocessed regions (DINOv2/v3)
5. **Analyze** behavior through dimensionality reduction and clustering (UMAP + DBSCAN)
6. **Annotate** clusters with the Cluster Annotator (grid video, labels, comments, auto-save)
7. **Analyze** further with Ethogram, Quality Metrics, and Group Comparison
8. **Export** results as a ZIP archive with selectable components

---

## Getting Help

- [FAQ & Troubleshooting](faq.md)
- [GitHub Issues](https://github.com/CASTLE-ai/castle-ai/issues)

---

## License

CASTLE is released under the [Apache 2.0 License](https://github.com/CASTLE-ai/castle-ai/blob/main/LICENSE.txt).
