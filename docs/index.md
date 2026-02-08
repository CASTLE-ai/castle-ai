# CASTLE

**Combined Approach for Segmentation and Tracking with Latent Extraction**

CASTLE is a **training-free** framework for unsupervised animal behavior analysis. It combines foundation models — SAM for segmentation, DeAOT for tracking, and DINOv2/v3 for feature extraction — to automatically discover behavioral patterns from video, without any manual labeling or model training. It works across species and experimental setups out of the box.

![CASTLE Pipeline](assets/Flowchart.png)

---

## Features

- 🎯 **Training-free** — no manual labeling or model training required
- 🐭 **Cross-species** — works on mice, rats, flies, C. elegans, and more
- 🧠 **Foundation model powered** — SAM + DeAOT + DINOv2/v3
- 🖥️ **Interactive GUI** — Gradio-based interface for the full analysis pipeline
- 🔬 **Hierarchical analysis** — explore behavior at multiple magnification scales
- 📊 **Publication-ready outputs** — CSV labels, UMAP plots, ethograms, SRT subtitles

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
Raw Video → SAM (segment) → DeAOT (track) → Align → DINOv2/v3 (features) → UMAP + DBSCAN (cluster)
```

1. **Segment** regions of interest with point-and-click (SAM)
2. **Track** ROIs across all video frames (DeAOT)
3. **Extract** visual features from tracked regions (DINOv2/v3)
4. **Analyze** behavior through dimensionality reduction and clustering (UMAP + DBSCAN)

---

## Getting Help

- [FAQ & Troubleshooting](faq.md)
- [GitHub Issues](https://github.com/CASTLE-ai/castle-ai/issues)

---

## License

CASTLE is released under the [Apache 2.0 License](https://github.com/CASTLE-ai/castle-ai/blob/main/LICENSE.txt).
