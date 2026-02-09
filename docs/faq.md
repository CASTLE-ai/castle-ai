# FAQ & Troubleshooting

---

## Installation

### CUDA not found / PyTorch not detecting GPU

**Symptom**: `torch.cuda.is_available()` returns `False`

**Solutions**:

1. Check that NVIDIA drivers are installed: `nvidia-smi`
2. Verify CUDA version compatibility: CASTLE requires CUDA 12.x
3. Reinstall PyTorch with the correct CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/)
4. If using Docker, ensure NVIDIA Container Toolkit is installed and the `deploy` section in `docker-compose.yml` is not commented out

### Google Drive download quota exceeded

**Symptom**: `gdown` fails with "Too many users have viewed or downloaded this file recently"

**Solutions**:

1. Download manually via browser — open the Google Drive link, sign in, and download
2. Wait a few hours — quotas reset periodically
3. Try with the fuzzy flag: `gdown --fuzzy '<URL>'`
4. Use a different Google account

### cuml installation fails

**Symptom**: `pip install cuml-cu12` fails or cuml import errors

**Solutions**:

1. Verify CUDA 12.x is installed: `nvcc --version`
2. cuml requires an NVIDIA GPU — it will not work on CPU-only systems
3. Try installing from NVIDIA's package index explicitly:
   ```bash
   pip install --extra-index-url https://pypi.nvidia.com cuml-cu12
   ```
4. On systems without GPU, CASTLE will still work but GPU-accelerated UMAP will be unavailable

### Import errors on startup

**Symptom**: `ModuleNotFoundError` when running `python app.py`

**Solutions**:

1. Ensure you're in the correct virtual environment
2. Reinstall dependencies: `pip install -r requirements.txt`
3. Check you're running from the repository root (not a subdirectory)

---

## Tracking

### Tracking loses the animal / mask drifts

**Solutions**:

1. **Add more labels** — label frames where tracking fails, especially frames with different postures or lighting
2. **Label diverse frames** — not just the first frame, but frames across the video with varied appearances
3. **Try SwinB model** — more accurate than R50, especially for challenging cases
4. **Use iterative refinement** — cancel bad tracking early, add labels at failure points, re-track from there
5. **Reduce frame range** — track shorter segments and verify quality before proceeding
6. **Aim for 5–30 prompt frames** — more diverse prompts improve robustness

### Tracking is very slow

**Solutions**:

1. Use the **R50** model instead of SwinB (faster, often sufficient)
2. Check GPU utilization (`nvidia-smi`) — if GPU is not being used, there may be a CUDA issue
3. Reduce video resolution if possible

---

## Feature Extraction

### Extraction is very slow

**Solutions**:

1. Increase **batch size** (e.g., 64, 128) if you have spare VRAM
2. Use `dinov2_vitb14_reg4_pretrain` (fastest) instead of `dinov3_vitl16` (slowest)
3. Enable **Skip existing files** to avoid re-processing
4. Check GPU utilization — ensure the GPU is actually being used

### "Please click Apply on Preprocess settings first"

You must click the **Apply** button after configuring preprocessing settings (Center ROI, Rotate, Remove Background) before running extraction.

---

## Behavior Analysis

### UMAP looks random / no structure visible

**Solutions**:

1. **Check your features** — if latent extraction had issues (many NaN frames), the embedding will be poor
2. **Try different n_neighbors** — start with Low Magnification 100, then experiment
3. **Verify preprocessing** — Center ROI and Remove Background often improve results significantly
4. **Increase data** — more frames and videos produce better embeddings
5. **Check tracking quality** — poor tracking → poor features → poor embedding

### Too many / too few clusters

Adjust the **eps** parameter in DBSCAN:

- **Too many clusters**: increase eps (e.g., 2.0, 3.0)
- **Too few clusters**: decrease eps (e.g., 0.5, 0.3)
- **All noise (-1)**: eps is too small, increase it

### Hierarchical analysis: how to zoom in

1. Run UMAP + DBSCAN at low magnification
2. Label and submit the broad clusters
3. Select one cluster from the **Select Cluster** dropdown
4. Re-run UMAP at intermediate or high magnification on just that cluster
5. Cluster and label the sub-behaviors
6. Submit again — results accumulate

---

## General

### What species does CASTLE support?

Any animal (or object) that SAM can segment. CASTLE is species-agnostic — it has been tested on mice, rats, flies, C. elegans, and gait analysis. If you can click on it and SAM can segment it, CASTLE can track and analyze it.

### Can I use CASTLE without a GPU?

Yes, but with significant limitations:

- **Tracking**: much slower (minutes → hours for a single video)
- **Feature extraction**: very slow on CPU
- **UMAP**: cuml-based GPU UMAP will not be available; a CPU fallback may be needed
- **Overall**: feasible for short demo videos, not practical for real datasets

### Can I process multiple videos at once?

Yes. Use **Batch Videos Tracking** (Step 2) to track multiple videos with shared ROI prompts. Feature extraction (Step 3) supports an "All" option to process all project videos. The Behavior Microscope (Step 4) automatically aggregates features across all videos.

### Where are my results stored?

All results are in `projects/{project_name}/`:

- Tracking masks: `track/{video}/mask_list.h5`
- Latent features: `latent/{model}/{video}_ROI_{id}_{model}.npz`
- Cluster results: `cluster/id.csv`, `cluster/time_series.csv`

See [Data Formats](technical/data-formats.md) for complete details.

### How do I cite CASTLE?

See the [Citation](citation.md) page for BibTeX entries.

---

## Still Need Help?

- [Open an issue on GitHub](https://github.com/CASTLE-ai/castle-ai/issues)
- Check existing issues for similar problems
