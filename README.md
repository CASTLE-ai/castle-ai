# CASTLE

CASTLE integrates the strengths of visual foundation models trained on large datasets possess open-world visual concepts, including Segment Anything (SA), DeAOT, and DINOv2, for one-shot image segmentation and unsupervised visual feature extraction. Furthermore, CASTLE employs unsupervised multi-stage and/or multi-layer UMAP clustering algorithms to distinguish behavioral clusters with unique temporal variability. 

# Install
```
git clone https://github.com/CASTLE-ai/castle-ai.git
cd castle-ai
python install .
```

# Example

## Image segmentation
```python
1+1
```

## Video objects segmentation
```python
from castle import generate_aot

tracker = generate_aot(ckpt_path, MODEL, DEVICE)
tracker.add_reference_frame(frame, mask, num_object)


new_mask = 
```

## Visual latent extractioin
```python
1+1
```

## UMAP & HDBSCAN analysis
```python
1+1
```