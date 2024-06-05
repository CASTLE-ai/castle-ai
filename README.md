# CASTLE


[![PyPI version](https://badge.fury.io/py/castle-ai.svg)](https://badge.fury.io/py/castle-ai)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CASTLE-ai/castle-ai/blob/main/notebooks/colab.ipynb)

CASTLE integrates the strengths of visual foundation models trained on large datasets possess open-world visual concepts, including Segment Anything (SA), DeAOT, and DINOv2, for one-shot image segmentation and unsupervised visual feature extraction. Furthermore, CASTLE employs unsupervised multi-stage and/or multi-layer UMAP clustering algorithms to distinguish behavioral clusters with unique temporal variability. 

# Install

## Step 1 install CASTLE core function
```
pip install castle-ai
```



## Step 2 (Speed Up) (not necessery)

for cuda12 user
```
pip install -U xformers --index-url https://download.pytorch.org/whl/cu121
pip install -U cudf-cu12 cuml-cu12 --extra-index-url=https://pypi.nvidia.com 
```

for cuda11 user
```
pip install -U xformers --index-url https://download.pytorch.org/whl/cu118
pip install -U cudf-cu11 cuml-cu11 --extra-index-url=https://pypi.nvidia.com 
```

## Step 3 Download Web UI
```
git clone https://github.com/CASTLE-ai/castle-ai.git
```

## Step 4 Download pretrain model

for Linux user
```
cd castle-ai/
bash download_ckpt.sh
```

for Windowns user
```
cd castle-ai/
mkdir ckpt
```
then using Web download model and put they into ckpt folder.


## Run
```
python app.py
```


# Reference

We would like to express our gratitude for the assistance received from the following projects during the development of this project.

[Segment Anything](https://github.com/facebookresearch/segment-anything.git)

[DeAOT & Segment-and-Track-Anything](https://github.com/z-x-yang/Segment-and-Track-Anything.git)

[DINOv2](https://github.com/facebookresearch/dinov2.git)