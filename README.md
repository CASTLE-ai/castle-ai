# CASTLE


[![PyPI version](https://badge.fury.io/py/castle-ai.svg)](https://badge.fury.io/py/castle-ai)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CASTLE-ai/castle-ai/blob/main/notebooks/colab.ipynb)

CASTLE integrates the strengths of visual foundation models trained on large datasets possess open-world visual concepts, including Segment Anything (SA), DeAOT, and DINOv2, for one-shot image segmentation and unsupervised visual feature extraction. Furthermore, CASTLE employs unsupervised multi-stage and/or multi-layer UMAP clustering algorithms to distinguish behavioral clusters with unique temporal variability. 

# Install

## Step 1 Install CASTLE Core Function
```
pip install castle-ai
```



## Step 2 Install xFormer and GPU Version of UMAP (Optional for Speed Up)

For CUDA 12 Users
```
pip install -U xformers --index-url https://download.pytorch.org/whl/cu121
pip install -U cudf-cu12 cuml-cu12 --extra-index-url=https://pypi.nvidia.com 
```


For CUDA 11 Users
```
pip install -U xformers --index-url https://download.pytorch.org/whl/cu118
pip install -U cudf-cu11 cuml-cu11 --extra-index-url=https://pypi.nvidia.com 
```


## Step 3 Download Web UI
```
git clone https://github.com/CASTLE-ai/castle-ai.git
```

## Step 4 Download Pretrained Model

### for Colab/Linux user

The model will download when you need to use it.

If you want to download models now
```
cd castle-ai/
mkdir ckpt
bash download_ckpt.sh 
```

### for other user
```
cd castle-ai/
mkdir ckpt
```

Then, download the models from the web by copying the links to the Chrome browser and downloading them.
```
https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth
https://drive.google.com/file/d/1g4E-F0RPOx9Nd6J7tU9AE1TjsouL4oZq/edit
https://drive.google.com/file/d/1QoChMkTVxdYZ_eBlZhK2acq9KMQZccPJ/edit
```

Afterward, place them into the ckpt folder.

```
castle-ai
├── castle
└── ckpt
    ├── dinov2_vitb14_reg4_pretrain.pth
    ├── R50_DeAOTL_PRE_YTB_DAV.pth
    ├── sam_vit_b_01ec64.pth
    └── SwinB_DeAOTL_PRE_YTB_DAV.pth

```



## Run
```
python app.py
```


# Reference

We would like to express our gratitude for the assistance received from the following projects during the development of this project.

[Segment Anything](https://github.com/facebookresearch/segment-anything.git)

[DeAOT & Segment-and-Track-Anything](https://github.com/z-x-yang/Segment-and-Track-Anything.git)

[DINOv2](https://github.com/facebookresearch/dinov2.git)