"""
castle/core/config.py
Central configuration for Castle AI.
"""

import colorsys
from pathlib import Path
from typing import Dict, List

# --- Paths ---
# C-01: Use pathlib for robust path resolution
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_CKPT_DIR = BASE_DIR / 'ckpt'
DEFAULT_PROJECT_DIR = BASE_DIR / 'projects'

# --- Model Checkpoints (Google Drive IDs) ---
# C-03: Add type hints
# AOT Models
CKPT_AOT_IDS: Dict[str, str] = {
    'r50_deaotl': '1QoChMkTVxdYZ_eBlZhK2acq9KMQZccPJ',
    'swinb_deaotl': '1g4E-F0RPOx9Nd6J7tU9AE1TjsouL4oZq'
}

# DINO Models — DINOv2 is loaded via torch.hub; DINOv3 via Google Drive (gdown).
CKPT_DINO_IDS: Dict[str, str] = {
    # DINOv2 (loaded via torch.hub)
    'dinov2_vitb14': '',
    'dinov2_vitl14': '',

    # DINOv3 (private Google Drive checkpoints)
    "dinov3_vitb16": "18doehnHWWnz9zBtOdgYZ3XMTpgPYbYZ6",
    "dinov3_vitl16": "195H5UHKJ0r4qRDY7Ly6WJrXGnpdlHMSu",
}

# Supported Models
SUPPORTED_MODELS: List[str] = [
    'dinov2_vitb14',
    'dinov2_vitl14',
    'dinov2_vitb14_reg4_pretrain',
    'dinov3_vitb16',
    'dinov3_vitl16',
    # 'dinov3_vits16' removed: no checkpoint id / filename / num_layers entry
    # (selecting it crashed at load) AND DINOv3Encoder mis-sized it to 768-d
    # while output_dim_for computes 384-d. Re-add only with all four wired up.
]

# DINOv3 Constants
DINOV3_CONSTANTS = {
    "PATCH_SIZE": 16,
    "TARGET_PATCHES_PER_SIDE": 37,
    "IMAGE_SIZE": 37 * 16,  # 592
    "IMAGENET_MEAN": (0.485, 0.456, 0.406),
    "IMAGENET_STD": (0.229, 0.224, 0.225),
    "MODEL_TO_CKPT_FILENAME": {
        "dinov3_vitb16": "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
        "dinov3_vitl16": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
    },
    "MODEL_TO_NUM_LAYERS": {
        "dinov3_vitb16": 12,
        "dinov3_vitl16": 24,
    },
}


# Error Messages (L-01: Internationalization preparation)
ERROR_MESSAGES = {
    'unsupported_model': "Unsupported model: {model}. Supported models: {supported}",
    'mask_not_found': "Mask file not found for video: {video}",
    'invalid_preprocess_type': "Invalid type for {param}: expected {expected}, got {actual}",
    'model_load_failed': "Failed to load model {model}: {error}",
    'video_read_failed': "Failed to read video {video}: {error}",
    'latent_save_failed': "Failed to save latent to {path}: {error}",
}


# --- Constants ---
VIDEO_EXTENSIONS: List[str] = ['.mp4', '.avi', '.mov', '.wmv', '.flv', '.mkv']

# --- Color Palette ---
# Unified palette for all visualization (cluster plots, masks, etc.)
# Used by: latent_explorer.py, explorer.py, plot.py
PALETTE_HEX: List[str] = [
    '#7AE4F0', '#FFD0EC', '#6EE368', '#C1B5EA', '#A7CCED', '#FBC471', '#9E83E3',
    '#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692',
    '#B6E880', '#FF97FF', '#FECB52',
    '#FD3216', '#00FE35', '#6A76FC', '#FED4C4', '#FE00CE', '#0DF9FF', '#F6F926',
    '#FF9616', '#479B55', '#EEA6FB', '#DC587D', '#D626FF', '#6E899C', '#00B5F7',
    '#B68E00', '#C9FBE5', '#FF0092', '#22FFA7', '#E3EE9E', '#86CE00', '#BC7196',
    '#7E7DCD', '#FC6955', '#E48F72',
    '#66c5cc', '#f6cf71', '#f89c74', '#dcb0f2', '#87c55f', '#9eb9f3', '#fe88b1',
    '#c9db74', '#8be0a4', '#b497e7', '#b3b3b3',
    '#e58606', '#5d69b1', '#52bca3', '#99c945', '#cc61b0', '#24796c', '#daa51b',
    '#2f8ac4', '#764e9f', '#ed645a', '#a5aa99',
]


# ---------------------------------------------------------------------------
# Colorblind-safe categorical palette for FIGURES (project decision 2026-06-20).
# Based on the Okabe-Ito 8-colour set — the de-facto colorblind-safe categorical
# standard for scientific figures (Okabe & Ito 2008). A single source of truth so
# every analysis figure colours cluster *i* the same way and is legible to readers
# with colour-vision deficiency. (The interactive UI cluster tree still uses its
# own name-hash palette; figures use this.)
# ---------------------------------------------------------------------------
OKABE_ITO: List[str] = [
    '#E69F00',  # orange
    '#56B4E9',  # sky blue
    '#009E73',  # bluish green
    '#F0E442',  # yellow
    '#0072B2',  # blue
    '#D55E00',  # vermillion
    '#CC79A7',  # reddish purple
    '#000000',  # black
]


# Distinct, well-separated absolute lightness levels used to extend the 8-hue
# Okabe-Ito set past 8 clusters. Six values → 48 unique colours before any
# repeat (far beyond realistic behaviour-cluster counts). Using an *absolute*
# lightness (rather than a relative darken/lighten) is what keeps even pure
# black at index 7 distinct across wraps — a relative shift can't move a colour
# that already sits at a lightness extreme, so clusters 7/15/23 would collide.
_WRAP_LIGHTNESS: List[float] = [0.78, 0.30, 0.62, 0.46, 0.88, 0.18]


def color_for_cluster(index: int) -> str:
    """Colorblind-safe hex colour for the *index*-th cluster in a figure.

    The first 8 clusters get the exact Okabe-Ito palette (the colourblind-safe
    guarantee that matters). Beyond 8, the base hue is reused at a distinct
    lightness from :data:`_WRAP_LIGHTNESS`, preserving hue/saturation so each
    stays as legible as 8+ simultaneous categories allow. Deterministic in
    ``index`` and unique for any realistic cluster count.
    """
    i = max(0, int(index))
    base = OKABE_ITO[i % len(OKABE_ITO)]
    wrap = i // len(OKABE_ITO)
    if wrap == 0:
        return base
    r, g, b = (int(base[k:k + 2], 16) / 255.0 for k in (1, 3, 5))
    h, _, s = colorsys.rgb_to_hls(r, g, b)
    lightness = _WRAP_LIGHTNESS[(wrap - 1) % len(_WRAP_LIGHTNESS)]
    extra = (wrap - 1) // len(_WRAP_LIGHTNESS)  # >48 clusters: nudge so still unique
    if extra:
        lightness = min(0.93, max(0.10, lightness + 0.05 * (extra if wrap % 2 else -extra)))
    r2, g2, b2 = colorsys.hls_to_rgb(h, lightness, s)
    return '#{:02x}{:02x}{:02x}'.format(int(r2 * 255), int(g2 * 255), int(b2 * 255))
