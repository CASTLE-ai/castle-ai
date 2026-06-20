"""
castle/core/config.py
Central configuration for Castle AI.
"""

import colorsys
import hashlib
import os
from pathlib import Path
from typing import Dict, List, Optional

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
    # Expected SHA-256 of each pretrained checkpoint. Verified after download /
    # before load so a corrupted or substituted weight file is caught instead of
    # silently producing different latents (reproducibility). The filename's
    # trailing hex (…-73cec8be) is the first 8 chars of these digests.
    "MODEL_TO_SHA256": {
        "dinov3_vitb16": "73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c",
        "dinov3_vitl16": "8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035",
    },
}


# --- Backbone version pinning (reproducibility) -------------------------------
# Pin the torch.hub backbone repos to explicit commits so a re-run pulls the
# SAME backbone code — and, for DINOv2, the same pretrained-weight URLs baked
# into that commit — rather than a moving ``main`` that can change results
# between runs. Captured 2026-06-20 from each repo's upstream ``main`` HEAD.
# Override per repo with env CASTLE_DINOV2_REF / CASTLE_DINOV3_REF (set empty to
# fall back to ``main``).
TORCH_HUB_REFS: Dict[str, str] = {
    'dinov2': '7764ea0f912e53c92e82eb78a2a1631e92725fc8',
    'dinov3': '346f38fee679c56a6888f91c51670fae61d364e0',
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


# ===========================================================================
# Unified dual-mode categorical palette (project decision 2026-06-20).
#
# ONE engine colours clusters everywhere — publication figures AND the
# interactive cluster tree / scatter / overlay video — so the whole tool stays
# visually consistent and a single toggle recolours all of it. Two ladders share
# the same extension math:
#   * 'colorblind' (DEFAULT) = Okabe-Ito, the de-facto colourblind-safe
#     categorical standard (Okabe & Ito 2008). Publication figures ship with it.
#   * 'normal' = VIVID, a punchier set for users who don't need colourblind
#     safety and prefer the original vibrant look.
# Figures index the ladder positionally (cluster 0,1,2…); the tree indexes it by
# an md5 hash of the cluster NAME (stable, sibling-distinct). Both flip together
# when the mode changes; exact cross-world hue identity is not guaranteed (the
# two index schemes differ), but each view is internally consistent.
# ===========================================================================
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

# Vibrant (non-colourblind-safe) ladder for 'normal' mode. Hues chosen at
# moderate lightness/saturation so the absolute-lightness wrap extension below
# keeps later clusters distinct (no near-extreme base entries).
VIVID: List[str] = [
    '#4C72B0',  # blue
    '#DD8452',  # orange
    '#55A868',  # green
    '#C44E52',  # red
    '#8172B3',  # purple
    '#937860',  # brown
    '#DA8BC3',  # pink
    '#8C8C8C',  # grey
    '#CCB974',  # gold
    '#64B5CD',  # cyan
    '#B07AA1',  # mauve
    '#FF9D4C',  # amber
]

# Neutral colour for unlabeled / container ('init') nodes — never a real cluster.
GREY_UNLABELED = 'grey'

# Distinct, well-separated absolute lightness levels used to extend a ladder past
# its length. Six values → len×6 unique colours before any repeat (far beyond
# realistic cluster counts). Using an *absolute* lightness (not a relative
# darken/lighten) keeps even pure black (Okabe-Ito index 7) distinct across
# wraps — a relative shift can't move a colour already at a lightness extreme.
_WRAP_LIGHTNESS: List[float] = [0.78, 0.30, 0.62, 0.46, 0.88, 0.18]

_PALETTE_LADDERS: Dict[str, List[str]] = {'colorblind': OKABE_ITO, 'normal': VIVID}
_DEFAULT_COLOR_MODE = 'colorblind'

# In-process override set by set_color_mode (lets a live UI toggle take effect
# with no restart). None → fall back to env CASTLE_COLOR_MODE, then the default.
_COLOR_MODE: Optional[str] = None


def get_color_mode() -> str:
    """Active colour mode: 'colorblind' (default) or 'normal'.

    Resolution order: in-process override (set_color_mode) > env
    CASTLE_COLOR_MODE (validated; bad value ignored) > default. Read fresh by
    every colour producer at render time, so a toggle recolours the next render.
    """
    if _COLOR_MODE is not None:
        return _COLOR_MODE
    env = os.environ.get('CASTLE_COLOR_MODE', '').strip().lower()
    if env in _PALETTE_LADDERS:
        return env
    return _DEFAULT_COLOR_MODE


def set_color_mode(mode: str) -> None:
    """Set the in-process colour mode. Raises ValueError on an unknown mode.

    Process-global (not per-session) — CASTLE is single-user/local, so the live
    UI toggle simply sets this and re-renders.
    """
    global _COLOR_MODE
    if mode not in _PALETTE_LADDERS:
        raise ValueError(
            f"Unknown color mode {mode!r}; expected one of {sorted(_PALETTE_LADDERS)}"
        )
    _COLOR_MODE = mode


def _ladder_color(index: int, base_list: List[str]) -> str:
    """Deterministic hex for *index* in *base_list*, extended past its length by
    distinct absolute lightness wraps, then cycled.

    Gives ``len(base_list) × (len(_WRAP_LIGHTNESS)+1)`` distinct colours (base hue
    + one per lightness level) before repeating — e.g. 56 (Okabe-Ito) / 84
    (VIVID). Cycling (rather than an unbounded lightness nudge) keeps the large
    md5-derived indices used for name-hashed tree colours well-distributed
    instead of collapsing them all to a lightness extreme. Identical to the prior
    positional behaviour for figure indices below the cycle length.
    """
    n = len(base_list)
    levels = len(_WRAP_LIGHTNESS)
    i = max(0, int(index))
    base = base_list[i % n]
    wrap = (i // n) % (levels + 1)  # 0 = base hue, 1..levels = lightness variants
    if wrap == 0:
        return base
    r, g, b = (int(base[k:k + 2], 16) / 255.0 for k in (1, 3, 5))
    h, _, s = colorsys.rgb_to_hls(r, g, b)
    lightness = _WRAP_LIGHTNESS[wrap - 1]
    r2, g2, b2 = colorsys.hls_to_rgb(h, lightness, s)
    return '#{:02x}{:02x}{:02x}'.format(int(r2 * 255), int(g2 * 255), int(b2 * 255))


def palette_color(index: int, mode: Optional[str] = None) -> str:
    """THE unified palette engine — hex for cluster *index* in the given *mode*.

    ``mode=None`` uses :func:`get_color_mode`. Deterministic in ``(index, mode)``.
    Accepts both small positional indices (figures) and large md5-derived
    indices (name-hashed tree colours).
    """
    ladder = _PALETTE_LADDERS.get(mode) or _PALETTE_LADDERS[get_color_mode()]
    return _ladder_color(index, ladder)


def color_for_cluster(index: int) -> str:
    """Mode-aware hex for the *index*-th cluster in a FIGURE (positional key).

    Thin wrapper over :func:`palette_color`; kept as the stable figure entry
    point so the figure modules need no edits to become mode-aware.
    """
    return palette_color(index)


def color_for_name(name: str) -> str:
    """Mode-aware hex for a cluster identified by NAME (interactive tree key).

    Hashes the (hierarchically-unique) name to a slot, so a cluster keeps the
    same slot — and a mode-consistent colour — across re-renders and siblings
    stay distinct. The unlabeled/container sentinels ('' / 'init') map to grey.
    """
    if not name or name == 'init':
        return GREY_UNLABELED
    idx = int(hashlib.md5(name.encode('utf-8')).hexdigest()[:8], 16)
    return palette_color(idx)
