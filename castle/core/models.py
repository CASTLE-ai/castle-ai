"""
castle/core/models.py
Unified Visual Encoder Interface.
"""

import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import List, Any, Optional, Union
import numpy as np
from torchvision import transforms
from PIL import Image
import torchvision.transforms.functional as TF

from castle.core.config import CKPT_DINO_IDS, DINOV3_CONSTANTS, DEFAULT_CKPT_DIR
from castle.core.environment import get_device
from castle.core.logging_config import setup_logger
from castle.utils.download import download_with_gdown

logger = setup_logger(__name__)

class VisualEncoder(ABC):
    """Base class for all visual encoders (DINOv2, DINOv3)."""
    
    def __init__(self, device: Optional[str] = None):
        self.device = device if device else get_device()
        self.model = None 
        self.n_feature = 768 
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def extract_features(self, batch_tensor: torch.Tensor) -> torch.Tensor:
        """Raw feature extraction from model."""
        pass
        
    def preprocess_batch(self, frame_batch: Union[torch.Tensor, List[np.ndarray], np.ndarray], masks: Optional[List[np.ndarray]] = None, roi_id: int = 1) -> torch.Tensor:
        """
        Prepares a batch of frames for the model.
        Default implementation for DINOv2-like models (Standard Resize/Norm).
        Can be overridden by subclasses (e.g. DINOv3).
        """
        if isinstance(frame_batch, list):
             processed = []
             for frame in frame_batch:
                 if isinstance(frame, np.ndarray):
                    img = TF.to_tensor(frame)
                 else:
                    img = frame
                 
                 # Align to patch size 14
                 h, w = img.shape[1], img.shape[2]
                 new_h = (h // 14) * 14
                 new_w = (w // 14) * 14
                 if new_h != h or new_w != w:
                     img = TF.resize(img, [new_h, new_w], interpolation=transforms.InterpolationMode.BICUBIC)
                 
                 processed.append(img)
             x = torch.stack(processed)
        else:
             x = frame_batch
             
        x = self.normalize(x)
        return x.to(self.device)

    def _weighted_pooling(self, features: torch.Tensor, masks: torch.Tensor, image_size: int, patch_size: int) -> torch.Tensor:
        """
        Computes weighted pooling of features using masks (DRY fix).
        
        Args:
            features: (B, N, C) Feature tensor from backbone
            masks: (B, H, W) Tensor of masks (input resolution)
            image_size: Model input resolution (e.g. 518, 592)
            patch_size: Patch size in pixels (e.g. 14, 16)
        """
        # Resize Masks (Nearest)
        masks_resized = F.interpolate(masks[:, None, ...], size=(image_size, image_size), mode='nearest')[:, 0]
        
        target_patches = image_size // patch_size
        
        # Downsample to Patch Grid
        # View as (B, Target, PatchSize, Target, PatchSize)
        w = masks_resized.view(masks_resized.size(0), target_patches, patch_size, target_patches, patch_size).sum(dim=(2, 4))
        sum_w = w.sum(dim=(1, 2), keepdim=True).clamp_min(1e-6)

        B, N, C = features.shape
        
        # Reshape feats to B, Target, Target, C
        feats = features.view(B, target_patches, target_patches, C).float()
        w = w.float()
        
        weighted_sum = (feats * w[..., None]).sum(dim=(1, 2))
        latents = weighted_sum / sum_w.view(B, 1)
        
        return latents

    def _multiscale_pooling(self, features: torch.Tensor, masks: torch.Tensor,
                            image_size: int, patch_size: int,
                            scales: Optional[List[int]] = None) -> torch.Tensor:
        """
        Multi-scale spatial pyramid pooling (SPP).

        Divides the patch grid into s×s regions for each scale s,
        computes mask-weighted average in each region, then concatenates.

        Args:
            features: (B, N, C) patch token features
            masks: (B, H, W) ROI masks at input resolution
            image_size: model input resolution
            patch_size: patch size in pixels
            scales: list of grid divisions [1, 2, 4] → 1×1 + 2×2 + 4×4

        Returns:
            (B, sum(s²) × C) concatenated multi-scale features
            e.g., scales=[1,2,4] → (B, 21 × 768) = (B, 16128)
        """
        if scales is None:
            scales = [1, 2, 4]
        # 1. Resize masks to image_size, then downsample to patch grid
        masks_resized = F.interpolate(
            masks[:, None, ...], size=(image_size, image_size), mode='nearest'
        )[:, 0]
        target_patches = image_size // patch_size

        # Patch-level weights: (B, target_patches, target_patches)
        w = masks_resized.view(
            masks.size(0), target_patches, patch_size, target_patches, patch_size
        ).sum(dim=(2, 4)).float()

        B, N, C = features.shape
        feats = features.view(B, target_patches, target_patches, C).float()

        results = []
        for s in scales:
            if s == 1:
                # Global pooling — identical to _weighted_pooling
                sum_w = w.sum(dim=(1, 2), keepdim=True).clamp_min(1e-6)
                weighted_sum = (feats * w[..., None]).sum(dim=(1, 2))
                results.append(weighted_sum / sum_w.view(B, 1))  # (B, C)
            elif target_patches % s == 0:
                # Evenly divisible: efficient reshape
                region_h = target_patches // s
                region_w = target_patches // s
                feats_regions = feats.view(B, s, region_h, s, region_w, C)
                w_regions = w.view(B, s, region_h, s, region_w)
                weighted = (feats_regions * w_regions[..., None]).sum(dim=(2, 4))  # (B, s, s, C)
                total = w_regions.sum(dim=(2, 4)).clamp_min(1e-6)  # (B, s, s)
                pooled = weighted / total[..., None]  # (B, s, s, C)
                results.append(pooled.reshape(B, s * s * C))  # (B, s²×C)
            else:
                # Non-divisible (e.g. 37 patches): adaptive region boundaries
                region_vecs = []
                for i in range(s):
                    for j in range(s):
                        h_start = (i * target_patches) // s
                        h_end = ((i + 1) * target_patches) // s
                        w_start = (j * target_patches) // s
                        w_end = ((j + 1) * target_patches) // s

                        region_feats = feats[:, h_start:h_end, w_start:w_end, :]
                        region_w_vals = w[:, h_start:h_end, w_start:w_end]

                        weighted = (region_feats * region_w_vals[..., None]).sum(dim=(1, 2))
                        total = region_w_vals.sum(dim=(1, 2)).clamp_min(1e-6)
                        region_vecs.append(weighted / total[:, None])  # (B, C)
                # Stack and flatten: (B, s² * C)
                results.append(torch.stack(region_vecs, dim=1).reshape(B, s * s * C))

        return torch.cat(results, dim=1)  # (B, sum(s²)×C)


    def extract_tensor_batch(self, frame_batch: Any, mask_batch: Any, roi_id: int) -> List[np.ndarray]:
        if self.model is None:
            self.load_model()
            
        with torch.no_grad():
            x = self.preprocess_batch(frame_batch, mask_batch, roi_id)
            features = self.extract_features(x)
            return features.cpu().numpy()

    def extract_batch_latent(self, frames: List[np.ndarray], masks: List[np.ndarray], select_roi: int) -> List[np.ndarray]:
        return self.extract_tensor_batch(frames, masks, select_roi)


class DINOv2Encoder(VisualEncoder):
    def __init__(self, model_type: str = 'dinov2_vitb14', device: Optional[str] = None):
        super().__init__(device)
        self.model_name = 'dinov2_vitb14_reg' if 'reg' in model_type else model_type
        self.batch_size = 16 # Default
        
        if 'vitb14' in self.model_name: self.n_feature = 768
        elif 'vitl14' in self.model_name: self.n_feature = 1024
        elif 'vits14' in self.model_name: self.n_feature = 384
        
        self.resolution = 518
        self.patch_len = self.resolution // 14 # 37
        
        assert self.resolution % 14 == 0, "Resolution must be divisible by patch size 14"


    def load_model(self):
        logger.info(f"Loading DINOv2: {self.model_name}")
        self.model = torch.hub.load('facebookresearch/dinov2', self.model_name)
        self.model.eval().to(self.device)

    def preprocess_batch(self, frame_batch, mask_batch, roi_id):
        if isinstance(frame_batch, torch.Tensor):
             frames_t = frame_batch.permute(0, 3, 1, 2).contiguous().float()
             frames_t.div_(255.0)
        else:
             frames_np = np.stack(frame_batch, axis=0)
             if frames_np.dtype != np.float32:
                  frames_np = frames_np.astype(np.float32) / 255.0
             frames_t = torch.from_numpy(frames_np).permute(0, 3, 1, 2).contiguous()
        
        frames_t = F.interpolate(frames_t, size=(self.resolution, self.resolution), mode='bilinear', align_corners=False, antialias=True)
        frames_t = self.normalize(frames_t)
        
        return frames_t.to(self.device)

    def extract_features(self, x, layers=None):
        """Extract features from specified layers.

        Args:
            layers: list of layer indices (0-indexed). None = last layer only
                    (uses forward_features for backward compatibility).
                    e.g., [3, 7, 11] for layers 4, 8, 12.
        """
        if layers is None:
            return self.model.forward_features(x)['x_norm_patchtokens']
        else:
            feats = self.model.get_intermediate_layers(x, n=layers, reshape=False, norm=True)
            if len(feats) == 1:
                return feats[0]
            return torch.cat(feats, dim=2)  # B, N, C_total

    def extract_tensor_batch(self, frame_batch, mask_batch, roi_id,
                             pooling='weighted_average', scales=None, layers=None):
         """Extract latent features with configurable pooling and layers.

         Args:
             pooling: 'weighted_average' (original) or 'multiscale'
             scales: list of ints for multiscale, e.g. [1, 2, 4].
             layers: list of layer indices. None = last layer only.
         """
         if self.model is None: self.load_model()
         
         x = self.preprocess_batch(frame_batch, mask_batch, roi_id)
         
         if not isinstance(mask_batch, torch.Tensor):
             mask_batch = torch.from_numpy(np.stack(mask_batch, axis=0))
         masks_t = (mask_batch.to(self.device) == roi_id).to(dtype=torch.float32)
         
         with torch.inference_mode():
             if self.device == 'cuda':
                 with torch.autocast(device_type='cuda', dtype=torch.float16):
                     feats = self.extract_features(x, layers=layers)
                     if pooling == 'multiscale' and scales:
                         latents = self._multiscale_pooling(feats, masks_t, self.resolution, 14, scales)
                     else:
                         latents = self._weighted_pooling(feats, masks_t, self.resolution, 14)
             else:
                 feats = self.extract_features(x, layers=layers)
                 if pooling == 'multiscale' and scales:
                     latents = self._multiscale_pooling(feats, masks_t, self.resolution, 14, scales)
                 else:
                     latents = self._weighted_pooling(feats, masks_t, self.resolution, 14)
             
         return latents.cpu().numpy()



class DINOv3Encoder(VisualEncoder):
    def __init__(self, model_type: str = 'dinov3_vitb16', device: Optional[str] = None):
        super().__init__(device)
        self.model_type = model_type
        self.filename = DINOV3_CONSTANTS['MODEL_TO_CKPT_FILENAME'].get(model_type, "")
        self.n_layers = DINOV3_CONSTANTS['MODEL_TO_NUM_LAYERS'].get(model_type, 12)
        
        if 'vitl' in model_type: self.n_feature = 1024
        else: self.n_feature = 768 # vitb usually 768
        
        # DINOv3 constants
        self.patch_size = DINOV3_CONSTANTS['PATCH_SIZE']
        self.target_patches = DINOV3_CONSTANTS['TARGET_PATCHES_PER_SIDE'] # 37
        self.image_size = DINOV3_CONSTANTS['IMAGE_SIZE'] # 592
        assert self.image_size % self.patch_size == 0, "Image size must be divisible by patch size"

        self.mean = DINOV3_CONSTANTS['IMAGENET_MEAN']
        self.std = DINOV3_CONSTANTS['IMAGENET_STD']

    def load_model(self):
        logger.info(f"Loading DINOv3: {self.model_type}")
        
        # 1. Download Checkpoint
        ckpt_path = DEFAULT_CKPT_DIR / self.filename
        if not ckpt_path.exists():
            logger.info(f"Checkpoint missing. Downloading {self.filename}...")
            file_id = CKPT_DINO_IDS.get(self.model_type)
            if not file_id:
                raise ValueError(f"No Google Drive ID for {self.model_type}")
            download_with_gdown(file_id, str(ckpt_path))
            
        # 2. Create Model Architecture (Hub)
        try:
             self.model = torch.hub.load('facebookresearch/dinov3', self.model_type, pretrained=False)
        except Exception:
             # Fallback: sometimes model names differ in hub vs filename
             # Attempt generic load or warn
             raise RuntimeError("Failed to load DINOv3 from torch.hub")

        # 3. Load Weights
        logger.info(f"Loading weights from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        
        state_dict = checkpoint
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint: state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint: state_dict = checkpoint['model']

        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device).eval()

    def _resize_transform(self, image_pil):
        # Implementation of resize_transform_dinov3
        w, h = image_pil.size
        target_size = self.image_size
        
        scale = target_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        img = TF.resize(image_pil, (new_h, new_w), interpolation=transforms.InterpolationMode.BICUBIC)
        min_s = min(img.size)
        img = TF.center_crop(img, (min_s, min_s))
        img = TF.resize(img, (target_size, target_size), interpolation=transforms.InterpolationMode.BICUBIC)
        return TF.to_tensor(img)

    def preprocess_batch(self, frame_list, mask_list, roi_id):
        if isinstance(frame_list, torch.Tensor):
             img_t = frame_list.permute(0, 3, 1, 2).float().div(255.0)
             
             B, C, H, W = img_t.shape
             target_size = self.image_size
             
             if H == W:
                 img_t = F.interpolate(img_t, size=(target_size, target_size), mode='bicubic', align_corners=False)
             else:
                 scale = target_size / max(H, W)
                 new_h, new_w = int(H * scale), int(W * scale)
                 
                 img_t = F.interpolate(img_t, size=(new_h, new_w), mode='bicubic', align_corners=False)
                 
                 # Center crop to square
                 min_s = min(new_h, new_w)
                 start_h = (new_h - min_s) // 2
                 start_w = (new_w - min_s) // 2
                 img_t = img_t[:, :, start_h:start_h+min_s, start_w:start_w+min_s]
                 
                 # Resize to target
                 img_t = F.interpolate(img_t, size=(target_size, target_size), mode='bicubic', align_corners=False)
             
             img_t = TF.normalize(img_t, mean=self.mean, std=self.std)
             
             return img_t.to(self.device)
             
        else:
            processed = []
            for frame in frame_list:
                if isinstance(frame, np.ndarray):
                     if frame.dtype != np.uint8 and frame.max() <= 1.0: frame = (frame * 255).astype(np.uint8)
                     elif frame.dtype != np.uint8: frame = frame.astype(np.uint8)
                     img_pil = Image.fromarray(frame)
                else:
                     img_pil = frame
                     
                img_t = self._resize_transform(img_pil) # C, H, W (0-1)
                img_t = TF.normalize(img_t, mean=self.mean, std=self.std)
                processed.append(img_t)
                
            return torch.stack(processed).to(self.device)

    def extract_features(self, x, layers=None):
        """Extract features from specified layers.

        Args:
            layers: list of layer indices (0-indexed). None = last layer only.
                    e.g., [3, 7, 11] for layers 4, 8, 12.
        """
        if layers is None:
            layer_ids = [self.n_layers - 1]
        else:
            layer_ids = layers

        with torch.inference_mode():
            if self.device == 'cuda':
                 with torch.autocast(device_type='cuda', dtype=torch.float16):
                      feats = self.model.get_intermediate_layers(x, n=layer_ids, reshape=True, norm=True)
            else:
                 feats = self.model.get_intermediate_layers(x, n=layer_ids, reshape=True, norm=True)

            if len(feats) == 1:
                x_out = feats[0]  # B, C, H, W
                B, C, H, W = x_out.shape
                return x_out.view(B, C, -1).permute(0, 2, 1)  # B, N, C
            else:
                # Concatenate features from multiple layers along channel dim
                combined = []
                for feat in feats:
                    B, C, H, W = feat.shape
                    combined.append(feat.view(B, C, -1).permute(0, 2, 1))  # B, N, C
                return torch.cat(combined, dim=2)  # B, N, C_total

    def extract_tensor_batch(self, frame_batch, mask_batch, roi_id,
                             pooling='weighted_average', scales=None, layers=None):
        """Extract latent features with configurable pooling and layers.

        Args:
            pooling: 'weighted_average' (original) or 'multiscale'
            scales: list of ints for multiscale, e.g. [1, 2, 4].
            layers: list of layer indices. None = last layer only.
        """
        if self.model is None: self.load_model()
        
        x = self.preprocess_batch(frame_batch, mask_batch, roi_id)
        
        if not isinstance(mask_batch, torch.Tensor):
            mask_batch = torch.from_numpy(np.stack(mask_batch, axis=0))
        masks_t = (mask_batch.to(self.device) == roi_id).to(dtype=torch.float32)
        
        with torch.no_grad():
             feats = self.extract_features(x, layers=layers)
             if pooling == 'multiscale' and scales:
                 latents = self._multiscale_pooling(feats, masks_t, self.image_size, self.patch_size, scales)
             else:
                 latents = self._weighted_pooling(feats, masks_t, self.image_size, self.patch_size)
             
        return latents.cpu().numpy()



# A-07: Model singleton cache — avoid reloading the same model
_model_cache: dict = {}
_MODEL_CACHE_MAX = 1  # Keep at most 1 model cached (the current one)


def _evict_model_cache():
    """Evict all models from cache and free GPU memory."""
    for old_key in list(_model_cache.keys()):
        logger.info(f"Evicting cached model: {old_key}")
        old_model = _model_cache.pop(old_key)
        # Delete model's underlying torch model to free GPU memory
        if hasattr(old_model, 'model') and old_model.model is not None:
            del old_model.model
            old_model.model = None
        del old_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_visual_encoder(model_name: str) -> VisualEncoder:
    """Get or create a visual encoder, with singleton caching.
    
    Keeps at most _MODEL_CACHE_MAX models cached. When the cache is full
    and a new model is requested, old models are evicted and GPU memory
    is freed via torch.cuda.empty_cache().
    """
    if model_name in _model_cache:
        logger.debug(f"Model cache hit: {model_name}")
        return _model_cache[model_name]
    
    # Evict old models if cache is full
    if len(_model_cache) >= _MODEL_CACHE_MAX:
        _evict_model_cache()
    
    # Create new encoder
    if 'dinov3' in model_name:
        encoder = DINOv3Encoder(model_name)
    else:
        encoder = DINOv2Encoder(model_name)
    
    _model_cache[model_name] = encoder
    logger.info(f"Created and cached encoder: {model_name}")
    return _model_cache[model_name]


def clear_model_cache():
    """Clear the model cache and free GPU memory."""
    _evict_model_cache()
    logger.info("Model cache cleared")
