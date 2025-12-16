from .download import download_file, download_with_gdown
from .video_io import ReadArray
from .video_align import get_mask
from tqdm import tqdm
import numpy as np
import gc
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import contextlib
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image

import platform
OS_SYS = platform.uname().system
if OS_SYS == 'Darwin':
    DEFAULT_DEVICE = 'mps'
elif torch.cuda.is_available():
    DEFAULT_DEVICE = 'cuda'
else:
    DEFAULT_DEVICE = 'cpu'


resolution = 518
patch_len = resolution // 14

class DinoV2latentGen:
    def __init__(self, model_cfg):
        self.device = model_cfg['device']
        print("Loading DinoV2 model...")
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
        self.model.eval().to(self.device)
        
        # Enable performance optimizations
        if self.device == 'cuda':
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        self.use_amp = (self.device == 'cuda')
        self.n_feature = self.model.embed_dim
        
    def batch_run(self, X):
        if isinstance(X, list):
            X = torch.stack(X)
        return self.run(X)
    
    def single_run(self, x):
        X = torch.unsqueeze(x, 0)
        return self.run(X)
        
    def run(self, X):
        # Model already on device and in eval mode from __init__
        with torch.no_grad():
            if X.device.type != self.device:
                X = X.to(self.device, non_blocking=(self.device == 'cuda'))
            autocast_ctx = torch.autocast(device_type='cuda', enabled=self.use_amp) if self.device == 'cuda' else contextlib.nullcontext()
            with autocast_ctx:
                result = self.model.forward_features(X)
        # Return features tensor on device to allow downstream GPU ops
        return result['x_norm_patchtokens'].detach()
    
    def __del__(self):
        if hasattr(self, 'model'):
            del self.model
        
   
class ObserverDINOv2:
    def __init__(self, dinov2_args):

        self.model = DinoV2latentGen(dinov2_args)
        self.n_feature = self.model.n_feature
        self.batch_size = dinov2_args['batch_size']

    def nan_latent(self):
        return np.full((self.n_feature), np.nan, dtype=np.float16)

    def extract_image_latent(self, frame, mask, select_roi):
        return self.extract_batch_latent([frame], [mask], select_roi)[0]

    def extract_batch_latent(self, frame_list, mask_list, select_roi):
        # This is now a backward-compatibility wrapper.
        # It converts lists to tensors and calls the new, efficient method.
        frames_np = np.stack(frame_list, axis=0) 
        frames_t = torch.from_numpy(frames_np)
        
        # Do NOT apply get_mask here. Pass the raw full masks.
        masks_np = np.stack(mask_list, axis=0)
        masks_t = torch.from_numpy(masks_np)

        return self.extract_tensor_batch(frames_t, masks_t, select_roi)

    def extract_tensor_batch(self, frames_t, masks_t, select_roi):
        # This method assumes frames_t and masks_t are already tensors
        use_cuda = (self.model.device == 'cuda')
        if use_cuda:
            frames_t = frames_t.pin_memory()
            masks_t = masks_t.pin_memory()
        
        frames_t = frames_t.to(self.model.device, non_blocking=use_cuda)
        masks_t = masks_t.to(self.model.device, non_blocking=use_cuda)
        
        # [GPU] Replicate get_mask logic on GPU: cv2.inRange(mask, val, val) is equivalent to (mask == val)
        # Assuming masks_t from DataLoader has shape (B, H, W, C) or (B, H, W)
        if masks_t.dim() == 4 and masks_t.shape[3] == 1: # Handle case with channel dim
            masks_t = masks_t.squeeze(-1)
        roi_masks_t = (masks_t == select_roi).float()


        # [優化] 3. 在 GPU 上做正規化
        # (B, H, W, C) -> (B, C, H, W) -> float -> div(255)
        frames_t = frames_t.permute(0, 3, 1, 2).float().div_(255.0)

        # Resize frames and masks to model resolution (GPU operation)
        x = F.interpolate(frames_t, size=(resolution, resolution), mode='bilinear', align_corners=False, antialias=True)
        x.sub_(0.5).div_(0.2)

        masks_resized = F.interpolate(roi_masks_t[:, None, ...], size=(resolution, resolution), mode='nearest')[:, 0]
        # Downsample 518x518 -> 37x37 by summing over 14x14 windows
        w = masks_resized.view(masks_resized.size(0), patch_len, 14, patch_len, 14).sum(dim=(2, 4))
        # avoid zero division if ROI mask is empty
        sum_w = w.sum(dim=(1, 2), keepdim=True).clamp_min(1e-6)  # (B, 1, 1)

        # Forward pass to get patch tokens: (B, 37*37, C)
        feats = self.model.run(x)
        B = feats.size(0)
        C = feats.size(-1)
        feats = feats.view(B, patch_len, patch_len, C).float()
        w = w.float()

        # Weighted average on GPU, then move to CPU numpy
        weighted_sum = (feats * w[..., None]).sum(dim=(1, 2))  # (B, C)
        latents = weighted_sum / sum_w.view(B, 1)  # Reshape sum_w from (B,1,1) to (B,1)
        return latents.detach().cpu().numpy()

    def extract_video_latent(self, video_path, mask_video_path, roi_rgb, batch_size=16):
        # TODO
        pass

    def __del__(self):
        if hasattr(self, 'model'):
            del self.model
        torch.cuda.empty_cache()


def download_dinov2_ckpt(model_type):
    if model_type == 'dinov2_vitb14_reg':
        ckpt_path = 'ckpt/dinov2_vitb14_reg4_pretrain.pth'
        download_file('https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth', ckpt_path)

        return ckpt_path
    else:
        raise ValueError(f"model_type mismatch {model_type}, expect dinov2_vitb14_reg4_pretrain.")


def download_dinov3_ckpt(model_type, notify_func=None):
    """
    Download DINOv3 model checkpoint
    """
    if model_type not in DINOV3_MODEL_TO_CKPT:
        available_models = ", ".join(DINOV3_MODEL_TO_CKPT.keys())
        raise ValueError(
            f"model_type mismatch {model_type}, expect one of: {available_models}"
        )
    
    ckpt_filename = DINOV3_MODEL_TO_CKPT[model_type]
    ckpt_path = f'ckpt/{ckpt_filename}'
    
    # Google Drive 文件 ID 映射
    gdrive_file_ids = {
        "dinov3_vitb16": "18doehnHWWnz9zBtOdgYZ3XMTpgPYbYZ6",
        "dinov3_vitl16": "195H5UHKJ0r4qRDY7Ly6WJrXGnpdlHMSu",
    }
    
    file_id = gdrive_file_ids.get(model_type)
    if file_id:
        download_with_gdown(file_id, ckpt_path, notify_func=notify_func)
    else:
        raise ValueError(f"No Google Drive file ID configured for {model_type}")
    
    return ckpt_path



def generate_dinov2(model_type='dinov2_vitb14_reg', device='', batch_size=16):
    if len(device) == 0:
        device = DEFAULT_DEVICE
    dinov2_args = {
        "model_type": model_type,
        "device": device,
        "batch_size": batch_size,
    }
    return ObserverDINOv2(dinov2_args)


# ============================================================================
# DINOv3 Support
# ============================================================================

# DINOv3 配置
DINOV3_GITHUB_LOCATION = "facebookresearch/dinov3"
DINOV3_LOCATION = os.getenv("DINOV3_LOCATION", DINOV3_GITHUB_LOCATION)

# DINOv3 模型配置
DINOV3_PATCH_SIZE = 16
DINOV3_TARGET_PATCHES_PER_SIDE = 37  # 目標：37x37 = 1369 patches
DINOV3_IMAGE_SIZE = DINOV3_TARGET_PATCHES_PER_SIDE * DINOV3_PATCH_SIZE  # 37 * 16 = 592
DINOV3_IMAGENET_MEAN = (0.485, 0.456, 0.406)
DINOV3_IMAGENET_STD = (0.229, 0.224, 0.225)

# 模型層數映射
DINOV3_MODEL_TO_NUM_LAYERS = {
    "dinov3_vits16": 12,
    "dinov3_vits16plus": 12,
    "dinov3_vitb16": 12,
    "dinov3_vitl16": 24,
    "dinov3_vith16plus": 32,
    "dinov3_vit7b16": 40,
}

# 模型名稱到 checkpoint 文件名的映射
DINOV3_MODEL_TO_CKPT = {
    "dinov3_vitl16": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
    "dinov3_vitb16": "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
}

# [已棄用] 舊的 CPU resize 函式，保留但不再使用，以避免破壞舊代碼引用
def resize_transform_dinov3(image: Image.Image, target_patches_per_side: int = DINOV3_TARGET_PATCHES_PER_SIDE, patch_size: int = DINOV3_PATCH_SIZE) -> torch.Tensor:
    target_size = target_patches_per_side * patch_size
    w, h = image.size
    scale = target_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    image_resized = TF.resize(image, (new_h, new_w), interpolation=transforms.InterpolationMode.BICUBIC)
    min_size = min(image_resized.size)
    image_resized = TF.center_crop(image_resized, (min_size, min_size))
    image_resized = TF.resize(image_resized, (target_size, target_size), interpolation=transforms.InterpolationMode.BICUBIC)
    return TF.to_tensor(image_resized)


class DinoV3latentGen:
    def __init__(self, model_cfg):
        self.device = model_cfg['device']
        model_name = model_cfg.get('model_type', 'dinov3_vitl16')
        ckpt_path = model_cfg.get('ckpt_path', None)
        notify_func = model_cfg.get('notify_func', None)
        self.model_name = model_name
        
        print(f"Loading DINOv3 model: {model_name}")
        print(f"Device: {self.device}")
        
        # --- Checkpoint loading logic (kept exactly as provided) ---
        if ckpt_path is None:
            BASE_DIR = Path(__file__).parent.parent.parent
            if model_name in DINOV3_MODEL_TO_CKPT:
                ckpt_filename = DINOV3_MODEL_TO_CKPT[model_name]
                ckpt_path = BASE_DIR / "ckpt" / ckpt_filename
            else:
                available_models = ", ".join(DINOV3_MODEL_TO_CKPT.keys())
                raise ValueError(f"DINOv3 model '{model_name}' does not have a checkpoint mapping configured.")
        
        if isinstance(ckpt_path, str):
            ckpt_path = Path(ckpt_path)
            if not ckpt_path.is_absolute():
                BASE_DIR = Path(__file__).parent.parent.parent
                ckpt_path = BASE_DIR / ckpt_path
        
        if not ckpt_path.exists():
            # Auto download logic
            if model_name in DINOV3_MODEL_TO_CKPT:
                if notify_func: notify_func(f"Automatically downloading {model_name}...")
                try:
                    downloaded_path = download_dinov3_ckpt(model_name, notify_func=notify_func)
                    BASE_DIR = Path(__file__).parent.parent.parent
                    ckpt_path = BASE_DIR / downloaded_path
                except Exception as e:
                    raise FileNotFoundError(f"Failed to download checkpoint: {e}")
            else:
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        
        print("Loading model architecture from torch.hub...")
        source = "local" if DINOV3_LOCATION != DINOV3_GITHUB_LOCATION else "github"
        
        try:
            try:
                self.model = torch.hub.load(repo_or_dir=DINOV3_LOCATION, model=model_name, source=source, force_reload=False, pretrained=False)
            except TypeError:
                self.model = torch.hub.load(repo_or_dir=DINOV3_LOCATION, model=model_name, source=source, force_reload=False)
        except Exception as hub_error:
            cache_dir = os.path.expanduser('~/.cache/torch/hub')
            local_repo = os.path.join(cache_dir, 'facebookresearch_dinov3_main')
            if os.path.exists(local_repo):
                print(f"Using local cached model from: {local_repo}")
                self.model = torch.hub.load(repo_or_dir=local_repo, model=model_name, source='local', force_reload=False)
            else:
                raise RuntimeError(f"Unable to load DINOv3 model. Error: {hub_error}")
        
        print(f"Loading checkpoint from: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
        else:
            state_dict = checkpoint
        
        self.model.load_state_dict(state_dict, strict=False)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.n_layers = DINOV3_MODEL_TO_NUM_LAYERS.get(model_name, 24)
        self.n_feature = self.model.embed_dim
        
        if self.device == 'cuda':
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        self.use_amp = (self.device == 'cuda')
        print(f"Model loaded successfully. Layers: {self.n_layers}, Dim: {self.n_feature}")
    
    def batch_run(self, X):
        if isinstance(X, list):
            X = torch.stack(X)
        return self.run(X)
    
    def single_run(self, x):
        X = torch.unsqueeze(x, 0)
        return self.run(X)
        
    def run(self, X):
        # Model already on device and in eval mode
        with torch.inference_mode():
            if X.device.type != self.device:
                X = X.to(self.device, non_blocking=(self.device == 'cuda'))
            
            device_type = 'cuda' if (isinstance(self.device, torch.device) and self.device.type == 'cuda') or self.device == 'cuda' else 'cpu'
            autocast_ctx = torch.autocast(device_type=device_type, dtype=torch.float32, enabled=self.use_amp)
            
            with autocast_ctx:
                feats = self.model.get_intermediate_layers(X, n=range(self.n_layers), reshape=True, norm=True)
                x = feats[-1]  # (B, embed_dim, H_patches, W_patches)
                B = x.size(0)
                dim = x.size(1)
                x = x.view(B, dim, -1).permute(0, 2, 1)  # (B, num_patches, embed_dim)
        
        return x.detach()
    
    def __del__(self):
        if hasattr(self, 'model'):
            del self.model


class ObserverDINOv3:
    def __init__(self, dinov3_args):
        self.model = DinoV3latentGen(dinov3_args)
        self.n_feature = self.model.n_feature
        self.batch_size = dinov3_args['batch_size']
        self.patch_len = DINOV3_TARGET_PATCHES_PER_SIDE  # 37

    def nan_latent(self):
        return np.full((self.n_feature), np.nan, dtype=np.float16)

    def extract_image_latent(self, frame, mask, select_roi):
        return self.extract_batch_latent([frame], [mask], select_roi)[0]

    def extract_batch_latent(self, frame_list, mask_list, select_roi):
        # This is now a backward-compatibility wrapper.
        # It converts lists to tensors and calls the new, efficient method.
        frames_np = np.stack(frame_list, axis=0) 
        frames_t = torch.from_numpy(frames_np)
        
        # Do NOT apply get_mask here. Pass the raw full masks.
        masks_np = np.stack(mask_list, axis=0)
        masks_t = torch.from_numpy(masks_np)

        return self.extract_tensor_batch(frames_t, masks_t, select_roi)

    def extract_tensor_batch(self, frames_t, masks_t, select_roi):
        device = self.model.device
        use_cuda = (device == 'cuda')

        if use_cuda:
            frames_t = frames_t.pin_memory()
            masks_t = masks_t.pin_memory()
            
        frames_t = frames_t.to(device, non_blocking=use_cuda)
        masks_t = masks_t.to(device, non_blocking=use_cuda)

        # [GPU] Replicate get_mask logic on GPU
        if masks_t.dim() == 4 and masks_t.shape[3] == 1:
            masks_t = masks_t.squeeze(-1)
        roi_masks_t = (masks_t == select_roi).float()

        # [GPU] 轉維度 + 正規化 (0-255 -> 0-1)
        frames_t = frames_t.permute(0, 3, 1, 2).float().div_(255.0)

        # [GPU] DINOv3 極速縮放 (Resize to 592x592)
        frames_resized = F.interpolate(
            frames_t, 
            size=(DINOV3_IMAGE_SIZE, DINOV3_IMAGE_SIZE), 
            mode='bilinear', 
            align_corners=False, 
            antialias=True
        )

        # [GPU] ImageNet 標準化 (Mean/Std)
        mean = torch.tensor(DINOV3_IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        std = torch.tensor(DINOV3_IMAGENET_STD, device=device).view(1, 3, 1, 1)
        frames_final = (frames_resized - mean) / std
        
        # Resize masks to patch grid size (37x37)
        masks_resized = F.interpolate(
            roi_masks_t[:, None, ...], 
            size=(DINOV3_IMAGE_SIZE, DINOV3_IMAGE_SIZE), 
            mode='nearest'
        )[:, 0]
        
        # Downsample 592x592 -> 37x37 by summing over 16x16 windows
        w = masks_resized.view(
            masks_resized.size(0), 
            self.patch_len, 
            DINOV3_PATCH_SIZE, 
            self.patch_len, 
            DINOV3_PATCH_SIZE
        ).sum(dim=(2, 4))
        
        sum_w = w.sum(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        
        # 模型推論
        feats = self.model.run(frames_final)  # (B, 37*37, embed_dim)
        B = feats.size(0)
        C = feats.size(-1)
        feats = feats.view(B, self.patch_len, self.patch_len, C).float()
        w = w.float()
        
        # 加權平均
        weighted_sum = (feats * w[..., None]).sum(dim=(1, 2))
        latents = weighted_sum / sum_w.view(B, 1)
        return latents.detach().cpu().numpy()
        
    def extract_video_latent(self, video_path, mask_video_path, roi_rgb, batch_size=16):
        pass

    def __del__(self):
        if hasattr(self, 'model'):
            del self.model
        torch.cuda.empty_cache()


def generate_dinov3(model_type='dinov3_vitl16', device='', batch_size=16, ckpt_path=None, notify_func=None):
    if len(device) == 0:
        device = DEFAULT_DEVICE
    
    if ckpt_path is None:
        try:
            import json
            from pathlib import Path
            BASE_DIR = Path(__file__).parent.parent.parent
            config_path = BASE_DIR / "castle" / "configs" / "model_config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                if 'dinov3_args' in config and 'path' in config['dinov3_args']:
                    config_path_str = config['dinov3_args']['path']
                    if config_path_str:
                        ckpt_path = BASE_DIR / config_path_str if not Path(config_path_str).is_absolute() else Path(config_path_str)
        except Exception as e:
            print(f"Warning: Could not load config: {e}")
    
    dinov3_args = {
        "model_type": model_type,
        "device": device,
        "batch_size": batch_size,
        "ckpt_path": str(ckpt_path) if ckpt_path else None,
        "notify_func": notify_func,
    }
    return ObserverDINOv3(dinov3_args)