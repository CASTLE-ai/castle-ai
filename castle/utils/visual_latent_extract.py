
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
        print(f"Device: {self.device}, AMP: {self.use_amp}, TF32: {self.device == 'cuda'}")
        
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
            autocast_ctx = torch.cuda.amp.autocast(enabled=self.use_amp)
            with autocast_ctx:
                result = self.model.forward_features(X)
        # Return features tensor on device to allow downstream GPU ops
        return result['x_norm_patchtokens'].detach()
    
    def __del__(self):
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
        # Stack frames: (B, H, W, 3) -> tensor (B, 3, H, W) in [0,1]
        frames_np = np.stack(frame_list, axis=0)
        if frames_np.dtype != np.float32:
            frames_np = frames_np.astype(np.float32) / 255.0
        frames_t = torch.from_numpy(frames_np).permute(0, 3, 1, 2).contiguous()

        # Prepare ROI masks and downsample to patch grid weights on GPU
        roi_masks_np = np.stack([get_mask(m, select_roi) for m in mask_list], axis=0)
        masks_t = torch.from_numpy(roi_masks_np)

        use_cuda = (self.model.device == 'cuda')
        if use_cuda:
            frames_t = frames_t.pin_memory()
            masks_t = masks_t.pin_memory()

        frames_t = frames_t.to(self.model.device, non_blocking=use_cuda)
        masks_t = masks_t.to(self.model.device, non_blocking=use_cuda, dtype=torch.float32)

        # Resize frames and masks to model resolution
        x = F.interpolate(frames_t, size=(resolution, resolution), mode='bilinear', align_corners=False, antialias=True)
        x.sub_(0.5).div_(0.2)

        masks_resized = F.interpolate(masks_t[:, None, ...], size=(resolution, resolution), mode='nearest')[:, 0]
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
    
    Args:
        model_type: Model type, options: 'dinov3_vitb16' or 'dinov3_vitl16'
        notify_func: Optional notification function for displaying messages in Gradio (e.g., gr.Info)
    
    Returns:
        ckpt_path: Checkpoint file path
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
    # 其他模型可以繼續添加
}


def resize_transform_dinov3(
    image: Image.Image,
    target_patches_per_side: int = DINOV3_TARGET_PATCHES_PER_SIDE,
    patch_size: int = DINOV3_PATCH_SIZE,
) -> torch.Tensor:
    """
    將圖像調整為固定的 patch 數量（37x37 = 1369 patches）
    
    Args:
        image: PIL Image
        target_patches_per_side: 目標每邊的 patch 數量（37）
        patch_size: patch 大小（16）
    
    Returns:
        tensor: 調整大小後的圖像 tensor (C, H, W)，確保為 37x37 patches
    """
    # 計算目標圖像尺寸
    target_size = target_patches_per_side * patch_size  # 37 * 16 = 592
    
    # 獲取原始圖像尺寸
    w, h = image.size
    
    # 計算縮放比例，保持寬高比，以較長邊為準
    scale = target_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # 調整大小
    image_resized = TF.resize(image, (new_h, new_w), interpolation=transforms.InterpolationMode.BICUBIC)
    
    # 中心裁剪為正方形
    min_size = min(image_resized.size)
    image_resized = TF.center_crop(image_resized, (min_size, min_size))
    
    # 最後調整為目標大小（592x592），確保得到 37x37 patches
    image_resized = TF.resize(image_resized, (target_size, target_size), interpolation=transforms.InterpolationMode.BICUBIC)
    
    return TF.to_tensor(image_resized)


class DinoV3latentGen:
    def __init__(self, model_cfg):
        self.device = model_cfg['device']
        model_name = model_cfg.get('model_type', 'dinov3_vitl16')
        ckpt_path = model_cfg.get('ckpt_path', None)
        notify_func = model_cfg.get('notify_func', None)  # 獲取通知函數
        self.model_name = model_name
        
        print(f"Loading DINOv3 model: {model_name}")
        print(f"Device: {self.device}")
        
        # 先確定 checkpoint 路徑並確保文件存在（如果不存在則下載）
        if ckpt_path is None:
            # 嘗試從配置中獲取路徑
            BASE_DIR = Path(__file__).parent.parent.parent
            # 根據模型名稱推斷 checkpoint 文件名
            if model_name in DINOV3_MODEL_TO_CKPT:
                ckpt_filename = DINOV3_MODEL_TO_CKPT[model_name]
                ckpt_path = BASE_DIR / "ckpt" / ckpt_filename
            else:
                # If no mapping exists, prompt user to configure checkpoint path
                available_models = ", ".join(DINOV3_MODEL_TO_CKPT.keys())
                raise ValueError(
                    f"DINOv3 model '{model_name}' does not have a checkpoint mapping configured.\n"
                    f"Please place the corresponding checkpoint file in the ckpt directory, or use one of the following supported models:\n"
                    f"{available_models}\n"
                    f"Or manually specify the checkpoint path via the ckpt_path parameter."
                )
        
        if isinstance(ckpt_path, str):
            ckpt_path = Path(ckpt_path)
            if not ckpt_path.is_absolute():
                BASE_DIR = Path(__file__).parent.parent.parent
                ckpt_path = BASE_DIR / ckpt_path
        
        # If checkpoint doesn't exist, try automatic download (before loading model architecture)
        if not ckpt_path.exists():
            message = f"Checkpoint file not found at: {ckpt_path}"
            print(message)
            if notify_func:
                try:
                    notify_func(message)
                except:
                    pass
            
            # Check if model type supports automatic download
            if model_name in DINOV3_MODEL_TO_CKPT:
                message = f"Automatically downloading {model_name} model checkpoint..."
                print(message)
                if notify_func:
                    try:
                        notify_func(message)
                    except:
                        pass
                
                try:
                    # Automatically download checkpoint, passing notification function
                    downloaded_path = download_dinov3_ckpt(model_name, notify_func=notify_func)
                    # Update ckpt_path to the downloaded path
                    BASE_DIR = Path(__file__).parent.parent.parent
                    ckpt_path = BASE_DIR / downloaded_path
                    message = f"Successfully downloaded checkpoint to: {ckpt_path}"
                    print(message)
                    if notify_func:
                        try:
                            notify_func(message)
                        except:
                            pass
                except Exception as e:
                    error_msg = f"DINOv3 checkpoint not found at: {ckpt_path}\nFailed to automatically download checkpoint: {e}\nPlease download the checkpoint file manually and place it in the ckpt directory."
                    if notify_func:
                        try:
                            notify_func(f"Download failed: {str(e)}")
                        except:
                            pass
                    raise FileNotFoundError(error_msg)
            else:
                error_msg = f"DINOv3 checkpoint not found at: {ckpt_path}\nModel '{model_name}' does not support automatic download.\nPlease download the checkpoint file manually and place it in the ckpt directory."
                if notify_func:
                    try:
                        notify_func(f"Model '{model_name}' does not support automatic download")
                    except:
                        pass
                raise FileNotFoundError(error_msg)
        
        # Now checkpoint exists, get model architecture from torch.hub
        # Note: torch.hub.load may try to download weights, but we will manually load local weights
        print("Loading model architecture from torch.hub...")
        source = "local" if DINOV3_LOCATION != DINOV3_GITHUB_LOCATION else "github"
        
        # Try to load model architecture using torch.hub (may try to download, but we'll handle errors)
        try:
            # Try to load model architecture, using pretrained=False if supported
            try:
                self.model = torch.hub.load(
                    repo_or_dir=DINOV3_LOCATION,
                    model=model_name,
                    source=source,
                    force_reload=False,
                    pretrained=False,  # Try not to auto-download weights
                )
            except TypeError:
                # If pretrained parameter is not supported, try without it
                # This may trigger download, but we'll catch the error
                self.model = torch.hub.load(
                    repo_or_dir=DINOV3_LOCATION,
                    model=model_name,
                    source=source,
                    force_reload=False,
                )
        except Exception as hub_error:
            # If torch.hub download fails (e.g., 403), try loading from local cache
            error_msg = str(hub_error)
            if "403" in error_msg or "Forbidden" in error_msg:
                print(f"Warning: torch.hub download blocked (403), attempting to load model architecture from local cache...")
            else:
                print(f"Warning: torch.hub.load failed: {hub_error}")
            
            # Try to use locally cached model architecture
            cache_dir = os.path.expanduser('~/.cache/torch/hub')
            local_repo = os.path.join(cache_dir, 'facebookresearch_dinov3_main')
            if os.path.exists(local_repo):
                print(f"Using local cached model from: {local_repo}")
                self.model = torch.hub.load(
                    repo_or_dir=local_repo,
                    model=model_name,
                    source='local',
                    force_reload=False,
                )
            else:
                raise RuntimeError(
                    f"Unable to load DINOv3 model architecture. torch.hub download failed (possibly blocked) and local cache does not exist.\n"
                    f"Please ensure you can access torch.hub or have dinov3 installed.\n"
                    f"Error: {hub_error}\n"
                    f"Tip: You can manually clone the dinov3 repository locally and set the DINOV3_LOCATION environment variable."
                )
        
        # 從本地 checkpoint 加載權重
        print(f"Loading checkpoint from: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        
        # 處理不同的 checkpoint 格式
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # 加載權重
        self.model.load_state_dict(state_dict, strict=False)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 獲取模型層數
        self.n_layers = DINOV3_MODEL_TO_NUM_LAYERS.get(model_name, 24)
        self.n_feature = self.model.embed_dim
        
        # Enable performance optimizations
        if self.device == 'cuda':
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        self.use_amp = (self.device == 'cuda')
        print(f"Model loaded successfully. Number of layers: {self.n_layers}, Embedding dim: {self.n_feature}")
        print(f"Device: {self.device}, AMP: {self.use_amp}, TF32: {self.device == 'cuda'}")
    
    def batch_run(self, X):
        if isinstance(X, list):
            X = torch.stack(X)
        return self.run(X)
    
    def single_run(self, x):
        X = torch.unsqueeze(x, 0)
        return self.run(X)
        
    def run(self, X):
        """
        提取 DINOv3 patch 特徵
        
        Args:
            X: 輸入圖像 tensor (B, 3, H, W)，已經標準化
        
        Returns:
            patch_features: patch 特徵 (B, num_patches, embed_dim)
        """
        # Model already on device and in eval mode from __init__
        with torch.inference_mode():
            if X.device.type != self.device:
                X = X.to(self.device, non_blocking=(self.device == 'cuda'))
            
            device_type = 'cuda' if (isinstance(self.device, torch.device) and self.device.type == 'cuda') or self.device == 'cuda' else 'cpu'
            autocast_ctx = torch.autocast(device_type=device_type, dtype=torch.float32, enabled=self.use_amp)
            
            with autocast_ctx:
                # 獲取所有層的特徵
                feats = self.model.get_intermediate_layers(
                    X, 
                    n=range(self.n_layers), 
                    reshape=True, 
                    norm=True
                )
                # 使用最後一層的特徵
                x = feats[-1]  # (B, embed_dim, H_patches, W_patches)
                B = x.size(0)
                dim = x.size(1)
                # 重塑為 (B, num_patches, embed_dim)
                x = x.view(B, dim, -1).permute(0, 2, 1)  # (B, num_patches, embed_dim)
        
        return x.detach()
    
    def __del__(self):
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
        """
        從批次圖像提取 latent，使用 DINOv3 模型
        
        Args:
            frame_list: 圖像列表，每個為 (H, W, 3) numpy array
            mask_list: mask 列表，每個為 (H, W) numpy array
            select_roi: ROI 索引
        
        Returns:
            latents: (B, embed_dim) numpy array
        """
        # 轉換圖像為 PIL Image 並預處理
        processed_images = []
        for frame in frame_list:
            if isinstance(frame, np.ndarray):
                # 確保是 uint8 格式
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
                image_pil = Image.fromarray(frame)
            else:
                image_pil = frame
            
            # 使用 DINOv3 的 resize_transform
            image_tensor = resize_transform_dinov3(
                image_pil, 
                DINOV3_TARGET_PATCHES_PER_SIDE, 
                DINOV3_PATCH_SIZE
            )
            # 標準化
            image_tensor = TF.normalize(
                image_tensor, 
                mean=DINOV3_IMAGENET_MEAN, 
                std=DINOV3_IMAGENET_STD
            )
            processed_images.append(image_tensor)
        
        # Stack 為 batch tensor
        frames_t = torch.stack(processed_images)  # (B, 3, 592, 592)
        
        # Prepare ROI masks
        roi_masks_np = np.stack([get_mask(m, select_roi) for m in mask_list], axis=0)
        masks_t = torch.from_numpy(roi_masks_np)
        
        use_cuda = (self.model.device == 'cuda')
        if use_cuda:
            frames_t = frames_t.pin_memory()
            masks_t = masks_t.pin_memory()
        
        frames_t = frames_t.to(self.model.device, non_blocking=use_cuda)
        masks_t = masks_t.to(self.model.device, non_blocking=use_cuda, dtype=torch.float32)
        
        # Resize masks to patch grid size (37x37)
        masks_resized = F.interpolate(
            masks_t[:, None, ...], 
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
        
        # avoid zero division if ROI mask is empty
        sum_w = w.sum(dim=(1, 2), keepdim=True).clamp_min(1e-6)  # (B, 1, 1)
        
        # Forward pass to get patch tokens: (B, 37*37, C)
        feats = self.model.run(frames_t)  # (B, 37*37, embed_dim)
        B = feats.size(0)
        C = feats.size(-1)
        feats = feats.view(B, self.patch_len, self.patch_len, C).float()
        w = w.float()
        
        # Weighted average on GPU, then move to CPU numpy
        weighted_sum = (feats * w[..., None]).sum(dim=(1, 2))  # (B, C)
        latents = weighted_sum / sum_w.view(B, 1)  # Reshape sum_w from (B,1,1) to (B,1)
        return latents.detach().cpu().numpy()

    def extract_video_latent(self, video_path, mask_video_path, roi_rgb, batch_size=16):
        # TODO
        pass

    def __del__(self):
        del self.model
        torch.cuda.empty_cache()


def generate_dinov3(model_type='dinov3_vitl16', device='', batch_size=16, ckpt_path=None, notify_func=None):
    """
    生成 DINOv3 observer
    
    Args:
        model_type: 模型名稱，可選 'dinov3_vits16', 'dinov3_vitb16', 'dinov3_vitl16', 'dinov3_vith16plus', 'dinov3_vit7b16'
        device: 設備，如果為空則自動選擇
        batch_size: 批次大小
        ckpt_path: checkpoint 文件路徑，如果為 None 則從配置文件或默認路徑加載
        notify_func: 可選的通知函數，用於在 Gradio 中顯示消息（例如 gr.Info）
    
    Returns:
        ObserverDINOv3 實例
    """
    if len(device) == 0:
        device = DEFAULT_DEVICE
    
    # 如果沒有指定 ckpt_path，嘗試從配置文件讀取
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
        "notify_func": notify_func,  # 傳遞通知函數
    }
    return ObserverDINOv3(dinov3_args)