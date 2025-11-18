
from .download import download_file
from .video_io import ReadArray
from .video_align import get_mask
from tqdm import tqdm
import numpy as np
import gc

import torch
import torch.nn.functional as F
import contextlib

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
        assert False, f"model_type mismatch {model_type}, expect dinov2_vitb14_reg4_pretrain."



def generate_dinov2(model_type='dinov2_vitb14_reg', device='', batch_size=16):
    if len(device) == 0:
        device = DEFAULT_DEVICE
    dinov2_args = {
        "model_type": model_type,
        "device": device,
        "batch_size": batch_size,
    }
    return ObserverDINOv2(dinov2_args)