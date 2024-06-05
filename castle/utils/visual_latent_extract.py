
from .download import download_file
from .video_io import ReadArray
from .video_align import get_mask
from tqdm import tqdm
import numpy as np

import torch
import torchvision.transforms as tt
from torch.utils.data import Dataset, DataLoader
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
img2tensor = tt.Compose([
    tt.ToTensor(), # range [0, 255] -> [0.0,1.0]
    tt.Resize((resolution, resolution), antialias=True),
    tt.Normalize(mean=0.5, std=0.2), # range [0.0,1.0] -> [-2.5, 2.5]

])

img2mask = tt.Compose([
    tt.ToTensor(), # range [0, 255] -> [0.0,1.0]
    tt.Resize((resolution, resolution), antialias=True),
])

class DinoV2latentGen:
    def __init__(self, model_cfg):
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
        self.device = model_cfg['device']
        self.n_feature = self.model.embed_dim
        print("Device: ", self.device)
        
    def batch_run(self, X):
        if type(X) == list:
            X = torch.stack(X)
        return self.run(X)
    
    def single_run(self, x):
        X = torch.unsqueeze(x, 0)
        return self.run(X)
        
    def run(self, X):
        self.model.eval()
        self.model.to(self.device)
        with torch.no_grad():
            X = X.to(self.device)
            result = self.model.forward_features(X)
        
        return result['x_norm_patchtokens'].detach().cpu().numpy()
    
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
        batch_latent = []
        mask_roi_list = [img2mask(get_mask(it, select_roi)) for it in mask_list]
        frame_list = [img2tensor(it) for it in frame_list]

        patch_feature = self.model.batch_run(frame_list)

        for img, mask in zip(patch_feature, mask_roi_list):
            latent = img.reshape((patch_len, patch_len, self.n_feature))
            small_mask = mask.reshape(resolution//14, 14, resolution//14, 14).sum(axis=(1, 3))
            sum_mask = small_mask.sum()
            result = small_mask[:, :, np.newaxis] * latent
            latent_mask_ave = result.sum(axis=0).sum(axis=0) / sum_mask
            batch_latent.append(latent_mask_ave)

        return batch_latent

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