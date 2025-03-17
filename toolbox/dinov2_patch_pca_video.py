#!/usr/bin/env python
"""
Usage:
    python this_script.py input_video.mp4 input_mask.h5 --output output_video.mp4 --roi 1 --fit_ratio 0.1 --seed 42 --chunk_size 16 --model dinov2_vitb14_reg

Description:
    此腳本利用指定的 DINOv2 模型（支援以下模型：
        dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14,
        dinov2_vits14_reg, dinov2_vitb14_reg, dinov2_vitl14_reg, dinov2_vitg14_reg）
    從影片中提取 patch latent 特徵，並根據遮罩檔案使用 cuML PCA 進行降維與著色，
    最後將結果以處理後的影片輸出。
    
    本版本利用 chunk 批次處理加速 frame 處理，且 chunk 大小可由 --chunk_size 指定。
"""

import argparse
import os
import cupy as cp
from tqdm import tqdm

import torch
import torchvision.transforms as tt
import torch.nn.functional as F

from cuml.decomposition import PCA
from castle.utils.h5_io import H5IO
from castle.utils.video_io import ReadArray, WriteArray

RESOLUTION = 518
PATCH_LEN = 37


def mask_filter(m, resolution=RESOLUTION):
    """
    Resize mask、重新 reshape 並篩選出有效區域：
    1. 轉成 tensor 並調整大小到 (RESOLUTION, RESOLUTION)，並縮放至 255.
    2. 重新 reshape 為 (resolution//14, 14, resolution//14, 14) 並在 axis 1 與 3 上取和。
    3. 閾值過濾：大於 100 為有效區域。
    """
    transform = tt.Compose([
        tt.ToTensor(),
        tt.Resize((RESOLUTION, RESOLUTION), antialias=True),
    ])

    m_resized = transform(m)[0] * 255.0
    m_reshaped = m_resized.reshape(resolution // 14, 14, resolution // 14, 14).sum(axis=(1, 3))
    m_filtered = (m_reshaped > 100).cpu().numpy()  # 轉回 numpy 陣列
    m_filtered = cp.asarray(m_filtered)            # 再轉成 cupy 陣列以便 GPU 運算
    return m_filtered

class DINOv2:
    def __init__(self, model_name="dinov2_vitb14_reg"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 根據 model_name 載入對應的模型
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.model.eval()
        self.model.to(self.device)
        self.num_features = self.model.num_features
    
        self.transform = tt.Compose([
            tt.ToTensor(),
            tt.Resize((RESOLUTION, RESOLUTION)),
            tt.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_patch_latents(self, vr, selected_indices, chunk_size=16):
        patch_latents = []
        # 以 chunk 為單位處理所選 frame
        for i in tqdm(range(0, len(selected_indices), chunk_size), desc="Extracting patch latents in chunks"):
            chunk_indices = selected_indices[i:i+chunk_size]
            frames = [vr[j] for j in chunk_indices]
            batch_tensor = torch.stack([self.transform(frame) for frame in frames]).to(self.device)
            with torch.no_grad():
                features = self.model.forward_features(batch_tensor)['x_norm_patchtokens'].detach()
            # features shape: (batch_size, num_patches, N_DIM)
            features_np = features.cpu().numpy()
            for feat in features_np:
                latent = cp.asarray(feat.reshape(-1, self.num_features))
                patch_latents.append(latent)
        patch_latents = cp.stack(patch_latents, axis=0)
        return patch_latents

    def get_patch_latent(self, vr, index):
        frame = vr[index]
        frame_tensor = self.transform(frame).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model.forward_features(frame_tensor)['x_norm_patchtokens'].detach()
        patch_latent = cp.asarray(features.cpu().numpy().reshape(-1, self.num_features))
        return patch_latent

    def get_patch_latents_batch(self, frames):
        frames_tensor = torch.stack([self.transform(frame) for frame in frames]).to(self.device)
        with torch.no_grad():
            features = self.model.forward_features(frames_tensor)['x_norm_patchtokens']
        features_np = features.cpu().numpy()
        batch_size = features_np.shape[0]
        patch_latents = cp.asarray(features_np.reshape(batch_size, PATCH_LEN, PATCH_LEN, self.num_features))
        return patch_latents

def main(video_path, mask_path, output_path, roi_id, fit_ratio, seed, chunk_size, model_name):
    # 讀取影片與遮罩檔案
    print("Reading video and mask files...")
    vr = ReadArray(video_path)
    fps = vr.fps
    mask = H5IO(mask_path)

    print("Extracting patch latent features using DINOv2 model:", model_name)
    vit = DINOv2(model_name=model_name)
    num_features = vit.num_features
    n_frames = len(vr)
    print(f"Number of frames: {n_frames}")

    # 根據 fit_ratio 隨機選取部分 frame 用於 PCA fitting
    sample_size = max(1, int(n_frames * fit_ratio))
    cp.random.seed(seed)
    selected_indices = cp.random.choice(n_frames, size=sample_size, replace=False)
    selected_indices = cp.sort(selected_indices)
    selected_indices_cpu = cp.asnumpy(selected_indices)
    print(f"Selected frames for PCA fitting (n={len(selected_indices_cpu)}): {selected_indices_cpu[:10]}")
    
    # 取得選取 frame 的 patch latents（以 chunk 方式處理）
    patch_selected = vit.get_patch_latents(vr, selected_indices_cpu, chunk_size)
    focus_patch_latents = []
    for i, index in enumerate(selected_indices_cpu):
        p = patch_selected[i].reshape((PATCH_LEN, PATCH_LEN, num_features))
        m = mask[index]
        m_filtered = mask_filter(m)
        focus_patch_latents.append(p[m_filtered])
    if focus_patch_latents:
        focus_patch_latents = cp.concatenate(focus_patch_latents, axis=0)
    else:
        focus_patch_latents = cp.array([])
    
    print("Fitting PCA on selected patch features using cuML...")
    pca = PCA(n_components=num_features)
    pca.fit(focus_patch_latents)
    
    # 取得所有選取 frame 經 PCA 轉換後的全局最小與最大值
    all_transformed = []
    for i, index in enumerate(selected_indices_cpu):
        p = patch_selected[i].reshape((PATCH_LEN, PATCH_LEN, num_features))
        m = mask[index]
        m_filtered = mask_filter(m)
        if m_filtered.sum() > 0:
            features = p[m_filtered]
            transformed = pca.transform(features)
            all_transformed.append(transformed)
    if all_transformed:
        all_transformed = cp.concatenate(all_transformed, axis=0)
        mi, mx = all_transformed.min(), all_transformed.max()
    else:
        mi, mx = 0, 1
    print(f"PCA value range from selected frames: min={mi:.3f}, max={mx:.3f}")

    if output_path is None:
        output_path = os.path.splitext(video_path)[0] + "_patch_pca.mp4"
    crf = 15
    out = WriteArray(output_path, fps, crf)

    height, width = vr[0].shape[:2]
    transform_resize = tt.Compose([
        tt.ToTensor(),
        tt.Resize((height, width)),
    ])
    alpha = 0.3

    print("Processing frames in chunks...")
    for i in tqdm(range(0, n_frames, chunk_size), desc="Processing frames in chunks"):
        end = min(i + chunk_size, n_frames)
        indices = list(range(i, end))
        frames_chunk = [vr[j] for j in indices]
        patch_latents_batch = vit.get_patch_latents_batch(frames_chunk)
        for j, frame_index in enumerate(indices):
            # 讀取原始 frame 與 mask，轉成 cupy 陣列以進行 GPU 運算
            f0 = cp.asarray(vr[frame_index])
            m0 = cp.asarray(mask[frame_index])
            m0 = cp.where(m0 != roi_id, 0, m0)
            m0 = cp.where(m0 == roi_id, 1, m0)
            # mask_filter 預期輸入為 numpy 陣列
            m0_filtered = mask_filter(cp.asnumpy(m0))
            p = patch_latents_batch[j]
            if m0_filtered.sum() > 0:
                features = p[m0_filtered]
                transformed = pca.transform(features)
                transformed = cp.clip(transformed, mi, mx)
                normalized = (transformed - mi) / (mx - mi)
                p[m0_filtered] = normalized
            # 將非 mask 區域的 patch 設為 0
            p = cp.where(m0_filtered[..., None], p, 0)

            p_cupy = p[:, :, 0:3]
            p_torch = torch.utils.dlpack.from_dlpack(p_cupy.toDlpack())
            p_torch = p_torch.permute(2, 0, 1).unsqueeze(0)

            p_resized_torch = F.interpolate(p_torch.float(), size=(height, width), mode="bilinear", align_corners=False)
            p_resized_torch = p_resized_torch.squeeze(0).permute(1, 2, 0)
            p_resized_torch = (p_resized_torch * 255).clamp(0, 255).to(torch.uint8)
            p_resized = cp.from_dlpack(torch.utils.dlpack.to_dlpack(p_resized_torch))

            binary_mask = cp.sum(p_resized, axis=2) > 0

            foreground = f0 * (1 - alpha) + p_resized * alpha
            f0 = cp.where(binary_mask[..., None], foreground, f0)
            mix = cp.concatenate([f0, p_resized], axis=1)
            out.append(cp.asnumpy(mix).astype("uint8"))
    
    out.close()
    print("Processing complete, output saved to:", output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Extract patch latents using the DINOv2 model, apply cuML PCA coloring based on a mask, and output a processed video."
    )
    parser.add_argument("video_path", help="Path to the input video")
    parser.add_argument("mask_path", help="Path to the input mask file (H5 format)")
    parser.add_argument("--output", help="Path to the output video (default: derived from input video)", default=None)
    parser.add_argument("--roi", type=int, help="ROI ID in the mask (default: 1)", default=1)
    parser.add_argument("--fit_ratio", type=float, help="Fraction of frames to use for PCA fitting (default: 0.1)", default=0.1)
    parser.add_argument("--seed", type=int, help="Random seed for frame selection (default: 42)", default=42)
    parser.add_argument("--chunk_size", type=int, help="Number of frames to process per chunk (default: 64)", default=64)
    parser.add_argument("--model", type=str, help="模型名稱，可選列表: dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14, dinov2_vits14_reg, dinov2_vitb14_reg, dinov2_vitl14_reg, dinov2_vitg14_reg", default="dinov2_vitb14_reg")
    args = parser.parse_args()
    main(args.video_path, args.mask_path, args.output, args.roi, args.fit_ratio, args.seed, args.chunk_size, args.model)
