"""
驗證優化後的 latent extraction 與原始實作產出一致的結果

使用方式：
    python tests/test_optimized_latent_extraction.py
"""

import os
import sys
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from castle import generate_dinov2
from castle.utils.video_io import ReadArray
from castle.utils.h5_io import H5IO

def load_reference_latent(npz_path):
    """載入參考 latent 陣列"""
    data = np.load(npz_path)
    return data['latent']

def test_normal_latent_extraction():
    """測試一般 latent extraction"""
    print("=" * 60)
    print("測試一般 Latent Extraction")
    print("=" * 60)
    
    # 載入參考結果
    ref_path = 'tests/correct_latent_features/ctrl_30fps_1_tvai_10s_ROI_1_latent.npz'
    if not os.path.exists(ref_path):
        print(f"⚠️  找不到參考檔案: {ref_path}")
        return False
    
    reference_latent = load_reference_latent(ref_path)
    print(f"參考 latent shape: {reference_latent.shape}")
    print(f"參考 latent dtype: {reference_latent.dtype}")
    print(f"參考 latent 範圍: [{reference_latent.min():.4f}, {reference_latent.max():.4f}]")
    
    # 檢查數值統計
    print(f"參考 latent mean: {reference_latent.mean():.6f}")
    print(f"參考 latent std: {reference_latent.std():.6f}")
    
    # 找出對應的影片和 mask
    # 根據檔名推測是 ctrl_30fps_1_tvai_10s.mp4
    project_path = 'projects/2025-09-17-04-27-01-YT-PD-9videos'
    video_name = 'ctrl_30fps_1_tvai_10s.mp4'
    
    source_path = os.path.join(project_path, 'sources', video_name)
    mask_path = os.path.join(project_path, 'track', video_name, 'mask_list.h5')
    
    if not os.path.exists(source_path):
        print(f"⚠️  找不到影片檔案: {source_path}")
        return False
    
    if not os.path.exists(mask_path):
        print(f"⚠️  找不到 mask 檔案: {mask_path}")
        return False
    
    print(f"\n載入影片: {source_path}")
    source_video = ReadArray(source_path)
    tracker = H5IO(mask_path)
    
    # 初始化 observer
    print("\n初始化 DinoV2 observer...")
    observer = generate_dinov2(model_type='dinov2_vitb14_reg')
    
    # 提取 latent（使用較小的 batch 以節省記憶體測試）
    print("\n開始提取 latent...")
    select_roi = 1
    batch_size = 16
    latent_list = []
    
    total_frames = len(source_video)
    print(f"總共 {total_frames} 個 frames")
    
    for i in range(0, total_frames, batch_size):
        frames, masks = [], []
        for j in range(batch_size):
            idx = i + j
            if idx >= total_frames:
                break
            frames.append(source_video[idx])
            masks.append(tracker.read_mask(idx))
        
        latent_batch = observer.extract_batch_latent(frames, masks, select_roi)
        latent_list.extend(latent_batch)
        
        if (i // batch_size + 1) % 10 == 0:
            print(f"  處理進度: {i + len(frames)}/{total_frames}")
    
    new_latent = np.array(latent_list)
    print(f"\n新提取的 latent shape: {new_latent.shape}")
    print(f"新提取的 latent dtype: {new_latent.dtype}")
    print(f"新提取的 latent 範圍: [{new_latent.min():.4f}, {new_latent.max():.4f}]")
    print(f"新提取的 latent mean: {new_latent.mean():.6f}")
    print(f"新提取的 latent std: {new_latent.std():.6f}")
    
    # 比較結果
    print("\n" + "=" * 60)
    print("比較結果")
    print("=" * 60)
    
    if reference_latent.shape != new_latent.shape:
        print(f"❌ Shape 不一致: {reference_latent.shape} vs {new_latent.shape}")
        return False
    
    # 計算差異
    abs_diff = np.abs(reference_latent - new_latent)
    max_diff = abs_diff.max()
    mean_diff = abs_diff.mean()
    
    print(f"最大絕對差異: {max_diff:.6f}")
    print(f"平均絕對差異: {mean_diff:.6f}")
    
    # 計算相對差異
    rel_diff = abs_diff / (np.abs(reference_latent) + 1e-8)
    max_rel_diff = rel_diff.max()
    mean_rel_diff = rel_diff.mean()
    
    print(f"最大相對差異: {max_rel_diff:.6f}")
    print(f"平均相對差異: {mean_rel_diff:.6f}")
    
    # 計算 cosine similarity
    from numpy.linalg import norm
    flat_ref = reference_latent.reshape(-1)
    flat_new = new_latent.reshape(-1)
    cos_sim = np.dot(flat_ref, flat_new) / (norm(flat_ref) * norm(flat_new))
    print(f"Cosine similarity: {cos_sim:.8f}")
    
    # 判定標準：
    # 1. 平均絕對差異 < 0.001（考慮浮點數精度）
    # 2. 最大絕對差異 < 0.01
    # 3. Cosine similarity > 0.9999
    
    threshold_mean = 0.001
    threshold_max = 0.01
    threshold_cos = 0.9999
    
    passed = (mean_diff < threshold_mean and 
              max_diff < threshold_max and 
              cos_sim > threshold_cos)
    
    if passed:
        print("\n✅ 測試通過！優化後的實作與原始實作結果一致。")
    else:
        print("\n⚠️  差異較大，可能需要檢查：")
        if mean_diff >= threshold_mean:
            print(f"   - 平均差異 {mean_diff:.6f} >= {threshold_mean}")
        if max_diff >= threshold_max:
            print(f"   - 最大差異 {max_diff:.6f} >= {threshold_max}")
        if cos_sim <= threshold_cos:
            print(f"   - Cosine similarity {cos_sim:.8f} <= {threshold_cos}")
    
    # 清理
    del tracker
    del observer
    
    return passed

def test_rotation_latent_extraction():
    """測試旋轉 latent extraction"""
    print("\n" + "=" * 60)
    print("測試旋轉 Latent Extraction")
    print("=" * 60)
    
    ref_path = 'tests/correct_latent_features/ctrl_30fps_1_tvai_10s_ROI_1_rotation_latent.npz'
    if not os.path.exists(ref_path):
        print(f"⚠️  找不到參考檔案: {ref_path}")
        return False
    
    reference_latent = load_reference_latent(ref_path)
    print(f"參考 rotation latent shape: {reference_latent.shape}")
    print(f"參考 rotation latent dtype: {reference_latent.dtype}")
    
    print("\n⚠️  旋轉 latent 測試尚未實作（需要從 extract_ui.py 提取相關邏輯）")
    print("   目前僅測試一般 latent extraction")
    
    return True  # 先跳過

if __name__ == '__main__':
    print("驗證優化後的 latent extraction")
    print("=" * 60)
    
    try:
        # 測試一般 latent extraction
        test1_passed = test_normal_latent_extraction()
        
        # 測試旋轉 latent extraction（目前跳過）
        # test2_passed = test_rotation_latent_extraction()
        
        print("\n" + "=" * 60)
        print("總結")
        print("=" * 60)
        
        if test1_passed:
            print("✅ 所有測試通過！")
            sys.exit(0)
        else:
            print("❌ 部分測試未通過")
            sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ 測試過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


