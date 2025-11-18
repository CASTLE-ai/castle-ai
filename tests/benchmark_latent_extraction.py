"""
效能基準測試：比較優化前後的速度

使用合成資料來測試不同 batch size 下的吞吐量
"""

import os
import sys
import time
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from castle import generate_dinov2
from castle.utils.video_align import get_mask


def benchmark_extraction(batch_sizes=[4, 8, 16, 32, 64], num_frames=300):
    """基準測試不同 batch size 的效能"""
    
    print("=" * 70)
    print("效能基準測試：Latent Extraction")
    print("=" * 70)
    
    # 檢查 CUDA
    if not torch.cuda.is_available():
        print("⚠️  CUDA 不可用，使用 CPU 進行測試（效能提升會較不明顯）")
        device = 'cpu'
    else:
        device = 'cuda'
        print(f"✅ 使用 GPU: {torch.cuda.get_device_name(0)}")
        print(f"   顯存總量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 生成合成測試資料
    print(f"\n生成測試資料: {num_frames} frames...")
    np.random.seed(456)
    H, W = 640, 480
    
    frames = []
    masks = []
    for i in range(num_frames):
        frame = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)
        mask = np.zeros((H, W), dtype=np.uint8)
        
        # 生成隨機 ROI
        cy, cx = H // 2 + np.random.randint(-50, 50), W // 2 + np.random.randint(-50, 50)
        for y in range(H):
            for x in range(W):
                if (y - cy)**2 + (x - cx)**2 < (min(H, W) // 4)**2:
                    mask[y, x] = 1
        
        frames.append(frame)
        masks.append(mask)
    
    print(f"✅ 測試資料準備完成")
    
    # 初始化 observer
    print(f"\n初始化 DinoV2 observer...")
    observer = generate_dinov2(model_type='dinov2_vitb14_reg', device=device)
    
    # 暖身（第一次執行會較慢，因為要編譯 kernels）
    print(f"\n暖身中...")
    _ = observer.extract_batch_latent(frames[:8], masks[:8], select_roi=1)
    if device == 'cuda':
        torch.cuda.synchronize()
    
    print("\n" + "=" * 70)
    print(f"{'Batch Size':<12} {'Time (s)':<12} {'FPS':<12} {'GPU Util':<12} {'Note'}")
    print("=" * 70)
    
    results = []
    
    for batch_size in batch_sizes:
        try:
            if device == 'cuda':
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
            
            latent_list = []
            
            start_time = time.time()
            
            for i in range(0, num_frames, batch_size):
                batch_frames = frames[i:i+batch_size]
                batch_masks = masks[i:i+batch_size]
                
                latent_batch = observer.extract_batch_latent(batch_frames, batch_masks, select_roi=1)
                latent_list.extend(latent_batch)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            
            elapsed = time.time() - start_time
            fps = num_frames / elapsed
            
            if device == 'cuda':
                peak_mem = torch.cuda.max_memory_allocated() / 1024**3
                note = f"{peak_mem:.1f} GB"
            else:
                note = "CPU mode"
            
            print(f"{batch_size:<12} {elapsed:<12.2f} {fps:<12.1f} {note:<12}")
            
            results.append({
                'batch_size': batch_size,
                'time': elapsed,
                'fps': fps
            })
            
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print(f"{batch_size:<12} {'OOM':<12} {'N/A':<12} {'記憶體不足'}")
                if device == 'cuda':
                    torch.cuda.empty_cache()
                break
            else:
                raise
    
    print("=" * 70)
    
    # 分析結果
    if len(results) >= 2:
        print("\n效能分析:")
        baseline = results[0]
        best = max(results, key=lambda x: x['fps'])
        
        speedup = best['fps'] / baseline['fps']
        print(f"  基準 (batch={baseline['batch_size']}): {baseline['fps']:.1f} FPS")
        print(f"  最佳 (batch={best['batch_size']}): {best['fps']:.1f} FPS")
        print(f"  加速比: {speedup:.2f}x")
        
        # 建議
        print("\n建議:")
        if device == 'cuda':
            if speedup > 1.5:
                print(f"  ✅ 建議使用 batch size = {best['batch_size']} 以獲得最佳效能")
            else:
                print(f"  ⚠️  效能提升有限，可能受限於：")
                print(f"     - I/O 瓶頸（硬碟讀取速度）")
                print(f"     - CPU 前處理（preprocess.transform）")
                print(f"     - 小型 GPU 或舊架構")
        else:
            print(f"  ℹ️  CPU 模式下效能提升較不明顯，建議使用 GPU")
    
    # 清理
    del observer
    if device == 'cuda':
        torch.cuda.empty_cache()
    
    print("\n" + "=" * 70)
    print("優化總結")
    print("=" * 70)
    print("已實施的優化:")
    print("  ✅ GPU 向量化前處理（resize, normalize）")
    print("  ✅ GPU 上完成 ROI mask 下採樣與加權平均")
    print("  ✅ 啟用 AMP (Automatic Mixed Precision)")
    print("  ✅ 啟用 TF32 (Tensor Float 32)")
    print("  ✅ 使用 pin_memory 與 non_blocking 傳輸")
    print("  ✅ 模型只載入一次，避免重複 to(device)")
    print("  ✅ 提高預設 batch size (8 -> 32)")
    print("\n預期整體效能提升: 1.5-3.0x")
    print("（實際提升取決於 GPU 型號、batch size 與 I/O 速度）")


if __name__ == '__main__':
    try:
        # 根據 GPU 記憶體調整測試的 batch sizes
        if torch.cuda.is_available():
            total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if total_mem_gb >= 24:
                batch_sizes = [4, 8, 16, 32, 64, 128]
            elif total_mem_gb >= 12:
                batch_sizes = [4, 8, 16, 32, 64]
            elif total_mem_gb >= 8:
                batch_sizes = [4, 8, 16, 32]
            else:
                batch_sizes = [4, 8, 16]
        else:
            batch_sizes = [4, 8, 16]
        
        benchmark_extraction(batch_sizes=batch_sizes, num_frames=300)
        
    except KeyboardInterrupt:
        print("\n\n測試被中斷")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 測試過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


