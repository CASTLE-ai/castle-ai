#!/usr/bin/env python3
"""
測試 DINOv3 模型檢查點下載功能
"""

import os
import sys
import shutil
from pathlib import Path

# 添加項目根目錄到路徑
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from castle.utils.visual_latent_extract import download_dinov3_ckpt, generate_dinov3

def test_dinov3_download():
    """測試 DINOv3 模型下載功能"""
    print("=" * 60)
    print("測試 DINOv3 模型檢查點下載功能")
    print("=" * 60)
    
    # 測試下載 dinov3_vitb16
    print("\n[測試 1] 下載 dinov3_vitb16 模型...")
    try:
        ckpt_path_b16 = download_dinov3_ckpt('dinov3_vitb16')
        print(f"✓ 成功下載到: {ckpt_path_b16}")
        
        # 檢查文件是否存在
        if os.path.exists(ckpt_path_b16):
            file_size = os.path.getsize(ckpt_path_b16) / (1024 * 1024)  # MB
            print(f"✓ 文件大小: {file_size:.2f} MB")
        else:
            print(f"✗ 錯誤: 文件不存在於 {ckpt_path_b16}")
            return False
    except Exception as e:
        print(f"✗ 錯誤: {e}")
        return False
    
    # 測試下載 dinov3_vitl16
    print("\n[測試 2] 下載 dinov3_vitl16 模型...")
    try:
        ckpt_path_l16 = download_dinov3_ckpt('dinov3_vitl16')
        print(f"✓ 成功下載到: {ckpt_path_l16}")
        
        # 檢查文件是否存在
        if os.path.exists(ckpt_path_l16):
            file_size = os.path.getsize(ckpt_path_l16) / (1024 * 1024)  # MB
            print(f"✓ 文件大小: {file_size:.2f} MB")
        else:
            print(f"✗ 錯誤: 文件不存在於 {ckpt_path_l16}")
            return False
    except Exception as e:
        print(f"✗ 錯誤: {e}")
        return False
    
    # 測試錯誤的模型類型
    print("\n[測試 3] 測試錯誤的模型類型...")
    try:
        download_dinov3_ckpt('invalid_model')
        print("✗ 錯誤: 應該拋出異常但沒有")
        return False
    except ValueError as e:
        print(f"✓ 正確捕獲錯誤: {e}")
    except Exception as e:
        print(f"✗ 意外的錯誤類型: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("所有測試通過！✓")
    print("=" * 60)
    return True

def test_auto_download():
    """測試自動下載功能（當模型不存在時）"""
    print("\n" + "=" * 60)
    print("測試自動下載功能（當使用模型但尚未下載時）")
    print("=" * 60)
    
    # 備份現有的 checkpoint（如果存在）
    ckpt_dir = project_root / "ckpt"
    backup_dir = project_root / "ckpt_backup"
    test_ckpt_b16 = ckpt_dir / "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
    test_ckpt_l16 = ckpt_dir / "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
    
    # 創建備份目錄
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    backup_dir.mkdir(exist_ok=True)
    
    # 備份文件（如果存在）
    if test_ckpt_b16.exists():
        shutil.move(str(test_ckpt_b16), str(backup_dir / test_ckpt_b16.name))
        print(f"已備份 {test_ckpt_b16.name}")
    
    if test_ckpt_l16.exists():
        shutil.move(str(test_ckpt_l16), str(backup_dir / test_ckpt_l16.name))
        print(f"已備份 {test_ckpt_l16.name}")
    
    try:
        # 測試自動下載 dinov3_vitb16
        print("\n[測試 4] 測試自動下載 dinov3_vitb16（當文件不存在時）...")
        if test_ckpt_b16.exists():
            test_ckpt_b16.unlink()
            print("已刪除現有文件以測試自動下載")
        
        # 嘗試生成模型，應該會自動下載
        print("嘗試生成 dinov3_vitb16 模型（應該會自動下載）...")
        try:
            observer = generate_dinov3(model_type='dinov3_vitb16')
            print("✓ 成功自動下載並生成模型")
            del observer
        except Exception as e:
            print(f"✗ 錯誤: {e}")
            return False
        
        # 驗證文件已下載
        if test_ckpt_b16.exists():
            file_size = test_ckpt_b16.stat().st_size / (1024 * 1024)  # MB
            print(f"✓ 文件已下載，大小: {file_size:.2f} MB")
        else:
            print("✗ 錯誤: 文件未下載")
            return False
        
    finally:
        # 恢復備份的文件
        if (backup_dir / test_ckpt_b16.name).exists():
            if test_ckpt_b16.exists():
                test_ckpt_b16.unlink()
            shutil.move(str(backup_dir / test_ckpt_b16.name), str(test_ckpt_b16))
            print(f"已恢復 {test_ckpt_b16.name}")
        
        if (backup_dir / test_ckpt_l16.name).exists():
            if test_ckpt_l16.exists():
                test_ckpt_l16.unlink()
            shutil.move(str(backup_dir / test_ckpt_l16.name), str(test_ckpt_l16))
            print(f"已恢復 {test_ckpt_l16.name}")
        
        # 清理備份目錄
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
    
    print("\n" + "=" * 60)
    print("自動下載測試通過！✓")
    print("=" * 60)
    return True

if __name__ == "__main__":
    # 測試基本下載功能
    success1 = test_dinov3_download()
    
    # 測試自動下載功能（可選，因為需要實際下載）
    print("\n是否測試自動下載功能？這將需要實際下載模型文件。")
    print("（如果不想測試，可以按 Ctrl+C 取消）")
    try:
        success2 = test_auto_download()
        success = success1 and success2
    except KeyboardInterrupt:
        print("\n跳過自動下載測試")
        success = success1
    
    sys.exit(0 if success else 1)

