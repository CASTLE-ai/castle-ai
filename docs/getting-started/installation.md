# 安裝指南

本頁面將引導您完成 CASTLE 的安裝過程。請根據您的系統環境選擇合適的安裝方式。

## 系統需求

在開始安裝之前，請確保您的系統滿足以下需求：

### 硬體需求

| 組件 | 最低需求 | 建議配置 |
|------|----------|----------|
| **CPU** | Intel i5 / AMD Ryzen 5 | Intel i7 / AMD Ryzen 7 |
| **記憶體** | 8GB RAM | 16GB+ RAM |
| **硬碟** | 5GB 可用空間 | 20GB+ SSD |
| **GPU** | 無 (可選) | NVIDIA GTX 1060+ / RTX 系列 |

### 軟體需求

- **作業系統**: Windows 10+, macOS 10.15+, Ubuntu 18.04+
- **Python**: 3.8, 3.9, 3.10, 3.11
- **套件管理器**: pip 或 conda

## 安裝方法

=== "pip 安裝 (推薦)"

    ### 1. 建立虛擬環境
    ```bash
    # 使用 venv
    python -m venv castle-env
    source castle-env/bin/activate  # Linux/Mac
    # castle-env\Scripts\activate  # Windows
    
    # 或使用 conda
    conda create -n castle-env python=3.10
    conda activate castle-env
    ```

    ### 2. 安裝 CASTLE
    ```bash
    pip install castle-behavior
    ```

    ### 3. 驗證安裝
    ```python
    import castle
    print(castle.__version__)
    # 應該顯示版本號，如: 2.1.0
    ```

=== "開發版本安裝"

    如果您想要使用最新的開發功能：

    ```bash
    git clone https://github.com/castle-project/castle.git
    cd castle
    pip install -e .
    ```

=== "Docker 安裝"

    使用 Docker 容器化部署：

    ```bash
    # 拉取映像
    docker pull castle/castle:latest
    
    # 執行容器
    docker run -it --rm -p 8888:8888 castle/castle:latest
    ```

## GPU 支援設定

如果您的系統有 NVIDIA GPU，建議安裝 CUDA 支援以加速分析：

### CUDA 安裝

1. **安裝 NVIDIA 驅動程式**
   - 前往 [NVIDIA 官網](https://www.nvidia.com/drivers) 下載最新驅動

2. **安裝 CUDA Toolkit**
   ```bash
   # Ubuntu
   sudo apt install nvidia-cuda-toolkit
   
   # 或下載官方安裝包
   wget https://developer.download.nvidia.com/compute/cuda/12.0.0/local_installers/cuda_12.0.0_525.60.13_linux.run
   sudo sh cuda_12.0.0_525.60.13_linux.run
   ```

3. **安裝 GPU 版本的 PyTorch**
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

4. **驗證 GPU 支援**
   ```python
   import torch
   print(torch.cuda.is_available())  # 應該返回 True
   print(torch.cuda.get_device_name(0))  # 顯示 GPU 名稱
   ```

## 可選套件

根據您的需求安裝額外功能：

### 影片處理增強
```bash
pip install castle-behavior[video]
```

### GUI 介面
```bash
pip install castle-behavior[gui]
```

### 完整功能 (包含所有可選套件)
```bash
pip install castle-behavior[all]
```

## 常見安裝問題

### 問題 1: pip 版本過舊
```bash
# 解決方案：升級 pip
python -m pip install --upgrade pip
```

### 問題 2: 權限不足
```bash
# 解決方案：使用用戶安裝
pip install --user castle-behavior
```

### 問題 3: 依賴衝突
```bash
# 解決方案：使用新的虛擬環境
python -m venv fresh-env
source fresh-env/bin/activate
pip install castle-behavior
```

### 問題 4: macOS M1/M2 晶片問題
```bash
# 解決方案：使用 conda 並指定 conda-forge 頻道
conda install -c conda-forge castle-behavior
```

## 驗證安裝

完成安裝後，執行以下檢查：

### 1. 基本功能測試
```python
import castle

# 檢查版本
print(f"CASTLE version: {castle.__version__}")

# 檢查核心模組
from castle import BehaviorAnalyzer
analyzer = BehaviorAnalyzer()
print("BehaviorAnalyzer 初始化成功")
```

### 2. 下載測試資料
```python
# 下載範例影片 (約 100MB)
castle.download_examples()
print("範例資料下載完成")
```

### 3. 執行簡單分析
```python
# 測試分析功能
results = analyzer.analyze("examples/test_video.mp4")
print(f"分析完成，發現 {len(results.behaviors)} 種行為模式")
```

## 效能優化建議

### 1. 記憶體優化
```python
# 設定批次大小 (根據您的 RAM 調整)
analyzer = castle.BehaviorAnalyzer(batch_size=16)  # 預設: 32
```

### 2. CPU 優化
```python
# 設定處理器核心數
analyzer = castle.BehaviorAnalyzer(n_workers=4)  # 預設: 自動檢測
```

### 3. GPU 記憶體管理
```python
# 啟用混合精度運算 (節省 GPU 記憶體)
analyzer = castle.BehaviorAnalyzer(mixed_precision=True)
```

## 更新 CASTLE

定期更新以獲得最新功能和修正：

```bash
# 更新至最新版本
pip install --upgrade castle-behavior

# 檢查更新
pip list --outdated | grep castle
```

## 解除安裝

如果需要完全移除 CASTLE：

```bash
# 解除安裝 CASTLE
pip uninstall castle-behavior

# 清理快取 (可選)
pip cache purge
```

---

!!! success "安裝完成！"
    恭喜您成功安裝 CASTLE！接下來可以：
    
    - 嘗試 [快速範例](quick-example.md)
    - 探索 [GUI 介面](gui-introduction.md)  
    - 開始第一個 [教學課程](../tutorials/)

!!! question "還有問題？"
    - 查看 [故障排除](../community/troubleshooting.md)
    - 在 [討論區](https://github.com/castle-project/castle/discussions) 提問
    - 回報 [Bug](https://github.com/castle-project/castle/issues)