# 常見問題 (FAQ)

這裡彙整了使用 CASTLE 時最常遇到的問題和解答。如果您的問題沒有在這裡找到答案，請參考 [故障排除指南](troubleshooting.md) 或在 [討論區](https://github.com/castle-project/castle/discussions) 提問。

## 🚀 安裝與設定

??? question "CASTLE 支援哪些作業系統？"
    **支援的作業系統**：
    
    - **Windows**: Windows 10, 11 (完全支援)
    - **macOS**: 10.15 (Catalina) 以上，支援 Intel 和 Apple Silicon
    - **Linux**: Ubuntu 18.04+, CentOS 7+, 大多數主流發行版
    
    詳細需求請參考 [系統需求](../getting-started/requirements.md)。

??? question "安裝時出現權限錯誤怎麼辦？"
    **解決方法**：
    
    ```bash
    # 方法 1: 使用用戶安裝
    pip install --user castle-behavior
    
    # 方法 2: 使用虛擬環境 (推薦)
    python -m venv castle-env
    source castle-env/bin/activate  # Linux/Mac
    # 或 castle-env\Scripts\activate  # Windows
    pip install castle-behavior
    ```

??? question "為什麼安裝後 import castle 失敗？"
    **常見原因與解決方案**：
    
    1. **套件名稱錯誤**
       ```python
       # 錯誤
       import castle
       
       # 正確
       import castle_behavior as castle
       ```
    
    2. **Python 路徑問題**
       ```bash
       # 檢查安裝位置
       pip show castle-behavior
       
       # 確認 Python 版本
       python --version
       pip --version
       ```
    
    3. **虛擬環境未啟用**
       - 確保在正確的虛擬環境中安裝和使用

## 🎬 影片處理

??? question "支援哪些影片格式？"
    **完全支援**：
    - MP4 (推薦)
    - AVI
    - MOV
    - MKV
    
    **有限支援**：
    - WMV
    - FLV
    
    **不支援**：
    - 專業相機 RAW 格式
    - 過度壓縮的格式
    
    建議使用 MP4 格式以獲得最佳相容性。

??? question "影片太大無法分析怎麼辦？"
    **解決策略**：
    
    1. **降低解析度**
       ```python
       analyzer = castle.BehaviorAnalyzer(
           input_resolution=(720, 720)  # 降低至 720p
       )
       ```
    
    2. **分段分析**
       ```python
       # 分析前 5 分鐘
       results = analyzer.analyze(
           video_path, 
           start_time=0, 
           end_time=300
       )
       ```
    
    3. **使用 GPU 記憶體優化**
       ```python
       analyzer = castle.BehaviorAnalyzer(
           batch_size=8,  # 減少批次大小
           mixed_precision=True  # 啟用混合精度
       )
       ```

??? question "分析速度太慢如何改善？"
    **優化建議**：
    
    | 方法 | 速度提升 | 設定方式 |
    |------|----------|----------|
    | **使用 GPU** | 5-10x | `device='cuda'` |
    | **降低解析度** | 2-3x | `input_resolution=(640, 480)` |
    | **增加批次大小** | 1.5-2x | `batch_size=64` |
    | **使用 SSD** | 1.2-1.5x | 將影片存放在 SSD |
    | **混合精度** | 1.3-1.7x | `mixed_precision=True` |

## 🐭 動物行為分析

??? question "如何提高動物偵測的準確度？"
    **調整策略**：
    
    1. **優化影片品質**
       - 充足均勻的光線
       - 高對比度背景
       - 穩定的相機設定
    
    2. **調整偵測參數**
       ```python
       analyzer = castle.BehaviorAnalyzer(
           detection_confidence=0.7,  # 提高信心閾值
           nms_threshold=0.4,  # 調整非極大值抑制
           min_animal_size=50,  # 設定最小動物大小
           max_animal_size=500  # 設定最大動物大小
       )
       ```
    
    3. **使用適當的動物類型設定**
       ```python
       results = analyzer.analyze(
           video_path,
           animal_type='mouse',  # 明確指定動物類型
           arena_type='open_field'  # 指定實驗場地
       )
       ```

??? question "分析結果中出現誤判怎麼辦？"
    **常見誤判與解決方案**：
    
    | 誤判類型 | 原因 | 解決方法 |
    |----------|------|----------|
    | **假陽性** | 背景物體被識別為動物 | 增加 `detection_confidence` |
    | **假陰性** | 動物未被識別 | 降低 `detection_confidence` |
    | **ID 跳躍** | 動物 ID 頻繁變換 | 增加 `tracking_threshold` |
    | **軌跡斷裂** | 追蹤中斷 | 調整 `max_disappeared_frames` |

??? question "可以同時分析多隻動物嗎？"
    **是的！** CASTLE 支援多動物分析：
    
    ```python
    # 設定最大動物數量
    analyzer = castle.BehaviorAnalyzer(max_animals=3)
    
    # 分析時指定動物數量
    results = analyzer.analyze(
        video_path,
        animal_count=2,  # 預期動物數量
        enable_identity_tracking=True  # 啟用個體識別
    )
    
    # 存取個別動物的結果
    animal_1_results = results.get_animal_results(animal_id=0)
    animal_2_results = results.get_animal_results(animal_id=1)
    ```

## 📊 結果分析與解讀

??? question "如何解讀行為分析結果？"
    **基本指標解釋**：
    
    | 指標 | 意義 | 正常範圍 (小鼠) |
    |------|------|-----------------|
    | **總移動距離** | 活動水平 | 1000-3000 cm |
    | **平均速度** | 運動強度 | 3-8 cm/s |
    | **中央區域時間** | 焦慮程度 | 10-30% |
    | **靜止時間** | 休息/整理行為 | 15-35% |
    | **探索時間** | 主動性 | 60-80% |
    
    **解讀技巧**：
    - 與對照組比較
    - 考慮實驗條件
    - 結合多項指標判斷
    - 注意個體差異

??? question "統計分析應該使用哪些方法？"
    **推薦統計方法**：
    
    1. **描述性統計**
       - 平均值 ± 標準誤差
       - 中位數和四分位距
       - 信賴區間
    
    2. **假設檢驗**
       ```python
       from scipy import stats
       
       # t-test for comparing two groups
       t_stat, p_value = stats.ttest_ind(group1_data, group2_data)
       
       # ANOVA for multiple groups
       f_stat, p_value = stats.f_oneway(group1, group2, group3)
       
       # Non-parametric tests
       u_stat, p_value = stats.mannwhitneyu(group1, group2)
       ```
    
    3. **效應量計算**
       - Cohen's d
       - Eta-squared (η²)
       - Cliff's delta

??? question "如何製作發表品質的圖表？"
    **圖表最佳實務**：
    
    ```python
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # 設定發表風格
    plt.style.use('seaborn-whitegrid')
    sns.set_context("paper", font_scale=1.2)
    
    # 建立高解析度圖表
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    # 使用專業色彩
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    # 添加統計顯著性標記
    castle.plot.add_significance_bars(ax, p_values, groups)
    
    # 匯出高品質圖片
    plt.savefig('behavior_results.png', dpi=300, bbox_inches='tight')
    ```

## ⚙️ 進階功能

??? question "如何自訂分析管線？"
    **建立客製化管線**：
    
    ```python
    from castle.pipeline import CustomPipeline
    
    # 定義自訂步驟
    pipeline = CustomPipeline([
        'video_preprocessing',
        'animal_detection', 
        'tracking',
        'custom_behavior_classifier',  # 您的客製化分類器
        'statistical_analysis'
    ])
    
    # 設定自訂參數
    pipeline.set_parameters({
        'custom_behavior_classifier': {
            'model_path': 'my_custom_model.pth',
            'threshold': 0.8
        }
    })
    
    # 執行分析
    results = pipeline.run(video_path)
    ```

??? question "可以整合其他分析工具嗎？"
    **是的！** CASTLE 設計為可擴展的：
    
    ```python
    # 與 DeepLabCut 整合
    import deeplabcut as dlc
    from castle.integrations import DeepLabCutIntegration
    
    # 使用 DLC 姿態估計 + CASTLE 行為分析
    dlc_results = dlc.analyze_videos(config_path, [video_path])
    castle_results = castle.analyze_with_poses(dlc_results)
    
    # 與 SLEAP 整合
    from castle.integrations import SLEAPIntegration
    sleap_integration = SLEAPIntegration(model_path)
    results = sleap_integration.analyze(video_path)
    ```

??? question "如何進行批次處理？"
    **大規模批次分析**：
    
    ```python
    from castle.batch import BatchProcessor
    
    # 設定批次處理器
    processor = BatchProcessor(
        n_workers=4,  # 平行處理數
        gpu_ids=[0, 1],  # 使用多個 GPU
        output_dir='batch_results/'
    )
    
    # 添加多個影片
    video_list = ['video1.mp4', 'video2.mp4', 'video3.mp4']
    for video in video_list:
        processor.add_video(video, animal_type='mouse')
    
    # 執行批次分析
    results = processor.run()
    
    # 匯出統合報告
    processor.export_summary('batch_summary.xlsx')
    ```

## 🔧 故障排除

??? question "GPU 顯示可用但分析仍然很慢？"
    **檢查清單**：
    
    1. **確認 GPU 正在使用**
       ```python
       import torch
       print(torch.cuda.is_available())  # 應該返回 True
       print(torch.cuda.current_device())  # 顯示當前 GPU
       
       # 在分析期間檢查 GPU 使用率
       # 使用 nvidia-smi 監控
       ```
    
    2. **檢查 CUDA 版本相容性**
       ```bash
       # 檢查 CUDA 版本
       nvcc --version
       
       # 檢查 PyTorch CUDA 版本
       python -c "import torch; print(torch.version.cuda)"
       ```
    
    3. **記憶體不足問題**
       ```python
       # 減少批次大小
       analyzer = castle.BehaviorAnalyzer(batch_size=8)
       
       # 清理 GPU 記憶體
       torch.cuda.empty_cache()
       ```

??? question "分析中途停止或崩潰？"
    **常見原因與解決方案**：
    
    | 錯誤類型 | 可能原因 | 解決方法 |
    |----------|----------|----------|
    | **記憶體不足** | 影片太大或批次太大 | 減少 `batch_size` 或降低解析度 |
    | **磁碟空間不足** | 暫存檔案過大 | 清理暫存目錄或指定其他位置 |
    | **影片損壞** | 檔案不完整 | 重新錄製或使用影片修復工具 |
    | **驅動程式過舊** | GPU 驅動版本過低 | 更新 NVIDIA 驅動程式 |

??? question "結果與預期差異很大？"
    **驗證步驟**：
    
    1. **檢查輸入資料品質**
       - 影片解析度和品質
       - 光線條件是否一致
       - 背景是否乾淨
    
    2. **驗證參數設定**
       ```python
       # 檢查當前參數
       print(analyzer.get_parameters())
       
       # 使用預設參數重新分析
       analyzer.reset_parameters()
       results = analyzer.analyze(video_path)
       ```
    
    3. **手動驗證關鍵時間點**
       - 隨機選擇幾個時間點
       - 人工檢查動物行為標註
       - 比較自動分析結果

## 📞 獲得更多幫助

如果這裡的解答無法解決您的問題：

### 🔍 進一步資源
- [故障排除詳細指南](troubleshooting.md)
- [API 完整文件](../api/)
- [GitHub 討論區](https://github.com/castle-project/castle/discussions)

### 💬 聯繫支援
- **GitHub Issues**: 回報 Bug 或功能請求
- **Email**: support@castle-project.org
- **Discord**: 即時聊天支援 (邀請連結)

### 📚 學習資源
- [YouTube 教學頻道](https://youtube.com/@castle-tutorials)
- [線上研討會](../community/webinars.md)
- [使用案例分享](../community/case-studies.md)

---

!!! tip "提問技巧"
    提問時請提供：
    - CASTLE 版本 (`castle.__version__`)
    - 作業系統版本
    - Python 版本
    - 完整的錯誤訊息
    - 最小重現範例

!!! info "文件更新"
    這個 FAQ 定期更新。如果您發現缺少的問題或有改進建議，歡迎 [提交 PR](contributing.md)！