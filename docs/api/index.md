# API 文件

CASTLE 提供完整而強大的 Python API，讓您能夠程式化地控制動物行為分析的每個環節。本章節將詳細介紹所有可用的類別、函數和參數。

## 🏗️ API 架構概覽

CASTLE API 採用模組化設計，主要分為以下幾個核心模組：

```
castle/
├── core/              # 核心分析引擎
│   ├── analyzer       # 主要分析器類別
│   ├── detector       # 動物偵測模組
│   ├── tracker        # 動物追蹤模組
│   └── classifier     # 行為分類模組
├── data/              # 資料處理模組
│   ├── loader         # 影片載入器
│   ├── preprocessor   # 資料前處理
│   └── postprocessor  # 結果後處理
├── models/            # 深度學習模型
│   ├── detection      # 偵測模型
│   ├── tracking       # 追蹤模型
│   └── behavior       # 行為識別模型
├── visualization/     # 視覺化模組
│   ├── plots          # 基本圖表
│   ├── interactive    # 互動式圖表
│   └── export         # 匯出功能
├── utils/             # 工具函數
│   ├── video          # 影片處理
│   ├── math           # 數學運算
│   └── io             # 輸入輸出
└── gui/               # 圖形使用者介面
    ├── main_window    # 主視窗
    ├── widgets        # UI 元件
    └── dialogs        # 對話框
```

## 🚀 快速開始

### 基本使用模式

```python
import castle

# 1. 初始化分析器
analyzer = castle.BehaviorAnalyzer()

# 2. 載入影片並分析
results = analyzer.analyze('video.mp4')

# 3. 檢視結果
print(results.summary())

# 4. 視覺化
castle.plot.trajectory(results)
```

### 進階配置

```python
# 自訂分析器配置
analyzer = castle.BehaviorAnalyzer(
    device='cuda',                    # 使用 GPU
    model_config='high_precision',    # 高精度模式
    batch_size=32,                    # 批次大小
    detection_threshold=0.7,          # 偵測閾值
    tracking_algorithm='deepsort'     # 追蹤演算法
)

# 詳細分析參數
results = analyzer.analyze(
    video_path='experiment.mp4',
    animal_type='mouse',              # 動物類型
    arena_type='open_field',          # 實驗場地
    start_time=10,                    # 開始時間(秒)
    end_time=300,                     # 結束時間(秒)
    roi_mask='arena_mask.png',        # ROI 遮罩
    background_model='adaptive',      # 背景模型
    output_dir='results/'             # 輸出目錄
)
```

## 📚 核心 API 模組

### [Core Functions](core.md)

核心分析功能，包含主要的分析器類別和方法：

- **`BehaviorAnalyzer`** - 主要分析器類別
- **`VideoProcessor`** - 影片處理器
- **`ResultsContainer`** - 結果容器類別
- **`AnalysisConfig`** - 分析配置管理

**主要功能**：
- 動物偵測與追蹤
- 行為模式識別  
- 軌跡分析
- 統計計算

### [GUI Components](gui.md)

圖形使用者介面相關的類別和元件：

- **`MainWindow`** - 主視窗類別
- **`VideoPlayer`** - 影片播放器元件
- **`AnalysisPanel`** - 分析控制面板
- **`ResultsViewer`** - 結果檢視器

**主要功能**：
- 影片載入與播放
- 即時分析監控
- 結果視覺化
- 參數調整介面

### [Utils](utils.md)

工具函數和輔助功能：

- **`video_utils`** - 影片處理工具
- **`math_utils`** - 數學運算工具
- **`io_utils`** - 檔案輸入輸出
- **`validation`** - 資料驗證

**主要功能**：
- 影片格式轉換
- 座標系轉換
- 檔案格式支援
- 資料完整性檢查

## 🎨 視覺化 API

### 基本圖表

```python
import castle.plot as cplot

# 軌跡圖
cplot.trajectory(results)

# 熱力圖  
cplot.heatmap(results)

# 行為時間軸
cplot.behavior_timeline(results)

# 統計圖表
cplot.behavior_statistics(results)
```

### 進階視覺化

```python
# 自訂圖表樣式
cplot.trajectory(
    results, 
    color_by='speed',          # 按速度著色
    show_arena=True,           # 顯示實驗場地
    alpha=0.7,                 # 透明度
    linewidth=2,               # 線條寬度
    figsize=(10, 8)            # 圖片大小
)

# 互動式圖表
cplot.interactive_timeline(
    results,
    show_video=True,           # 同步顯示影片
    export_format='html'       # 匯出為 HTML
)
```

## 🔧 配置與參數

### 全域配置

```python
import castle.config as cfg

# 設定全域配置
cfg.set_global_device('cuda')
cfg.set_cache_directory('/tmp/castle_cache')
cfg.set_log_level('INFO')

# 檢視目前配置
print(cfg.get_current_config())
```

### 模型配置

```python
# 使用預設配置
analyzer = castle.BehaviorAnalyzer(config='default')

# 使用高精度配置
analyzer = castle.BehaviorAnalyzer(config='high_precision')

# 自訂配置
custom_config = {
    'detection': {
        'model_name': 'yolov8n',
        'confidence_threshold': 0.5,
        'nms_threshold': 0.4
    },
    'tracking': {
        'algorithm': 'deepsort',
        'max_disappeared': 30,
        'max_distance': 50
    },
    'behavior': {
        'clustering_method': 'hierarchical',
        'n_clusters': 'auto',
        'time_window': 5.0
    }
}

analyzer = castle.BehaviorAnalyzer(config=custom_config)
```

## 📊 資料結構

### 分析結果

```python
# 結果物件結構
class AnalysisResults:
    def __init__(self):
        self.metadata = {}          # 分析元資料
        self.trajectories = []      # 軌跡資料
        self.behaviors = []         # 行為標籤
        self.statistics = {}        # 統計指標
        self.annotations = []       # 標註資訊
        
    # 主要方法
    def summary(self):              # 結果摘要
    def get_behavior(self, name):   # 取得特定行為
    def save_to_csv(self, path):    # 匯出 CSV
    def save_to_json(self, path):   # 匯出 JSON
```

### 行為資料

```python
# 行為物件結構
class BehaviorSegment:
    def __init__(self):
        self.name = ""              # 行為名稱
        self.start_time = 0.0       # 開始時間
        self.end_time = 0.0         # 結束時間
        self.confidence = 0.0       # 信心程度
        self.properties = {}        # 額外屬性
        
    @property
    def duration(self):             # 持續時間
        return self.end_time - self.start_time
        
    @property
    def frames(self):               # 對應幀數
        return self._frame_range
```

## 🔌 外掛系統

### 建立客製化外掛

```python
from castle.plugins import PluginBase

class CustomBehaviorPlugin(PluginBase):
    def __init__(self):
        super().__init__()
        self.name = "custom_behavior"
        self.version = "1.0.0"
    
    def process(self, trajectory_data):
        # 實作您的客製化行為分析邏輯
        behavior_labels = self.analyze_custom_behavior(trajectory_data)
        return behavior_labels
    
    def analyze_custom_behavior(self, data):
        # 您的分析算法
        pass

# 註冊外掛
castle.plugins.register(CustomBehaviorPlugin())
```

### 使用外掛

```python
# 啟用特定外掛
analyzer = castle.BehaviorAnalyzer(
    plugins=['custom_behavior', 'social_interaction']
)

# 檢視可用外掛
print(castle.plugins.list_available())
```

## 🌐 整合其他工具

### DeepLabCut 整合

```python
from castle.integrations import DeepLabCutIntegration

# 初始化整合
dlc_integration = DeepLabCutIntegration(
    config_path='/path/to/dlc/config.yaml'
)

# 使用 DLC 姿態 + CASTLE 行為分析
results = dlc_integration.analyze_with_poses('video.mp4')
```

### SLEAP 整合

```python
from castle.integrations import SLEAPIntegration

sleap_integration = SLEAPIntegration(
    model_path='/path/to/sleap/model'
)

results = sleap_integration.analyze('video.mp4')
```

## 📝 API 參考索引

### 快速查找

| 功能 | 類別/函數 | 說明 |
|------|-----------|------|
| **基本分析** | `BehaviorAnalyzer` | 主要分析器 |
| **影片載入** | `VideoLoader` | 影片檔案載入 |
| **結果檢視** | `ResultsContainer` | 分析結果容器 |
| **軌跡繪製** | `plot.trajectory()` | 繪製動物軌跡 |
| **行為統計** | `stats.behavior_summary()` | 行為統計摘要 |
| **資料匯出** | `export.to_csv()` | 匯出 CSV 格式 |
| **批次處理** | `BatchProcessor` | 大量影片處理 |
| **GUI 啟動** | `launch_gui()` | 啟動圖形介面 |

### 詳細文件

<div class="api-reference-grid" markdown="1">

[![Core Functions](../assets/api-icons/core.svg)](core.md)
**[Core Functions](core.md)**
核心分析功能和主要類別

[![GUI Components](../assets/api-icons/gui.svg)](gui.md)
**[GUI Components](gui.md)**  
圖形使用者介面元件

[![Utils](../assets/api-icons/utils.svg)](utils.md)
**[Utils](utils.md)**
工具函數和輔助功能

</div>

## 🔍 程式碼範例

### 完整分析範例

```python
import castle
import matplotlib.pyplot as plt

# 設定分析器
analyzer = castle.BehaviorAnalyzer(
    device='cuda' if castle.cuda_available() else 'cpu',
    detection_model='yolov8s',
    tracking_algorithm='deepsort',
    behavior_model='hierarchical_clustering'
)

# 執行分析
results = analyzer.analyze(
    video_path='mouse_experiment.mp4',
    animal_type='mouse',
    arena_type='open_field',
    save_intermediate=True,
    output_dir='analysis_results/'
)

# 產生報告
report = castle.generate_report(results)
report.save('experiment_report.pdf')

# 視覺化關鍵結果
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

castle.plot.trajectory(results, ax=axes[0,0])
castle.plot.heatmap(results, ax=axes[0,1])
castle.plot.speed_profile(results, ax=axes[1,0])
castle.plot.behavior_pie(results, ax=axes[1,1])

plt.tight_layout()
plt.savefig('analysis_summary.png', dpi=300)
plt.show()
```

---

## 📖 更多資源

- **[教學課程](../tutorials/)** - 逐步學習 API 使用
- **[範例程式碼](../examples/)** - 實際應用範例
- **[GitHub Repository](https://github.com/castle-project/castle)** - 原始碼和問題回報
- **[討論區](https://github.com/castle-project/castle/discussions)** - 社群支援

!!! tip "API 使用建議"
    - 先從基本的 `BehaviorAnalyzer` 開始
    - 閱讀每個模組的詳細文件
    - 參考範例程式碼學習最佳實務
    - 善用 IDE 的自動補全功能

!!! info "版本相容性"
    目前 API 文件對應 CASTLE v2.1.0。不同版本間的 API 變更請參考 [更新日誌](../community/changelog.md)。

**準備開始使用 API？** [查看核心函數文件 →](core.md){ .md-button .md-button--primary }