# Castle-AI 架構文件

**版本**: 1.0  
**最後更新**: 2025-12-16

---

## 設計原則

### 1. 關注點分離 (Separation of Concerns)
- **Core 層**: 純 Python 邏輯，不依賴任何 UI 框架
- **UI 層**: 僅負責 Gradio 介面與事件綁定
- **Utils 層**: 共用工具函式

### 2. 依賴倒置 (Dependency Inversion)
- Core 定義 Protocol (ProgressCallback, NotificationCallback)
- UI 層注入具體實作 (gr.Progress, gr.Info)

### 3. 配置集中化 (Centralized Configuration)
- 所有路徑、模型 ID、常數集中於 `castle/core/config.py`
- 消除 Magic Numbers

---

## 模組依賴圖

\`\`\`mermaid
graph TD
    UI[castle/ui/] --> Core[castle/core/]
    Core --> Utils[castle/utils/]
    Core --> Config[config.py]
    
    subgraph "Core Layer"
        Config
        Extractor[extractor.py]
        Cluster[cluster.py]
        Models[models.py]
        Data[data.py]
    end
    
    subgraph "UI Layer"
        ExtractUI[extract_ui.py]
        ClusterUI[cluster_page_ui.py]
    end
    
    ExtractUI --> Extractor
    ExtractUI --> Data
    ClusterUI --> Cluster
    Extractor --> Models
\`\`\`

---

## 核心介面

### ProgressCallback Protocol
\`\`\`python
class ProgressCallback(Protocol):
    def __call__(self, progress: float, desc: str = None) -> None: ...
\`\`\`
**用途**: 解耦進度報告，允許 Core 層在不依賴 Gradio 的情況下報告進度。

### NotificationCallback Protocol
\`\`\`python
class NotificationCallback(Protocol):
    def __call__(self, message: str, level: str = "info") -> None: ...
\`\`\`
**用途**: 通用訊息通知，支援 info/warning/error 等級。

### VisualEncoder ABC
\`\`\`python
class VisualEncoder(ABC):
    @abstractmethod
    def load_model(self): ...
    
    @abstractmethod
    def extract_features(self, batch_tensor: torch.Tensor) -> torch.Tensor: ...
    
    def extract_tensor_batch(self, frames, masks, roi_id) -> np.ndarray: ...
\`\`\`
**用途**: 統一 DINOv2/v3 介面，支援 Duck Typing。

---

## 資料流程

### 1. 特徵提取流程
\`\`\`
Video File → VideoDataset → DataLoader → VisualEncoder → Latent .npz
                ↓
            Preprocess (crop/rotate)
                ↓
            ROI Masking
\`\`\`

### 2. 行為聚類流程
\`\`\`
Multiple Latent Files → LatentAggregator → UMAP → Clustering → Subtitles
                                ↓
                        Temporal Binning
\`\`\`

---

## 目錄結構

\`\`\`
castle/
├── core/                    # 核心邏輯層 (UI-agnostic)
│   ├── config.py           # 集中配置
│   ├── logging_config.py   # 統一日誌
│   ├── interfaces.py       # Protocol 定義
│   ├── data.py             # 資料結構
│   ├── extractor.py        # 特徵提取
│   ├── cluster.py          # 行為聚類
│   └── models.py           # 模型介面
├── ui/                      # UI 層 (Gradio)
│   ├── extract_ui.py
│   └── cluster_page_ui.py
└── utils/                   # 共用工具
    ├── video_io.py
    ├── video_manager.py
    └── ...
\`\`\`

---

## 關鍵設計決策

### 為何使用 Protocol 而非繼承？
- **彈性**: 允許任何實作，不強制繼承
- **測試性**: 容易 mock
- **解耦**: Core 不需要知道 UI 的存在

### 為何 VisualEncoder 使用 ABC？
- **強制介面**: 確保所有編碼器實作必要方法
- **型別安全**: 支援靜態型別檢查
- **文檔化**: 明確定義預期行為

### 為何不將 EmbeddingScatterPlot 完全移至 Core？
- **繪圖邏輯**: 與 Matplotlib 緊密耦合
- **UI 互動**: 需要 Gradio 事件處理
- **妥協方案**: 僅將計算邏輯 (KDTree) 移至 Core

---

## 擴展指南

### 新增支援的模型
1. 在 `config.py` 的 `SUPPORTED_MODELS` 新增條目
2. 在 `models.py` 實作對應的 Encoder 子類別
3. 更新 `get_visual_encoder` 工廠函式

### 新增 UI 頁面
1. 在 `castle/ui/` 建立新檔案
2. 僅處理 Gradio 元件與事件
3. 呼叫 Core 層函式執行邏輯
4. 使用 Protocol 注入回調

---

**維護者**: Hsu Lab  
**聯絡**: [專案 Repository]
