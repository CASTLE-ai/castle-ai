# CASTLE 優化提案 v2.0

> **提案人**: Kurisu (AI Architect)  
> **日期**: 2026-02-21  
> **決策者**: IsonaEi  
> **Branch**: `dev`  
> **現狀**: 110 non-vendored files, ~20,500 LOC, 410 tests, 0 lint errors  

---

## 目錄

1. [設計原則](#1-設計原則)
2. [現狀分析](#2-現狀分析)
3. [Phase 0: KIT/SKE 前處理整合](#3-phase-0-kitske-前處理整合)
4. [Phase 1: UX 大改善 + Docker 部署](#4-phase-1-ux-大改善--docker-部署)
5. [Phase 2: 效能優化](#5-phase-2-效能優化)
6. [Phase 3: 程式碼簡化](#6-phase-3-程式碼簡化)
7. [Phase 4: 功能強化](#7-phase-4-功能強化)
8. [跨 Phase 任務：文檔更新](#8-跨-phase-任務文檔更新)
9. [Sub-agent 管理策略](#9-sub-agent-管理策略)
10. [排程與人力](#10-排程與人力)
11. [風險評估](#11-風險評估)
12. [決策紀錄](#12-決策紀錄)

---

## 1. 設計原則

### 1.1 使用者友善 —「國中生和 80 歲 PI 都會用」

| 原則 | 實踐方式 |
|------|----------|
| **Zero-configuration** | 每一步都有合理預設值，不需要懂參數 |
| **Progressive disclosure** | 基本用法 3 個按鈕搞定，進階選項隱藏在 accordion |
| **即時回饋** | 每個操作有 progress bar、預覽、清楚的錯誤訊息 |
| **引導式流程** | 步驟之間自動提示下一步，禁用未就緒的按鈕 |
| **冷啟動零障礙** | 安裝→啟動→完成分析，全程不需要讀文件或問人 |

### 1.2 效能優化 —「合理發揮硬體效能」

| 原則 | 實踐方式 |
|------|----------|
| **GPU memory 精管** | 精準管理 VRAM，不佔用不需要的顯存 |
| **Lazy loading** | 需要時才載入模型，用完釋放 |
| **Pipeline parallelism** | I/O 和 compute 重疊執行 |
| **Smart cache** | 重複計算結果快取，content-hash 避免浪費 |

### 1.3 Linus Torvalds 程式哲學

| 原則 | 實踐方式 |
|------|----------|
| **Simple > Clever** | 程式碼要蠢到看一眼就懂 |
| **Good taste** | 正確的抽象層次，消除特殊情況而非處理特殊情況 |
| **No premature abstraction** | 不要為了「可能需要」而加框架 |
| **Data structures > Algorithms** | 先設計好資料結構，程式碼自然簡單 |
| **Working code ships** | 每個 commit 都要能跑、能測 |

### 1.4 CASTLE 設計目的

| 核心理念 | 說明 |
|----------|------|
| **Training-free** | 不需要標註資料、不需要訓練模型 |
| **Species-agnostic** | 任何動物、任何場景 |
| **Pixel-level information** | 利用 DINOv2 視覺特徵，不僅限於關鍵點 |
| **Hierarchical behavioral discovery** | 從粗到細自動發現行為模式 |

---

## 2. 現狀分析

### 2.1 架構概覽

```
castle/
├── core/           # 核心演算法 (extractor, cluster, ethogram, metrics, models...)
├── service/        # Service Layer (project, tracking, extraction, clustering, annotation...)
├── ui/             # Gradio 前端 (7 tabs: Project→Export)
├── desktop/        # PyQt6 前端 (7 tabs: 對齊 Gradio)
├── cli/            # CLI 介面 (castle command)
├── visualization/  # 繪圖工具
├── utils/          # 工具函數 (video_io, h5_io, video_align)
├── aot/            # [vendored] DeAOT tracking
└── sam/            # [vendored] SAM segmentation
```

### 2.2 現有前處理管線

```
目前的 Preprocess class:
  center_roi_switch → 將 frame 中心對齊到 ROI centroid
  rotate_roi_tail_switch → 依據 tail ROI 旋轉（固定角度）
  remove_background_switch → 背景置黑
  center_roi_crop_width/height → 固定裁切大小 (e.g. 300×300)
```

**缺失**: 沒有低通濾波、沒有動態裁切、沒有穩定化虛擬攝影機。

### 2.3 KIT/SKE 管線（Raiso 開發，待整合）

```
SAM+DeAOT → mask → centroid x(t), orientation θ(t)
       ↓
Zero-phase Butterworth lowpass (fc=0.25 Hz, order=2)
       ↓
x_c(t), θ_c(t)  — smooth camera trajectory
       ↓
Dynamic crop: max(300, 2 × (dist + 75))  [px]
       ↓
warpAffine: translate to x_c, rotate by θ_c
       ↓
Resize → 518 × 518 → DINOv2
```

**關鍵參數**:

| 參數 | 值 | 說明 |
|------|-----|------|
| Filter type | Zero-phase Butterworth (filtfilt) | 無延遲 (IsonaEi 決策) |
| Filter order | 2 | 最小保證平滑二階導數 |
| Cutoff frequency | 0.25 Hz | 4 秒週期，對齊行為 bout 時間尺度 |
| Sampling rate | 30 fps | 影片幀率 |
| Min crop size | 300 px | 最大 zoom |
| Margin | 75 px | 動物體寬的 119% |
| Output size | 518 × 518 | DINOv2 ViT-B/14 patch 對齊 (37×14) |

**來源**:
- `/mnt/AI-Assistant/260210-lowpass-justification/` — 完整論文等級的理論基礎
- `/home/raiso/playground/260109-dynamic-lowpass-crop/generate_dynamic_crop_video.py` — 參考實作
- `/home/raiso/playground/251228-angular-velocity-lowpass-video-generate/generate_filtered_data.py` — 濾波函數

---

## 3. Phase 0: KIT/SKE 前處理整合 ✅ DONE

**優先級**: ★★★ 最高  
**目標**: 將 Raiso 的 Stabilized Virtual Camera 前處理整合進 CASTLE，成為標準前處理選項

### P0-1: Core — StabilizedCamera Module

**新增 `castle/core/stabilized_camera.py`**

```
Agent-0A「Signal Processing Engineer」
├── 人設: 熟悉 scipy.signal, Butterworth filter, zero-phase (filtfilt)
├── 限制: 只碰 castle/core/stabilized_camera.py (新檔案)
│
├── P0-1a: StabilizedCamera class
│   ├── __init__(positions, angles, fps, fc=0.25, order=2,
│   │           margin=75, min_crop=300, output_size=518)
│   ├── _apply_zero_phase_lowpass(data) → filtered data
│   │   └── 使用 scipy.signal.filtfilt (非 lfilter)
│   ├── compute_trajectory() → self.pos_filtered, self.angle_filtered
│   └── get_crop_size(frame_idx) → int (dynamic crop size)
│
├── P0-1b: Frame generation
│   ├── generate_frame(frame, frame_idx) → stabilized cropped frame
│   │   ├── 取 lowpass position/angle
│   │   ├── 計算 dist = ‖raw - filtered‖
│   │   ├── crop_size = max(min_crop, 2 × (dist + margin))
│   │   ├── warpAffine (translate + rotate)
│   │   └── resize → output_size × output_size
│   └── get_diagnostics() → dict (crop stats, HP residual RMS, etc.)
│
└── P0-1c: Utility functions
    ├── extract_centroids_from_masks(mask_h5_path, roi_id) → positions array
    ├── extract_orientations_from_masks(mask_h5_path, roi_ids) → angles array
    └── preview_stabilization(video_path, ..., duration=10) → preview video path
```

### P0-2: Integration — 接入 Extraction Pipeline

```
Agent-0B「Integration Engineer」
├── 人設: 熟悉 CASTLE architecture, service layer, Gradio/PyQt
├── 限制: 修改 castle/core/data.py, castle/core/extractor.py,
│         castle/service/extraction_service.py
│         castle/ui/extract_ui.py, castle/desktop/widgets/extract_panel.py
│         castle/cli/extract_cmd.py
│
├── P0-2a: Preprocess 擴充
│   ├── 新增 stabilized_camera_switch: bool = False
│   ├── 新增 stabilized_camera_fc: float = 0.25
│   ├── 新增 stabilized_camera_order: int = 2
│   ├── 新增 stabilized_camera_margin: int = 75
│   ├── 新增 stabilized_camera_min_crop: int = 300
│   └── 向下相容：原有參數不變，新功能預設關閉
│
├── P0-2b: Extractor 修改
│   ├── extract_roi_latent_from_video() 偵測 stabilized_camera_switch
│   ├── 啟用時：先掃描全影片取 centroid/angle，初始化 StabilizedCamera
│   ├── Dataset 的 transform 改為呼叫 StabilizedCamera.generate_frame()
│   └── Latent 檔名加 tag: "stab_fc025" 等
│
├── P0-2c: Gradio UI (Extract tab)
│   ├── 「🎥 Stabilized Virtual Camera」Accordion (預設展開)
│   ├── 開關 switch
│   ├── fc slider (0.05–2.0, default 0.25, step 0.05)
│   ├── margin slider (25–200, default 75)
│   ├── min_crop slider (100–600, default 300)
│   ├── 「👁 Preview」按鈕 → 產生 10 秒預覽影片
│   └── 診斷資訊區 (crop 分佈, HP residual)
│
├── P0-2d: PyQt UI (Extract panel)
│   └── 同步 Gradio 的所有控制項到 PyQt
│
└── P0-2e: CLI
    └── castle extract --stabilized-camera --fc 0.25 --margin 75
```

### P0-3: Tests & Diagnostics

```
Agent-0C「Test & Diagnostics Engineer」
├── 人設: TDD-driven, 關注邊界條件和科學正確性
├── 限制: 只碰 tests/, 不碰 production code
│
├── P0-3a: 單元測試 (castle/core/stabilized_camera.py)
│   ├── test_zero_phase_filter: 驗證 filtfilt 無延遲
│   ├── test_static_trajectory: 靜止動物 → crop 不變
│   ├── test_fast_movement: 快速移動 → crop 擴大
│   ├── test_circular_motion: 圓周運動 → 穩定追蹤
│   ├── test_boundary_conditions: 邊緣附近不 crash
│   ├── test_fc_sweep: 不同 fc 的 crop 分佈正確性
│   └── test_frame_generation: output shape = (518, 518, 3)
│
├── P0-3b: 整合測試
│   ├── test_stabilized_extraction_pipeline: video → stabilize → latent
│   └── test_backward_compat: 舊參數仍然正常
│
└── P0-3c: 診斷工具
    ├── 預覽影片: side-by-side (原始 vs 穩定化)
    └── 診斷圖表: crop 分佈、HP residual、speed-crop correlation
```

### P0-QC: Quality Control

```
Agent-QC0「QC Reviewer」
├── 觸發: P0-1, P0-2, P0-3 全部完成後
├── 檢查項:
│   ├── pytest tests/ -x -q → 全過
│   ├── ruff check castle/ → 0 errors
│   ├── Import smoke test → 全模組正常
│   ├── Gradio / PyQt 功能一致性
│   ├── zero-phase filter 科學正確性抽驗
│   └── 文檔是否更新
└── 不通過 → 回報 → 修復 → 再 QC
```

### P0-Docs: 文檔更新

```
Agent-Docs0
├── README: 新增 Stabilized Camera 功能說明
├── docs/tutorials/: 新增前處理教學
├── docs/technical/architecture.md: 更新 pipeline 圖
└── docs/reference/api.md: StabilizedCamera API
```

---

## 4. Phase 1: UX 大改善 + Docker 部署 ✅ DONE

**優先級**: ★★ 高  
**目標**: 冷啟動零障礙，國中生到 80 歲 PI 不需要問人不需要看文件

### P1-1: ~~Wizard Mode — Gradio~~ ❌ Cancelled 2026-05-16
> **取消理由**：違反 human-in-the-loop 哲學。Step 3「一鍵全自動 Track → Extract → Cluster」跳過 cluster 階段必要的使用者互動（看 scatter、調 eps、看代表幀、賦予標籤）。已於 2026-05-16 整批移除（`castle/ui/wizard_ui.py`）。設計史保留於下：

```
Agent-1A「UX Architect — Gradio」
├── 人設: 以「80歲教授第一次用」為設計標準
├── 限制: castle/ui/wizard_ui.py (新檔), castle/ui/main_ui.py,
│         castle/service/auto_config.py (新檔)
│
├── P1-1a: 歡迎畫面 + 引導
│   ├── 偵測：啟動時無專案 → 自動顯示 Wizard tab
│   ├── 歡迎訊息 + 「3 步完成分析」動畫
│   └── 「載入範例資料」按鈕（內建 demo video）
│
├── P1-1b: 3 步驟精靈
│   ├── Step 1:「拖入影片」(drag & drop file upload)
│   │   └── 支援 mp4/avi/mov，自動偵測 fps 和解析度
│   ├── Step 2:「在影片中點擊動物」(SAM interactive prompt)
│   │   └── 點一下 → 自動分割 + 追蹤
│   ├── Step 3:「按下開始」(一鍵全自動)
│   │   └── Track → Extract (auto-config) → Cluster (auto) → 完成
│   └── 完成畫面：行為時間軸 + 叢集預覽 + 引導到 Annotator
│
├── P1-1c: Auto Config Service
│   ├── 自動推薦 preprocess 參數:
│   │   ├── 影片解析度 → crop size, margin
│   │   ├── fps → fc (lowpass cutoff)
│   │   ├── 動物大小 (from SAM mask) → min_crop
│   │   └── 可用 VRAM → batch_size
│   └── auto_config.py: recommend_config(video_info, mask_info, gpu_info) → dict
│
└── P1-1d: Pipeline Dashboard
    ├── 每一步的狀態: ✅ 完成 / 🔄 進行中 / ⏳ 等待 / ❌ 失敗
    ├── 預估剩餘時間
    └── 可隨時切到對應 tab 查看詳情
```

### P1-2: ~~Wizard Mode — PyQt~~ ❌ Cancelled 2026-05-16
> 同 P1-1 取消理由。已於 2026-05-16 整批移除（`castle/desktop/widgets/wizard_panel.py`）。設計史保留於下：

```
Agent-1A-PyQt「UX Architect — PyQt」
├── 人設: PyQt6 專家，鏡射 Gradio Wizard 的所有功能
├── 限制: castle/desktop/widgets/wizard_panel.py (新檔),
│         castle/desktop/main_window.py
│
├── P1-2a: WizardPanel 實作
│   ├── QStackedWidget 實現 3 步驟切換
│   ├── Step 1: QDragDrop 影片上傳
│   ├── Step 2: 影片播放 + 滑鼠點擊 SAM prompt
│   ├── Step 3: 一鍵啟動 + QProgressBar
│   └── 完成畫面
│
└── P1-2b: Pipeline Dashboard (PyQt 版)
    └── QTreeWidget 或 QTableWidget 顯示各步狀態
```

### P1-3: 錯誤訊息 + Tooltip + 引導

```
Agent-1B「Accessibility Engineer」
├── 人設: 專注人性化錯誤訊息，讓不懂技術的人也看得懂
├── 限制: 所有 castle/ui/*.py 和 castle/desktop/widgets/*.py
│
├── P1-3a: 錯誤訊息改善 (Gradio)
│   ├── 每個 gr.Error/gr.Warning → "發生了什麼 + 怎麼修" 格式
│   ├── 範例: "找不到影片檔案" → "找不到影片。請確認已在步驟 1 上傳影片，
│   │        或檢查檔案路徑是否正確。"
│   └── 技術名詞替換: "UMAP failed" → "行為分群過程遇到問題"
│
├── P1-3b: 錯誤訊息改善 (PyQt)
│   └── 同步 Gradio 的所有改善到 PyQt (QMessageBox)
│
├── P1-3c: 前置條件檢查 (Gradio + PyQt)
│   ├── Tab 切換時自動驗證上一步是否完成
│   ├── 未完成 → 顯示「請先完成 [步驟名稱]」+ 跳轉按鈕
│   └── 按鈕 disabled + tooltip 說明原因
│
├── P1-3d: Tooltip (Gradio)
│   ├── 每個參數旁加 ℹ️ 說明:
│   │   ├── 用途（做什麼的）
│   │   ├── 建議值（大多數情況用多少）
│   │   └── 影響範圍（改了會怎樣）
│   └── Gradio: 使用 gr.Info 或 elem_id + tooltip
│
└── P1-3e: Tooltip (PyQt)
    └── QWidget.setToolTip() 同步所有提示文字
```

### P1-4: Docker 一鍵部署

```
Agent-1C「DevOps Engineer」
├── 人設: Docker + NVIDIA Container Toolkit 專家
├── 限制: Dockerfile, docker-compose.yml, scripts/docker-entrypoint.sh
│
├── P1-4a: Dockerfile
│   ├── Multi-stage build:
│   │   ├── Stage 1: nvidia/cuda:12.x-runtime → Python 3.10
│   │   ├── Stage 2: pip install -r requirements.txt
│   │   └── Stage 3: copy castle code + entrypoint
│   ├── 自動下載 checkpoints (SAM, DeAOT, DINOv2)
│   ├── 支援 GPU mode 和 CPU-only mode
│   └── Image size 最小化 (清 pip cache, 合併 layers)
│
├── P1-4b: docker-compose.yml
│   ├── GPU passthrough (deploy.resources.reservations.devices)
│   ├── Volume mount: projects/ 資料夾
│   ├── Port: 7860 (Gradio)
│   └── 環境變數: CASTLE_DEVICE=auto, CASTLE_DATA=/data
│
├── P1-4c: 一行指令
│   ├── `docker run --gpus all -p 7860:7860 -v $(pwd)/projects:/data castle-ai/castle`
│   ├── CPU-only: `docker run -p 7860:7860 -v $(pwd)/projects:/data castle-ai/castle`
│   └── README 中的 quick start section
│
└── P1-4d: CI/CD
    └── GitHub Actions: build + push to Docker Hub on tag
```

### P1-QC + P1-Docs

```
Agent-QC1 → tests + lint + consistency + UX audit (使用 ui-audit skill)
Agent-Docs1 → README Docker section, wizard tutorial, UX 指南
```

---

## 5. Phase 2: 效能優化 ✅ DONE

**優先級**: ★★ 高  
**目標**: 降低 VRAM usage, 提升 throughput, 避免重複計算

### P2-1: GPU Memory Management

```
Agent-2A「GPU Systems Engineer」
├── 人設: 熟悉 CUDA, PyTorch memory management, model lifecycle
├── 限制: castle/core/models.py, castle/core/environment.py,
│         castle/core/extractor.py
│
├── P2-1a: Model Lifecycle Manager
│   ├── ModelRegistry singleton: 統一管理 SAM/DeAOT/DINOv2
│   ├── load_model(name) → model (lazy, 需要時才載)
│   ├── unload_model(name) → 主動釋放 CUDA cache
│   └── context manager: with ModelRegistry.use('dinov2') as model:
│
├── P2-1b: Auto Batch Size
│   ├── 偵測可用 VRAM (torch.cuda.mem_get_info)
│   ├── 根據模型大小 + frame size 計算最大 batch
│   └── 自動降級: OOM → halve batch → retry
│
└── P2-1c: CUDA Cache Eviction
    ├── 每個 pipeline stage 完成後: torch.cuda.empty_cache()
    ├── GC 觸發: gc.collect() before model loading
    └── 顯存監控 log: 每 100 frames 記錄 VRAM 使用
```

### P2-2: Pipeline Parallelism

```
Agent-2B「Pipeline Engineer」
├── 人設: Producer-Consumer 模式、concurrent.futures 專家
├── 限制: castle/core/extractor.py, castle/service/extraction_service.py
│
├── P2-2a: 三級 Pipeline
│   ├── Stage 1 (I/O thread): VideoReader decode → frame queue
│   ├── Stage 2 (CPU thread): Preprocess / StabilizedCamera → tensor queue
│   ├── Stage 3 (GPU): DINOv2 inference → latent queue
│   └── 使用 queue.Queue + threading (不用 multiprocessing，避免 CUDA fork)
│
├── P2-2b: Content-Hash Cache
│   ├── cache_key = hash(video_path + preprocess_config + model_name)
│   ├── 如果 latent 已存在且 cache_key 匹配 → skip
│   └── 存在 latent 目錄: .cache_manifest.json
│
└── P2-2c: 增量更新
    ├── 專案新增影片時只處理新的 → 不重跑整個專案
    └── 刪除影片時清理對應 latent + cluster
```

### P2-QC + P2-Docs

```
Agent-QC2 → tests + lint + benchmark (before/after speed, VRAM)
Agent-Docs2 → 效能調校文檔, GPU 需求說明
```

---

## 6. Phase 3: 程式碼簡化 ✅ DONE

**優先級**: ★ 中  
**目標**: 更少的 code、更少的 bug surface、更好的可讀性

### P3-1: Data Structure Refactor

```
Agent-3A「Data Architect」
├── 人設: 資料建模專家，Torvalds 哲學信徒
├── 限制: castle/core/project.py, castle/core/data.py,
│         castle/service/project_service.py
│
├── P3-1a: ProjectData Dataclass
│   ├── 取代散落的 storage_path + project_name 拼接
│   ├── @dataclass ProjectData:
│   │   ├── root: Path
│   │   ├── sources_dir: Path (auto-computed)
│   │   ├── track_dir: Path
│   │   ├── latent_dir: Path
│   │   ├── cluster_dir: Path
│   │   ├── config: ProjectConfig
│   │   └── videos: list[VideoInfo]
│   └── ProjectData.from_path(storage_path, project_name) → ProjectData
│
└── P3-1b: ClusterData 統一
    ├── 整合 cluster_.npz / time_series_*.csv / id.csv
    ├── @dataclass ClusterData:
    │   ├── labels: np.ndarray (flat leaf assignments)
    │   ├── hierarchy: dict (tree structure)
    │   ├── names: dict[int, str]
    │   └── colors: dict[int, tuple]
    └── ClusterData.load(cluster_dir, session_id=None)
```

### P3-2: 消除特殊情況

```
Agent-3B「Simplification Engineer」
├── 人設: "Special cases aren't special enough to break the rules"
├── 限制: castle/core/cluster.py,
│         castle/utils/video_io.py
│   (P0-A' 2026-05-16: castle/core/auto_cluster.py 已移除)
│
├── P3-2a: Device Factory
│   ├── 現狀: 每個函數都有 if cpu/mps elif cuda else 分支
│   ├── 改為: DeviceFactory.get_umap(device), .get_dbscan(device)
│   └── 一處定義，全部引用
│
├── P3-2b: VideoReader 簡化
│   └── 統一 av backend，移除 cv2 fallback 的複雜分支
│
└── P3-2c: UI Handler 瘦身 (Gradio + PyQt)
    ├── 所有業務邏輯 → service layer
    ├── Gradio handler: widget → service call → update widget
    └── 目標: 每個 handler < 20 行
```

### P3-QC + P3-Docs

```
Agent-QC3 → tests + lint + architecture consistency
Agent-Docs3 → 更新 architecture.md, api.md
```

---

## 7. Phase 4: 功能強化 ✅ DONE

**優先級**: 依功能決定  
**決策**: P4-2 (Real-time streaming) 暫緩

### P4-1: 多動物支援 (Multi-Subject) ✅ DONE

```
castle/core/multi_subject.py        — SubjectTrack, MultiSubjectProject
castle/analysis/social_features.py  — pairwise distance, orientation, approach score, events
castle/analysis/group_ethogram.py   — build_group_ethogram, plot_group_ethogram
```

### P4-3: Batch Processing ✅ DONE

```
castle/core/batch.py       — BatchConfig (from_yaml), BatchRunner (run, generate_summary)
castle/cli/batch_cmd.py    — castle batch run / status / report
```

### P4-4: Report Generation ✅ DONE

```
castle/analysis/report.py  — ReportGenerator (self-contained HTML with inline plots)
```

### P4-QC + P4-Docs ✅ DONE

```
docs/technical/architecture.md  — Phase 4 modules + API reference sections added
docs/reference/api.md           — Full Phase 4 API documentation added
README.md                       — Phase 4 changelog + CLI commands updated
docs/getting-started/quickstart.md — Batch processing + multi-subject sections added
docs/OPTIMIZATION_PROPOSAL_v2.md   — All phases marked ✅ DONE
```

---

## 8. 跨 Phase 任務：文檔更新

**每個 Phase 結束後，必須有一個 Docs Agent 執行：**

| Phase | 文檔更新範圍 |
|-------|-------------|
| P0 | StabilizedCamera API, 前處理教學, architecture 圖更新 |
| P1 | Docker quick start, Wizard 教學, UX 指南, 冷啟動體驗文檔 |
| P2 | 效能調校指南, GPU 需求, batch_size 建議 |
| P3 | Architecture.md 大改, API reference 更新 |
| P4 | 新功能教學 (multi-subject, batch, report) |

**文檔 QC 標準：**
- 所有 tab 名稱和 code 一致
- 所有 CLI 指令實際可執行
- 所有 API 描述和程式碼一致
- 沒有引用已移除/改名的功能

---

## 9. Sub-agent 管理策略

### 9.1 Agent Prompt 結構

每個 sub-agent 的 prompt 遵循以下結構：

```
## Task: [任務名稱]

### Role
你是 [角色名稱]，專精於 [專長領域]。

### Environment
Working dir: /mnt/AI-Assistant/ei-castle-dev/
Git: GIT_DIR=... GIT_WORK_TREE=...
Python: source activate.sh && PYTHONPATH=. python

### Objective
[明確目標，預期產出]

### File Ownership (嚴格遵守)
✅ 可以修改: [列出檔案]
❌ 不可修改: [列出檔案] (其他 agent 負責)

### Reference (先讀再做)
[列出需要先理解的現有程式碼]

### Deliverables
1. [具體產出 1]
2. [具體產出 2]

### Quality Gates
- pytest tests/ -x -q → 全過
- ruff check → 0 errors  
- git commit + push

### Constraints
- 使用 shutil.copyfile (not copy/copy2) — CIFS
- [其他限制]
```

### 9.2 衝突避免

| 策略 | 說明 |
|------|------|
| **File Ownership** | 每個 agent 有嚴格的檔案清單，不可跨界 |
| **Interface-first** | 需要跨模組合作時，先定好 interface 再各自實作 |
| **Sequential commits** | git push 前先 pull --rebase |
| **No shared state files** | 避免兩個 agent 同時修改同一個 import |

### 9.3 QC 循環

```
實作 Agent 完成
       ↓
QC Agent 執行檢查
       ↓
  ┌─ PASS → 進入下一 Phase
  └─ FAIL → 回報問題清單
              ↓
        修復 Agent 處理
              ↓
        QC Agent 再次檢查
              ↓
           (重複直到 PASS)
```

### 9.4 失敗處理

| 狀況 | 處理方式 |
|------|----------|
| Agent 卡死 (>10 min 無回應) | 殺掉，拆更小的任務重派 |
| Tests 失敗 | 回報具體 failure，派修復 agent |
| Git 衝突 | 手動 resolve，或 rebase |
| API 超載 (model unavailable) | 切換 model (sonnet fallback) |

---

## 10. 排程與人力

### 10.1 時間線

```
Phase 0 (KIT/SKE)     ████████░░░░░░░░░░░░░░░░  Week 1
Phase 1 (UX+Docker)    ░░░░░░░░████████████░░░░  Week 2-3
Phase 2 (效能)          ░░░░░░░░████████████░░░░  Week 2-3 (平行)
Phase 3 (簡化)          ░░░░░░░░░░░░░░░░████████  Week 3-4
Phase 4 (功能)          ░░░░░░░░░░░░░░░░░░░░████  Week 4+
```

### 10.2 每 Phase 人力配置

| Phase | Implementation | QC | Docs | Total |
|-------|:-:|:-:|:-:|:-:|
| P0 | 3 agents (0A, 0B, 0C) | 1 | 1 | 5 |
| P1 | 3 agents (1A, 1A-PyQt, 1B) + 1 (1C Docker) | 1 | 1 | 6 |
| P2 | 2 agents (2A, 2B) | 1 | 1 | 4 |
| P3 | 2 agents (3A, 3B) | 1 | 1 | 4 |
| P4 | 3 agents (4A, 4B, 4C) | 1 | 1 | 5 |

**注意**: Agent 不是同時跑的人，而是依序或小批平行的 sub-agent session。每次最多跑 2-3 個 implementation agent 以避免 git 衝突。

---

## 11. 風險評估

| 風險 | 可能性 | 影響 | 緩解策略 |
|------|:------:|:----:|----------|
| KIT 整合改動 extractor 核心 | 中 | 高 | 新增 StabilizedCamera 作為平行路徑，不修改現有邏輯 |
| 多 agent 修改衝突 | 中 | 中 | 嚴格 file ownership + sequential push |
| Sub-agent 執行卡死 | 高 | 低 | 拆小任務 (每個 < 5 min)，卡死就殺掉重派 |
| CIFS shutil.copy 問題 | 低 | 低 | 全面使用 shutil.copyfile (已落實) |
| Docker build 環境差異 | 中 | 中 | CI/CD 測試 + multi-arch build |
| Wizard 自動推薦參數不準 | 中 | 中 | 保守預設 + 可手動覆蓋 + 「進階設定」accordion |
| PyQt 和 Gradio 功能 drift | 高 | 中 | 每個 Phase QC 包含一致性檢查 |

---

## 12. 決策紀錄

| 日期 | 決策 | 決策者 |
|------|------|--------|
| 2026-02-21 | Phase 排序: P0→P1→P2→P3→P4 | IsonaEi |
| 2026-02-21 | P4-2 (Real-time streaming) 暫緩 | IsonaEi |
| 2026-02-21 | KIT 使用 zero-phase filter (filtfilt) | IsonaEi |
| 2026-02-21 | 安裝 ClawHub skills: `gradio` + `ui-audit` | IsonaEi |
| 2026-02-21 | 每個 Phase 結束必須更新文檔 | IsonaEi |
| 2026-02-21 | 前端改善必須同時覆蓋 Gradio + PyQt | IsonaEi |
| 2026-02-21 | 新增 Docker 一鍵部署 | IsonaEi |
| 2026-02-21 | 冷啟動體驗列為 P1 核心目標 | IsonaEi |

---

*提案版本: v2.0 | 最後更新: 2026-02-21*  
*完成狀態: Phase 0 ✅ Phase 1 ✅ Phase 2 ✅ Phase 3 ✅ Phase 4 ✅ — 全部完成*  
*El Psy Kongroo.*
