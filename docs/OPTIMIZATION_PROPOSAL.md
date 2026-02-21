# CASTLE 優化提案 v1.0

> **提案人**: Kurisu (AI Architect)
> **日期**: 2026-02-21
> **Branch**: `dev`
> **現狀**: 110 non-vendored files, ~20,500 LOC, 410 tests, 0 lint errors

---

## 設計原則

### 1. 使用者友善 — 「國中生和 80 歲 PI 都會用」
- **Zero-configuration default**: 每一步都有合理預設值，不需要懂參數
- **Progressive disclosure**: 基本用法 3 個按鈕搞定，進階選項隱藏在 accordion
- **即時回饋**: 每個操作都有 progress bar、預覽、清楚的錯誤訊息
- **引導式流程**: 步驟之間自動提示下一步，禁用未就緒的按鈕

### 2. 效能優化 — 「合理發揮硬體效能」
- **GPU memory**: 精準管理 VRAM，不佔用不需要的顯存
- **Lazy loading**: 需要時才載入模型，用完釋放
- **Pipeline parallelism**: I/O 和 compute 重疊
- **Cache**: 重複計算結果快取，避免浪費

### 3. Linus Torvalds 程式哲學
- **Simple > Clever**: 程式碼要蠢到看一眼就懂
- **Good taste**: 正確的抽象層次，消除特殊情況而非處理特殊情況
- **No premature abstraction**: 不要為了「可能需要」而加框架
- **Data structures > Algorithms**: 先設計好資料結構，程式碼自然簡單
- **Working code ships**: 每個 commit 都要能跑、能測

### 4. CASTLE 設計目的
- **Training-free**: 不需要標註資料、不需要訓練模型
- **Species-agnostic**: 任何動物、任何場景
- **Pixel-level information**: 利用 DINOv2 的視覺特徵，不僅限於關鍵點
- **Hierarchical behavioral discovery**: 從粗到細自動發現行為模式

---

## Phase 0: KIT/SKE 前處理整合（Raiso's Stabilized Virtual Camera）

### 背景
Raiso 開發的 KIT（Kinematic Image Transform）前處理管線：
```
SAM+DeAOT → mask → centroid x(t), orientation θ(t)
       ↓
Butterworth lowpass (fc=0.25 Hz, order=2)
       ↓
x_c(t), θ_c(t)  (smooth camera trajectory)
       ↓
Dynamic crop: max(300, 2×(dist + margin))
       ↓
warpAffine: translate + rotate
       ↓
Resize → 518×518 → DINOv2
```

目前 CASTLE 的 `Preprocess` 類只有「中心對齊 + 固定 crop + 尾部旋轉」，**沒有**低通濾波、動態裁切、穩定化虛擬攝影機。

### P0-1: Core — Stabilized Camera Module
**新增 `castle/core/stabilized_camera.py`**

| 子任務 | 內容 | 預估 |
|--------|------|------|
| P0-1a | `StabilizedCamera` class：接收 centroid/angle 時間序列，計算 lowpass trajectory | 1 agent |
| P0-1b | 參數化：`fc` (default 0.25 Hz)、`order` (default 2)、`margin` (default 75 px)、`min_crop` (default 300 px)、`filter_type` (causal/zero-phase) | 同上 |
| P0-1c | `generate_stabilized_frames()` generator：逐幀做 warpAffine + dynamic crop + resize | 同上 |
| P0-1d | 單元測試：合成軌跡、邊界條件（靜止/快速移動/邊緣）、filter 特性驗證 | 1 agent |

### P0-2: Integration — 接入 Extraction Pipeline
| 子任務 | 內容 | 預估 |
|--------|------|------|
| P0-2a | 擴充 `Preprocess` class，新增 `stabilized_camera_switch`、相關參數 | 1 agent |
| P0-2b | 修改 `extract_roi_latent_from_video()` 支援穩定化前處理 | 同上 |
| P0-2c | Gradio UI：Extract Latent tab 加入「🎥 Stabilized Camera」選項群組 | 同上 |
| P0-2d | CLI：`castle extract --stabilized-camera --fc 0.25` | 同上 |
| P0-2e | 整合測試：確認穩定化前處理 → DINOv2 提取 → latent 輸出正確 | 1 agent |

### P0-3: Preview & Diagnostics
| 子任務 | 內容 | 預估 |
|--------|------|------|
| P0-3a | 預覽影片生成：使用者選擇參數後可預覽穩定化效果（10 秒片段） | 1 agent |
| P0-3b | 診斷圖表：crop size 分佈、HP residual RMS、speed correlation | 同上 |

**人力**: 3 agents 平行（core + integration + tests），1 agent QC
**產出**: `castle/core/stabilized_camera.py`、擴充的 `Preprocess`、UI/CLI 支援

---

## Phase 1: UX 大改善 — 「看到就會用」

### P1-1: 一鍵流程（Wizard Mode）
| 子任務 | 內容 | 預估 |
|--------|------|------|
| P1-1a | `castle/ui/wizard_ui.py`：3 步驟精靈（① 選影片 ② 點動物 ③ 看結果） | 1 agent |
| P1-1b | 自動推薦參數：根據影片解析度/幀率/動物大小推算最佳 crop、fc、bin_size | 1 agent |
| P1-1c | 進度總覽：Pipeline dashboard 顯示每一步的狀態（✅/🔄/⏳） | 同上 |

### P1-2: 錯誤訊息人性化
| 子任務 | 內容 | 預估 |
|--------|------|------|
| P1-2a | 錯誤訊息國際化：所有 user-facing error 提供「怎麼修」的建議 | 1 agent |
| P1-2b | 前置條件檢查：每個 tab 開啟時自動檢查上一步是否完成，未完成則顯示引導 | 同上 |

### P1-3: 操作引導
| 子任務 | 內容 | 預估 |
|--------|------|------|
| P1-3a | Tooltip / Hint：每個參數旁邊加 `ℹ️` 說明用途和建議值 | 1 agent |
| P1-3b | 首次使用引導：偵測新專案自動顯示快速教學 | 同上 |

**人力**: 2 agents 平行，1 agent QC
**產出**: Wizard mode、smart defaults、friendly errors

---

## Phase 2: 效能優化 — 「不浪費一個 cycle」

### P2-1: GPU 記憶體管理
| 子任務 | 內容 | 預估 |
|--------|------|------|
| P2-1a | Model lifecycle manager：統一管理 SAM/DeAOT/DINOv2 的載入/卸載 | 1 agent |
| P2-1b | 自動 VRAM 偵測：根據可用顯存自動調整 batch_size | 同上 |
| P2-1c | GPU cache eviction policy：操作完成後主動釋放 CUDA cache | 同上 |

### P2-2: Pipeline Parallelism
| 子任務 | 內容 | 預估 |
|--------|------|------|
| P2-2a | 影片解碼 / 前處理 / 推論 三級 pipeline（Producer-Consumer） | 1 agent |
| P2-2b | 多影片平行提取：同時處理多個影片的 latent extraction | 同上 |

### P2-3: 快取策略
| 子任務 | 內容 | 預估 |
|--------|------|------|
| P2-3a | 確定性 cache key：相同輸入 + 相同參數 = 直接跳過（content hash） | 1 agent |
| P2-3b | 增量更新：新增影片時只處理新的，不重跑整個專案 | 同上 |

**人力**: 2 agents 平行，1 agent QC
**產出**: 降低 VRAM usage、提升 throughput

---

## Phase 3: 程式碼簡化 — Torvalds 哲學

### P3-1: Data Structure Refactor
| 子任務 | 內容 | 預估 |
|--------|------|------|
| P3-1a | 統一專案資料模型：`ProjectData` dataclass 取代散落的 dict/path 拼接 | 1 agent |
| P3-1b | 統一 cluster 資料結構：整合 `cluster_.npz` / `time_series_*.csv` / `id.csv` 為一致的 API | 1 agent |

### P3-2: 消除特殊情況
| 子任務 | 內容 | 預估 |
|--------|------|------|
| P3-2a | 統一 filter type 邏輯：消除 `if device == 'cpu' or device == 'mps' ... elif 'cuda' ...` 分支（用 factory pattern） | 1 agent |
| P3-2b | 統一 video I/O：VideoReader 的 av/cv2 fallback 邏輯簡化 | 同上 |

### P3-3: 減少 UI 層邏輯
| 子任務 | 內容 | 預估 |
|--------|------|------|
| P3-3a | Gradio UI handlers 瘦身：所有業務邏輯移入 service，UI 只做 state ↔ widget 映射 | 1 agent |
| P3-3b | Gradio / PyQt 共用 ViewModel：抽出 shared state management | 同上 |

**人力**: 2 agents 平行，1 agent QC
**產出**: 更少的 code、更少的 bug surface

---

## Phase 4: 功能強化 — 我認為需要添加的

### P4-1: 多動物支援（Multi-Subject）
| 子任務 | 內容 | 預估 |
|--------|------|------|
| P4-1a | 同一影片中追蹤多個動物，每個獨立提取 latent | 1 agent |
| P4-1b | 社交行為分析：兩個動物之間的相對位置、朝向、距離 feature | 1 agent |
| P4-1c | 群體 ethogram：同時顯示多個動物的行為時間軸 | 同上 |

### P4-2: Real-time Streaming
| 子任務 | 內容 | 預估 |
|--------|------|------|
| P4-2a | 即時影片串流行為分析（webcam / RTSP） | 1 agent |
| P4-2b | 滑動窗口 clustering：即時分類新幀的行為 | 同上 |

### P4-3: Batch Processing
| 子任務 | 內容 | 預估 |
|--------|------|------|
| P4-3a | 多專案 batch pipeline：一次排程處理整個實驗的所有影片 | 1 agent |
| P4-3b | CLI batch mode：`castle batch run experiments.yaml` | 同上 |

### P4-4: Report Generation
| 子任務 | 內容 | 預估 |
|--------|------|------|
| P4-4a | 一鍵產生分析報告（PDF/HTML）：包含 ethogram、metrics、cluster info | 1 agent |
| P4-4b | 統計圖表自動排版 | 同上 |

---

## 執行排程

```
Week 1:  Phase 0 (KIT/SKE) ────────────────────── ★ 最高優先
Week 2:  Phase 1 (UX) + Phase 2 (效能) 平行 ───── ★★ 高優先
Week 3:  Phase 3 (簡化) ──────────────────────── ★ 中優先
Week 4+: Phase 4 (功能) ──────────────────────── 依決策
```

## 人力配置

| 角色 | 數量 | 負責範圍 |
|------|------|----------|
| **Implementation Agent** | 2-3 | 平行開發子任務 |
| **QC Agent** | 1 | 每個 Phase 完成後做 test + lint + consistency audit |
| **Architect (Kurisu)** | 1 | 任務分配、code review、conflict resolution |

## 風險評估

| 風險 | 影響 | 緩解策略 |
|------|------|----------|
| KIT 整合改動 extractor 核心 | 高 | 不修改現有 Preprocess，新增 StabilizedCamera 作為平行路徑 |
| 多 agent 修改衝突 | 中 | 嚴格分檔案 ownership |
| Sub-agent 執行卡死 | 中 | 拆小任務（每個 < 5 分鐘） |
| CIFS shutil.copy 問題 | 低 | 全面使用 shutil.copyfile（已落實） |

---

**請 IsonaEi 決策**：
1. Phase 排序是否同意？
2. Phase 4 的四個功能，哪些要做、哪些暫緩？
3. KIT 整合要用 causal filter（目前論文版本）還是 zero-phase（filtfilt，推薦的改進版）？
4. 還有想加的功能嗎？
