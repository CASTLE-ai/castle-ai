# CASTLE Improvement Plan

> 本文件記錄所有待改進項目，含深度程式碼審計結果、最佳實踐研究、及具體改進方案。
> 最後更新：2026-02-15 v3.0
> 狀態：**規劃階段 — 待 IsonaEi 確認後實作**

---

## 優先順序矩陣

| # | 項目 | 重要性 | 工作量 | 狀態 |
|---|------|--------|--------|------|
| **C-01** | **Service Layer 抽象** | 🔴 High | Large | 📋 待實作（所有前端的前提） |
| **A-06** | **Latent 抽取方式改進** | 🔴 High | Medium | 📋 待討論方案 |
| **A-07** | **深度模型效能優化** | 🔴 High | Medium | 📋 待實作 |
| **A-03** | **Tracking Mask 後處理改善** | 🟡 Medium | Small | 📋 待實作 |
| **A-01** | **CLI 前端** | 🔴 High | Medium | 📋 待 C-01 完成後實作 |
| **A-04** | **Cluster Annotator（Stage 4 延伸）** | 🔴 High | Large | 📋 待討論 |
| **B-01** | **Latent/LocalLatent 資料視覺化分離** | 🟡 Medium | Medium | 📋 隨 C-01 一起做 |
| **A-02** | **資料處理管線效能優化** | 🟡 Medium | Medium | 📋 部分已完成 |
| **B-02** | **階層式聚類 UI + Tree View** | 🟡 Medium | Medium | 📋 |
| **B-03** | **Session 完整恢復** | 🟡 Medium | Medium | 📋 |
| **B-05** | **統一配置管理** | 🟡 Medium | Medium | 📋 |
| **C-02** | **VideoReader LRU cache** | 🟡 Medium | Small | 📋 |
| **C-04** | **測試覆蓋補強** | 🟡 Medium | Medium | 📋 |
| **C-05** | **UMAP 異步執行** | 🟡 Medium | Medium | 📋 |
| **C-06** | **Device 偵測統一** | 🟢 Low | Small | 📋 |
| **C-07** | **post_track / batch_track 重複碼** | 🟢 Low | Small | 📋 |
| **C-03** | **NUM_WORKERS 智慧設定** | 🟢 Low | Small | 📋 |
| **B-04** | **Undo/Redo 機制** | 🟢 Low | Large | 📋 |
| **A-05** | **PyQt Desktop 前端** | 🟢 Low | Large | ⏸️ 暫停（CLI > Web > Desktop） |
| F-01~F-09 | Quick Fixes（9 項） | — | — | ✅ 全部完成 |

---

## 一、新增項目（IsonaEi 2/15 提出）

### A-06. Latent 抽取方式改進 [Medium] 🔴 High

#### 現狀分析

目前 `_weighted_pooling()` 的做法（`castle/core/models.py` L85-107）：
1. 將 ROI mask resize 到模型輸入尺寸（518×518 或 592×592）
2. 下採樣到 patch grid（37×37）
3. 每個 patch 的權重 = mask 在該 patch 覆蓋的像素數
4. **所有 patch token 做 mask-weighted average → 輸出一個 768-dim 向量**

```python
# 核心邏輯（簡化）
weighted_sum = (patch_features * mask_weights).sum(dim=spatial)
latent = weighted_sum / total_weight  # → (768,) 單一向量
```

#### 問題

**整個 ROI 被壓縮成一個 768-dim 向量，丟失了所有空間資訊。**

- ROI 內不同部位（頭、身體、尾巴、四肢）的 patch token 被平均掉
- 動物的姿態資訊（四肢的空間排列）在平均後消失
- 小面積但重要的結構（例如尾巴末端）被大面積結構（身體）主導
- 無法區分「頭在左邊 vs 頭在右邊」的差異（除非用 rotation alignment）

#### 可能的改進方案（待討論）

**方案 A：保留空間 patch tokens（不做 pooling）**
- 直接使用 37×37 的 patch grid，只取 mask 覆蓋的 patches
- Latent 維度變高（N_patches × 768），但保留完整空間資訊
- 問題：不同幀的 ROI 位置不同 → patch 數量不一致 → 需要 padding 或 adaptive pooling
- 適用場景：需要精細姿態區分的行為

**方案 B：區域分割 pooling（Regional Pooling）**
- 將 ROI 分成 K 個子區域（例如上/中/下，或用 mask 的幾何中心做放射狀分割）
- 每個子區域做獨立 weighted average → K × 768-dim 向量
- 保留粗略空間結構，且維度固定
- 問題：K 怎麼定義？固定分區 vs 動態分區？

**方案 C：多尺度 pooling（Multi-scale）**
- 同時計算 global average + 2×2 grid average + 4×4 grid average
- 拼接成 (1 + 4 + 16) × 768 = 21 × 768 的向量
- SPPNet 式的空間金字塔策略
- 優點：固定維度、保留多尺度空間資訊

**方案 D：Generalized Mean Pooling (GeM)**
- `GeM(x, p) = (mean(x^p))^(1/p)`，p=1 等於 average，p→∞ 等於 max
- p 可學習或設為超參數
- 比 average pooling 更強調顯著特徵
- 論文參考：Group Generalized Mean Pooling for ViT (CVPR 2023)

**方案 E：保留 top-K 顯著 patch tokens**
- 對 mask 內的 patch tokens 按 L2 norm 排序，取 top-K
- 拼接或做 attention pooling
- 保留最「特殊」的 patches，忽略平淡的背景

**方案 F：多層特徵融合**
- 目前只取最後一層的 patch tokens
- DINOv2 的淺層 → 紋理/邊緣，深層 → 語義
- 取多層（例如 layer 4, 8, 12）做拼接，增加特徵豐富度

**我的建議**：先試 **方案 C（多尺度 pooling）** 或 **方案 D（GeM）**——改動最小，效果可量化。如果需要更精細的姿態區分，再考慮方案 A 或 B。

#### 實驗設計（驗證改進是否有效）
- 對照組：現有 weighted average
- 指標：下游聚類的 silhouette score、同一行為 cluster 的 intra-class variance
- 資料：CTRL30 OFT 資料集

---

### A-07. 深度模型效能優化 [Medium] 🔴 High

#### 目前使用深度模型的階段

| 階段 | 模型 | 用途 | 效能現狀 |
|------|------|------|----------|
| Stage 2 Tracking | DeAOT (r50/swinb) | 逐幀 ROI 追蹤 | batch=16, `track_batch()` 有做 batch encoding |
| Stage 2 Label | SAM (vit_b) | 點擊分割 ROI | 單幀推理，即時互動 |
| Stage 3 Extract | DINOv2/v3 (vitb/vitl) | 提取 latent | batch processing, DataLoader 多線程 |

#### 發現的效能問題

**1. AOT Tracker — batch encoding 但 sequential propagation**
- `track_batch()` 在 `video_object_segment.py` L117-145：
  - ✅ `self.model.encode_image(image_batch_tensor)` 是 batch 操作
  - ❌ 但 propagation（記憶體匹配 + 解碼）是逐幀循序的 → batch encoding 的好處有限
  - 這是 AOT 架構的根本限制（tracking 需要前一幀的 mask 作為 memory）

**2. DINOv2/v3 — 沒有用 `torch.inference_mode()` 一致**
- `DINOv3Encoder.extract_features()` 用了 `torch.inference_mode()` + `torch.autocast(float16)` ✅
- `DINOv2Encoder.extract_tensor_batch()` 只用了 `torch.no_grad()` ❌
- `torch.inference_mode()` 比 `no_grad()` 更快（不保留 version counter）
- DINOv2 也沒有用 `float16 autocast` → 效能損失

**3. DINOv3 預處理 — 三次 resize 操作**
- `preprocess_batch()` 做了：resize → center_crop → resize
- 這是兩次 bicubic interpolation + 一次 crop，可以合併

**4. 模型載入 — 每次 extraction 都可能重新載入**
- `extract_tensor_batch()` 檢查 `self.model is None` → lazy load
- 但如果 LatentAggregator 為每個影片建新 encoder，就會重複載入
- 目前看起來 UI 層會持有同一個 encoder 實例 → 不一定重複，但沒有保證

**5. SAM — 使用 `vit_b` 固定**
- `generate_sa(model_type='vit_b')` 硬編碼在 `label_ui.py`
- SAM2 已經出了，效能和精度都更好
- 但 SAM 只用於互動式標註（偶爾使用），優先級不高

#### 改善方案

| 問題 | 方案 | 預估提升 | 工作量 |
|------|------|----------|--------|
| DINOv2 沒用 inference_mode + float16 | 加入 `torch.inference_mode()` + `torch.autocast` | 10-30% 推理加速 | Small |
| DINOv3 三次 resize | 合併成一次（計算最終尺寸直接 resize） | ~5% preprocessing 加速 | Small |
| 模型載入沒有 singleton 保證 | 加入 model registry / singleton pattern | 避免重複載入 | Small |
| AOT sequential propagation | 結構限制，無法根本改變 | — | — |
| SAM → SAM2 | 替換分割模型 | 更好的分割品質 | Medium |

---

## 二、原始項目（IsonaEi 提出，更新版）

### A-01. CLI 前端 [Medium] 🔴 High

**前提**：需先完成 C-01（Service Layer）

**CLI 定位**：自動化 pipeline 工具，不是互動式介面。

**子命令設計**（typer-based）：
```
castle init <project_name> --storage <path>
castle add-videos <project> --source <dir_or_files>
castle track <project> --model r50_deaotl [--skip-existing]
castle extract <project> --model dinov3_vitb16 --roi 1 [--batch-size 32]
castle cluster <project> --umap-preset high-100-50 --eps 1.0
castle export <project> --format csv,srt
castle info <project>    # 顯示專案狀態
```

**不適合 CLI 的操作**：
- SAM 點擊標註 ROI（需要視覺互動）
- UMAP 結果視覺確認（需要人看圖）
- 手動調 epsilon（需要反覆嘗試）

### A-03. Tracking Mask 後處理改善 [Small] 🟡 Medium

**現狀問題**（code review 結果）：
1. 閾值硬編碼 `area * 0.1`，不可配置
2. 多個 reference frame 時，閾值被最後一個覆蓋（應取 mean/max/median）
3. 只在 tracking 時使用，Extract 階段沒有 post-filter 選項
4. `print()` 而非 `logging`

**改善方案**：
- 閾值策略改為可配置：`{method: 'relative', ratio: 0.1}` 或 `{method: 'absolute', min_area: 50}`
- 多 reference frame 取 median area 作為基準
- 提供獨立 API：`castle.core.mask_filter.filter_largest_component(mask, threshold)`
- 讓 Extract 階段也可選擇性啟用

**多 ROI 設計考量**：
- 每個 ROI ID 獨立做 largest component（目前已是如此 ✅）
- 但多個 ROI 之間可能重疊 → 過濾後需要處理衝突（取 ROI priority 或面積大者）

### A-04. Cluster Annotator [Large] 🔴 High

**定位**：Stage 4 延伸 tab

**核心功能**：
- Gallery view：每個 cluster 顯示 N 個代表幀（隨機抽樣或按 distance-to-centroid 排序）
- 影片片段預覽：每個 cluster 的 top-K bout 自動剪成短片
- 批次標註表格：一次看所有 cluster，直接編輯名稱和描述
- 標註 schema：`{cluster_id, behavior_name, description, confidence, annotator, timestamp}`
- 與現有 `id.csv` 整合或擴展

---

## 三、架構問題（B 系列）

### B-01. Latent/LocalLatent 資料視覺化分離 [Medium]
- 把 `plot_embedding()`, `plot_name_embedding()`, `plot_syllables()` 移到 `castle.visualization` 模組
- 隨 C-01 Service Layer 一起重構

### B-02. 階層式聚類 UI + Tree View [Medium]
- 加入 tree widget 顯示 `root → root_a0 → root_a0_b1` 的層級
- 點擊 tree node 跳到對應的 UMAP view

### B-03. Session 完整恢復 [Medium]
- 保存 UMAP embedding 結果（目前 `cluster_{name}.npz` 有存，但 restore 沒有讀回）
- 恢復時重建 `LocalLatent` 和 `EmbeddingScatterPlot`，不需重跑 UMAP

### B-05. 統一配置管理 [Medium]
- UMAP/DBSCAN 參數寫入 `config.json` 或獨立 `cluster_config.json`
- 方便 CLI 和 Session restore 讀取

### B-04. Undo/Redo 機制 [Large] 🟢 Low priority
- Command Pattern：每個操作封裝成 Command 物件
- 操作歷史 stack
- 暫不需要——先做好 Session restore 更實際

---

## 四、程式碼品質（C 系列）

### C-01. Service Layer 抽象 [Large] 🔴 **最高優先**

**這是所有前端（CLI/Web/Desktop）的前提。**

**設計草案**：
```python
# castle/service/project_service.py
class ProjectService:
    def create_project(name, storage_path) -> ProjectInfo
    def add_videos(project, video_paths) -> List[str]
    def get_project_info(project) -> ProjectInfo

# castle/service/tracking_service.py  
class TrackingService:
    def track(project, video, model, start, stop, ...) -> TrackingResult
    def get_tracking_status(project, video) -> TrackingStatus

# castle/service/extraction_service.py
class ExtractionService:
    def extract_latent(project, video, model, roi, preprocess, ...) -> str
    def extract_crop_video(project, video, roi, preprocess, ...) -> str

# castle/service/clustering_service.py
class ClusteringService:
    def initialize(project, roi, bin_size, model) -> SessionState
    def run_umap(session, cluster_name, umap_config) -> EmbeddingResult
    def run_dbscan(session, eps) -> ClusterResult
    def label_cluster(session, cluster_id, name) -> None
    def submit(session) -> ExportResult
    def restore_session(project) -> SessionState
```

**原則**：
- Service Layer 只依賴 Core（不依賴 UI）
- 返回值是 dataclass / dict，不是 Gradio update 物件
- 狀態管理在 Service 內，UI 只負責呈現

### C-02 ~ C-07：其他程式碼品質項目

（與之前版本相同，此處省略。詳見 git history `261f0d8`。）

- C-02: VideoReader LRU cache
- C-03: NUM_WORKERS 智慧設定
- C-04: 測試覆蓋補強
- C-05: UMAP 異步執行
- C-06: Device 偵測統一
- C-07: 重複碼消除

---

## 五、已完成的 Quick Fixes（v2.0 完成）

| # | 項目 | Commit |
|---|------|--------|
| F-01 | KDTree 快取 | `0d0bcb3` |
| F-02 | interpolate_missing_points 向量化 | `ad42804` |
| F-03 | H5IO context manager + logging | `b12d7dc` |
| F-04 | video_io_old deprecated | `e075d94` |
| F-05 | 調色盤統一到 config.py | — |
| F-07 | config.py SUPPORTED_MODELS typo | `078df85` |
| F-08 | find_closest_point 向量化 | `c2e9c66` |
| F-09 | explorer.py merge() bug | `591ec25` |
| C-08 | H5IO print → logging | `b12d7dc` |

---

## 六、建議實作順序

```
Phase 1: 基礎架構（先做這些，其他都依賴它們）
  ├── C-01 Service Layer
  ├── B-01 Latent/LocalLatent 資料視覺化分離
  ├── C-06 Device 偵測統一
  └── C-07 重複碼消除

Phase 2: 功能改進
  ├── A-06 Latent 抽取方式改進（實驗 + 整合）
  ├── A-07 深度模型效能優化（inference_mode, float16, resize 合併）
  ├── A-03 Tracking Mask 後處理改善
  └── B-05 統一配置管理

Phase 3: CLI + 聚類改進
  ├── A-01 CLI 前端（基於 Service Layer）
  ├── A-04 Cluster Annotator（Stage 4 延伸 tab）
  ├── B-02 階層式聚類 Tree View
  └── B-03 Session 完整恢復

Phase 4: 進階優化
  ├── A-02 資料處理管線效能（C-02, C-03, C-05）
  ├── C-04 測試覆蓋補強
  └── A-05 Desktop 前端（如果需要）
```

---

## 七、下一步

**等待 IsonaEi 確認**：
1. A-06 Latent 抽取方式：選哪個方案？還是先做實驗比較？
2. A-07 的 small fixes（inference_mode, float16）可以直接做嗎？
3. 建議的 Phase 1-4 順序是否同意？
4. Service Layer 的 API 設計草案需要更細化嗎？
