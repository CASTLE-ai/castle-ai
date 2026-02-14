# CASTLE Improvement Plan

> 本文件記錄所有待改進項目，含深度程式碼審計結果、最佳實踐研究、及具體改進方案。
> 最後更新：2026-02-15
> 版本：v2.0（Deep Audit Edition）

---

## 優先順序矩陣

| 項目 | 重要性 | 工作量 | 分類 | 狀態 |
|------|--------|--------|------|------|
| F-01 KDTree 快取 | High | Small | [Quick Fix] | ✅ Done |
| F-02 interpolate_missing_points 向量化 | High | Small | [Quick Fix] | ✅ Done |
| F-03 H5IO 資源管理改進 | High | Small | [Quick Fix] | ✅ Done |
| F-04 video_io_old.py 死碼清理 | Medium | Small | [Quick Fix] | ✅ Done |
| F-05 explorer.py / latent_explorer.py 調色盤重複 | Medium | Small | [Quick Fix] | ✅ Done |
| F-06 型別提示補充 | Medium | Small | [Quick Fix] | ✅ Done |
| F-07 config.py 格式錯誤修正 | High | Small | [Quick Fix] | ✅ Done |
| F-08 find_closest_point 向量化 | Medium | Small | [Quick Fix] | ✅ Done |
| F-09 merge() 方法 bug 修正 | High | Small | [Quick Fix] | ✅ Done |
| A-01 多前端架構（CLI/Web/Desktop） | High | Large | [Epic] | 🔄 進行中 |
| A-02 資料處理管線效能優化 | High | Medium | [Medium] | 📋 規劃中 |
| A-03 Tracking Mask 後處理 | Medium | Medium | [Medium] | ✅ 已實作於 tracking_manager |
| A-04 Cluster Annotator | High | Large | [Large] | 📋 規劃中 |
| A-05 PyQt Desktop 前端 | High | Large | [Epic] | 🔄 進行中 |
| B-01 Latent/LocalLatent 資料視覺化分離 | Medium | Medium | [Medium] | 📋 規劃中 |
| B-02 階層式聚類 UI 改善 | Medium | Medium | [Medium] | 📋 規劃中 |
| B-03 Session 完整恢復 | Medium | Medium | [Medium] | 📋 規劃中 |
| B-04 Undo/Redo 機制 | Low | Large | [Large] | 📋 規劃中 |
| B-05 統一配置管理 | Medium | Medium | [Medium] | 📋 規劃中 |
| C-01 Service Layer 抽象 | High | Large | [Large] | 📋 規劃中 |
| C-02 VideoReader 快取策略改善 | Medium | Small | [Quick Fix] | 📋 規劃中 |
| C-03 NUM_WORKERS 智慧設定 | Low | Small | [Quick Fix] | 📋 規劃中 |
| C-04 test coverage 補強 | Medium | Medium | [Medium] | 📋 規劃中 |
| C-05 UMAP 異步執行 | Medium | Medium | [Medium] | 📋 規劃中 |
| C-06 device 偵測統一 | Medium | Small | [Quick Fix] | 📋 規劃中 |
| C-07 post_track_ui 重複碼消除 | Low | Small | [Quick Fix] | 📋 規劃中 |
| C-08 H5IO print 改 logging | Low | Small | [Quick Fix] | ✅ Done |

---

## 一、原始改進項目（IsonaEi 提出）

### A-01. 多前端架構（CLI / Web / Desktop）[Epic]

**目標**：將 CASTLE 拆分為 CLI / Web / Desktop 三種操作模式。

**現狀問題**：
- UI 和邏輯高度耦合：`cluster_page_ui.py`（695 行）混合 Gradio 事件處理、matplotlib 繪圖、CSV I/O、KDTree 查找
- `Latent`/`LocalLatent` 內嵌 `plot_embedding()` 等繪圖方法
- Preprocess 配置用 Dropdown 字串 `'True'/'False'`
- 沒有統一 API 層

**技術方向**：
- 抽出 `castle.service` 層（function signatures，不依賴 UI）
- CLI：`typer` 套件（比 click 更現代，自動生成 help，type hints 整合佳）
- Web：保留 Gradio（快速迭代）或遷移 FastAPI + 前端
- Desktop：**PySide6**（推薦，LGPL 授權，Qt 官方維護）

**研究結論**：
- SLEAP 使用 PySide2 (legacy) → 已遷移 PySide6，架構為 MainWindow + 多 Panel
- DeepLabCut 使用 wxPython（較老舊，但穩定）
- napari 使用 Qt (magicgui)，是 Python 科學桌面應用的典範
- **建議跟隨 SLEAP/napari 路線，使用 PySide6**

### A-02. 資料處理管線效能優化 [Medium]

**現狀問題**（審計確認）：
1. `LatentAggregator.get_frame()` 每次呼叫都開新 `VideoReader` → 應改用 LRU cache
2. `find_nearest_embedding()` 每次都建新 KDTree → **已修復 (F-01)**
3. `interpolate_missing_points()` O(n²) list comprehension → **已修復 (F-02)**
4. matplotlib server-side render 到 JPEG 再傳前端
5. `NUM_WORKERS` 硬編碼 `cpu_count() // 2`
6. UMAP 同步阻塞

**改善方向**：
- `get_frame()` 加入 LRU cached VideoReader pool（C-02）
- UMAP 放入 background thread + progress callback（C-05）
- 前端互動改用 client-side rendering（pyqtgraph/vispy）

### A-03. Tracking Mask 後處理：最大連通區域過濾 [Medium]

**現狀**：已在 `ROITracker._smart_filter()` 中實作。
- 使用 `cv2.connectedComponentsWithStats()` per-object filtering
- 基於 reference frame 面積的 10% 作為閾值
- 此實作看起來合理，但僅在 tracking 時啟用

**待改進**：
- 提供獨立的 post-processing API（讓 Extract 階段也可用）
- 支援配置閾值（而非寫死 10%）

### A-04. Cluster Annotator [Large]

**目標**：讓使用者在聚類後審閱、標註行為語義。

**現狀**：命名流程簡陋，一次一個 cluster → 輸入 name → Enter。

**建議**：
- Gallery view：grid layout 顯示每個 cluster 的 N 個代表幀
- 影片片段預覽：每 cluster 的 top-K bout
- 批次標註表格（DataGrid）
- 標註 schema：`{cluster_id, behavior_name, description, confidence, annotator, timestamp}`
- **PyQt Desktop 版本最適合做此功能**

---

## 二、深度程式碼審計發現

### 架構層級問題

#### B-01. Latent/LocalLatent 資料視覺化耦合 [Medium]
- `latent_explorer.py` 的 `Latent.plot_syllables()`, `LocalLatent.plot_embedding()`, `plot_name_embedding()` 都在資料類別內
- `explorer.py` 的 `Latent.plot()`, `FocusLatent.plot()` 同樣問題
- **建議**：抽出到 `castle.visualization` 模組

#### B-02. 階層式聚類 UI 不直觀 [Medium]
- Stage 4 核心操作是遞迴的（選 cluster → UMAP → DBSCAN → 命名 → submit → 再選子 cluster）
- UI 是線性佈局，沒有樹狀結構視覺化
- **建議**：加入 tree widget 顯示聚類層級

#### B-03. Session 恢復不完整 [Medium]
- `restore_session()` 可讀回 cluster assignment，但不能恢復 embedding
- 使用者重新開啟需重跑 UMAP

#### B-04. 沒有 Undo/Redo [Large]
- 所有操作不可逆，沒有操作歷史
- **建議**：Command Pattern

#### B-05. 配置管理分散 [Medium]
- `config.json` 管專案設定，UMAP/DBSCAN 參數在 UI 層字串裡
- 沒有統一的配置序列化/反序列化

### 程式碼品質問題

#### C-01. Service Layer 缺失 [Large]
- 現有的 `core/extractor.py` 是部分 service layer，但不完整
- `cluster_page_ui.py` 直接操作 `Latent`/`LocalLatent` 物件
- 需要定義 `castle.service.clustering_service`, `castle.service.extraction_service` 等

#### C-02. LatentAggregator.get_frame() 效能問題 [Quick Fix]
- 每次呼叫開新 `VideoReader` context manager
- **建議**：維護 VideoReader pool with LRU eviction

#### C-03. NUM_WORKERS 硬編碼 [Quick Fix]
- `extractor.py`: `NUM_WORKERS = os.cpu_count() // 2`
- `tracking_manager.py`: `num_workers = max(1, int(os.cpu_count() * 0.2))`
- 不一致且不考慮 GPU memory
- **建議**：移到 config.py 或 environment.py

#### C-04. 測試覆蓋不足 [Medium]
- 現有 12 個測試檔案，但多為 integration tests
- 缺少 unit tests for：Preprocess, LatentAggregator, Latent/LocalLatent, H5IO
- 無 UI 測試

#### C-05. UMAP 異步執行 [Medium]
- 大資料量時 UI 完全卡住
- **建議**：QThread (PyQt) 或 ThreadPoolExecutor

#### C-06. Device 偵測重複 [Quick Fix]
- `environment.py` 有 `Environment._detect_device()`
- `image_segment.py`, `video_object_segment.py`, `latent_explorer.py`, `myumap.py` 各自偵測
- **建議**：統一使用 `castle.core.environment.env.device`

#### C-07. post_track_ui / batch_track_ui 重複碼 [Quick Fix]
- `plot_basic_mask_info()` 和 `generate_csv_analysis()` 有近乎相同的邏輯
- `read_label()` 和 `read_roi_labels()` 功能重複
- **建議**：統一到一個模組

#### C-08. H5IO 使用 print 而非 logging [Quick Fix] ✅ Done
- `read_config()` 和 `write_config()` 使用 `print()`
- **修復**：改為 `logger.debug()`

### Bug 修正

#### F-07. config.py SUPPORTED_MODELS 格式錯誤 [Quick Fix] ✅ Done
- 第 36 行存在明顯的字串格式問題：
  ```python
  'dinov2_vitb1ain',                                                                                                      'dinov2_vitb14_reg4_pretrain',
  ```
- `'dinov2_vitb1ain'` 應該是 `'dinov2_vitb14'`，且 `reg4_pretrain` 被空白和換行分隔到另一行

#### F-09. explorer.py merge() 方法 bug [Quick Fix] ✅ Done
- `Latent.merge()` 引用了 `cid2` 而非 `cids`，是明顯的拼字錯誤：
  ```python
  def merge(self, cids):
      self.syllables[self.syllables == cid2] = cid1  # ← BUG: cid2, cid1 未定義
  ```

### 效能問題

#### F-01. KDTree 每次重建 [Quick Fix] ✅ Done
- `find_nearest_embedding()` 每次呼叫都 `KDTree(data)`
- **修復**：改為在 `EmbeddingScatterPlot.__init__` 建一次，後續重用

#### F-02. interpolate_missing_points O(n²) [Quick Fix] ✅ Done
- 用 list comprehension 找前後最近點
- **修復**：numpy 向量化，使用 `np.searchsorted`

#### F-08. find_closest_point 線性搜尋 [Quick Fix] ✅ Done
- `video_align.py` 中 `find_closest_point()` 用 Python for-loop 遍歷 contour
- **修復**：改用 numpy 向量化運算

### 資源管理問題

#### F-03. H5IO 資源管理不完善 [Quick Fix] ✅ Done
- `reset_count > 5000` 才重新開啟檔案，邏輯不直觀
- `__del__` 中的 `f.id.valid` 檢查可能在程式結束時失敗
- **修復**：加入 context manager 支援

#### F-04. video_io_old.py 死碼 [Quick Fix] ✅ Done
- 整個檔案是 `video_io.py` 的舊版本，`ReadArray` 和 `WriteArray` 已有新實作
- **修復**：標記為 deprecated，加入棄用警告

### 程式碼重複

#### F-05. 調色盤定義重複 [Quick Fix] ✅ Done
- `explorer.py` 和 `latent_explorer.py` 各自定義了相同的 `_palette`
- `plot.py` 也有 `_palette_hex`
- **修復**：統一到 `castle.core.config` 或專門的 palette 模組

---

## 三、最佳實踐研究結論

### 3.1 Desktop Framework 選擇
- **推薦：PySide6**
  - LGPL 授權（PyQt6 是 GPL，對學術開源限制較大）
  - Qt 官方維護，API 一致
  - 與 pyqtgraph、vispy 完全相容
  - napari、SLEAP 等科學工具都使用 Qt 生態

### 3.2 UMAP 視覺化最佳實踐
- **pyqtgraph ScatterPlotItem**：適合 <100K 點，Qt 原生整合
- **vispy SceneCanvas**：OpenGL 加速，適合 >100K 點
- **建議**：先用 pyqtgraph（簡單整合），必要時切換 vispy
- 互動功能：lasso selection, hover tooltip, click-to-inspect

### 3.3 CLI 設計模式
- **typer** > **click**（2025+ 推薦）
  - 自動型別推斷、自動 --help 生成
  - 支援 rich 整合、progress bars
  - Snakemake/Nextflow 風格的 pipeline 定義
- 子命令結構：`castle track`, `castle extract`, `castle cluster`, `castle annotate`

### 3.4 SLEAP 架構參考
- SLEAP 使用 sleap-io（資料 I/O 層）+ sleap-nn（模型層）+ sleap label（GUI）
- 清楚的 CLI subcommands：`sleap label`, `sleap nn-train`, `sleap nn-track`
- **CASTLE 可以參考此架構**：castle-core + castle-desktop + castle-cli

---

## 四、PyQt Desktop 前端規劃 [Epic]

### 4.1 架構設計
```
castle/desktop/
├── __init__.py
├── __main__.py          # Entry point: python -m castle.desktop
├── app.py               # QApplication setup
├── main_window.py       # Main window with tab navigation
├── widgets/
│   ├── __init__.py
│   ├── project_panel.py     # Stage 0: Project management
│   ├── source_panel.py      # Stage 1: Video upload
│   ├── tracking_panel.py    # Stage 2: ROI tracking
│   ├── extract_panel.py     # Stage 3: Latent extraction
│   ├── microscope_panel.py  # Stage 4: Behavior Microscope (priority)
│   └── annotator_panel.py   # Stage 4b: Cluster Annotator
├── components/
│   ├── __init__.py
│   ├── embedding_view.py    # pyqtgraph scatter plot
│   ├── video_player.py      # Video frame viewer
│   ├── cluster_tree.py      # Hierarchical cluster tree
│   └── syllable_bar.py      # Behavior timeline bar
└── services/
    ├── __init__.py
    └── worker_threads.py    # QThread wrappers for UMAP, extraction
```

### 4.2 Stage 4 Behavior Microscope（優先）
- **pyqtgraph ScatterPlotItem** for embedding visualization
  - Click → nearest point → show frame
  - Lasso selection for manual cluster editing
  - Color by cluster assignment
- **QTreeWidget** for hierarchical cluster navigation
- **QThread** for async UMAP computation
- 即時互動（無需 matplotlib render → JPEG → 傳送循環）

### 4.3 啟動方式
```bash
python -m castle.desktop              # 啟動桌面版
python -m castle.desktop --project X  # 直接開啟專案
```

---

## 五、下一步行動

1. ✅ 深度程式碼審計完成
2. ✅ 最佳實踐研究完成
3. ✅ Quick Fixes 實作完成（F-01 ~ F-09）
4. 🔄 PyQt Desktop 前端 shell 建立中（A-05）
5. 📋 Service Layer 設計與實作（C-01）
6. 📋 CLI 建立（typer-based）
7. 📋 Test coverage 補強（C-04）
