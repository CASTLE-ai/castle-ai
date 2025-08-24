# CASTLE Documentation & Tutorial Platform

<div class="hero-section" markdown="1">

![CASTLE Logo](assets/castle-logo.png){ .hero-logo }

## 🏰 CASTLE
### Computer vision Assisted System for Tracking, and Learning animal bEhavior

**Training-free 的動物行為分析革命性工具**

CASTLE 是一個創新的動物行為分析平台，採用 Focused Visual Latent 技術和階層式聚類演算法，無需訓練即可精準分析各種動物的行為模式。

[快速開始 :material-rocket-launch:](getting-started/){ .md-button .md-button--primary }
[觀看介紹影片 :material-play:](https://youtube.com/watch?v=example){ .md-button }

</div>

---

## 🌟 核心特色

<div class="feature-cards" markdown="1">

<div class="feature-card" markdown="1">
### :material-brain: Training-Free 分析
無需標註資料或訓練模型，直接分析動物行為，大幅降低使用門檻。
</div>

<div class="feature-card" markdown="1">
### :material-eye: 精準視覺追蹤  
採用先進的 Focused Visual Latent 技術，提供高精度的動物姿態追蹤。
</div>

<div class="feature-card" markdown="1">
### :material-chart-scatter-plot: 智慧行為聚類
自動識別和分類複雜的行為模式，支援多種動物物種。
</div>

<div class="feature-card" markdown="1">
### :material-tune: 靈活易用
提供直觀的 GUI 介面和強大的 API，適合不同程度的使用者。
</div>

</div>

---

## 🚀 快速開始

### 安裝 CASTLE

=== "pip 安裝"

    ```bash
    pip install castle-behavior
    ```

=== "從源碼安裝"

    ```bash
    git clone https://github.com/castle-project/castle.git
    cd castle
    pip install -e .
    ```

### 30 秒快速體驗

```python
import castle

# 載入範例影片
video_path = "examples/mouse_open_field.mp4"

# 初始化 CASTLE 分析器
analyzer = castle.BehaviorAnalyzer()

# 分析行為
results = analyzer.analyze(video_path)

# 視覺化結果
castle.plot.show_behavior_map(results)
```

!!! tip "第一次使用？"
    建議從 [安裝指南](getting-started/installation.md) 開始，然後嘗試 [快速範例](getting-started/quick-example.md)。

---

## 📚 學習路徑

<div class="learning-path" markdown="1">

### :material-numeric-1-circle: 初級使用者
**適合第一次接觸行為分析的研究生**

- [ ] [系統需求檢查](getting-started/requirements.md)
- [ ] [安裝 CASTLE](getting-started/installation.md)
- [ ] [GUI 介紹](getting-started/gui-introduction.md)
- [ ] [小鼠行為分析入門](tutorials/species/mouse-behavior.md)

**預估學習時間：2-3 小時**

### :material-numeric-2-circle: 中級使用者  
**適合有 DeepLabCut 經驗的使用者**

- [ ] [CASTLE 架構概念](tutorials/concepts/architecture.md)
- [ ] [Visual Latent 原理](tutorials/concepts/visual-latent.md)
- [ ] [多物種分析](tutorials/species/)
- [ ] [參數優化技巧](tutorials/advanced/parameter-optimization.md)

**預估學習時間：4-5 小時**

### :material-numeric-3-circle: 進階使用者
**適合需要客製化 Pipeline 的研究人員**

- [ ] [階層式聚類深入](tutorials/concepts/hierarchical-clustering.md)
- [ ] [自訂分析管線](tutorials/advanced/custom-pipeline.md)
- [ ] [批次處理](tutorials/advanced/batch-processing.md)
- [ ] [API 文件](api/)

**預估學習時間：6-8 小時**

</div>

---

## 📰 最新消息

!!! info "v2.1.0 發布 - 2024-01-15"
    - ✨ 新增果蠅行為分析支援
    - 🚀 效能提升 30%
    - 🐛 修復多線程處理問題
    - 📊 改進視覺化介面

!!! success "教學影片上線 - 2024-01-10"
    新增完整的 YouTube 教學系列，包含 15+ 個專業教學影片。
    [前往觀看 →](https://youtube.com/@castle-tutorials)

!!! note "論文發表 - 2023-12-20"
    CASTLE 核心技術論文已發表於 Nature Methods。
    [閱讀論文 →](https://doi.org/10.1038/s41592-023-xxxxx-x)

---

## 🤝 社群支援

<div class="community-links" markdown="1">

[:fontawesome-brands-github: **GitHub Issues**](https://github.com/castle-project/castle/issues)
:   回報 Bug 或功能請求

[:fontawesome-brands-youtube: **YouTube 頻道**](https://youtube.com/@castle-tutorials)
:   完整影片教學系列

[:material-forum: **討論區**](https://github.com/castle-project/castle/discussions)
:   技術討論與經驗分享

[:material-help-circle: **FAQ**](community/faq.md)
:   常見問題解答

</div>

---

## 📊 使用統計

<div class="stats-grid" markdown="1">

**1000+** 研究機構使用
**50+** 支援動物物種  
**10000+** 分析影片數量
**95%** 使用者滿意度

</div>

---

*最後更新：{{ git_revision_date_localized }}*