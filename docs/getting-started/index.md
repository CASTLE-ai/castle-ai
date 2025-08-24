# 快速開始

歡迎使用 CASTLE！這個章節將幫助您快速上手動物行為分析。

## 🎯 開始之前

在開始使用 CASTLE 之前，請確保您已經：

- [x] 檢查 [系統需求](requirements.md)
- [x] 準備好待分析的影片檔案  
- [x] 了解基本的動物行為分析概念

## 📋 快速檢查清單

<div class="checklist" markdown="1">

### 1. 系統準備
- [ ] Python 3.8+ 已安裝
- [ ] 足夠的硬碟空間 (建議 >5GB)
- [ ] GPU 支援 (選用但推薦)

### 2. 安裝 CASTLE
- [ ] 完成 [安裝程序](installation.md)
- [ ] 驗證安裝成功

### 3. 首次使用
- [ ] 嘗試 [快速範例](quick-example.md)
- [ ] 熟悉 [GUI 介面](gui-introduction.md)

</div>

## 🚀 三步驟開始使用

### Step 1: 安裝
```bash
pip install castle-behavior
```

### Step 2: 載入範例
```python
import castle
castle.download_examples()  # 下載範例影片
```

### Step 3: 分析
```python
analyzer = castle.BehaviorAnalyzer()
results = analyzer.analyze("examples/mouse_oft.mp4")
```

!!! success "恭喜！"
    您已經完成第一個動物行為分析！接下來可以：
    
    - 探索 [教學課程](../tutorials/) 學習更多功能
    - 查看 [GUI 介紹](gui-introduction.md) 了解圖形介面
    - 瀏覽 [範例資料](../examples/) 嘗試不同場景

## 📚 學習資源

| 資源類型 | 適合對象 | 預估時間 |
|---------|---------|----------|
| [系統需求](requirements.md) | 所有使用者 | 5 分鐘 |
| [安裝指南](installation.md) | 所有使用者 | 15 分鐘 |
| [快速範例](quick-example.md) | 初學者 | 10 分鐘 |
| [GUI 介紹](gui-introduction.md) | 非程式使用者 | 20 分鐘 |

## ❓ 遇到問題？

如果在開始使用過程中遇到任何問題：

- 查看 [FAQ](../community/faq.md) 常見問題
- 參考 [故障排除](../community/troubleshooting.md) 指南
- 在 [GitHub Issues](https://github.com/castle-project/castle/issues) 回報問題

---

**準備好了嗎？** [開始安裝 CASTLE →](installation.md){ .md-button .md-button--primary }