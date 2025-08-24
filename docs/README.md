# CASTLE Documentation Project

這是 CASTLE 行為分析工具的完整文檔專案，採用 MkDocs + Material for MkDocs 建立的現代化文檔網站。

## 📁 專案結構

```
docs/
├── index.md                    # 首頁
├── getting-started/            # 快速開始
│   ├── index.md               # 快速開始主頁
│   ├── requirements.md        # 系統需求
│   ├── installation.md        # 安裝指南
│   ├── quick-example.md       # 快速範例
│   └── gui-introduction.md    # GUI 介紹
├── tutorials/                  # 教學課程
│   ├── index.md               # 教學主頁
│   └── concepts/              # 核心概念
│       └── index.md           # 概念總覽
├── api/                       # API 文件
│   └── index.md               # API 文件主頁
├── examples/                  # 範例資料
├── community/                 # 社群資源
│   └── faq.md                 # 常見問題
├── stylesheets/               # 自訂樣式
│   └── extra.css             # 額外 CSS 樣式
└── javascripts/               # 自訂腳本
    └── extra.js              # 額外 JavaScript 功能
```

## 🚀 快速開始

### 1. 安裝依賴

```bash
pip install -r requirements.txt
```

### 2. 本地預覽

```bash
mkdocs serve
```

然後在瀏覽器中打開 http://127.0.0.1:8000

### 3. 建構網站

```bash
mkdocs build
```

生成的靜態網站將位於 `site/` 目錄中。

## 🌟 功能特色

### 🎨 視覺設計
- **現代化 UI**: 基於 Material Design 的美觀介面
- **響應式設計**: 完美支援桌面、平板、手機
- **深色模式**: 自動或手動切換深色主題
- **自訂樣式**: 專門為 CASTLE 設計的視覺元素

### 📱 使用者體驗
- **快速搜尋**: 全文搜尋功能，支援中英文
- **導航清晰**: 階層式導航結構，易於使用
- **互動式元素**: 可展開的 FAQ、可勾選的學習清單
- **進度追蹤**: 自動保存學習進度到本地儲存

### 🔧 技術功能
- **語法高亮**: Python 程式碼自動語法高亮
- **一鍵複製**: 程式碼區塊支援快速複製
- **數學公式**: 支援 LaTeX 數學公式渲染
- **圖表支援**: Mermaid 圖表和自訂圖形

### 🌐 多媒體整合
- **影片嵌入**: YouTube 教學影片無縫整合
- **圖片優化**: 自動圖片壓縮和響應式載入
- **互動圖表**: Plotly 互動式資料視覺化
- **程式碼範例**: 完整可執行的程式碼示例

## 📚 內容概覽

### 🏠 首頁 (index.md)
- Hero Section 展示 CASTLE 核心價值
- 四大特色功能說明
- 快速開始指引
- 學習路徑導航
- 最新消息和社群連結

### ⚡ 快速開始 (getting-started/)
- **系統需求**: 詳細的硬軟體需求說明
- **安裝指南**: 多種安裝方式和故障排除
- **快速範例**: 10分鐘完整教學範例
- **GUI 介紹**: 圖形介面完整使用指南

### 🎓 教學課程 (tutorials/)
- **核心概念**: CASTLE 架構和技術原理
- **物種教學**: 針對不同動物的專門教學
- **進階應用**: 客製化和高級功能
- **實作專案**: 完整的實際應用範例

### 🔧 API 文件 (api/)
- **核心功能**: 主要 API 類別和方法
- **GUI 組件**: 圖形介面相關 API
- **工具函數**: 輔助功能和工具
- **程式碼範例**: 實際使用示例

### 🤝 社群資源 (community/)
- **FAQ**: 最常見問題解答
- **故障排除**: 詳細的問題解決指南
- **貢獻指南**: 如何參與專案開發

## 🔧 開發指南

### 添加新頁面

1. 在適當的目錄中創建 `.md` 文件
2. 在 `mkdocs.yml` 的 `nav` 部分添加連結
3. 使用一致的 Markdown 格式和結構

### 自訂樣式

- 編輯 `docs/stylesheets/extra.css`
- 使用 CSS 變量保持主題一致性
- 確保響應式設計相容性

### 添加互動功能

- 編輯 `docs/javascripts/extra.js`
- 使用現有的工具函數
- 保持向後相容性

### 更新配置

- 主要配置在 `mkdocs.yml`
- 新增外掛需更新 `requirements.txt`
- 測試本地建構無誤後提交

## 🚀 部署

### GitHub Pages 自動部署

專案已配置 GitHub Actions 自動部署：

1. **推送觸發**: 當 `main` 分支的文檔相關檔案變更時自動觸發
2. **建構檢查**: 自動檢查連結有效性和 HTML 結構
3. **效能測試**: Lighthouse 效能評測
4. **自動部署**: 成功建構後自動部署到 GitHub Pages

### 手動部署

```bash
# 建構並部署到 GitHub Pages
mkdocs gh-deploy

# 或建構到本地目錄
mkdocs build
```

## 📊 品質保證

### 內容品質
- ✅ 中文繁體內容完整
- ✅ 程式碼範例已測試
- ✅ 外部連結定期檢查
- ✅ 圖片和影片正常載入

### 技術品質
- ✅ 響應式設計測試
- ✅ 跨瀏覽器相容性
- ✅ 載入效能優化
- ✅ SEO 優化設定

### 使用者體驗
- ✅ 導航結構清晰
- ✅ 搜尋功能完善
- ✅ 無障礙設計
- ✅ 互動功能穩定

## 🔮 未來計畫

### 內容擴充
- [ ] 更多物種教學範例
- [ ] 進階 API 文件完善
- [ ] 視頻教學內容製作
- [ ] 多語言支援 (英文版本)

### 功能增強
- [ ] 離線使用支援
- [ ] 互動式程式碼編輯器
- [ ] 社群評論系統
- [ ] 個人化學習記錄

### 技術升級
- [ ] PWA (Progressive Web App) 支援
- [ ] 更豐富的互動圖表
- [ ] AI 輔助內容搜尋
- [ ] 即時協作編輯

## 🤝 貢獻

歡迎參與 CASTLE 文檔的改進！

### 如何貢獻
1. Fork 專案
2. 創建功能分支
3. 提交變更
4. 發起 Pull Request

### 貢獻類型
- 📝 內容改進和錯誤修正
- 🎨 視覺設計和 UX 改善
- 💡 新功能建議和實作
- 🐛 Bug 回報和修復
- 🌐 翻譯和在地化

### 聯繫方式
- GitHub Issues: 技術問題和功能請求
- GitHub Discussions: 社群討論和意見交流
- Email: documentation@castle-project.org

---

## 📄 授權

本文檔專案遵循 [MIT License](../LICENSE) 授權條款。

**建立日期**: 2024年1月  
**最後更新**: {{ git_revision_date_localized }}  
**版本**: v1.0.0