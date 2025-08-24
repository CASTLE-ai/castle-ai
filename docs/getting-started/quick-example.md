# 快速範例

通過這個 10 分鐘的快速範例，您將學會如何使用 CASTLE 分析小鼠在開放場實驗 (Open Field Test) 中的行為。

## 學習目標

完成這個範例後，您將能夠：

- [x] 載入和分析影片檔案
- [x] 理解 CASTLE 的基本工作流程  
- [x] 視覺化行為分析結果
- [x] 匯出分析數據

⏱️ **預計時間**: 10-15 分鐘

## 準備工作

### 1. 確認安裝
確保您已經完成 [CASTLE 安裝](installation.md)：

```python
import castle
print(f"CASTLE 版本: {castle.__version__}")
```

### 2. 下載範例資料
```python
# 下載範例影片檔案 (約 50MB)
castle.download_examples()

# 檢查下載的檔案
import os
print("可用範例檔案:")
for file in os.listdir("examples/"):
    if file.endswith(('.mp4', '.avi')):
        print(f"  - {file}")
```

## 🎬 影片教學

<div class="video-container">
<iframe width="560" height="315" 
        src="https://www.youtube.com/embed/example_video_id" 
        frameborder="0" allowfullscreen>
</iframe>
</div>

*如果無法觀看影片，請繼續閱讀以下步驟說明。*

## 步驟說明

### Step 1: 匯入套件並初始化

```python
import castle
import matplotlib.pyplot as plt
import pandas as pd

# 初始化行為分析器
analyzer = castle.BehaviorAnalyzer(
    device='cuda' if castle.cuda_available() else 'cpu'  # 自動選擇 GPU 或 CPU
)

print(f"使用設備: {analyzer.device}")
print(f"可用記憶體: {castle.get_memory_info()}")
```

### Step 2: 載入範例影片

```python
# 載入小鼠開放場實驗影片
video_path = "examples/mouse_open_field_5min.mp4"

# 檢查影片基本資訊
video_info = castle.get_video_info(video_path)
print(f"影片長度: {video_info['duration']:.1f} 秒")
print(f"影片解析度: {video_info['width']}x{video_info['height']}")
print(f"幀率: {video_info['fps']:.1f} FPS")
```

!!! info "關於範例影片"
    這個影片包含一隻 C57BL/6 小鼠在 40x40cm 開放場中 5 分鐘的行為記錄。
    實驗條件：白天、頂視角度、300 lux 照明。

### Step 3: 執行行為分析

```python
# 開始分析 (這可能需要 1-3 分鐘)
print("開始分析影片，請稍候...")

results = analyzer.analyze(
    video_path=video_path,
    animal_type='mouse',           # 指定動物類型
    arena_type='open_field',       # 實驗場地類型
    show_progress=True             # 顯示進度條
)

print(f"分析完成！發現 {len(results.behaviors)} 種行為模式")
```

### Step 4: 查看分析結果

```python
# 檢視基本統計資訊
print("\n=== 基本統計 ===")
print(f"總幀數: {results.total_frames}")
print(f"分析時長: {results.duration:.1f} 秒")
print(f"平均移動速度: {results.avg_speed:.2f} cm/s")
print(f"總移動距離: {results.total_distance:.1f} cm")

# 檢視發現的行為類型
print("\n=== 發現的行為模式 ===")
for behavior in results.behaviors:
    duration = behavior.total_duration
    frequency = behavior.frequency
    print(f"- {behavior.name}: {duration:.1f}秒 ({frequency} 次)")
```

### Step 5: 視覺化結果

```python
# 1. 軌跡圖
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
castle.plot.trajectory(results)
plt.title('小鼠移動軌跡')

# 2. 熱力圖
plt.subplot(1, 3, 2)
castle.plot.heatmap(results)
plt.title('位置熱力圖')

# 3. 行為時間軸
plt.subplot(1, 3, 3)
castle.plot.behavior_timeline(results)
plt.title('行為時間軸')

plt.tight_layout()
plt.show()
```

### Step 6: 詳細行為分析

```python
# 分析特定行為：探索行為 (探頭、嗅聞)
exploration_behavior = results.get_behavior('exploration')

print(f"\n=== 探索行為分析 ===")
print(f"探索時間佔比: {exploration_behavior.time_percentage:.1f}%")
print(f"探索區域偏好:")

# 區域偏好分析 (中央 vs 邊緣)
center_time = results.center_time_percentage
edge_time = 100 - center_time

print(f"  中央區域: {center_time:.1f}%")
print(f"  邊緣區域: {edge_time:.1f}%")
print(f"  焦慮指數: {results.anxiety_index:.2f}")  # 越高表示越焦慮

# 運動模式分析
print(f"\n=== 運動模式分析 ===")
print(f"靜止時間: {results.immobility_time:.1f}秒 ({results.immobility_percentage:.1f}%)")
print(f"慢速移動: {results.slow_movement_time:.1f}秒")
print(f"快速移動: {results.fast_movement_time:.1f}秒")
```

### Step 7: 高級視覺化

```python
# 建立綜合分析圖表
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 左上：速度變化
axes[0, 0].plot(results.speed_over_time)
axes[0, 0].set_title('速度變化曲線')
axes[0, 0].set_xlabel('時間 (秒)')
axes[0, 0].set_ylabel('速度 (cm/s)')

# 右上：行為分佈圓餅圖
behavior_durations = [b.total_duration for b in results.behaviors]
behavior_names = [b.name for b in results.behaviors]
axes[0, 1].pie(behavior_durations, labels=behavior_names, autopct='%1.1f%%')
axes[0, 1].set_title('行為時間分佈')

# 左下：距離中心的距離變化
axes[1, 0].plot(results.center_distance_over_time)
axes[1, 0].set_title('距離中心的距離')
axes[1, 0].set_xlabel('時間 (秒)')
axes[1, 0].set_ylabel('距離 (cm)')

# 右下：軌跡密度圖
castle.plot.trajectory_density(results, ax=axes[1, 1])
axes[1, 1].set_title('軌跡密度分析')

plt.tight_layout()
plt.show()
```

### Step 8: 匯出結果

```python
# 匯出詳細資料到 CSV
results.save_to_csv('mouse_oft_results.csv')

# 匯出行為標註 (可用於其他軟體)
results.save_annotations('mouse_oft_annotations.json')

# 匯出視覺化圖表
castle.plot.save_summary_report(
    results, 
    'mouse_oft_report.pdf',
    include_video_frames=True  # 包含關鍵幀截圖
)

print("結果已匯出:")
print("  - mouse_oft_results.csv (詳細數據)")
print("  - mouse_oft_annotations.json (行為標註)")
print("  - mouse_oft_report.pdf (分析報告)")
```

## 🎯 結果解讀

### 正常行為模式
在健康的 C57BL/6 小鼠中，您應該觀察到：

| 行為 | 正常範圍 | 說明 |
|------|----------|------|
| **探索時間** | 60-80% | 主動探索環境 |
| **中央區域時間** | 10-30% | 反映焦慮水平 |
| **平均速度** | 3-8 cm/s | 正常活動水平 |
| **靜止時間** | 15-35% | 包括整理毛髮等行為 |

### 異常指標警示

!!! warning "注意異常模式"
    如果觀察到以下情況，可能需要進一步調查：
    
    - 中央區域時間 <5% (高焦慮)
    - 總移動距離 <500cm (活動不足)  
    - 長時間靜止 >50% (可能的健康問題)
    - 重複性刻板行為

## 📊 資料解釋範例

根據剛才的分析，假設我們得到以下結果：

```python
# 範例結果解釋
print("=== 實驗結果解釋 ===")
print(f"實驗動物表現出正常的探索行為")
print(f"中央區域停留時間 {center_time:.1f}% 在正常範圍內")
print(f"運動模式顯示健康的活動水準")

if results.anxiety_index > 0.7:
    print("⚠️  焦慮指數較高，建議檢查實驗環境")
elif results.anxiety_index < 0.3:
    print("ℹ️  動物表現較為放鬆")
else:
    print("✅ 動物行為表現正常")
```

## 🚀 下一步

完成這個快速範例後，建議您：

### 即時行動
1. **嘗試其他範例** - 探索不同的範例影片
2. **調整參數** - 嘗試修改分析參數看看效果
3. **自己的影片** - 使用您自己的實驗影片

### 深入學習
1. **了解核心概念** - 閱讀 [CASTLE 架構](../tutorials/concepts/architecture.md)
2. **學習參數調整** - 查看 [參數優化指南](../tutorials/advanced/parameter-optimization.md)
3. **探索 GUI** - 嘗試 [圖形使用者介面](gui-introduction.md)

### 進階應用
1. **批次處理** - 學習 [批次分析多個影片](../tutorials/advanced/batch-processing.md)
2. **自訂管線** - 建立 [客製化分析流程](../tutorials/advanced/custom-pipeline.md)
3. **不同物種** - 嘗試 [其他動物分析](../tutorials/species/)

## ❓ 常見問題

??? question "為什麼我的分析時間比預期長？"
    可能原因：
    - 使用 CPU 而非 GPU (GPU 可加速 5-10 倍)
    - 影片解析度過高 (建議調整至 720p)
    - 記憶體不足導致頻繁交換檔案

??? question "如何提高分析準確度？"
    建議方法：
    - 確保影片品質清晰
    - 調整適當的動物類型參數
    - 根據實驗環境調整背景分割參數
    - 使用更高品質的攝影設備

??? question "可以分析多隻動物嗎？"
    是的！CASTLE 支援多動物分析：
    ```python
    analyzer = castle.BehaviorAnalyzer(max_animals=2)
    results = analyzer.analyze(video_path, animal_count=2)
    ```

---

🎉 **恭喜！** 您已經成功完成第一個 CASTLE 行為分析。

**準備探索更多功能？** [前往教學課程 →](../tutorials/){ .md-button .md-button--primary }