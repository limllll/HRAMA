# CARLA-SUMO 聯合仿真使用指南（自動模式）

## 概述

本系統實現了 CARLA 和 SUMO 的聯合仿真，CARLA 中的車輛會自動同步到 SUMO 中進行交通仿真分析。系統啟動時會自動運行兩個仿真器，每一幀都會保持同步，確保兩個仿真器的時間一致性。

## 主要功能

### 1. 單向同步 (CARLA → SUMO)
- CARLA 中的所有車輛自動在 SUMO 中生成對應的車輛
- 車輛位置、朝向每幀同步更新
- 支持車輛的動態添加和移除

### 2. 時間同步
- 使用相同的 `fixed_delta_seconds` 確保時間步長一致
- CARLA world tick 後立即執行 SUMO 同步
- 支持同步和非同步模式

### 3. 自動重置
- 每個新回合自動重置 SUMO 狀態
- 清理上一回合的所有車輛
- 重新建立車輛映射關係

## 使用方法

### 1. 啟動 T 型路口訓練（自動帶 SUMO 聯合仿真）

```bash
cd src/macad_gym/agents/train/
python Tjunction_ppo_train.py
```

### 2. 命令行參數

- `--verbose`: 啟用詳細日誌輸出 (可選)

### 3. 環境配置

在環境配置中添加以下選項：

```python
env_config = {
    "env": {
        # SUMO 聯合仿真已默認啟用，無需額外配置
        "verbose": True,            # 詳細日誌
        "sync_server": True,        # 同步模式（推薦）
        "fixed_delta_seconds": 0.05 # 時間步長
    }
}
```

## 技術實現

### 1. 橋接器架構

```
CARLA World ←→ CarlaSumoBridge ←→ SUMO TraCI
     ↓              ↓                ↓
  車輛狀態      OpenDRIVE 轉換    SUMO 網路
```

### 2. 同步流程

1. CARLA 執行動作並更新世界狀態
2. CARLA world.tick() 執行
3. 立即調用 `step_and_sync_from_carla()`
4. SUMO 執行 `simulationStep()`
5. 同步所有 CARLA 車輛到 SUMO

### 3. 車輛映射

- CARLA actor_id → SUMO vehicle_id (`carla_{actor_id}`)
- 自動處理車輛生成和銷毀
- 支持不同車輛類型的映射

## 調試和監控

### 1. 日誌輸出

啟用 verbose 模式查看詳細信息：
```
SUMO 橋接器啟動成功 (GUI: True)
SUMO 中有 3 輛車輛正在仿真
SUMO 橋接器已重置，準備新的仿真回合
```

### 2. SUMO GUI 監控

啟用 `--sumo-gui` 可以實時查看：
- SUMO 網路拓撲
- 車輛實時位置
- 交通流量統計

### 3. 錯誤處理

系統會自動處理以下錯誤：
- SUMO 連接失敗 → 降級為純 CARLA 模式
- 車輛同步失敗 → 跳過該車輛，繼續其他車輛
- 網路轉換失敗 → 使用默認網路配置

## 性能優化建議

### 1. 同步頻率
- 使用較大的 `fixed_delta_seconds` (如 0.1) 可以提高性能
- 對於實時應用，推薦 0.05 秒

### 2. SUMO 配置
- 關閉 SUMO GUI 可以顯著提高性能
- 使用較簡單的車輛模型

### 3. 網路優化
- 預先生成並緩存 SUMO 網路文件
- 使用本地 SUMO 安裝避免網路延遲

## 故障排除

### 1. SUMO 未安裝
```
錯誤：SUMO 橋接器啟動失敗
解決：安裝 SUMO 並設置環境變量 SUMO_HOME
```

### 2. 端口衝突
```
錯誤：TraCI 連接失敗
解決：檢查 SUMO 端口是否被占用，重啟系統
```

### 3. 地圖不匹配
```
錯誤：OpenDRIVE 轉換失敗
解決：確保 CARLA 地圖支持 OpenDRIVE 導出
```

## 擴展功能

### 1. 雙向同步
未來可以擴展為 SUMO → CARLA 的反向同步：
- 在 SUMO 中生成的車輛同步到 CARLA
- 交通信號燈狀態同步

### 2. 數據分析
利用 SUMO 的分析能力：
- 交通流量統計
- 路網效率分析
- 排放量計算

### 3. 多地圖支持
- 支持更多 CARLA 地圖
- 自動化網路轉換流程

## 注意事項

1. **系統要求**：需要同時安裝 CARLA 和 SUMO
2. **性能影響**：聯合仿真會增加約 20-30% 的計算開銷
3. **穩定性**：建議在生產環境中進行充分測試
4. **版本兼容**：確保 CARLA 和 SUMO 版本兼容

## 聯繫支持

如遇到問題，請提供以下信息：
- CARLA 版本
- SUMO 版本
- 錯誤日誌
- 系統配置
