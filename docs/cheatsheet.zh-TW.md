# BestConnectX 操作速查

[English](cheatsheet.md) · [README](../README.zh-TW.md)

從 repo 根目錄執行。只有執行訓練指令才會開始訓練。

## 建置與第一盤

```sh
uv sync --locked
uv run python -m engine.native
uv run python -m engine selfplay --preset tictactoe --games 1 --device cpu
```

Windows 需要已安裝的 MSVC x64 C++ 建置工具。建置使用配套 PyTorch wheel；修改原生程式或 PyTorch 後重新建置。

## 預設：15×15 五子棋

附帶設定以本機 Windows 硬體為起點：**多核心 CPU、系統記憶體、支援 CUDA 的 GPU**。`device = "auto"` 在本機選擇 CUDA。已測試的起始值是 6 個原生工作執行緒、64 盤並行、batch 128，以及 64 通道／6 區塊的殘差模型。Replay 視窗為 20,000 個局面，磁碟上限為 20 GiB。這些是實用預設，不是已證明的最佳設定；注意力仍為獨立實驗。小型 CPU 檢查可用 `--device cpu --workers 1`。

```sh
# 開始或續訓；兩小時上限、20 GiB 容量限制
uv run python train.py

# 成果與對戰，不控制訓練
uv run python dashboard.py --open-browser

# 查看狀態或要求安全停止
uv run python -m engine status --data data/connect5-15x15-s1-o1-v1
uv run python -m engine stop --data data/connect5-15x15-s1-o1-v1
```

Ctrl+C 也會要求安全保存。等待保存訊息後再關閉。瀏覽器網址為 **http://127.0.0.1:8765**；對戰前先停止訓練。方向鍵移動焦點，Enter／Space 落子。

## 選擇棋規與模型大小

| Preset | 棋規 | 通道／區塊 |
| --- | --- | --- |
| `tictactoe` | 3×3、連三、一子回合 | 8 / 1 |
| `gomoku` | 15×15、連五、一子回合 | 64 / 6 |
| `gomoku-small` | 相同五子棋棋規 | 32 / 2 |
| `gomoku-large` | 相同五子棋棋規 | 128 / 10 |
| `connect6` | 19×19、連六；開局一子，之後每回合兩子 | 64 / 6 |
| `gomoku-pooled`、`gomoku-attention` | 使用實驗模型的五子棋 | 64 / 6 |
| `connect6-pooled`、`connect6-attention` | 使用實驗模型的六子棋 | 64 / 6 |

```sh
uv run python train.py --preset gomoku-large --data data/large
uv run python train.py --preset connect6
uv run python dashboard.py --preset connect6 --open-browser

# 實驗用池化價值／單一注意力區塊：使用獨立 run
uv run python train.py --preset gomoku-attention --data data/gomoku-attention
uv run python train.py --preset connect6-attention --data data/connect6-attention
```

相同棋規的不同模型大小會共用預設棋規目錄。**模型大小實驗須以 `--data` 指定獨立目錄。** 獲勝立即停止，包含兩子回合只下部分棋子的情況。

架構包括 `residual`（預設）、`pooled`（全域價值頭）及 `attention`（池化價值加一個四-head 區塊）。注意力通道數須為四的倍數。新架構須新 run；續訓恢復保存的選擇。見[模型細節及檢查](model-options.zh-TW.md)與 [ADR](adr/0002-shared-board-model-experiments.zh-TW.md)。

## 簡單設定

使用 [`configs/experiments.toml`](../configs/experiments.toml)，或複製成自己的設定檔。修改 `[defaults]` 或 `[presets.NAME]`：

```toml
[presets.my-model]
channels = 96
blocks = 8
workers = 2
parallel = 32
```

```sh
uv run python train.py --config configs/experiments.toml --preset gomoku-small --data data/small --simulations 32
uv run python train.py --channels 96 --blocks 8 --data data/model-96
```

自己的檔案及新增 preset 使用 `--config FILE --preset my-model`。新 run 的明確 CLI 旗標覆寫設定檔；未知鍵或錯誤型別會被拒絕。仍可使用正方形棋盤旗標，例如 `--board-size "15*15" --connect 5`。

## 學習率

**新 run** 可設定 TOML `learning_rate`，或使用：

```sh
uv run python train.py --data data/gomoku-lr3e4 --learning-rate 0.0003
```

預設為固定學習率 `0.001` 的 AdamW。續訓即使傳入 `--learning-rate`，也會恢復已保存的學習率；修改設定檔不會更改它。沒有自動排程，也沒有支援續訓時覆寫學習率的介面。`0.0003` 是範例，不是已量測的最佳值。見[學習率及較新方法](learning-rates.zh-TW.md)。

## 續訓與常用上限

```sh
uv run python train.py --data data/large --hours 0.25 --workers 2 --parallel 32
uv run python dashboard.py --data data/large --port 8766 --open-browser
```

| 旗標 | 意義／預設 |
| --- | --- |
| `--hours 0.25` | 15 分鐘 session；預設兩小時 |
| `--device cpu` | CPU 檢查；`auto` 在可用時選擇 CUDA |
| `--workers 6` | 原生 CPU 工作執行緒，不是並行棋局 |
| `--parallel 64` | 並行自我對弈棋局，不是 optimizer batch |
| `--batch 128` | 學習 minibatch |
| `--learning-rate 0.001` | 新 run 的 AdamW 基礎學習率；續訓恢復保存值 |
| `--simulations 64` | 每落子搜尋模擬數 |
| `--channels 64 --blocks 6` | 模型寬度與殘差深度 |
| `--architecture attention` | 實驗用池化價值加單一注意力區塊；預設 `residual` |
| `--replay-limit 20000` | 近期訓練樣本數上限 |
| `--max-games 1000` | 到此**累計**棋局數停止，不是額外 1,000 盤 |
| `--seconds 0.25` | Gate 每完整回合上限，不是每子上限 |
| `--disk-gib 20` | 磁碟容量上限；停止，不刪除檔案 |

續訓恢復模型、棋規及學習設定。執行時限、裝置、工作執行緒及並行量可更改；新棋規或模型形狀須新 run。`--seconds` 更改未來 gate，上限變更涉及未完成 gate 時須明確使用 `--restart-gate`；一般續訓不使用此旗標。

## 要求 X 盤完整棋局，不學習

```sh
uv run python -m engine selfplay --preset gomoku --games 10 --parallel 8 --workers 2 --model heuristic --device cpu
uv run python -m engine selfplay --data data/connect5-15x15-s1-o1-v1 --games 10 --model latest
```

`selfplay` 輸出摘要 JSON，不保存棋譜或訓練狀態；`--parallel`、`--workers`、`--tactical-ms` 亦可覆寫生成設定。需要 checkpoint 及續訓時使用 `train.py`。非預設棋規的已保存遊戲，`selfplay` 須傳入相符 preset 或棋規旗標。

## 檔案與檢查

`latest.pt` 保存有界 replay、進行中棋局、權重、optimizer 及續訓狀態；`selfplay-stats.json` 只有最近 1,000 盤摘要。保留凍結模型及評估棋譜，不建立已完成自我對弈棋譜封存或永久逐局 replay 匯出。

```sh
uv run python -m unittest discover -s tests -v
uv run --group browser python tests/ui_smoke.py
uv run python -m experiments.benchmark_native --device cpu --output .native-cache/checks.json

# 使用者啟動的 GPU 檢查，限制程序時間
uv run python -m experiments.benchmark_native --device cuda --batch 64 --simulations 64 --wall-seconds 90 --output .native-cache/cuda-checks.json

# 完成既有凍結 gate，不學習；略過 80/20 排程
uv run python -m engine evaluate --data data/connect5-15x15-s1-o1-v1 --report gate-model-00002000.json
```

瀏覽器檢查需要已安裝 Chrome，不下載瀏覽器。可用 `CHROME_EXECUTABLE` 指定已安裝的瀏覽器。測試及自我對弈勝率不能證明棋力，見[檢查與量測](migration-validation.zh-TW.md)。
