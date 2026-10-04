# BestConnectX 操作速查

[English](cheatsheet.md) · [README](../README.zh-TW.md)

以下指令都在專案資料夾執行。只有跑 `train.py` 才會開始訓練。

## 建置，先檢查一盤

```sh
uv sync --locked
uv run python -m engine.native
uv run python -m engine selfplay --preset tictactoe --games 1 --device cpu
```

Windows 需要 MSVC x64 C++ 建置工具，建置會使用 PyTorch 配套的程式庫。改過原生程式或 PyTorch 後，記得重新建置。

## 15×15 六子棋：訓練、停止、續訓

**`connect6` preset 是 19×19。要跑 15×15，請加上 `--board-size "15*15"`。** 其他棋規不變：連六獲勝、開局一子，之後每回合兩子。

```sh
# 開始訓練；想從頭練新模型，就換一個新目錄
uv run python train.py --preset connect6 --board-size "15*15" --data data/connect6-15x15 --device cuda --hours 2

# 在另一個終端機開啟同一個 run 的 dashboard
uv run python dashboard.py --data data/connect6-15x15 --open-browser

# 看進度，或要求安全停止
uv run python -m engine status --data data/connect6-15x15
uv run python -m engine stop --data data/connect6-15x15

# 續訓：會恢復保存的棋規、模型和學習設定
uv run python train.py --data data/connect6-15x15 --device cuda --hours 2
```

Ctrl+C 也會要求安全停止。**看到 Saved 後再關終端機。** 兩小時限制算的是訓練迴圈，啟動和最後存檔還會多花一點時間。目錄裡已有 checkpoint，就會續訓，不會換成全新模型。想做另一個實驗，請改 `--data`。

Dashboard 網址是 **http://127.0.0.1:8765**。要跟 bot 下棋，先暫停訓練。方向鍵移動焦點，Enter／Space 落子。第二個 dashboard 可加 `--port 8766`。沒有 CUDA？把 `--device cuda` 換成 `--device cpu`；小型 CPU 檢查也可以加 `--workers 1`。

### 如果續訓提示未完成 gate 的程式碼已變更

**只有 CLI 提示需要時才用**，不是每次續訓都加：

```sh
uv run python train.py --data data/connect6-15x15 --device cuda --hours 2 --restart-gate
```

它會封存舊比賽報告，再用目前的程式碼重跑這次比較。Learner 的權重、optimizer 和訓練資料都會保留。一般續訓不用這個旗標。

## 其他 presets 和模型實驗

| Preset | 棋規／模型 |
| --- | --- |
| `tictactoe` | 3×3 連三，8 通道／1 區塊 |
| `gomoku` | 15×15 自由五子棋，64 / 6；預設選擇 |
| `gomoku-small` | 相同五子棋棋規，32 / 2 |
| `gomoku-large` | 相同五子棋棋規，128 / 10 |
| `connect6` | 19×19 六子棋，64 / 6 |
| `gomoku-pooled`、`connect6-pooled` | 相同棋規，池化價值頭，64 / 6 |
| `gomoku-attention`、`connect6-attention` | 相同棋規，池化價值加一個四-head 注意力區塊，64 / 6 |

```sh
uv run python train.py
uv run python dashboard.py --open-browser
uv run python train.py --preset gomoku-large --data data/large
uv run python train.py --preset connect6 --data data/connect6-19x19
uv run python train.py --preset gomoku-attention --data data/gomoku-attention
uv run python train.py --preset connect6-attention --board-size "15*15" --data data/connect6-15x15-attention
```

**同樣棋規會共用預設目錄，即使模型不同也一樣。** 模型實驗一定要另外指定 `--data`。換架構、模型大小或棋盤大小，都要新 run。Attention 的通道數必須是四的倍數。這些變體能運作，但還沒證明棋力更好。詳見[模型選擇](model-options.zh-TW.md)。

## 設定：哪些可以改？

起始值已在配備多核心 CPU 和支援 CUDA 的 GPU 的 Windows 桌上型電腦上檢查。為保護隱私，不公開確切硬體規格。它們是起點，不是已證明的最佳設定。`auto` 在 CUDA 可用時會選 CUDA。

| 旗標 | 用途 |
| --- | --- |
| `--hours 0.25` | 訓練迴圈可跑 15 分鐘；預設兩小時 |
| `--device cpu` | 選 CPU；也可選 `auto` 或 `cuda` |
| `--workers 6` | 原生 CPU 執行緒數；續訓可改 |
| `--parallel 64` | 同時進行的自我對弈棋局；續訓可改 |
| `--batch 128` | 每次學習更新的樣本數；續訓沿用 checkpoint |
| `--learning-rate 0.001` | 新 run 的固定 AdamW 學習率；續訓沿用保存值 |
| `--simulations 64` | 每次落子的搜尋模擬數；續訓沿用保存值 |
| `--channels 64 --blocks 6` | 模型寬度和殘差深度；更改要新 run |
| `--architecture attention` | 實驗模型設計；預設 `residual` |
| `--replay-limit 50000` | 近期已完成棋局的局面；新 run 預設 50,000，續訓沿用保存值 |
| `--max-games 1000` | 到**累計 1,000 盤**停止，不是再多練 1,000 盤 |
| `--snapshot-every 1000` | 每幾盤保存固定里程碑；續訓沿用保存值 |
| `--seconds 0.25` | 比賽每完整回合的時間，不是每顆棋子 |
| `--disk-gib 20` | 磁碟容量上限；滿了停止，不刪檔 |

新 run 最多保留 **50,000 個 replay 局面**，不是 50,000 盤；滿了會移除最舊局面。更多歷史會增加 RAM 和存檔時間，還沒證明棋力更強。已有 checkpoint 會保留原上限，即使傳另一個 `--replay-limit` 也不會覆寫。

**`--parallel` 和 `--batch` 做的是不同工作。** 改 batch 也會改變舊樣本被重複抽到的頻率。續訓時傳 `--batch 256` 或新的學習率，不會覆寫 checkpoint。這類比較請使用新目錄。詳見[學習率](learning-rates.zh-TW.md)。

新 run 的明確旗標會覆寫 [`configs/experiments.toml`](../configs/experiments.toml)。自己的設定檔用 `--config FILE --preset NAME`。未知鍵、錯誤型別或超出範圍的值會被拒絕。想續訓而不重新套用 preset 預設值：

```sh
uv run python train.py --data data/large --hours 0.25 --workers 2 --parallel 32
uv run python dashboard.py --data data/large --port 8766 --open-browser
```

`--seconds` 會改未來的比賽時間。要更改未完成比賽的上限，必須用 `--restart-gate`，避免把新舊條件下的結果混在一起。

## 只產生棋局，不學習

```sh
uv run python -m engine selfplay --preset gomoku --games 10 --parallel 8 --workers 2 --model heuristic --device cpu
uv run python -m engine selfplay --preset connect6 --board-size "15*15" --data data/connect6-15x15 --games 10 --model latest
```

這會輸出摘要 JSON，不更新權重、不保存棋譜，也不是可續訓的訓練流程。`--model` 可以選 `heuristic`、`latest` 或 `best`。已保存的非預設棋規，需要帶上相符 preset／棋規。執行前先暫停其他 GPU 工作。

## 存檔和比賽，簡單說

- **`latest.pt`：接著學。** 完整狀態，包含 optimizer、近期樣本和未完棋局。約每 60 秒在安全點保存，啟動、里程碑和正常停止時也會存；長操作可能延後。
- **編號里程碑：留下固定對手。** `models/model-00001000.pt` 代表完成 1,000 盤自我對弈，不是更新 1,000 次。它不能恢復完整 learner。
- **`best.pt`：跟接受的模型下棋。** `incumbent.json` 決定它對應哪份里程碑；匯出中斷可在續訓時修復。第一份里程碑還沒驗證棋力。
- **後續升級：100 盤至少 55 分。** 50 組開局交換黑白，棋譜要通過檢查，最大完整回合超時為 0.1 秒。勝一盤得 1 分，和棋得 0.5 分。這是暫定篩檢，不是棋力證明。
- **升級沒過，仍繼續學。** 只有 best 不變。時間目標大約是 80% 訓練／20% 評估，未完成比賽可以先等，訓練不用停住。
- **保留檔案，滿了停止。** 不自動刪掉舊里程碑。`selfplay-stats.json` 只有最近 1,000 盤摘要，不含棋譜；近期 replay 留在 checkpoint，比賽棋譜留給驗證用。

## 跑檢查

```sh
uv run python -m unittest discover -s tests -v
uv run --frozen --group browser python tests/ui_smoke.py
uv run python -m experiments.benchmark_native --device cpu --output .native-cache/checks.json

# 使用者可自行跑的 GPU 檢查，設有程序時間上限
uv run python -m experiments.benchmark_native --device cuda --batch 64 --simulations 64 --wall-seconds 90 --output .native-cache/cuda-checks.json

# 接著跑既有比賽，不學習；略過 80/20 排程
uv run python -m engine evaluate --data data/connect6-15x15 --report gate-model-00002000.json
```

瀏覽器檢查用已安裝的 Chrome 和暫存 CPU run，不下載瀏覽器。可用 `CHROME_EXECUTABLE` 指定已安裝的瀏覽器。`evaluate` 要有那份既有報告才能跑。測試通過、loss 下降、自我對弈勝率，都不等於棋力結果。詳見[驗證紀錄](migration-validation.zh-TW.md)和[訓練量測](selfplay-training-guidance-2026.zh-TW.md)。
