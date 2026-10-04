# BestConnectX

[English](README.md) · [操作速查](docs/cheatsheet.zh-TW.md) · [原生引擎](docs/native-engine.zh-TW.md) · [檢查與量測](docs/migration-validation.zh-TW.md)

本機、從頭訓練的連棋 bot。**C++ 執行完整對弈及模型推論；Python／PyTorch 更新模型並管理實驗。** 瀏覽器負責成果檢視及對戰，不提供訓練控制。

預設為 **15×15 自由五子棋、64 通道、6 個殘差區塊**。殘差區塊將輸入加回學習結果；寬度及深度可透過設定調整。不附帶已訓練 bot 權重。

五子棋及六子棋也可選用 `pooled` 與 `attention` 模型，保留全部八個遊戲特徵、原生推論及安全續訓。見[模型 presets 與檢查](docs/model-options.zh-TW.md)及[架構決策](docs/adr/0002-shared-board-model-experiments.zh-TW.md)。

## 安裝與建置

```sh
uv sync --locked
uv run python -m engine.native
uv run python dashboard.py --open-browser
```

必須建置原生引擎。Windows 使用已安裝的 MSVC x64 C++ 工具，以及 PyTorch wheel 內的標頭／程式庫。本路徑不需要另裝 CUDA SDK、CMake 或匯出模型。已使用 PyTorch 2.11.0+cu128 驗證 Windows CPU／CUDA。Unix 建置路徑需要 C++17 編譯器，本機未驗證該路徑。

Dashboard 網址為 **http://127.0.0.1:8765**，Ctrl+C 關閉。另一個 dashboard 可用 `--port 8766`。所有指令在 repo 根目錄執行；修改原生來源或 PyTorch 後須重新建置。

## 訓練、停止、續訓

以下指令**由你執行時**才啟動訓練：

```sh
uv run python train.py
uv run python -m engine status --data data/connect5-15x15-s1-o1-v1
uv run python -m engine stop --data data/connect5-15x15-s1-o1-v1
```

Ctrl+C 或 `stop` 會要求安全 checkpoint。等待保存訊息後再關閉。強制終止可能遺失上次保存後的工作。

重複訓練指令即可續訓。自訂 run 使用 `train.py --data PATH` 時會恢復已保存的棋規、模型及學習設定。明確指定的執行時限、裝置、工作執行緒數及棋局並行量可調整；不同模型形狀或棋規必須使用新的 run 目錄。

```sh
uv run python train.py --preset gomoku-large --data data/larger-model
uv run python train.py --data data/larger-model --hours 0.25 --workers 2
uv run python dashboard.py --data data/larger-model --open-browser
```

設定見 [`configs/experiments.toml`](configs/experiments.toml)。`--config FILE` 選擇其他檔案；`--preset NAME` 選擇命名 preset。新 run 的明確 CLI 旗標覆寫設定檔。不承諾精確重跑，也不提供舊 checkpoint 匯入器。

## 其他棋規 presets

```sh
# 第一盤檢查：不更新模型、不保存棋譜
uv run python -m engine selfplay --preset tictactoe --games 1 --device cpu

# 要求原生引擎完成棋局；只輸出摘要，不產生落子檔案
uv run python -m engine selfplay --preset gomoku --games 10 --model heuristic --device cpu

# 後續六子棋實驗
uv run python train.py --preset connect6
uv run python dashboard.py --preset connect6 --open-browser
```

僅支援正方形棋盤，邊長 2..25，`--connect` 為 2..邊長。四個方向中連成指定長度**或更長**即勝，沒有禁手或交換開局。黑棋首回合及後續回合各允許一或兩子；獲勝立即停止，包含兩子回合只下第一子的情況。不同棋規使用獨立預設 run 目錄。

## 對弈與學習

```text
C++：棋規 → 戰術及經檢查的證明 → 批次 PUCT／推論 → 真實終局
Python：近期訓練樣本 → 模型更新 → 安全 checkpoint 及凍結模型
C++：安全點接收更新權重 → 下一個對弈批次
```

**MCTS**（蒙地卡羅樹搜尋）探索後續局面並回傳價值。**PUCT** 是其中的分支選擇規則，在估計價值、模型落子先驗及訪問次數之間取得平衡。**TSS** 是威脅空間搜尋，用來尋找強制獲勝計畫。攻方搜尋有界；獨立檢查必須涵蓋所有可阻擋攻擊的回應，並排除對手先獲勝。未完成證明代表未知，不代表必敗。證明只引導落子；訓練標籤只取自真實終局。

- C++ 推論使用 FP32；CUDA 學習使用 BF16，CPU 學習使用 FP32。
- 模型更新與棋局生成交替執行，搜尋批次期間不更新權重。
- Loss 為策略交叉熵加上價值均方誤差。學習時抽樣八種旋轉／反射，不儲存八份資料。
- 預設 64 盤並行、6 個 CPU 工作執行緒、每落子 64 次模擬、optimizer batch 128。這些是不同設定；更多執行緒不一定更快。
- 保留有效時間 80% 訓練／20% 評估目標，每四秒訓練累積一秒評估額度。沒有待評估 gate 時，全部有效時間用於訓練。
- 每完成 1,000 盤保存凍結模型。第一個是初始比較基準，不代表已證實棋力。
- 後續模型以 50 組開局、交換顏色，共 100 盤挑戰目前最佳。升級須得分超過 50%，且完整回合超時不超過 0.1 秒；和棋得 0.5 分。預設每完整回合思考 0.25 秒。
- Gate 可續跑，資料不進入訓練 replay。凍結權重、棋規及程式碼檢查保護每次比較。只有修改程式碼或上限後，才用 `--restart-gate` 封存並重啟未完成 gate。

Dashboard 顯示實驗／模型資訊、搜尋／推論／學習／保存實測時間、loss 與 gate 圖，以及最近 1,000 盤已完成訓練棋局摘要。摘要只含勝者、長度、回合數及落子來源計數，不含落子或棋盤歷史。自我對弈勝率描述資料，不是棋力。對戰前先停止訓練。支援雙方顏色、完整 bot 回合，以及 best／latest／凍結模型選擇。方向鍵移動焦點，Enter／Space 落子。請求限本機並使用同源 token。

## 儲存

| Run 內路徑 | 用途 |
| --- | --- |
| `run.json` | 已保存棋規、模型形狀、設定及環境資訊 |
| `latest.pt` | 權重、optimizer、有界 replay、進行中棋局／證明計畫、待更新／metrics、計數器及 RNG |
| `selfplay-stats.json` | 最多 1,000 盤摘要，checkpoint 後發布 |
| `models/`、`incumbent.json` | 凍結模型與目前最佳版本 |
| `metrics/`、`gate-model-*.json`、`gate-archive/` | Loss 資料、可續跑評估棋譜及已封存 gate 報告 |

不產生永久 `selfplay/` 棋譜封存、逐局 `replay/` 匯出或原始碼 ZIP。評估棋譜仍保留供 gate 檢查；近期 replay 樣本及未完成棋局可以保存在 checkpoint。

保存為原子操作，預設磁碟上限 20 GiB。容量不足時保留前一份 checkpoint，不默默清理資料。有界 replay 淘汰舊樣本是正常學習行為，不是刪除既有檔案。生成的 run 與 `.native-cache/` 不進 Git；刪除後無法由 Git 還原。

## 檢查

```sh
uv run python -m unittest discover -s tests -v
uv run --group browser python tests/ui_smoke.py
uv run python -m experiments.benchmark_native --device cpu --output .native-cache/checks.json
```

瀏覽器檢查使用已安裝 Chrome、CPU 及暫存 run，不下載瀏覽器。[驗證報告](docs/migration-validation.zh-TW.md)記錄分階段檢查及配對搜尋時間。短測試**不能**證明每訓練小時的棋力增益。

現行需求：[已核准共識](docs/migration-requirements.zh-TW.md)。模型設計：[研究比較](docs/model-options.zh-TW.md)。[詞彙表](CONTEXT.zh-TW.md)定義遊戲共用語言。過時的 Python 引擎報告及效能測試已移除；獨立 Python 參考實作仍保留供正確性測試。
