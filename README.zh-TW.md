# BestConnectX

[English](README.md) · [指令速查](docs/cheatsheet.zh-TW.md) · [Bot 怎麼想](figures/how-bot-thinks.html)

在自己的電腦上訓練連棋 bot，再用瀏覽器跟它下棋。**C++ 負責棋規、搜尋和模型推論；Python／PyTorch 負責學習和存檔。** 專案沒有附上訓練好的權重。

預設是 **15×15 自由五子棋**，搭配 **64 通道、6 個殘差區塊的模型**。也支援六子棋，以及兩種實驗中的模型設計。

## 先跑起來

在專案資料夾執行：

```sh
uv sync --locked
uv run python -m engine.native
uv run python dashboard.py --open-browser
```

Dashboard 會開在 **http://127.0.0.1:8765**。語言選單可切換 **English／繁體中文**，瀏覽器會記住你的選擇。還沒訓練時，bot 用的是沒有學習權重的規則式選點。瀏覽器用來下棋、看結果；訓練則在終端機執行。

Windows 需要 MSVC x64 C++ 建置工具。建置直接使用 PyTorch 附的程式庫，不必另外裝 CUDA SDK 或 CMake。已用 PyTorch 2.11.0+cu128 檢查 Windows CPU／CUDA。Unix 路徑需要 C++17 編譯器，目前還沒在這裡驗證。改過原生程式或 PyTorch 後，記得重新建置。

## 訓練 15×15 六子棋

`connect6` preset 預設是 **19×19**。要訓練 15×15，別漏掉棋盤大小：

```sh
uv run python train.py --preset connect6 --board-size "15*15" --data data/connect6-15x15 --device cuda --hours 2
uv run python dashboard.py --data data/connect6-15x15 --open-browser
```

黑棋第一回合下一子，之後每回合下兩子。連成六子或更多就贏；一旦獲勝，回合立刻結束，不必把第二子下完。

如果這個目錄已經有 checkpoint，會接著續訓。想從頭做新實驗，就換一個 `--data` 目錄。沒有 CUDA 的話，改用 `--device cpu`。

要跑預設五子棋，只要：

```sh
uv run python train.py
```

## 停下來，之後再接著練

在訓練的終端機按 **Ctrl+C**，等到看到 **Saved** 再關閉。也可以在另一個終端機要求停止：

```sh
uv run python -m engine status --data data/connect6-15x15
uv run python -m engine stop --data data/connect6-15x15
uv run python train.py --data data/connect6-15x15 --device cuda --hours 2
```

續訓會恢復存好的棋規、模型、optimizer 和學習設定。你可以改這次的執行時間、裝置、CPU 工作執行緒和同時進行的棋局數。**但重新傳入 `--batch` 或 `--learning-rate`，不會改掉存檔裡的值。** 換棋規或模型大小，需要新目錄。

如果續訓提示未完成比賽的程式碼已變更，再看 [`--restart-gate`](docs/cheatsheet.zh-TW.md) 的說明。一般續訓不用加它。

## 訓練時到底在做什麼？

1. 最新模型自己跟自己下棋。C++ 先檢查戰術，再搭配神經網路做 MCTS 搜尋。
2. 每個已完成棋局的局面，都把**八種旋轉／反射視角**放入 replay。棋盤和策略一起變換，真實終局結果不變。新 run 預設 **400,000 筆**，可保留 50,000 個原始局面；每次學習仍抽原本的 batch 大小。
3. Python 用這些樣本更新模型。CLI 顯示的 **optimizing** 就是在學習，不是在打比賽。
4. 預設每完成 **10,000 盤**，保存一份不再更改的對弈模型。既有 run 沿用保存的間隔。檔名數字算的是棋局，不是學習更新次數。
5. 第一份里程碑先當初始 best。每次開始新比賽，都用**最新固定里程碑**挑戰目前 best，不再依序補比積壓的舊候選。已開始的比較先用原模型完成，全部里程碑檔案仍保留。

每次比賽有 **100 盤**：50 組相同開局，交換黑白。候選要拿到**至少 55 分**、全部棋譜通過檢查，而且完整回合超時最多 **0.1 秒**，才會升級。贏一盤得 1 分，和棋得 0.5 分。預設每完整回合思考 **0.25 秒**，包含六子棋的兩次落子；雙方使用相同模擬上限，預設 64 次。

這是實用的退步篩檢，**不是棋力變強的統計證明**。比賽沒過，也不會把訓練中的模型退回舊版。自我對弈繼續用最新權重，只有 best 保持不變。

初始基準之後，每到一個里程碑，訓練就暫停，直到**整場 100 盤比賽**完成，才恢復自我對弈與學習。安全停止或本次時間到期會保存未完成比賽；續跑時先比完，再繼續學習。

想用圖理解搜尋？打開 [Bot 怎麼想](figures/how-bot-thinks.html)。**MCTS** 探索接下來的局面；**PUCT** 決定先探索哪條分支；**TSS** 尋找必勝戰術，但找到後還要獨立檢查。

MCTS 的訪問次數相同時，先看模型的先驗機率；連先驗也相同、或戰術選擇等價時，就用帶 seed 的隨機選擇。TSS 會重用已搜尋完的相同局面，並在原本的時間／節點上限內依序嘗試 **2、3、4 個攻方回合**。超時只代表未知，接著照常選點。

## 模型檔案該用哪一個？

| Run 目錄內的檔案 | 拿來做什麼 |
| --- | --- |
| `latest.pt` | 接著訓練：包含權重、optimizer、近期樣本、未完棋局、待辦更新和隨機狀態 |
| `best.pt` | 跟目前接受的模型下棋；第一份基準還沒驗證棋力 |
| `models/model-00010000.pt` 等 | 跟固定里程碑下棋；不能恢復完整學習狀態 |
| `incumbent.json` | 記錄目前哪份里程碑才是 best，以這份為準 |
| `run.json`、`status.json` | 保存的設定和目前進度 |
| `selfplay-stats.json`、`metrics/` | 近期棋局摘要和 loss 歷史 |
| `gate-model-*.json`、`gate-archive/` | 可接著跑的比賽棋譜和已封存的比較 |

`latest.pt` **每個模型里程碑只保存一次**（新 run 預設每完成 10,000 盤），安全停止時也會保存。啟動或續訓不會重寫，里程碑之間不再定時保存。即時進度仍持續更新；棋局摘要和 loss 歷史隨 checkpoint 發布。停電或強制關閉，會失去上次完整存檔之後的工作。

存檔採用原子替換。預設磁碟上限是 **20 GiB**；滿了就停止，不會偷偷刪掉舊模型。下完的自我對弈棋譜不另外封存，近期樣本和未完棋局留在 checkpoint；比賽棋譜則保留給驗證用。如果 `best.pt` 匯出中斷，續訓會依 `incumbent.json` 修復。

## 下棋、調設定、跑檢查

要讓 dashboard 的 bot 下棋，先暫停訓練。可以選 best、latest 或編號里程碑，以及每完整回合 **0.5、1、2、4、8、16 秒**的思考時間。六子棋 bot 會完成整個回合。方向鍵移動焦點，Enter／Space 落子。Dashboard 用 Ctrl+C 關閉；要再開一個就加 `--port 8766`。請求限本機，並使用同源 token。

起始設定是 **6 個 CPU 工作執行緒、64 盤同時進行、每次落子 64 次模擬、學習 batch 128**。這些是不同控制，不是同一種 batch。執行緒更多、GPU 更忙，不代表 bot 一定更強。預設學習率固定 **0.001**，沒有自動排程。

新 run 可以改 [`configs/experiments.toml`](configs/experiments.toml)，或直接傳入旗標。`pooled` 和 `attention` 是選用實驗，請各自使用新目錄。棋盤必須是正方形，邊長 2–25。五子棋沒有禁手或交換開局，超過五子也算贏。

```sh
uv run python -m unittest discover -s tests -v
uv run --frozen --group browser python tests/ui_smoke.py
```

瀏覽器檢查使用已安裝的 Chrome、CPU 和暫存 run，不必下載瀏覽器。Python 參考實作還留著，是因為測試要用它檢查 C++，不是正式對弈的備援引擎。

- [操作速查](docs/cheatsheet.zh-TW.md)：直接可用的指令、presets 和續訓規則。
- [原生引擎](docs/native-engine.zh-TW.md)：程式分工和安全檢查。
- [模型選擇](docs/model-options.zh-TW.md)：殘差、池化和注意力。
- [學習率](docs/learning-rates.zh-TW.md)：哪些時候可以改。
- [訓練決策與量測](docs/selfplay-training-guidance-2026.zh-TW.md)：存檔開銷和短時間速度比較。
- [驗證紀錄](docs/migration-validation.zh-TW.md)、[核准的移植範圍](docs/migration-requirements.zh-TW.md)和[詞彙表](CONTEXT.zh-TW.md)。
