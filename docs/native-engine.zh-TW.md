# 引擎怎麼分工

[English](native-engine.md) · [指令速查](cheatsheet.zh-TW.md) · [圖解](../figures/how-bot-thinks.html)

簡單說：**C++ 下棋，Python 訓練和管理 run。** 沒有另外的 actor 服務，也不會一邊搜尋、一邊讓 learner 改權重。

## 想看程式，從哪裡找？

| 檔案 | 工作 |
| --- | --- |
| `engine/runtime.h`、`engine/runtime.cpp` | 棋規、棋盤、威脅、合法動作、戰術搜尋、證明檢查、MCTS／PUCT、工作執行緒和完整 bot 回合 |
| `engine/inference.cpp` | LibTorch 模型和 CPU／CUDA 推論 |
| `engine/native.cpp` | PUCT 選點，以及有參考比對的特徵／計數核心 |
| `engine/native.py`、`engine/runtime.py` | 建置、載入程式庫，透過 `ctypes` 傳遞具型別陣列 |
| `engine/network.py`、`engine/selfplay.py` | 定義 learner、準備樣本、更新權重；也保留測試用參考對弈 |
| `engine/game.py`、`engine/search.py`、`engine/tactics.py`、`engine/tss.py` | 共用型別和獨立 Python 正確性參考實作 |
| `engine/training.py`、`engine/generation.py` | 訓練、比賽、存檔，以及要求完整棋局 |
| `engine/config.py`、`engine/cli.py`、`engine/web.py` | TOML presets、指令和本機 dashboard |

Python 保管可存檔的棋局狀態。正式決策由 C++ 做：複製搜尋用棋盤、落子、跨棋局準備特徵，再整批推論。Python 的停止 callback 只檢查該不該停，不負責下棋。

JSON bot 指令會在 C++ 裡完成一／兩子的整個回合。訓練和比賽則會在落子之間交還控制，讓未完棋局能保存。Python 對弈參考實作是測試工具，不是正式備援路徑。

## 建置和同步權重

```sh
uv sync --locked
uv run python -m engine.native
```

快取鍵包含原生來源／標頭內容和 PyTorch 版本。Windows 使用 MSVC `/MD` 和已安裝 PyTorch 的程式庫，先由 PyTorch 載入 runtime，再載入原生程式庫。Windows CPU／CUDA 已檢查，Unix 路徑還沒驗證。

Python 和 C++ 的層名稱、形狀要一致。學習後，下一批棋局生成會把權重和 batch normalization buffers 複製到獨立原生模型；搜尋期間保持不變。比賽用的模型則在整次比較中都固定。原本殘差模型的 C 建立介面保留不變，變體使用另一個入口。

全部模型都吃相同八個遊戲特徵平面。`residual` 是預設；`pooled` 換掉價值頭；`attention` 再加一個四-head 棋盤注意力區塊。格式 2 的 checkpoint 沒有架構欄位，就當作 residual。換架構要新 run，詳見[模型選擇](model-options.zh-TW.md)。

原生推論用 **FP32**（32 位元浮點數）。CUDA 學習透過 PyTorch autocast 用 **BF16**，CPU 學習仍用 FP32。這條路徑沒有 TorchScript、AOTInductor、CUDA graph 或自訂 GPU kernel。

## 設定和學習

[`configs/experiments.toml`](../configs/experiments.toml) 放預設值和 presets。新 run 的明確旗標會覆寫檔案值。續訓沿用已保存的學習／搜尋設定；可以明確改執行緒數或並行量，不必丟掉未完棋局。換模型或棋規要新 run，也不保證計時實驗每次完全一樣。

| 設定 | 允許範圍／預設 |
| --- | --- |
| 棋盤邊長／連線長度 | 2–25／2–邊長；只支援正方形 |
| 開局／後續回合棋子數 | 各為 1 或 2 |
| 通道／殘差區塊 | 4–256／0–32；預設 64 / 6 |
| 同時進行棋局／CPU 執行緒 | 1–128／1–12；預設 64 / 6 |
| 學習 batch | 2–4096；預設 128 |
| 近期 replay | 1–200,000 個局面；新 run 預設 50,000，續訓保留原上限 |

Learner 用 AdamW，預設學習率固定 **0.001**，weight decay 是 **0.0001**。一般每完成八盤觸發 32 次更新，較大的完成批次會按比例往上增加工作。預設前 16 盤用規則式選點。學習時抽近期已完成棋局，旋轉／反射棋盤，再降低策略交叉熵和價值均方誤差。[學習率設定](learning-rates.zh-TW.md)適用新 run，不會覆寫續訓值。

## 安全檢查不會省掉

- **證明要獨立檢查。** 攻方搜尋可以限制候選數，驗證端則另行重建能阻擋攻擊的回應，包含合法第二子補位，也要檢查對手是否先贏。無效證明不能拿來下棋；時間或節點用完只代表未知，改走一般搜尋。
- **勝負來自真的終局。** 戰術證明只引導落子，價值標籤要等棋局真的勝、和、負。
- **先保存恢復狀態，再發布摘要。** 格式 2 的 `latest.pt` 包含權重、optimizer、有上限的 replay、未完棋局／計畫、待辦工作、計數和隨機狀態。發布摘要或 metrics 失敗，可從它重試。舊格式會被拒絕，沒有舊 run 匯入器。
- **約一分鐘存一次，不是每輪學習都存。** 棋局生成／更新的安全點可自動保存，長學習週期內也能保存。每次成功保存後才重設計時器；啟動、里程碑、正常停止仍會存，長操作可能延後。
- **Best 是對弈模型，不是續訓檔。** `incumbent.json` 選定固定里程碑，`best.pt` 原子匯出它。續訓可修復缺少或中斷的匯出。後續升級要在 100 盤經驗證的配對對局拿至少 55 分，回合超時最多 0.1 秒；第一份基準還沒驗證棋力。
- **不偷偷刪檔。** 保留原子替換、獨占 run lock、磁碟上限、固定里程碑和比賽棋譜檢查。不建立永久的已完成自我對弈棋譜封存。

重構程式可能讓未完成比賽的來源 checksum 不符。續訓提示這個問題時，再走明確的[重啟流程](cheatsheet.zh-TW.md)。不要混用新舊程式碼的結果，也不要自動重啟 gate。

## 時間數字怎麼看？

搜尋時間包含戰術和特徵準備。推論時間包含傳輸和等結果，不只是 GPU kernel。學習、存檔另外計時。權重同步／載入不算在暖機後的搜尋中位數，但會算進整個效能測試程序的預算。

最近的存檔頻率比較，把 checkpoint 次數從 12 降到 2，短時間吞吐量約快 38%。這不是新的搜尋速度或棋力結果，詳見[訓練量測與回歸檢查](selfplay-training-guidance-2026.zh-TW.md)。多執行緒可能幫到大批次，也可能拖慢小批次；選設定請看[實測結果](migration-validation.zh-TW.md)，別只看使用率。
