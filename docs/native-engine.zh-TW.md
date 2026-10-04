# 原生引擎與實驗設定

[English](native-engine.md) · [操作速查](cheatsheet.zh-TW.md)

**Checkpoint** 是續訓所需的保存狀態。**Replay** 指近期訓練樣本。**Optimizer** 利用 loss gradients 更新權重，gradients 表示可降低 loss 的方向。**推論（inference）** 在不更改權重的情況下預測分數。

## 語言分工

| 檔案 | 用途 |
| --- | --- |
| `engine/runtime.h`、`engine/runtime.cpp` | 棋規、快取連線視窗、棋盤、威脅、合法動作、啟發式、TSS、獨立證明檢查、PUCT、抽樣、工作執行緒及完整 bot 回合 |
| `engine/inference.cpp` | C++ 殘差模型、權重 buffers，以及 LibTorch tensor 操作的 CPU／CUDA 推論 |
| `engine/native.cpp` | 既有特徵／計數及 PUCT 核心；舊 forest 保留供參考檢查 |
| `engine/native.py`、`engine/runtime.py` | 建置／載入，以及標準程式庫 `ctypes` 的具型別狀態傳輸 |
| `engine/network.py`、`engine/selfplay.py` | Python 模型定義、樣本準備及 optimizer 更新；舊 Python 對弈函式只作參考工具 |
| `engine/training.py`、`engine/generation.py` | 要求工作、計數棋局、管理學習／評估及保存狀態 |
| `engine/config.py`、`engine/cli.py`、`engine/web.py` | TOML 設定、指令及本機 dashboard |

Python 保存可序列化棋局狀態以管理工作，正式棋規計算及決策使用 C++。搜尋自行複製及落子葉節點棋盤，不呼叫 Python。工作執行緒跨根節點準備葉節點／特徵，再以一次原生呼叫評估整批。停止 callback 是管理檢查，不是棋類或模型呼叫。

JSON bot 指令在 C++ 內完成一／兩子完整回合。訓練及 gate 在安全落子點交還控制以保存未完成棋局。不建立獨立 actor 服務，也不並行執行 learner。

## 建置及模型一致性

```sh
uv sync --locked
uv run python -m engine.native
```

二進位快取鍵涵蓋所有原生來源／標頭內容及 PyTorch 版本。Windows 使用 Release 風格的 MSVC `/MD`，連結已安裝 wheel 的程式庫；先匯入 PyTorch 載入配套 runtime，再載入原生程式。已驗證 Windows CPU／CUDA，Unix 建置路徑尚未驗證。

Python 與 C++ 模型保留棋盤形狀，層名稱及形狀一致。設定包括通道數、殘差區塊數及棋盤邊長。Python 將權重及 batch normalization 的 running buffers 複製到獨立原生推論模型；optimizer 更新後，下一次棋局生成會在搜尋前刷新 buffers。凍結 gate 模型不受 learner 刷新。

**FP32** 是 32 位元浮點運算，原生 CPU／CUDA 推論使用 FP32。**BF16** 是較小的浮點格式，CUDA 學習透過 PyTorch autocast 使用。本版不使用 TorchScript、模型匯出、AOTInductor、CUDA graphs 或自訂 GPU kernel。

## 設定與上限

[`configs/experiments.toml`](../configs/experiments.toml) 包含 `[defaults]` 與 `[presets.NAME]`。鍵名對應 CLI destinations，例如 `board_size`、`channels`、`blocks`、`workers`、`parallel`。錯誤鍵、型別及範圍在訓練前被拒絕；新 run 的明確旗標覆寫檔案值。

- 棋盤邊長 2..25、連線長度 2..邊長，開局／後續回合各一或兩子。
- 模型 4..256 通道、0..32 殘差區塊。附帶 large preset 為 128/10，不是模型最大上限；大型實驗前須檢查記憶體。
- 並行根節點 1..128、原生工作執行緒 1..12、學習 batch 2..4096。
- 近期 replay 1..200,000 樣本，預設 20,000；進行中棋局樣本也占用 RAM／checkpoint 空間。
- 預設學習率 0.001、bootstrap 16 盤；每完成八盤累積 32 次更新。較大的完成批次保留同比例更新工作。

續訓恢復已保存學習／搜尋設定。明確更改工作執行緒或並行量時，保留未完成棋局並輪替進入新批次；更改模型／棋規須新 run。設定系統不承諾計時實驗可精確重現。

## 證明及保存安全

攻方候選可以排序及限制。證明檢查不依賴搜尋端 cover 生成器，獨立重建防守回應，包含合法第二子填充及對手先獲勝。無效證明不能引導落子；時間／節點上限代表未知，回到一般搜尋。價值標籤須等棋盤達到真實終局才產生。

Checkpoint 格式 2 保存有界 replay、進行中棋局與計畫／策略，以及權重、optimizer、待 metrics／更新、計數器及 RNG。拒絕舊格式，不提供舊 run 匯入器或永久已完成棋局封存。

先保存 checkpoint，再發布摘要／metrics；發布失敗時可由 checkpoint 重試。保留原子寫入、獨占 run lock、磁碟上限、不可變模型及 gate 棋譜檢查，不加入默默刪檔行為。

## 量測限制

原生時間分開統計搜尋／戰術／特徵的實際時間與推論時間。推論包含必要 CPU／GPU 傳輸及等待輸出，不只是 GPU kernel 活動。Learner 及 checkpoint 時間另以高精度時鐘量測；建置後的程序啟動及權重刷新計入外層測試程序預算，不計入暖機後穩態搜尋中位數。

更多工作執行緒可能改善大批次、降低小批次效率。依[實測結果](migration-validation.zh-TW.md)選設定，不只看 CPU／GPU 使用率。長期棋力比較仍由使用者執行。
