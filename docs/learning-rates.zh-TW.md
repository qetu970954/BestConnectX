# 學習率與續訓

[English](learning-rates.md) · [操作速查](cheatsheet.zh-TW.md) · [網路研究](network-research-2026.zh-TW.md)

檢查日期：2026-10-04。**學習率仍然重要。本引擎可設定新 run 的學習率；續訓會恢復已保存的學習率。目前沒有學習率排程。**

## 目前行為

| 設定 | 已實作的行為 |
| --- | --- |
| Optimizer | AdamW，weight decay 為 `0.0001` |
| 預設基礎學習率 | `0.001` |
| 新 run 的設定方式 | TOML `learning_rate` 或 CLI `--learning-rate`；明確旗標優先 |
| 允許範圍 | 大於零且不超過 `0.1` 的有限數值；這是輸入驗證上限，不是實用學習率建議 |
| 策略／價值頭 | 共用一個 optimizer 及一個基礎學習率 |
| 自動調整 | 沒有 warmup、衰減、plateau 規則或各輸出頭的獨立學習率 |
| 續訓 | 恢復學習設定及 AdamW 狀態，包含學習率與梯度歷史的估計值 |

上述行為來自[設定載入](../engine/config.py)、[CLI 驗證](../engine/cli.py)、[optimizer 建立與恢復](../engine/training.py)及[權重更新](../engine/selfplay.py)。

基礎學習率控制權重更新的尺度。AdamW 依梯度歷史調整更新，但仍會使用基礎 `lr`，不代表不必選擇學習率。[目前 PyTorch AdamW 文件](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html)。

## 設定新 run 的學習率

每個實驗使用新的目錄。只有執行以下指令時才會開始訓練：

```sh
uv run python train.py --data data/gomoku-lr3e4 --learning-rate 0.0003
uv run python train.py --preset connect6 --data data/connect6-lr3e4 --learning-rate 0.0003
```

也可在選定的 TOML preset 設定：

```toml
[presets.gomoku]
learning_rate = 0.0003
```

`0.0003` 是範例，並非任一棋規下已量測的最佳值。即使使用相同 optimizer，五子棋與六子棋仍須分別評估。

既有 run 會以 checkpoint 保存的學習設定及 optimizer 狀態為準，連明確傳入的 `--learning-rate` 都會被忽略。修改 TOML 或 `run.json` 不會更改 checkpoint 中的學習率。本版沒有支援續訓時覆寫學習率的介面。一般續訓方式如下：

```sh
uv run python train.py --data data/gomoku-lr3e4
```

一次暫存 CPU 檢查使用 TOML 學習率 `0.0002`，以 CLI `0.0003` 覆寫，完成八盤井字棋及兩次 optimizer 更新，再以 CLI `0.0001` 續訓至九盤。保存的設定與 optimizer groups 都保留 `0.0003`。沒有更改既有訓練 run。此檢查驗證設定及續訓行為，不驗證學習品質。

## 較新方法改變了什麼？

**排程**會隨訓練進度調整基礎學習率。PyTorch 提供 cosine 衰減，以及在選定指標停止改善時降低學習率的規則。這些仍是受支援的選項，但本引擎沒有使用。[CosineAnnealingLR](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html)、[ReduceLROnPlateau](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html)。

Schedule-free optimization 不需要衰減排程，但作者仍要求調整學習率。其實作另需處理 optimizer 的 train／eval 狀態，並注意 batch normalization buffers。這會影響本專案的原生權重刷新、模型快照與續訓，因此不能直接替換就視為升級。[作者實作及注意事項](https://github.com/facebookresearch/schedule_free)。

## 本引擎的建議

保留固定學習率 AdamW 作為比較基準。可先在兩個新 run 比較 `0.001` 與 `0.0003`，保持模型、搜尋預算、更新比例及總訓練時間相同。依各自棋規，檢查保留局面的價值誤差，以及交換黑白的對戰結果。這是實驗提案，不是降低學習率會改善棋力的證據。

價值 loss 低，不足以構成降低學習率的理由：bot 學習時，自我對弈局面與結果也會變動。Plateau 規則需要穩定的驗證目標；訓練 loss 很低，仍可能在其他局面預測不準。見[五子棋先手必勝的結果及限制](gomoku-first-player.zh-TW.md)。

若之後加入排程，應跨續訓累計 optimizer 更新次數，並與 AdamW 一起保存排程狀態。每次 session 的 `--hours` 上限不是完整訓練週期。學習率、新輸入表示及較大模型應分別測試，才能量測各自效果。
