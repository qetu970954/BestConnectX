# 想改學習率，怎麼做？

[English](learning-rates.md) · [指令速查](cheatsheet.zh-TW.md) · [網路研究](network-research-2026.zh-TW.md)

**新 run 可以選學習率；續訓沿用保存值。** 預設是固定學習率 **0.001** 的 AdamW，weight decay 為 **0.0001**，沒有自動學習率排程。

學習率控制每次權重更新的幅度。AdamW 也會參考梯度歷史，但還是需要選一個基礎學習率。策略頭和價值頭共用同一個 optimizer、同一個 rate。詳見 [PyTorch AdamW 文件](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html)。

## 開一個學習率實驗

分別使用新目錄：

```sh
uv run python train.py --data data/gomoku-lr3e4 --learning-rate 0.0003
uv run python train.py --preset connect6 --board-size "15*15" --data data/connect6-15x15-lr3e4 --learning-rate 0.0003
```

也可以放在 preset：

```toml
[presets.gomoku]
learning_rate = 0.0003
```

新 run 的明確旗標會覆寫 TOML。接受範圍是大於零、不超過 0.1 的有限數值；這是輸入上限，**不是建議的調參範圍**。`0.0003` 只是範例，還沒量出它對哪個棋規最好。

## 續訓不會換掉學習率

```sh
uv run python train.py --data data/gomoku-lr3e4
```

這會恢復保存的學習設定和 AdamW 狀態，包含學習率和梯度歷史估計。加上 `--learning-rate 0.0001` 也不會覆寫；改 TOML 或 `run.json` 同樣不會改 checkpoint。目前沒有支援續訓時覆寫學習率的方式。

一次暫存 CPU 檢查確認了這點：新井字棋 run 的 TOML 0.0002 被 CLI 0.0003 覆寫。完成八盤／兩次更新後，帶 CLI 0.0001 續訓，設定和 optimizer groups 都仍是 0.0003。既有 run 沒有被改動。這是在驗證行為，不是學習品質。程式見[設定](../engine/config.py)、[CLI](../engine/cli.py)、[訓練](../engine/training.py)和[更新](../engine/selfplay.py)。

## 需要換新 optimizer 或排程嗎？

不能只因為 loss 低就換。Bot 邊學，自己的對弈資料也邊變；訓練 loss 低，仍可能在其他局面預測不準、下得不強。見[五子棋先手必勝的結果和限制](gomoku-first-player.zh-TW.md)。

PyTorch 支援 [cosine 衰減](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html)和 [ReduceLROnPlateau](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html)，但本引擎都沒用。Plateau 規則需要穩定的驗證目標，不能只盯著一直變動的自我對弈 loss。

[Schedule-free optimization](https://github.com/facebookresearch/schedule_free)省掉的是衰減排程，不是學習率調整。還要處理 optimizer 的 train／eval 和 batch normalization buffers，會影響原生權重同步、固定模型和續訓，不是直接替換就能升級。

先把固定學習率 AdamW 當基準。要比較 0.001 和 0.0003，用獨立 run，保持模型、搜尋預算、更新比例和總訓練時間相同。五子棋、六子棋分開比較保留局面的價值誤差和交換黑白對戰。這是實驗提案，不是較低學習率比較強的證據。

之後若加排程，要保存排程狀態，跨續訓累計更新次數。某次 session 的 `--hours` 不是完整訓練週期。學習率、輸入和模型大小分開改，才看得出各自效果。
