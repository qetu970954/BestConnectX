# 模型該選哪一種？

[English](model-options.md) · [指令速查](cheatsheet.zh-TW.md) · [圖解](../figures/how-bot-thinks.html)

**先用 `residual` 就好。** `pooled` 和 `attention` 也能用 C++ 推論，但這裡還沒證明它們棋力更好。三種都用相同八個輸入平面，都能跑五子棋或六子棋。

## 它們差在哪裡？

| 架構 | 改了什麼 | Presets |
| --- | --- | --- |
| `residual` | 空間卷積和殘差區塊；價值頭把棋盤特徵攤平 | `gomoku`、`connect6` |
| `pooled` | 主體／策略頭相同，價值頭改成全盤特徵平均 | `gomoku-pooled`、`connect6-pooled` |
| `attention` | 池化價值頭，再於殘差主體後加一個四-head 注意力區塊 | `gomoku-attention`、`connect6-attention` |

**殘差區塊**會把輸入加回學到的結果。**通道數**決定主體多寬，**區塊數**決定殘差部分多深。**策略頭**評分哪裡值得落子，**價值頭**從目前玩家的角度預測勝負。

池化頭先投影成 16 通道，對每個通道做全盤平均，再接 64 單元隱藏層。注意力區塊用 pre-normalization、可學習二維相對位置 bias、兩倍通道寬的 GELU feedforward，不用 dropout。Attention 通道數要是四的倍數。這是自己的小型實驗，不是 KataGo 或 ResTNet 的實作。

池化讓價值頭少了直接的空間細節。注意力能讓遠處格子互相影響，但會多花運算。兩者都不是免費升級，池化也不代表權重能跨棋盤大小或棋規使用。

## 指令

每個實驗都給它一個**新目錄**：

```sh
uv run python train.py --preset gomoku-pooled --data data/gomoku-pooled
uv run python train.py --preset gomoku-attention --data data/gomoku-attention
uv run python train.py --preset connect6-pooled --board-size "15*15" --data data/connect6-15x15-pooled
uv run python train.py --preset connect6-attention --board-size "15*15" --data data/connect6-15x15-attention
```

Connect6 presets 沒改大小時是 19×19。同樣棋規的變體共用預設目錄，所以模型實驗不要省掉 `--data`。也可以直接設定 `--architecture residual|pooled|attention`、`--channels` 和 `--blocks`。

續訓用 `train.py --data PATH`，不用再選另一個 preset。它會恢復保存的架構、權重和 optimizer。換架構、寬度、深度或棋盤大小，都要新 run。格式 2 的 checkpoint 沒有架構欄位，就當作 residual。沒有權重轉移、預訓練影像初始化、自動擴張或架構搜尋功能。

## 模型有多大？

從 [`Network`](../engine/network.py) 量得，固定 64 通道／6 個殘差區塊：

| 架構 | 15×15 參數 | 19×19 參數 |
| --- | ---: | ---: |
| `residual` | 477,764 | 495,172 |
| `pooled` | 450,898 | 450,898 |
| `attention` | 487,734 | 489,846 |

其他殘差大小：

| 棋盤 | 通道／區塊 | 參數 | 用途 |
| --- | --- | ---: | --- |
| 3×3 | 8 / 1 | 3,084 | 快速檢查 |
| 15×15 | 32 / 2 | 68,516 | 小型基準 |
| 15×15 | 96 / 8 | 1,366,500 | 較大實驗 |
| 15×15 | 128 / 10 | 2,993,028 | `gomoku-large`，不是允許的最大大小 |

權重檔案小，不保證推論快或訓練省記憶體。訓練還要保存 activations、梯度、optimizer 狀態和暫存區。密集卷積成本大致隨寬度平方成長；加區塊則讓主體成本大致線性增加。

## 已經檢查了什麼？

2026-10-04 模型變體階段通過 **72 個 CPU 測試**。涵蓋 3×3、15×15、19×19、黑白雙方、六子棋落子階段、非零位置 bias、真實權重更新、完整回合、固定模型和續訓。重現並修復 DLL 存取違規後，保留原本 C 模型建立介面，已保存的五子棋 best／latest 也能完成合法 CPU 回合。

以下是**當時的** CPU 搜尋中位數：兩個根節點、四次模擬、一個工作執行緒、三次暖機後執行。不包含啟動、載入或學習：

| 遊戲 | 殘差 | 池化 | 注意力 |
| --- | ---: | ---: | ---: |
| 15×15 五子棋 | 8.60 ms | 9.72 ms | 16.98 ms |
| 19×19 六子棋 | 11.60 ms | 8.95 ms | 16.86 ms |

池化沒有每次都比較快；注意力在這個小型 CPU 批次更花時間。這不是完整棋局吞吐量或棋力結果。各架構也完成合法棋局和暫存井字棋訓練／續訓檢查。

一次 attention CUDA 檢查，在三種棋盤通過 FP32 學習端／原生輸出比對、完整棋局、BF16 學習、更新權重一致性和續訓。使用八個根節點／八次模擬，每種棋規更新 32 次，大棋盤 batch 為 128。暫存 run 從八盤續訓至九盤，並保留未完棋局。成功程序耗時 23.874 秒，失敗嘗試另花 4.772 秒；加上先前移植檢查，當次核准的 300 秒 GPU 預算約用了 133 秒。約 254／437 MiB 是累計 PyTorch 程序配置峰值，不是總 VRAM 或隔離的單模型用量。

原始本機紀錄在 `.native-cache/model-{residual,pooled,attention}-cpu.json` 和 `.native-cache/model-attention-cuda.json`，不進 Git。[驗證紀錄](migration-validation.zh-TW.md)列出其他檢查，不代表每次修改都重新量了這些時間。

```sh
uv run python -m experiments.benchmark_native --architecture attention --device cpu --repeats 3 --batch 2 --simulations 4 --wall-seconds 60 --output .native-cache/model-attention-cpu.json
uv run python -m experiments.benchmark_native --architecture attention --device cuda --repeats 1 --batch 8 --simulations 8 --training-presets tictactoe,gomoku,connect6 --wall-seconds 60 --output .native-cache/model-attention-cuda.json
```

GPU 指令是給使用者自行執行的範例，不是額外授權 agent 跑 GPU。測試程序時間包含啟動和暖機，長期棋力比較需要另外授權。

## 什麼結果才值得換預設？

先確認更新前後兩端輸出一致、棋局能下完、續訓安全。再依各棋規比較交換黑白的對戰，以及相同總訓練時間後的棋力。模型更大、loss 更低，還不能決定這件事。

SE、瓶頸殘差和棋盤版 ConvNeXt 仍是研究點子，不是可用選項。論文用來選設計，不匯入別人的棋類引擎或訓練好 bot 權重。背景見[網路研究](network-research-2026.zh-TW.md)、[學習率](learning-rates.zh-TW.md)和[架構決策](adr/0002-shared-board-model-experiments.zh-TW.md)。

## 來源

1. [Silver 等人，AlphaZero](https://discovery.ucl.ac.uk/id/eprint/10069050/1/alphazero_preprint.pdf)：殘差棋盤模型；256 通道／19 區塊不是我們的本機預設。
2. [He 等人，殘差網路](https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)：殘差和瓶頸設計。
3. [Hu 等人，squeeze-and-excitation](https://openaccess.thecvf.com/content_cvpr_2018/papers/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.pdf)：通道控制，量測來自影像任務。
4. [Liu 等人，ConvNeXt](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.pdf)：影像模型結果，不是六子棋棋力證據。
