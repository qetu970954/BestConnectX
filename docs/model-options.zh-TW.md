# C++ 引擎的模型選擇

[English](model-options.md)

**狀態：基準及實驗變體均已實作，全部使用 C++ 推論。15×15 預設仍為 64 通道／6 區塊的一般殘差網路。五子棋及六子棋可選池化價值頭與注意力變體；尚未建立棋力改善的證據。**

較新的棋類證據、五子棋／六子棋共用輸入及後續實驗見 [2026-10-04 網路研究](network-research-2026.zh-TW.md)。[學習率設定及較新 optimizer 方法](learning-rates.zh-TW.md)另有說明。

## 目標

不必修改棋規或搜尋程式碼，就能調整模型大小。

較大的模型可能學到更豐富的局面特徵，但每步搜尋成本也較高。只有提高每小時棋力增益時，擴大模型才有價值。

以下論文都不能直接證明其模型會改善本專案的五子棋棋力。影像模型論文的結果來自影像任務，不是五子棋。

## 術語

- **殘差區塊（residual block）**：一組模型層將輸入加回運算結果，有助於訓練更深的模型。[2]
- **通道數（channels）**：每個棋盤位置的特徵圖數量。增加通道數會加寬模型。
- **策略頭（policy head）**：為各個可能落子提供分數的模型部分。
- **價值頭（value head）**：從局面預測對局結果的模型部分。
- **SE 區塊**：使用全盤摘要，調整各特徵通道權重的小型模型部分。[3]
- **瓶頸區塊（bottleneck block）**：在昂貴的空間運算前縮減通道寬度，之後再恢復的殘差區塊。[2]

## 簡要比較

| 模型種類 | 主要概念 | 本專案適用性 | 狀態／建議 |
| --- | --- | --- | --- |
| 一般殘差網路 | 保持棋盤形狀，堆疊各含兩層 3×3 卷積的殘差區塊。 | 最接近既有模型與 AlphaZero 設計。深度及寬度是直接的設定項目。[1][2] | 建議作為基準。 |
| SE 殘差網路 | 在殘差區塊加入全盤通道摘要與小型學習閘門。 | 小幅修改可能幫助全盤資訊處理，但額外 pooling 與模型呼叫仍有成本。[3] | 先做新版研究中的池化價值頭檢查，再測此小型主體變體；尚未證實棋力增益。 |
| 瓶頸殘差網路 | 使用 1×1、3×3、1×1 層，讓 3×3 層維持較窄。 | 可能降低寬模型的算術成本，但小批次下新增層數可能增加啟動成本。[2] | 寬模型變昂貴時再考慮，應量測實際時間，而非只比較運算數。 |
| ConvNeXt 式網路 | 使用 depthwise 空間層、通道混合及其他現代卷積設計。 | 值得研究，但論文中的影像模型會縮減空間尺寸。棋盤策略需要另行調整設計。[4] | 不直接匯入影像模型或其權重，棋盤版本延後處理。 |

AlphaZero 論文描述的主體包含 19 個殘差區塊與 256 通道，但使用的硬體遠多於本專案。這不是建議的本機預設大小。[1]

## 以本專案實作為基礎的模型大小

以下參數數量是在推論檢查前，於 CPU 建立 [`Network`](../engine/network.py) 計算所得。後續 CPU／CUDA 輸出及學習檢查見[驗證報告](migration-validation.zh-TW.md)。

下列數字使用 `residual` 架構的策略／價值頭。已實作的池化及注意力變體，另於下方列出參數數量。

| 棋盤 | 通道數 | 殘差區塊數 | 可訓練參數 | 建議用途 |
| --- | --- | --- | --- | --- |
| 3×3 | 8 | 1 | 3,084 | 第一盤及跨語言介面檢查 |
| 15×15 | 32 | 2 | 68,516 | 既有小型基準 |
| 15×15 | 64 | 6 | 477,764 | 預設五子棋模型 |
| 15×15 | 96 | 8 | 1,366,500 | 較大模型實驗 |
| 15×15 | 128 | 10 | 2,993,028 | 上方範例，不是必要預設 |
| 19×19 | 64 | 6 | 495,172 | 六子棋基準 |

[`configs/experiments.toml`](../configs/experiments.toml) 已實作的 preset：

```toml
[presets.gomoku]
channels = 64
blocks = 6
```

模型主體保留棋盤的空間形狀，輸出每個格子的策略分數，以及每個局面的單一價值。

## 已實作的實驗

三種架構都使用相同八個輸入平面及每格一個策略分數。`pooled` 將價值頭改為 16 通道投影、全域平均池化及 64 單元隱藏層。`attention` 使用同樣價值頭，再於殘差主體之後加一個四-head Transformer 區塊，包含可學習二維相對位置 bias、layer normalization 及兩倍通道寬的 GELU feedforward 路徑。不使用 dropout。這是由研究啟發的小型本機實驗，不是實作 KataGo 或 ResTNet。

| 架構 | 15×15 參數 | 19×19 參數 | Presets |
| --- | ---: | ---: | --- |
| `residual` | 477,764 | 495,172 | `gomoku`、`connect6` |
| `pooled` | 450,898 | 450,898 | `gomoku-pooled`、`connect6-pooled` |
| `attention` | 487,734 | 489,846 | `gomoku-attention`、`connect6-attention` |

以上是在 CPU 建立已實作的 64 通道／6 區塊模型量測所得。池化失去絕對布局；注意力增加全域互動及運算。池化價值頭不代表 checkpoint 能跨棋盤大小或棋規使用。注意力通道數須為四的倍數。

```sh
uv run python train.py --preset gomoku-pooled --data data/gomoku-pooled
uv run python train.py --preset gomoku-attention --data data/gomoku-attention
uv run python train.py --preset connect6-pooled --data data/connect6-pooled
uv run python train.py --preset connect6-attention --data data/connect6-attention
```

只有執行指令時才會開始訓練。每種架構使用新目錄；同一棋規的各變體共用原本棋規預設目錄。也可使用 `--architecture residual|pooled|attention`。不指定 preset 的續訓會恢復架構、權重及 optimizer；明確更改架構會被拒絕。缺少架構欄位的既有 checkpoint 仍代表 `residual`。

見[決策樹及取捨](adr/0002-shared-board-model-experiments.zh-TW.md)。下方記錄功能檢查及時間量測，不代表棋力已改善。

參數數量不能證明模型能放進 GPU 記憶體。訓練還需要保存 activations、梯度、optimizer 狀態及暫存緩衝區。

固定棋盤大小及區塊數時，密集卷積的成本大致隨寬度平方成長。固定寬度時，主體成本大致隨區塊數線性成長。權重檔案小，不代表執行快。

## 可用模型設定

- `--architecture residual|pooled|attention`：主體及輸出頭設計。
- `--channels`：特徵圖寬度。
- `--blocks`：殘差深度；注意力變體仍只有一個注意力區塊。

訓練與 C++ 推論共用模型設定、層形狀、權重及 buffers。更改架構、寬度、深度或棋盤大小必須使用新 run。沒有模型轉移、預訓練影像初始化、自動擴張或架構搜尋框架。

## 模型大小的檢查

1. 使用相同權重與輸入，比對 C++ 與學習端輸出。
2. 執行一次真實權重更新，再確認 C++ 推論取得最新權重及 buffers。
3. 量測啟動、單局推論、批次搜尋、學習更新與 GPU 記憶體峰值。
4. 檢查模型、optimizer、replay、未完棋局與 RNG 狀態的停止／續訓。
5. 分開報告效能與棋力。短測試不能建立五子棋每小時棋力增益的結論。

GPU 測試預算為 GPU 程序累計五分鐘實際時間。長時間模型棋力比較需要使用者自行執行，或另行明確授權。

## 2026-10-04 本機檢查

原生建置成功，**72 項 CPU 單元測試全部通過，耗時 28.864 秒**。新案例在 3×3、15×15、19×19 檢查兩種變體，包含真實 optimizer 更新前後、非零相對位置 bias、黑白雙方、六子棋兩個落子階段、完整 bot 回合、凍結模型與續訓。保留原本模型建立 C 介面。重現並修復存取違規後，已保存的五子棋 best 及 latest 模型也完成 CPU bot 回合。

下列 CPU 搜尋中位數使用**兩個根節點、四次模擬、一個原生工作執行緒及三次暖機後重複**；各架構內使用相同輸入及權重，比對 FP32 推論與學習端。大棋盤使用 64 通道／6 個殘差區塊。這是[移植驗證報告](migration-validation.zh-TW.md)所列機器上的小型搜尋量測，不含啟動、模型載入與學習。

| 遊戲 | 殘差 | 池化價值 | 池化價值加注意力 |
| --- | ---: | ---: | ---: |
| 15×15 五子棋 | 8.60 ms | 9.72 ms | 16.98 ms |
| 19×19 六子棋 | 11.60 ms | 8.95 ms | 16.86 ms |

池化沒有一致加速；此小批次下，注意力增加 CPU 成本。這不是完整棋局吞吐量、訓練品質或相同時間下棋力的比較。各 CPU 架構檢查也完成三種棋盤的合法棋局，以及暫存井字棋訓練／續訓檢查。

**Attention CUDA 檢查通過**三種棋盤的 FP32 學習端／原生一致性、完整原生棋局，以及 BF16 學習後的更新權重一致性及續訓。搜尋使用八個根節點／八次模擬，各棋規執行 32 次 optimizer 更新，大棋盤 batch 為 128。各暫存 run 由八盤續訓到九盤；大棋盤保留 63 盤未完棋局。這仍是功能檢查，沒有持久訓練或棋力宣稱。

成功 CUDA 程序包含啟動及暖機，耗時 **23.874 秒**。一次 sandbox 失敗嘗試另使用 **4.772 秒**，本次新增 CUDA 程序時間共 **28.646 秒**。加上先前移植約 104 秒，合計記錄時間約 133 秒，低於既有五分鐘上限。PyTorch 配置記憶體峰值至五子棋階段約 254 MiB，至六子棋約 437 MiB；是累計程序峰值，不含 driver／context 及非 PyTorch 記憶體，不是模型單獨用量。

本機原始紀錄：`.native-cache/model-{residual,pooled,attention}-cpu.json` 與 `.native-cache/model-attention-cuda.json`。可重新執行相同 CPU 檢查，或自行要求有界 CUDA 檢查：

```sh
uv run python -m experiments.benchmark_native --architecture attention --device cpu --repeats 3 --batch 2 --simulations 4 --wall-seconds 60 --output .native-cache/model-attention-cpu.json
uv run python -m experiments.benchmark_native --architecture attention --device cuda --repeats 1 --batch 8 --simulations 8 --training-presets tictactoe,gomoku,connect6 --wall-seconds 60 --output .native-cache/model-attention-cuda.json
```

## 原創性要求

以論文選擇設計，在本專案實作棋盤模型。不複製棋類引擎，不匯入已訓練 bot 權重。

## 第一手來源

1. [Silver 等人，AlphaZero 論文與補充資料](https://discovery.ucl.ac.uk/id/eprint/10069050/1/alphazero_preprint.pdf)：架構與訓練細節。
2. [He 等人，Deep Residual Learning for Image Recognition，CVPR 2016](https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)：殘差與瓶頸設計。
3. [Hu 等人，Squeeze-and-Excitation Networks，CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/papers/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.pdf)：通道閘門及其量測成本。
4. [Liu 等人，A ConvNet for the 2020s，CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.pdf)：ConvNeXt 設計及影像任務結果。
