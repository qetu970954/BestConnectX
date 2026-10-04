# 五子棋：先手必勝與價值 loss

[English](gomoku-first-player.md)

## 答案

**是的。無限制 15×15 五子棋已證明：從初始棋盤出發，黑方正確下棋可以強制獲勝。** 這與 repo 主要棋規相符：黑方先手、每回合一子、連五或以上即勝，沒有禁手或交換規則。[1，第 2、7 節]

這不代表任何開局或後續失誤後黑方仍必勝；它表示黑方從初始局面有一套策略，能應對白方所有回應並獲勝。

## 第一手證據

Allis、van den Herik、Huntjens 在 1993 年 AAAI 論文 *Go-Moku Solved by New Search Techniques* 報告此結果。[1]

- 第 2 節定義 15×15 棋盤與無限制棋規；長連（六子或以上）對兩色都是勝利。
- 第 7 節報告無限制版本，以及只有恰好五子才算勝的 variant B，都已解出。原文寫道：「Both variants have been solved.」。黑方獲勝樹被保存，並檢查一致性及完整性。
- 第 5–8 節描述 Victoria 結合威脅空間搜尋及證明數搜尋；只有有界 TSS 仍可能漏掉獲勝路線。
- 第 1 節給出具體提醒：先前冠軍程式執黑，仍走進已證明黑方必敗的局面。初始局面必勝不能抵消後續壞棋。

Allis 的博士論文 *Searching for Solutions in Games and Artificial Intelligence* 於 1994 年出版，大學官方紀錄確認其著作及年份。[2][3]

| 棋規 | 所引結果適用範圍 |
| --- | --- |
| 無限制 15×15，連五或以上即勝 | 黑方有初始局面獲勝策略 |
| 15×15，兩色都須恰好連五 | 此論文也證明黑方必勝 |
| Renju，黑方有專屬禁手 | 不可直接套用此結果 |
| Swap／Swap2 開局規則 | 不可直接套用此結果 |
| 任意中後盤或隨機起始局面 | 初始局面證明不為所有這些局面指定價值 |

本筆記不宣稱其他棋規目前是否已被解出。

## 對訓練的意義

黑方成績明顯較好，本身不是 bug 或過擬合的證據。如果大部分自我對弈由黑方獲勝，模型可能很容易擬合結果並得到極低價值 loss；先手必勝結果讓此現象較不意外。

但必須區分兩種價值：

- **完美對弈價值：** 雙方都做出最佳決策時的結果。
- **實際棋局價值目標：** 該盤實際玩家及落子選擇產生的真實終局結果。

本 repo 以後者作為訓練標籤，從輪到落子的玩家視角取勝 +1、和 0、敗 −1。標籤來自真實終局，不取自初始局面定理或未完成 TSS 證明。見 [`engine/selfplay.py`](../engine/selfplay.py) 及 [`engine/training.py`](../engine/training.py)。

實際玩家可能犯錯，自我對弈及評估也使用不同起始局面與落子方式。因此訓練／評估價值誤差落差，可能反映棋局分布改變，不只代表記憶訓練資料；只看 loss 無法區分原因。

中後盤若已有立即獲勝落子，就是更強的檢查：無論對手多強，當前玩家都可立即獲勝。此時負價值預測就是錯誤，即使空棋盤是黑方必勝。

先前 CPU 檢查的 8,000 盤模型，最後訓練 minibatch 價值 loss 幾乎為零，但 300 個已保存評估局面的價值 MSE 約 0.493。這是轉移能力及價值品質的警訊，不是經典過擬合的證明；黑方必勝定理不能免除局面層級檢查。

## 本引擎的限制

本引擎有自行實作的有界 C++ TSS 與獨立證明檢查，不附帶 Victoria 獲勝資料庫或完整證明數求解器。已發表獲勝策略不代表此 bot 能遵循它。

保留目前棋規及真實終局標籤；先用獨立棋局及已知勝敗局面檢查價值誤差，再改模型。本次研究沒有更改訓練設定。

## 來源

1. L. V. Allis、H. J. van den Herik、M. P. H. Huntjens（1993），[*Go-Moku Solved by New Search Techniques*](https://cdn.aaai.org/Symposia/Fall/1993/FS-93-02/FS93-02-001.pdf)，AAAI Technical Report FS-93-02。見第 1、2、5–8 節。
2. Maastricht University，[官方論文紀錄](https://cris.maastrichtuniversity.nl/en/publications/searching-for-solutions-in-games-and-artificial-intelligence/)，DOI [10.26481/dis.19940923la](https://doi.org/10.26481/dis.19940923la)。
3. L. Victor Allis（1994），[*Searching for Solutions in Games and Artificial Intelligence*](https://project.dke.maastrichtuniversity.nl/games/files/phd/SearchingForSolutions.pdf)。
