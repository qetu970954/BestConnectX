# 連棋遊戲學習

可設定連棋遊戲、自我對弈及模型比較的領域詞彙。此詞彙表描述領域，不描述程式碼。

[English](CONTEXT.md)

## Language

**Rule configuration**：
正方形連棋遊戲的完整棋規：棋盤邊長、獲勝連線長度、開局棋子數及後續每回合棋子數。不同棋規定義不同遊戲，不得混用棋力結果或樣本。
_Avoid_: Board size alone

**Freestyle Gomoku**：
黑方先下，雙方每回合一子，連續五子或以上即勝。兩色同規則，無禁手或交換開局；目前主要棋盤為 15×15。
_Avoid_: Connect6, Renju

**Connect6**：
標準 19×19 六子棋，黑方開局一子，之後每回合兩子。任一連線方向六子或以上即勝。

**Square symmetry**：
四種旋轉及可選反射形成的八種視角。棋盤及策略一起變換，玩家及結果不變，不構成獨立棋局。

**Bot**：
依指定棋規選擇合法落子的完整對弈系統，包含其模型及搜尋（若有）。
_Avoid_: Weights alone, model alone

**Playing strength**：
在明示棋規、運算限制及思考時間下贏得棋局的能力，這些條件是比較的一部分。
_Avoid_: Model size, training accuracy, GPU usage

**Weights**：
模型學到的數值，不等同學習或搜尋規則。

**Learner**：
從 replay window 學習並更新權重的模型，不一定是已接受的 incumbent。
_Avoid_: Best model

**Self-play actor**：
用於產生訓練棋局的 bot 版本，可使用 learner 權重而尚未成為 incumbent。
_Avoid_: Incumbent when only game generation is meant

**Turn**：
一位玩家放置允許棋子數的機會。開局可能不同，獲勝可提早結束，完整回合時限包含全部落子。
_Avoid_: Move when it is unclear whether one placement or a turn is meant

**Incumbent**：
指定 run 遊戲中已接受的最佳 bot，或後續比較前的初始基準。
_Avoid_: Latest model

**Candidate**：
尚未取得取代 incumbent 資格的候選版本。

**Promotion**：
通過必要棋局、得分及計時檢查後，將 candidate 接受為 incumbent。

**Initial baseline**：
第一個里程碑形成的比較起點，不代表已證實棋力。
_Avoid_: Validated strong bot

**Milestone model**：
在指定累計完成棋局數保存的凍結模型，不可覆寫，也不一定是目前最佳。
_Avoid_: Latest checkpoint

**Promotion score**：
候選每盤得分，勝 1、和 0.5、敗 0。不同於純勝率，也不是 Elo。
_Avoid_: Win rate, Elo

**Replay window**：
學習使用的有界近期樣本，包含局面、策略目標及真實終局結果。淘汰視窗內舊樣本不代表刪除既有檔案。
_Avoid_: Permanent game archive

**Self-play summary**：
已完成棋局的勝者、長度、回合數及落子來源計數，不含落子列表或棋盤歷史，不能重播棋局。
_Avoid_: Kifu, saved move record

**Active game**：
須保留狀態、樣本及計畫以安全續訓的未完成棋局；其樣本尚無最終價值標籤。

**Tactical certificate**：
指定搜尋範圍內經檢查的強制獲勝計畫，必須涵蓋所有可阻擋攻擊的對手回應，並排除對手先獲勝。
_Avoid_: A threat alone

**Unknown tactical result**：
尚未建立 certificate 的局面，不代表必敗，也不代表沒有獲勝計畫。

**Training session**：
使用者啟動、有界且可停止／續跑的學習與評估期間。Session 結束不代表 bot 已達最終棋力。
