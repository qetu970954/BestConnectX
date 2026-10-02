# 棋類訓練

本專案正方形連棋遊戲、自我對弈訓練及既有六子棋實驗所使用的領域詞彙。

## Language

**Rule configuration**:
一個連棋遊戲的完整棋規：正方形棋盤大小、獲勝連線長度、黑棋開局回合棋子數及之後每回合棋子數。不同配置是不同遊戲，棋力及訓練資料不能不加區分地混用。
_Avoid_: Board size alone

**Square symmetry**:
正方形棋盤的四種旋轉與是否反射所構成的八種空間變換。變換棋盤與策略目標不改變玩家或結果；等價視角不是獨立對局。

**9×9 自由五子棋（freestyle Gomoku）**:
黑方在空的 9×9 棋盤先下，雙方輪流，每回合放置一子。橫、直或斜向連續五子以上即勝，黑白同規則，無禁手與交換開局規則。
_Avoid_: 六子棋、連珠（Renju）

**Connect6**:
The standard game on a 19×19 board: Black places one opening stone, then players alternate turns of two stones; six or more consecutive stones of one color horizontally, vertically, or diagonally wins.

**Playing strength**:
在指定棋規、運算資源與思考時間下，bot 贏得對局的能力。比較棋力時必須明示這些條件。
_Avoid_: Model size, training accuracy

**Bot**:
The complete game-playing system that chooses legal placements under a rule configuration, including its learned evaluation and search when present.
_Avoid_: Weights alone, model alone

**Weights**:
The numerical parameters learned during a model's training. They are distinct from the algorithm used to train or evaluate the model.

**Turn**:
一位玩家依棋規放置棋子的機會；黑棋開局回合可與後續回合具有不同棋子數，獲勝即停止放置。完整回合的思考時間限制涵蓋該回合全部落子。
_Avoid_: Move when it is unclear whether one placement or a complete turn is meant

**Incumbent**:
在指定遊戲與棋規下，目前已接受的最佳 bot 版本；候選版本須在同一棋規下與其比較。
_Avoid_: Latest model

**Candidate**:
A proposed bot version that has not yet earned replacement of the incumbent through evaluation.

**Promotion**:
Acceptance of a candidate as the new incumbent after the required playing-strength and rules checks.

**Initial baseline**:
第一個里程碑形成的初始比較對手；它可以作為正式比較起點，但尚未以升級對戰證明棋力。
_Avoid_: Validated strong bot

**Milestone model**:
在指定累計完成自我對弈數量保存的不可覆寫模型版本。它不等同於最新訓練狀態或目前最佳版本。
_Avoid_: Latest checkpoint

**Promotion score**:
候選在升級對戰中的得分率，勝、和、敗分別計 1、0.5、0。它與純勝率不同，也不是 Elo。
_Avoid_: Win rate, Elo

**Self-play record**:
一盤自我對弈的實際落子、開局及终局結果，供重播與檢查。它與含策略目標的內部訓練資料不同。

**Training dataset**:
保存的自我對弈訓練樣本，包含棋盤狀態、策略目標及對局結果；它不受目前 replay 視窗大小限制。

**Replay window**:
目前最佳化時抽樣使用的近期訓練資料子集。淘汰視窗內的舊樣本不代表刪除保存的完整訓練資料。

**Tactical certificate**:
A placement plan with a checked proof that the player can force a win within the stated search horizon. The proof must account for every opponent reply that can block the attack and rule out an earlier opponent win. A threat alone is not a certificate.

**Unknown tactical result**:
A position for which no tactical certificate has been established. It does not mean the position is lost or that no winning plan exists.

**Proof-adjudicated self-play game**:
在棋盤達到終局前，以確切戰術證明判定結果的訓練棋局；這是舊六子棋實驗使用的方式。新訓練以證明引導落子，仍須實際下到終局。

**Training session**:
A manually started, bounded period of learning and evaluation that can be stopped and resumed later. Finishing a session does not mean the bot has reached its final strength.
