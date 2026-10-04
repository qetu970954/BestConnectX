# Connection-game learning

Terms for configurable connection games, self-play, and model comparisons. This glossary describes the domain, not the code.

[繁體中文](CONTEXT.zh-TW.md)

## Language

**Rule configuration**:
The full rules of a square-board connection game: board edge, winning line length, opening stones, and later stones per turn. Different rules define different games; their strength results and samples must not be mixed.
_Avoid_: Board size alone

**Freestyle Gomoku**:
A game where Black starts and each player places one stone per turn; five or more in a row wins. Both colors use the same rules, with no forbidden moves or swap opening; the current main board is 15×15.
_Avoid_: Connect6, Renju

**Connect6**:
The standard 19×19 game where Black places one opening stone, then turns have two stones. Six or more in a row wins in any line direction.

**Square symmetry**:
One of eight views formed by four rotations and an optional reflection. Board and policy change together; player and result do not, and the views are not separate games.

**Bot**:
The full playing system that chooses legal placements under the rules, including its model and search when present.
_Avoid_: Weights alone, model alone

**Playing strength**:
The ability to win games under stated rules, compute limits, and thinking time. Those conditions are part of any comparison.
_Avoid_: Model size, training accuracy, GPU usage

**Weights**:
The numbers learned by a model. They are not the learning or search rules.

**Turn**:
One player's chance to place the allowed number of stones. The opening may differ; a win ends the turn early, and a full-turn time limit includes every placement.
_Avoid_: Move when it is unclear whether one placement or a turn is meant

**Incumbent**:
The accepted best bot version for the run's game, or its initial baseline before later comparisons.
_Avoid_: Latest model

**Candidate**:
A proposed version that has not yet earned replacement of the incumbent.

**Promotion**:
Acceptance of a candidate as incumbent after the required game, score, and timing checks.

**Initial baseline**:
The first milestone used as a comparison starting point. It is not a claim of proven strength.
_Avoid_: Validated strong bot

**Milestone model**:
A frozen model saved at a stated total completed-game count. It cannot be overwritten and is not necessarily the current best.
_Avoid_: Latest checkpoint

**Promotion score**:
The candidate's points per game: win 1, draw 0.5, loss 0. It differs from pure win rate and is not Elo.
_Avoid_: Win rate, Elo

**Replay window**:
The bounded recent samples used for learning, with board positions, policy targets, and real terminal results. Removing old samples from this window does not mean deleting existing files.
_Avoid_: Permanent game archive

**Self-play summary**:
A completed game's winner, length, turn count, and move-source counts. It contains no move list or board history and cannot replay the game.
_Avoid_: Kifu, saved move record

**Active game**:
An unfinished game whose state, samples, and plans must be retained for safe resume. Its samples have no final value label yet.

**Tactical certificate**:
A plan with a checked forced-win proof within its stated horizon. It must include every opponent reply that can block the attack and rule out an earlier opponent win.
_Avoid_: A threat alone

**Unknown tactical result**:
A position with no established certificate. It does not mean loss or the absence of a winning plan.

**Training session**:
A user-started, bounded period of learning/evaluation that can stop and resume. Its end does not mean the bot has reached final strength.
