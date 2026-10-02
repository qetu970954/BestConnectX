# Recent self-play / board-game training research

Reviewed 2026-10-02 against primary arXiv papers. This note supersedes the warm-start/pretrained-engine advice in `connect6-research.md` for the current work. Research ideas are allowed; copied game engines and pretrained bot weights are not.

## Decision for this delivery

Keep the original **policy/value network + batched PUCT self-play + independently verified TSS**, with terminal outcomes, durable datasets and frozen candidate-versus-best matches. First establish reliable training/evaluation; do not replace the learner merely because a newer paper reports a better result in another game.

The new code supports square boards only. Four rotations and optional reflection give eight spatial symmetries; the same transformation is applied to board planes and policy targets. Augmentation is sampled during training, without storing eight copies. Value labels are invariant. Eight equivalent views do not constitute eight independent games, and symmetric boards can yield identical views.

## 2026 papers worth investigating

### 1. Engineering Efficient Self-Play Chess: Search, Replay, and Throughput Under Limited Compute

Source: [arXiv:2609.37447](https://arxiv.org/abs/2609.37447), [full paper](https://arxiv.org/html/2609.37447v1), particularly §§2–4 and the reproducibility appendix. Submitted 2026-09-27.

- Reports training from random initialization for **2.5 days on eight GPUs**, ingesting 3.25 million games. Its 6.32-million-parameter final model reports 3,251 **benchmark Elo** against a fixed-node Stockfish 13 ladder at 100,000 searches per move. This is not a universal Elo rating or a consumer-GPU two-hour result.
- Separates game actors/search, replay materialization, learning, model publication, and evaluation. Describes native search, batched inference, retained trees, growing replay, archived-state restarts and deployment-throughput experiments.
- Explicitly warns that faster inference, lower stored-data loss, or a locally attractive optimization need not improve the complete learning loop. Its final result cannot be attributed to one isolated component.

**Use here:** preserve the game/training/dashboard separation, recorded datasets, and immutable comparison models. These engineering principles are implemented; the paper's native/TensorRT pipeline, graph search, adaptive replay, model growth and restart-state recipe are **not** implemented. Profile the current Python search before selecting a native rewrite. The published compute alone is about 480 GPU-hours, far beyond a short agent validation allowance.

### 2. Revisiting Regularized Policy Optimization for Stable and Efficient RL in Two-Player Games (KLENT)

Source: [arXiv:2602.10894](https://arxiv.org/abs/2602.10894), [full paper](https://arxiv.org/html/2602.10894v2), §§4, 5 and 6. The primary page lists acceptance at ICML 2026; v2 is dated 2026-05-21.

- Combines reverse-KL and entropy regularization with an action-value network and λ-return targets. Practical policy improvement is computed analytically instead of through MCTS; its main training procedure starts with random networks.
- Experiments cover Animal Shogi, Gardner Chess, 9×9 Go, Hex and Othello. Reports up to **4× training efficiency** versus its Gumbel AlphaZero baseline in the paper's comparison.
- The main horizontal axis is **simulator evaluations**, and evaluation uses reactive policies to match test-time resources. This is not a demonstrated 4× wall-clock reduction on this project's GPU, nor a Gomoku/Connect6 result.
- Convergence results rely on the paper's theoretical assumptions; they do not guarantee convergence of an arbitrary finite neural implementation. A preliminary bias/variance experiment uses a pretrained Go checkpoint; that experiment is distinct from its main from-random learning procedure.

**Possible later experiment:** an original KLENT learner with its own action-value targets, in a separate data directory. Do not relabel current AlphaZero replay as KLENT data, and do not copy checkpoints. This changes the learner and is not a small PUCT parameter tweak. Not implemented or locally benchmarked.

### 3. The Surprising Effectiveness of Approximate Value Iteration in Self-Play (AVI)

Source: [arXiv:2609.09094](https://arxiv.org/abs/2609.09094), [full paper](https://arxiv.org/html/2609.09094v1), §§3–5. Submitted 2026-09-08.

- Replaces MCTS-generated targets with one-step value iteration using a frozen target value function and greedy/ε-greedy self-play. On Connect Four and Hex(7×7), evaluates value error, regret and played-out results against exact oracles.
- Reports more accurate values and competitive greedy policies at lower neural-call cost; larger-game Othello/Go experiments are presented separately without exact oracles.
- Its controlled comparison matches data/optimizer work and reports a **forward-equivalent compute proxy**, not elapsed wall time. AVI enumerates one-step successors; Connect Four has at most seven actions, whereas a free-placement square board can have hundreds.
- Reports repeated seeds (20 AVI and five AlphaZero runs on the solved games). Its cross-inference experiment uses two backbones and is explicitly diagnostic rather than a free-speed hybrid.

**Possible later experiment:** a value-only training baseline after the current saved-data pipeline is stable. For Connect6, a successor placement can retain the same player: value perspective must change only when the player changes, not after every placement. This is already handled by the current PUCT backup but must be handled separately in any AVI targets. Not implemented or locally benchmarked.

### 4. Chessformer: A Unified Architecture for Chess Modeling

Source: [arXiv:2605.19091](https://arxiv.org/abs/2605.19091). The abstract reports geometric attention bias, square-token representations, source/destination policy heads, and over 100 Elo when integrated into Leela Chess Zero.

**Caution:** that chess-specific architecture/engine integration does not establish a stronger from-scratch Gomoku bot on this hardware. This abstract-level architectural lead was not selected; neither its engine nor its weights are imported.

## Earlier leads and important limits

### Search-contempt

Source: [arXiv:2504.07757](https://arxiv.org/abs/2504.07757), [full paper](https://arxiv.org/html/2504.07757v1), 2025.

Keeps root-player PUCT while freezing/sampling opponent-node visit distributions after a threshold. It changes self-play position distribution and reports strength improvements with existing chess engines, especially odds chess. Consumer-GPU, much-cheaper from-scratch training is a **proposed opportunity**, not a completed training result demonstrated for this project. Asymmetric search also complicates tree reuse between moves.

**Decision:** retain symmetric PUCT for now. Test any original search-contempt implementation in isolation with equal-time frozen matches and saved configurations; no claimed speedup before measurement.

### AlphaGateau / graph representation and board-size transfer

Source: [arXiv:2410.23753](https://arxiv.org/abs/2410.23753), [full paper](https://arxiv.org/html/2410.23753v1), 2024.

Uses graph attention with edge features and demonstrates transfer from 5×5 chess to 8×8 chess. Its experiments use Gumbel MuZero and multiple RTX A5000 GPUs; training configurations are not a repeated-seed consumer-GPU benchmark.

**Decision:** a fixed-size compact CNN is sufficient for this delivery. Square board sizes are configurable, but every rule configuration gets separate weights/data; automatic cross-size transfer is **not** claimed. Graph networks and Gumbel search remain alternatives, not implemented features.

### Amortized Planning with Large-Scale Transformers

Source: [arXiv:2402.04494](https://arxiv.org/abs/2402.04494), 2024.

Reports searchless chess strength via **supervised Stockfish-16 distillation**: 10 million games, roughly 15 billion labeled data points, and models up to 270 million parameters. Its 2,895 Lichess blitz Elo is against humans, not this project's self-play gate or a universal engine rating.

**Decision:** not a drop-in alternative to original from-scratch self-play. No Stockfish corpus, external engine, or pretrained weights were imported.

## What to measure before changing the algorithm

1. Real completed games/hour, replay supply, optimizer updates/hour, and CPU/search versus inference time on the user's hardware.
2. Candidate-versus-best results with identical rules, paired legal openings, equal full-turn budgets and recorded timing failures. Loss is diagnostic, not strength evidence.
3. Keep existing baseline/candidate weights and archived targets; compare changes without erasing previous runs. More ambitious efficiency claims need repeated runs and external opponents.

No GPU experiment or strength comparison of these research alternatives was performed in this delivery. Their papers motivate experiments; they do not authorize extra compute or establish the strongest attainable bot.
