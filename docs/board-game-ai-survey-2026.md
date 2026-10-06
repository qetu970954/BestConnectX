# Recent AI approaches for abstract strategy games

Reviewed 2026-10-06. Scope: deterministic, two-player, perfect-information games such as chess, Go, Hex, Gomoku, and Connect6. Emphasis: playing strength per total training hour and per full-turn thinking time on one machine, not only parameter count or simulator throughput.

**Conclusion:** keep the current run as a baseline. The best justified next search experiment is Gumbel AlphaZero; the cheapest existing architecture experiment is pooled value. KLENT is an important new, peer-reviewed alternative for resource-efficient training, but requires a separate learning implementation. None of the verified sources establishes a superior recipe for this repo's 19×19 Connect6.

This survey did not start training, run GPU benchmarks, modify engine code, or change existing run artifacts. It extends the [network survey](network-research-2026.md) and [self-play guidance](selfplay-training-guidance-2026.md).

## What this repo already does

- A residual policy/value network with eight board planes; the default 19×19 model has 64 channels, six residual blocks, and about 495,000 parameters. Optional pooled-value and one-block attention variants already exist. [Local network](../engine/network.py), [model choices](model-options.md).
- Native PUCT MCTS, root Dirichlet noise, and normalized root visit counts as policy targets; the default is 64 simulations per placement. The native search builds a fresh tree for each call. [Native search](../engine/runtime.cpp), [defaults](../configs/experiments.toml).
- Immediate tactics and independently verified threat-space proofs guide moves; completed games supply actual terminal outcomes. Replay stores eight symmetry views per position. [Training](../engine/training.py), [self-play](../engine/selfplay.py), [runtime interface](../engine/runtime.py).
- Connect6 turns are already decomposed into sequential placements. The remaining-stones feature distinguishes intermediate turn states; value backup changes sign according to player identity, not blindly after each placement. [Board model](../engine/game.py), [native backup](../engine/runtime.cpp).

This is an AlphaZero-style tactical/neural hybrid, not a plain neural policy. The relevant question is which part to improve, rather than whether to replace everything with a newer model.

## Evidence and applicability

### 1. Gumbel AlphaZero: better planning with few simulations

**Source:** Danihelka et al., *Policy improvement by planning with Gumbel*, ICLR 2022. This is older foundational work, retained because recent studies still use it as a strong comparator. [Author-hosted paper][1].

The method samples root actions without replacement using Gumbel-Top-k, allocates search through sequential halving, and constructs improved policy targets using action values. The full method also changes non-root selection. The paper reports substantial improvements over earlier approaches when simulations are few relative to available actions, with experiments in Go, chess, and Atari. Its policy-improvement guarantee depends on correctly evaluated action values; it does not imply every approximate neural search improves every position. [1, abstract and Sections 1–5].

**Why it fits:** this repo searches a board with up to 361 placement actions using 64 simulations. A root cannot visit every legal action in that budget, and visit-count targets can be sparse. This is the setting the paper directly addresses, although tactical filtering reduces the action space in some positions.

**Recommendation:** highest-priority new search algorithm to compare. Retain the game engine, tactical checks, network, and real terminal outcomes. Change action selection and policy targets together; merely replacing Dirichlet noise with Gumbel noise is not Gumbel AlphaZero. No speed or strength gain is established locally.

The official [MiniZero framework][2] provides an implementation reference for AlphaZero and Gumbel AlphaZero. Its documented environments include Gomoku, but not Connect6; it is not a drop-in replacement for this Windows project. [2].

### 2. KLENT: remove look-ahead search during training

**Source:** Ota et al., *Revisiting Regularized Policy Optimization for Stable and Efficient Reinforcement Learning in Two-Player Games*, ICML 2026. The February preprint was titled *Resource-Efficient Model-Free Reinforcement Learning for Board Games*; use the revised paper and conference record, not just that early version. [Conference record][3], [revised full paper][4].

KLENT combines reverse-KL regularization, entropy regularization, and lambda-return action-value targets. It builds an improved policy analytically instead of running MCTS to produce training targets. The practical algorithm learns policy and action-value outputs using an on-policy collection/fitting loop. [4, Sections 4 and 5].

**Reported result:** averaged across Animal Shogi, Gardner Chess, 9×9 Go, Hex, and Othello, KLENT reaches a 50% win rate against anchored opponents at about **75 million simulator evaluations**, versus **300 million** for Gumbel AlphaZero. This is a fourfold reduction in that budget metric, not a fourfold Connect6 wall-clock speedup. The revised paper also reports competitive learning against AlphaZero in 19×19 Go, and explores MCTS at test time. Its conclusions explicitly do not establish superior asymptotic strength with unlimited compute. [4, Sections 6.1, 6.3, 7, and Appendix M].

**Why it matters:** search is expensive during self-play. Model-free learning could collect much more experience within a desktop budget while retaining search when playing, a different trade-off from making MCTS faster.

**Recommendation:** strongest new alternative training paradigm found in this survey, but a separate research experiment, not a flag change. This repo currently has a scalar state-value head and replay of older games; KLENT needs action-value predictions, different targets, and its collection/update regime. Preserve same-player Connect6 placement semantics in returns and policy updates. No Connect6 experiment appears in this paper.

### 3. Rapfi: cheap incremental neural evaluation plus alpha-beta

**Source:** Jin, Duan, and Hang, *Rapfi: Distilling Efficient Neural Network for the Game of Gomoku*, 2025. [Full paper][5], [official engine][6].

Rapfi learns directional local patterns, distills them into a codebook-based network, and updates affected features incrementally. Combined with carefully tuned alpha-beta search, the paper reports stronger Gomoku play than CNN-based Katagomo under CPU-limited conditions without GPU accelerators. This is evidence for a complete representation/inference/search system, not an isolated architecture swap. [5, abstract and Sections 3–5].

Its supervised dataset contains roughly **30.8 million Katagomo-generated positions**. Therefore it is not evidence that a tiny model trained from nothing on a desktop will achieve the same result quickly. [5, Section 4.1].

**Recommendation:** most relevant route if CPU deployment or batch-one move speed becomes the main goal. Connect6 needs rule-specific directional patterns and careful handling of two-stone combinations. Keep this as a separate engine direction, not a replacement for the current GPU-batched learner. Reusing ideas is much less work than reproducing Rapfi's exporter, incremental state, and search.

### 4. Convolution plus attention: local tactics and global relationships

**Source:** *Bridging Local and Global Knowledge via Transformer in Board Games* (ResTNet), IJCAI 2025. [Conference paper][7].

ResTNet interleaves residual and Transformer blocks. In its 9×9 Go architecture comparison, RRTRRT obtains **60.80%** against the selected KataGo opponents, versus **54.60%** for six residual blocks. These are separate win rates against common opponents, not a direct 60.80% head-to-head win rate against the residual baseline. Reported inference latency increases from **3.067 to 4.566 ms**. Each architecture's training uses roughly **200 GTX 1080 Ti GPU hours**. Pure attention is weaker in that comparison. [7, Section 4.1 and Table 1].

The paper also evaluates 19×19 Go and 19×19 Hex: the hybrid has higher win rates against their reference opponents. The 19×19 Go experiment uses supervised human-game data; the Hex experiment uses Gumbel AlphaZero self-play. These are different training settings and neither is Connect6. [7, Table 2].

**Recommendation:** attention is a reasonable experiment for distant interacting threats, not an automatic upgrade. This repo already implements a small attention variant, but it is not ResTNet: it adds one block after the residual trunk. The simplest immediate architecture comparison is existing residual versus pooled value, then attention, each in a separate directory after the current session. No local variant has established higher strength.

### 5. Current KataGo: architecture efficiency and the training recipe both matter

**Source:** official architecture and method documentation, including Transformer networks introduced in August 2026. [Architectures][8], [methods][9], [original training-efficiency paper][10].

The current architecture table lists a roughly **28.5M-parameter Transformer** at similar peak strength to a **72.8M-parameter convolutional network**, and a roughly **70.4M Transformer** near the peak rating of a **232.5M convolutional network**. Those ratings use **equal search visits**, not equal thinking time, and do not isolate architecture from all training-history differences. Evaluation cost is backend- and hardware-dependent. KataGo's internal Elo scale is not a human chess rating. [8, summary and notes].

This supports investigating board-aware attention and bottlenecks; it does not support blindly replacing this repo's approximately 0.5M model with a large Transformer.

KataGo's broader training recipe also offers ideas: randomized search budgets for data generation, decoupling exploratory visits from policy targets, global context, and auxiliary predictions. The original paper reports a large combined Go training-efficiency improvement; no individual trick inherits the entire gain. [9–10].

**Recommendation:** after the baseline, consider one search/data improvement at a time. Connect6-specific threat or line targets are a hypothesis inspired by auxiliary learning, not a published Connect6 gain. Existing immediate-threat input planes already supply some domain knowledge. Do not copy Go ownership or score objectives into a game without those semantics.

### 6. Searchless Transformers: strong teacher distillation, not free planning

**Source:** Ruoss et al., *Amortized Planning with Large-Scale Transformers: A Case Study on Chess*, NeurIPS 2024. [Conference paper][11], [official code][12].

The authors train models up to **270M parameters** on ChessBench: **10 million games** and approximately **15 billion annotated data points**, using Stockfish 16 as teacher. Their no-explicit-search policy reaches **2895 Lichess blitz Elo against humans**. The paper also shows imperfect distillation of Stockfish; this is not a claim to replace a full-strength searching Stockfish engine. [11, abstract and dataset/evaluation sections].

**Recommendation:** attractive when a strong same-game teacher and a large dataset already exist, or when low-latency policy inference is essential. It shifts substantial planning work into data generation and training. For a fresh Connect6 learner without that teacher, it is not the first route. A board Transformer is not necessarily a language model or an LLM reasoning agent.

## Connect6-specific implications

After Black's opening, 360 cells are empty: a full two-stone turn has 360 × 359 / 2 = **64,620 unordered pairs**, ignoring early termination on a first-stone win. Sequential placement avoids a quadratic output head, but does not eliminate the combinatorial search problem. This repo already uses that decomposition; recommending it as a new feature would be redundant. [Local rules and board model](../engine/game.py).

Different orders of two placements can reach the same complete-turn position. Shared neural evaluation could avoid duplicate work if those states are actually visited. A full graph search is more complex: shared states do not justify blindly sharing all parent-edge statistics. The older [Monte-Carlo Graph Search for AlphaZero paper][13] evaluates chess and crazyhouse and illustrates this distinction; it does not establish a Connect6 gain.

Start by measuring duplicated neural evaluations before adding a cache or graph. A Connect6 state key must distinguish rules, board contents, player, and placements remaining. Retain first-stone terminal detection and player-aware backups throughout any search change.

### Coverage limitation

Search located a potentially relevant 2026 SSRN item, *Accelerating AlphaZero Training for Connect6 via Behavioral Cloning Initialization and Prior Action Masking in Resource-Constrained Environments* ([listing][14]). Both publisher fetch attempts returned HTTP 403. Its numerical claims and experiment design were not verified and are **not used as evidence for the recommendations**. In particular, this survey does not repeat its advertised Elo or short-training strength claims. No inspected source establishes a superior end-to-end 19×19 Connect6 recipe for this repository.

## Recommended experiment order

| Priority | Experiment | Reason | Scope |
| --- | --- | --- | --- |
| Now | Preserve current residual training as baseline | Otherwise there is no meaningful comparator | No training or engine changes |
| Cheapest existing option | Residual vs pooled, then small attention | Already implemented; isolates a modest architecture change | Separate run directories |
| Best next search change | Gumbel AlphaZero | Direct evidence for low-simulation planning | Native/reference search and training-target changes |
| Next data experiment | Search-budget allocation or auxiliary targets | Learning efficiency need not require a bigger model | One controlled change at a time |
| New training paradigm | KLENT | Recent peer-reviewed resource-efficiency evidence | Separate trainer and action-value model |
| CPU-first deployment | Rapfi-style incremental evaluator | Directly relevant Gomoku CPU evidence | Larger, rule-specific engine project |

Do not replace the known, exact game dynamics with a learned model merely to use MuZero. Its motivation is different; the Gumbel paper distinguishes known-model AlphaZero from learned-model MuZero. This survey provides no evidence that learning Connect6 dynamics would improve the current exact-engine setup. [1].

For each comparison:

1. Validate native/learner agreement and both Connect6 placement phases.
2. Compare strength after **equal total training time**, including data generation and evaluation pauses, not just equal updates.
3. Play common-opening, color-swapped matches at **equal full-turn thinking time**. Report simulation caps and observed timing; equal caps are not equivalent to equal time when model speeds differ.
4. Include uncertainty, independent openings or held-out opponents, and more than one seed where feasible. The built-in 55-point/100-game promotion gate is a provisional regression filter, not definitive evidence of superiority. [Local evaluation policy](selfplay-training-guidance-2026.md).
5. Record completed games, original positions (not eightfold symmetry count), neural evaluations, memory, and tactical failure cases. More self-play games or lower loss alone does not mean stronger play.

## Primary sources

[1]: https://davidstarsilver.wordpress.com/wp-content/uploads/2025/04/gumbel-alphazero.pdf
[2]: https://github.com/rlglab/minizero
[3]: https://proceedings.mlr.press/v306/ota26a.html
[4]: https://arxiv.org/html/2602.10894v2
[5]: https://arxiv.org/html/2503.13178
[6]: https://github.com/dhbloo/rapfi
[7]: https://www.ijcai.org/proceedings/2025/0828.pdf
[8]: https://github.com/lightvector/KataGo/blob/master/docs/NetworkArchitectures.md
[9]: https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md
[10]: https://arxiv.org/html/1902.10565
[11]: https://proceedings.neurips.cc/paper_files/paper/2024/file/78f0db30c39c850de728c769f42fc903-Paper-Conference.pdf
[12]: https://github.com/google-deepmind/searchless_chess
[13]: https://arxiv.org/html/2012.11045
[14]: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6730709

- [1] Danihelka et al., ICLR 2022: Gumbel policy improvement.
- [2] MiniZero official implementation and environment documentation.
- [3–4] Ota et al., ICML 2026: KLENT and revised theoretical/empirical results.
- [5–6] Rapfi paper (2025) and official engine.
- [7] ResTNet, IJCAI 2025.
- [8–10] KataGo official architecture/method documentation and original efficiency study; some techniques predate 2024.
- [11–12] Ruoss et al., NeurIPS 2024, and official searchless-chess implementation.
- [13] Czech et al., older graph-search study; not presented as recent work.
- [14] Unverified Connect6 research lead, excluded from supporting evidence.
