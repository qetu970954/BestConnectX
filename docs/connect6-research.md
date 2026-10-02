# Strong Connect6 on one CUDA-capable GPU

Research date: 2026-09-28. Objective: maximize measured playing strength per local GPU-hour, not network size. This is a research recommendation and a CPU experiment, **not a trained champion**.

## Decision

**Reuse a working neural/search engine and pretrained weights before paying for self-play from scratch. Aim for a small policy/value network, batched search, and exact tactical checks.**

1. **Strength reference: MiniZero / Gumbel AlphaZero with threat-space input features.** MiniZero won Connect6 gold in the 2024 and 2025 Computer Olympiads. The latter had only three entrants: this is credible tournament evidence, not proof of superiority over every existing bot. [1,2]
2. **Practical warm-start candidate: Nebula Zero.** Its source includes Connect6 rules, C++ search, TensorRT, supervised/self-play training, evaluation gates, and a link to pretrained weights. Validate the weights and their terms before choosing it as the incumbent. Its README claims a student competition championship; that does not establish strength relative to MiniZero. [4]
3. **Do not start with a giant model, MuZero, an LLM, a new distributed trainer, or cloud GPUs.** Connect6 has cheap exact rules. A learned dynamics model adds work we do not need; money should go into measured search/training improvements.
4. **Replace the incumbent only after a held-out, equal-wall-clock match gate.** A smaller loss, a faster kernel, and synthetic tactical recall are not playing-strength evidence.

No price or training-duration claim below implies that world-leading strength is achievable within that budget. Public evidence does not establish the compute required to beat MiniZero24 on this GPU.

## Verified local environment

- The project directory was empty: no existing engine or incumbent checkpoint to replace.
- Local checks used a CUDA-capable GPU. Exact device, driver, memory, power, and utilization details are omitted for privacy.
- WSL Ubuntu is installed; its current allocation reports approximately 15 GiB RAM and 12 logical CPUs. These are guest-visible resources, not a measurement of all host RAM.
- WSL has Python 3 and g++; Windows has uv-managed Python 3.14.3. The ordinary Windows `python` command points to the Store alias, so the experiment uses `uv run --no-project`.
- No GPU training, checkpoint download, TensorRT build, or competitive match was run. GPU compatibility and throughput still need smoke tests.
- NVIDIA-SMI's CUDA number is driver capability, not proof of an installed toolkit or compatible PyTorch. PyTorch 2.7 introduced Blackwell support with CUDA 12.8 wheels. Use a current compatible build; do not copy MiniZero's old PyTorch 1.13/CUDA 11.6 environment unchanged. [3,9]

## What the strongest useful sources say

| Approach | Evidence | Use here |
|---|---|---|
| MiniZero / Gumbel AlphaZero | 2024/2025 Connect6 gold; reports explicitly mention threat-space inputs. Current source implements Connect6. [1–3] | Main algorithmic reference and target opponent. Ask authors for an accessible champion checkpoint/configuration. |
| Nebula Zero | MIT source, autoregressive two-stone policy, C++ MCTS, TensorRT, replay and paired evaluation. Weights linked on ModelScope. [4] | Cheapest plausible warm start, subject to checkpoint availability, licensing and benchmarks. |
| NCTU6 | Developer history documents alpha-beta, single/double-threat search, relevance zones, TD learning and dependency-based search; historical competition results. [5] | Tactical methods and an external opponent if available. Not evidence that old versions remain strongest. |
| Gumbel planning | Paper targets policy improvement when search visits few actions, using sampling without replacement and sequential halving. [6] | Compare against PUCT at small search budgets. No guarantee that approximate learned values improve every move. |
| KataGo efficiency methods | Author paper/methods describe playout-cap randomization, target pruning and auxiliary tasks. [7,8] | Borrow independently and ablate. Reported Go compute savings are not a Connect6 speedup guarantee. |
| Descent / Athénan | 2025 report describes value-only minimax learning and reuse of search-generated positions; its Connect6 entry was bronze. [2] | Credible alternative research direction, not the first rebuild on limited funds. |

### Important source corrections

Search-provider summaries repeatedly said MiniZero did not support Connect6 because its README's game list omits it. **Direct source inspection disproves that.** At commit `394b2e483d00cb658d5a24ccca297f864c3280c7`, the `connect6` environment exists and implements: [3]

- 361 single-placement actions;
- consecutive same-player placements within a normal turn;
- six-or-more-in-a-row termination after each placement;
- 24 input planes: 16 history planes, four threat planes, four player/remaining-placement planes.

Its action header explicitly rejects MuZero's no-argument `nextPlayer()` path. Do not assume every framework algorithm supports this environment.

The public environment is not proof that the exact tournament weights/configuration are public. I did not locate or validate a MiniZero24 Connect6 checkpoint.

Nebula source was inspected at `de71333cb3da247900ec3cb5991334e513f5ac19`. Its pair-policy experiment **already** computes conditional second-stone information from one trunk evaluation. Calling that a new invention would be wrong. Its historical report gives 34:26 in two combined evaluation batches, and later lower losses without a successful promotion. Its throughput figures use two A800s, not an CUDA-capable GPU. [4]

The ModelScope page required JavaScript and was not readable through the fetch tool. Weight availability, exact model size, quality and model/data licensing remain unverified. MiniZero code is Apache-2.0; Nebula code is MIT. Those code licenses do not automatically license all linked weights or game records.

## Connect6 architecture: preserve the actual game

Assume standard 19×19 Connect6: Black opens with one stone, subsequent turns have two stones, six or more wins. Confirm tournament-specific openings and time controls before claiming compatibility. MiniZero provides an implementation reference. [3]

With 360 empty cells, a normal turn has `360*359/2 = 64,620` unordered pairs. A full pair softmax is not the first thing to build.

**Initial implementation choice: retain the existing engine's sequential placement model.** Model `(board, player, stones remaining)` explicitly. Do not flip value sign automatically after every placement: the same player can act again. Stop immediately when the first placement wins. Mask occupied intersections and test the opening, partial turns, overlines, full-board draws and serialization.

For a future pair head, factorize `P(a,b|s) = P(a|s)P(b|s,a)`. An unordered pair's mass is the sum of its two ordered probabilities, not their maximum. Handle terminal first placements separately. A sequential tree and a pair tree have different simulation costs, so equal node counts alone are not fair comparisons.

**Tactics before approximation:**

1. Find your immediate one/two-stone wins first.
2. Find every opponent one/two-stone completion set in the 924 six-cell board windows.
3. A defensive pair must intersect every opponent completion set. This is a size-at-most-two hitting-set problem.
4. If there is no such pair and no immediate own win, the opponent can win on its next turn. This conclusion is exact for a nonterminal normal-turn state, not a proof about a deeper restricted search.
5. If there is a singleton defense, its second stone is strategically important. Do not permanently replace all legal fillers with one convenient filler.
6. Threat-space search beyond immediate wins needs a distinct `unknown` result on timeout. Incomplete candidate search must not report an unproved loss.

Threat-aware search is established prior art, not our novelty claim. Hashing must include side and remaining placements; if the neural input includes history, board-only reuse also needs careful evaluation.

## Most economical training route

These are **starting hypotheses**, not benchmarked optimal hyperparameters.

### Stage 0: acquire and freeze a baseline before training

- Try Nebula's published weights and run its existing tests and a complete small generation→training→evaluation cycle under WSL2. Keep original source/checkpoint hashes and evaluation results.
- Disable hosted experiment logging; use local JSON/CSV. Rebuild TensorRT on the actual GPU rather than trusting a downloaded engine.
- Evaluate the frozen network/search before changing architecture. If weights are unavailable or unusable, use MiniZero's Connect6 Gumbel AlphaZero environment as the fallback, with a materially larger training cost expectation.
- Ask MiniZero/CLAP/NCTU6 authors for runnable opponents or a match interface. Lack of an accessible opponent limits any eventual strength claim.

### Stage 1: spend at most a few hours measuring

Measure end-to-end games/hour, evaluated leaves/second, training examples/second, peak RAM/VRAM, move latency, and GPU utilization. Keep tactical scanning, CPU search, batching waits and model loading inside the appropriate wall-clock totals.

Use the existing model first. If training a new compact model becomes necessary, compare 6–10 residual blocks and 64–128 channels rather than assuming a large transformer is better. The selected network must maximize strength at fixed training time and fixed move time, not validation accuracy alone.

Suggested initial sweep:

| Setting | Initial candidates | Reason |
|---|---|---|
| Concurrent self-play games | 8, 16, 32 | Fill inference batches without monopolizing RAM. |
| Inference batch | 16, 32, 64 | Small networks can be CPU/launch limited. Measure latency as well. |
| Training microbatch | 64, 128, 256 if it fits | Use mixed precision and leave display/engine headroom. |
| Search budget | 32, 64, 128 simulations per placement | Test low-budget Gumbel vs PUCT; log total full-turn work. |
| CPU search threads | 4, 8 | Start below guest CPU count; profile contention. |
| Replay | 100k–300k compact positions initially | Increase only after measured storage/RAM needs. |

A float32 `24×19×19` input is 34,656 bytes. Expanding 300k such states already costs approximately 10.4 GB before policies and Python/container overhead. Store compact boards/records on disk and encode minibatches; do not pre-expand a million positions in this WSL guest.

Alternate generation and optimization phases on one GPU first. Multiple self-play games can share inference batches; separate simultaneous training/inference workers can fight for VRAM. Start with the upstream schedule and profile before adding concurrency.

### Stage 2: warm start, then bounded self-play

- Continue the validated checkpoint at a conservative learning rate; do not discard its learned strength for an aesthetically cleaner architecture.
- If only game records are available, verify permission/provenance and pretrain a policy; use final results or trustworthy teacher search for values. Split by game/opening, not random positions from the same game.
- Augment with the eight board symmetries, transforming all policy/auxiliary targets consistently.
- Keep a normal self-play replay stream plus a bounded tactical stream. Never label a merely threat-looking state as a proven win.
- Use verified wins/losses as exact labels; keep approximate search values distinguishable from solved outcomes.
- Initially disable resignations or audit them against completed games, so training does not manufacture incorrect terminal labels.
- Compare against several frozen older models and a tactical engine to expose self-play blind spots. Preserve a held-out opening set.

Borrow KataGo-style variable search budgets only as a separate experiment. For example, try 25% deeper and 75% cheaper searches, but preserve the distinction between policy-target quality and terminal outcome data. Do not blindly imitate exploratory visits or transfer Go-specific score/ownership targets. [7,8]

### Stage 3: allow more compute only after improvement

Use short pilots, then a bounded 24–72 GPU-hour run if the loop is correct and produces measurable gains. This is an experiment budget, **not a predicted time to championship strength**. Stop or revert if held-out strength per GPU-hour does not improve.

At a constant 250 W GPU draw, 72 hours is 18 kWh and seven days is 42 kWh, excluding the rest of the computer. Multiply measured whole-system kWh by the local tariff for actual cost. At an illustrative $0.20/kWh, those GPU-only figures are $3.60 and $8.40. Power-limit tuning may improve efficiency, but was not tested or applied.

No paid API, cloud GPU, or new hardware purchase is recommended yet. Optimize the actual bottleneck before spending money; a larger model that generates fewer useful games can be a net loss.

## Our first experiment: exact tactical candidate injection

**Question:** does adding exact winning completions and minimal defensive covers repair a small candidate shortlist cheaply?

Implementation: `experiments/threat_filter.py`, Python stdlib only. This is a research probe, not a full AI or a replacement for an upstream engine.

Baseline: rank cells by a deterministic attack/defense six-window heuristic; consider all 120 pairs from the top 16 cells. Treatment: retain those pairs and inject representative immediate winning/defensive pairs. The treatment uses one filler for singleton sets solely to measure the existence of a tactical escape; it is not a complete strategic candidate generator.

Validation actually run:

- Exhaustive comparison of the cover algorithm with brute force on **1,562** hypergraphs with up to three one/two-cell edges over six cells.
- **2,747** independent pair-placement/win checks on 30 synthetic 6×6 boards, plus explicit horizontal, vertical, diagonal and overline checks.
- **600** nonterminal tactical states sampled from legally sequenced random 19×19 games concentrated in the central 9×9, with seeds 6, 5070 and 2026. Check wins after each placement. The games are not expert games; neighboring samples are correlated.

| Seed | Immediate own win | Defensible threat | No immediate escape | Baseline successes | Injection successes |
|---|---:|---:|---:|---:|---:|
| 6 | 162 | 33 | 5 | 195/195 | 195/195 |
| 5070 | 164 | 30 | 6 | 194/194 | 194/194 |
| 2026 | 157 | 34 | 9 | 191/191 | 191/191 |
| Total | 483 | 97 | 20 | **580/580** | **580/580** |

Recorded run: Python 3.14.3, Windows, approximately 1.83 seconds total. Median injection time was approximately 7–8 microseconds **excluding board scanning and ranking**. These are CPU microbenchmarks, not engine throughput.

**Conclusion: no measured recall improvement over this already threat-aware baseline. No promotion.** The checks support the small cover algorithm's correctness on the tested cases, not competitive strength. The dataset is easy for this baseline, has no neural policy and does not measure deeper tactics. Keep the exact logic as a correctness reference; do not advertise it as a new stronger AI.

Reproduce:

```bash
uv run --no-project experiments/threat_filter.py > experiments/threat_filter_results.json
```

The experiment needs no installed third-party packages. Timings vary across runs. Raw results are committed alongside the script as a local artifact; no GPU training result is implied.

## Further ideas: hypotheses, not established discoveries

### A. Pair-order marginalization and shared trunk evaluation

Compare the existing conditional pair-head approach with sequential full-network evaluation, then test training on unordered pair probability mass. Both orders should contribute when both are nonterminal. Match by GPU-hours and move time, not by steps. **Shared-trunk conditional second-stone evaluation already exists in Nebula; only specific modifications and their results would be ours.** [4]

Potential win: more self-play per hour without loss of pair interaction modeling. Risk: inaccurate intermediate-state values or order-dependent history features erase any speed gain.

### B. Tactical-residual training and adaptive compute

Treat provable immediate outcomes as solved, and spend learning/search on the uncertain remainder. Add a small auxiliary target for minimum defensive cover size (0, 1, 2, impossible), computed from exact windows. Compare against MiniZero's existing threat input planes; extra supervision might be redundant.

Then test whether uncertainty plus tactical constraints can allocate a fixed overall search budget better than fixed simulations. Keep hard per-move limits. Do not treat low entropy as correctness or skip a legal move solely because the network dislikes it.

Ablations: frozen baseline; exact immediate guard; auxiliary target only; adaptive budget only. Only combine individually useful changes. Current tactical-proxy experiment does not establish any of these gains.

### C. Hard-negative replay from real engine disagreements

Collect positions where the frozen network's preferred move differs from a bounded tactical solver or stronger search. Verify outcomes and oversample a capped fraction during training. Keep ordinary self-play to avoid training only on rare puzzles.

Potential win: fix costly tactical errors with fewer games. Risk: solver bias, incorrectly treating unknown as loss, and repeated positions leaking across training/evaluation. This draws on established search distillation, not a claimed new learning paradigm.

## Promotion protocol: what would justify replacement?

1. Freeze the incumbent, candidate, code commits, seeds, opening set, rule interpretation and hardware limits.
2. Development: small paired screening matches to reject regressions cheaply. Do not call those a championship test.
3. Final gate: predeclare one held-out test, initially **200 independent openings played both ways = 400 games**. Use identical move-time and memory limits, including tactical-search cost. Candidate and incumbent must not compete simultaneously for an uncontrolled share of the same GPU.
4. Score wins 1, draws 0.5. Treat each color-swapped opening pair as the statistical unit; use a paired bootstrap confidence interval. Do not pretend the two games of an opening are independent Bernoulli trials.
5. Require the lower bound of a predeclared 95% interval to exceed 50%, no illegal moves/crashes, and no material tactical-suite regression. A result near 50% is inconclusive, not evidence of equality. Small gains may need many more games.
6. Repeated tuning against the same final suite invalidates its held-out status. Use new held-out openings or a predeclared sequential/multiple-testing method, rather than peeking until a candidate passes.
7. Test the winner against external neural and classical engines at more than one time control. Report inaccessible opponents rather than asserting they were beaten. Checkpoint promotion against ourselves is not a world-strength claim.
8. Preserve the previous checkpoint and logs; switch an incumbent manifest atomically only after passing. Never overwrite the only known-good artifact.

## Next practical milestone

Obtain and validate an upstream checkpoint, run a reproducible GPU smoke test and frozen baseline match, then profile one small self-play/training cycle. That is worth doing before another architecture or an overnight training commitment.

## Sources

[1] Cohen-Solal & Cazenave, **2024 Computer Olympiad report**. PDF downloaded and text inspected; identifies MiniZero's Connect6 victory and threat features.
https://www.lamsade.dauphine.fr/~cazenave/papers/ReportAthenanOlympiad2024.pdf

[2] Cohen-Solal & Cazenave, **2025 Computer Olympiad report**, published as ICGA Journal 48 (2026). PDF downloaded and text inspected; Connect6 placements, three entrants, MiniZero algorithm and Athénan description.
https://www.lamsade.dauphine.fr/~cazenave/papers/ReportAthenanOlympiad2025.pdf

[3] RLG Lab, **MiniZero source and training docs**, inspected commit `394b2e483d00cb658d5a24ccca297f864c3280c7`.
https://github.com/rlglab/minizero/tree/394b2e483d00cb658d5a24ccca297f864c3280c7/minizero/environment/connect6
https://github.com/rlglab/minizero/blob/394b2e483d00cb658d5a24ccca297f864c3280c7/docs/Training.md
https://github.com/rlglab/minizero/blob/394b2e483d00cb658d5a24ccca297f864c3280c7/LICENSE

[4] Nebula Zero, **source, training docs and pair-policy experiments**, inspected commit `de71333cb3da247900ec3cb5991334e513f5ac19`. Historical performance is author-reported, not independently reproduced here.
https://github.com/lzy-wj/Nebula-Zero-Connect6/tree/de71333cb3da247900ec3cb5991334e513f5ac19
https://github.com/lzy-wj/Nebula-Zero-Connect6/blob/de71333cb3da247900ec3cb5991334e513f5ac19/experiments/pair_policy/README.md
https://github.com/lzy-wj/Nebula-Zero-Connect6/blob/de71333cb3da247900ec3cb5991334e513f5ac19/experiments/pair_policy/RESULTS.md
Weight link from authors, not validated: https://modelscope.cn/models/Lazyshu/Nebula_Zero_Connect6

[5] I-Chen Wu / CGI Lab, **Introduction to Connect6 Programs Developed at CGI Lab**. Developer account of NCTU6 methods and historical competition record.
https://cgi.lab.nycu.edu.tw/~icwu/aigames/NCTU6.1.html

[6] Danihelka et al., **Policy Improvement by Planning with Gumbel**, ICLR 2022. Author-hosted paper surfaced by search; OpenReview page extraction failed, so no detailed numerical reproduction claim is made.
https://davidstarsilver.wordpress.com/wp-content/uploads/2025/04/gumbel-alphazero.pdf
https://openreview.net/forum?id=bERaNdoegnO

[7] David J. Wu, **Accelerating Self-Play Learning in Go**, 2019.
https://arxiv.org/abs/1902.10565

[8] KataGo author-maintained **KataGoMethods.md**. Describes playout-cap randomization, policy target pruning, auxiliary targets and implementation details.
https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md

[9] PyTorch, **2.7 release announcement**. Blackwell support and CUDA 12.8 wheels.
https://pytorch.org/blog/pytorch-2-7/

[10] NVIDIA, **CUDA-capable GPU family specifications**. Local device capacity and driver were also inspected directly with `nvidia-smi`.
https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5070-family/

Additional literature surfaced but not used as verified numerical evidence: **Deep learning approaches to the game of Connect6** (2024), publisher fetch blocked; and a recent resource-constrained behavioral-cloning preprint, not assessed for reproducibility.
https://www.sciencedirect.com/science/article/pii/S1875952124000752
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6730709
