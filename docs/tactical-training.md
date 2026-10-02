# Tactical search embedded in neural training

Initial tactical self-play integration checked on 2026-09-29. The accepted opponent and promotion standard were not changed. **Follow-up:** recursive TSS is now implemented in Python and integrated into self-play and candidate play; see [TSS reference notes](tss-python.md) and [ADR 0001](adr/0001-python-reference-and-native-search.md).

## What now happens before neural search

`connect6/training.py:selfplay_batch` classifies each active self-play position:

1. **Immediate win:** select a legal winning completion and produce exact winning supervision.
2. **Unavoidable immediate loss:** only after excluding our own immediate win, verify that no defensive set fits the remaining placements. Teach value −1 from the current player's perspective, with no policy preference among losing moves.
3. **Proven fork/TSS:** first use the bounded one-turn solver, then try TSS within the same tactical budget. TSS checks counterwins and every complete reply that blocks the immediate threats; unverified or incomplete trees are discarded.
4. **Single forced placement:** make a one-hot policy target without spending MCTS simulations on a choice that cannot change.
5. **Unknown:** keep normal batched neural PUCT, including exploration. Discovery failure/timeout never means a loss or permission to discard unproven moves. During the first 32 bootstrap games, unknown positions retain the original heuristic teacher instead.

Only unresolved neural roots are sent to the GPU. The fork-discovery budget defaults to 2 ms per position, with high-resolution cooperative cancellation. Exact immediate checks and certificate verification have additional bounded board-scanning cost. A running scan cannot be interrupted halfway, so this is not a hard real-time guarantee.

## How the network learns the tactics

For a two-stone winning certificate, replay receives **both** placements:

- The original position with a one-hot target for the first stone.
- The position after that stone, still the same player with one placement remaining, and a target for the second stone.
- Both receive value +1; the remaining certificate is retained with each sample for auditing.

Earlier positions in the same game receive the proven winning outcome from their own player's perspective. Thus the tactical solver supplies training targets rather than merely overriding the bot at play time.

A proved-losing position receives an all-zero policy target and value −1. Training normalizes policy loss over policy-bearing records only, so these examples teach the value head without inventing a preferred losing move. The shared trunk still learns from the value gradient.

The training game can end when its outcome is proven. This is recorded as **proof adjudication**, not by falsely setting a nonterminal board's `done` flag. Actual evaluation matches still play to a terminal position and do not enter replay.

## Resume and scope

- Existing weights, optimizer, replay, active games and pending evaluation gates remain usable.
- Old checkpoints adopt the tactical budget on their first upgraded resume. New checkpoints retain it as a run setting.
- `selfplay_counts` records the sources of committed decisions. `counted_from_game` identifies where counters began for migrated runs.
- The dashboard distinguishes tactical/TSS-adjudicated finishes, forced placements and neural decisions. The overall completed-game count includes proof adjudications.
- `--tactical-ms 0` disables bounded fork discovery only; exact immediate wins/losses and forced-placement shortcuts remain active.
- The accepted heuristic's algorithm and evaluation's played-out-result rule remain unchanged. New neural candidates use the separately recorded TSS setting; no candidate has been promoted.

## Checks actually run

**The original 19 tests passed** for the first self-play integration. The TSS follow-up now passes **26 unit tests** and is documented separately in [TSS reference notes](tss-python.md).

- Only unresolved positions reaching a recording network.
- Both stone targets, turn-phase encoding and certificate validity.
- Real gradient updates on proof-derived targets.
- Value-only loss examples producing zero policy-head gradient and a nonzero value-head gradient.
- Own immediate wins taking precedence over an opponent's threats.
- Unknown/cancelled searches preserving normal exploration or safely discarding uncommitted batches.
- Full training/checkpoint/resume through a simulated older checkpoint schema, with stored proof targets reverified after loading.

The real-browser check also exercises the existing start/stop/checkpoint and candidate-preview paths.

## Historical CUDA-capable GPU training run (before recursive TSS)

The existing main run was resumed, rather than reset:

| Metric | Before | After |
|---|---:|---:|
| Finished/adjudicated self-play games | 48 | 64 |
| Optimizer updates | 160 | 224 |
| Replay positions | 12,399 | 13,087 |

New committed decisions recorded during this session:

- 243 normal neural/MCTS decisions.
- 14 proven-win adjudications.
- 1 proven-loss adjudication.
- 3 forced placements without MCTS.

Thus **18 committed root decisions bypassed neural search**, and 15 games did not need to play out their proved result. This is not a claim that all computation on those games was avoided: their earlier positions still needed search, and the solver itself costs CPU time.

The session reported approximately 14.1 seconds after setup and saved cleanly. The final training loss was approximately 2.922; it is not a strength metric. Earlier unfinished histories were preserved and received outcome labels when their games finished. The accepted tactical heuristic was unchanged. This run predates recursive TSS.

## Historical frozen-network microbenchmark (before recursive TSS)

`experiments/benchmark_selfplay.py` compared the previous search-all-roots path with the new decision path on the same frozen network, alternating measurement order after warmup. Five measurements per path were recorded. Source, raw timings, workload descriptions and checkpoint hash are in `selfplay-benchmark.json`.

| Eight-root workload | Previous median | Tactical median | Neural positions evaluated, median |
|---|---:|---:|---:|
| Quiet openings | 252.947 ms | 262.199 ms | 520 → 520 |
| Selected tactical mix | 269.222 ms | 204.845 ms | 490 → 195 |

For this historical pre-TSS run, the selected tactical workload was about **24% faster** with about **60% fewer neural position evaluations**. Quiet positions were about **3.7% slower**. The updated TSS-integrated measurements are in [tss-python.md](tss-python.md).

These are decision-generation microbenchmarks, not overall training throughput or convergence measurements. The tactical examples are selected and correlated. Batch utilization, position distribution and CPU overhead matter; this does not establish a general speedup or stronger learned play.

## Reproduce

```sh
uv run python -m unittest discover -s tests -v
uv run --group browser python tests/ui_smoke.py
uv run python -m experiments.benchmark_selfplay --checkpoint data/latest.pt
uv run python -m connect6 train --hours 2
```

Use a separate `--data` directory for a new budget/ablation configuration. Next evidence needed: sustained self-play efficiency and a held-out strength gate for the newly trained neural candidate. The neural model has not passed that gate yet.
