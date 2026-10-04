# How the engine is put together

[繁體中文](native-engine.zh-TW.md) · [Commands](cheatsheet.md) · [Visual guide](../figures/how-bot-thinks.html)

The short version: **C++ plays; Python trains and manages the run.** There is no separate actor service and no concurrent learner changing weights during search.

## Where to look in the code

| File | Job |
| --- | --- |
| `engine/runtime.h`, `engine/runtime.cpp` | Rules, boards, threats, legal moves, tactical search, proof checking, MCTS/PUCT, workers, and complete bot turns |
| `engine/inference.cpp` | LibTorch models and CPU/CUDA inference |
| `engine/native.cpp` | PUCT selection and reference-checked feature/count kernels |
| `engine/native.py`, `engine/runtime.py` | Build/load the library and pass typed arrays through `ctypes` |
| `engine/network.py`, `engine/selfplay.py` | Define the learner, prepare samples, and update weights; also retain test-only reference playing |
| `engine/game.py`, `engine/search.py`, `engine/tactics.py`, `engine/tss.py` | Shared types and independent Python correctness references |
| `engine/training.py`, `engine/generation.py` | Training, tournaments, saves, and requests for complete games |
| `engine/config.py`, `engine/cli.py`, `engine/web.py` | TOML presets, commands, and the local dashboard |

Python keeps serializable game state. C++ makes production game decisions: it copies leaf boards, plays moves, prepares features across roots, and evaluates a batch. The Python stop callback only checks whether work should stop.

The JSON bot command completes its whole one-/two-stone turn in C++. Training and tournaments return control between placements so unfinished games can be saved. Python playing references are test tools, not a production fallback.

## Build and weight updates

```sh
uv sync --locked
uv run python -m engine.native
```

The cache key includes native source/header content and the PyTorch version. Windows uses MSVC `/MD` and the installed PyTorch libraries. PyTorch loads its runtime before the native library. Windows CPU/CUDA are checked; the Unix path is not.

Python and C++ use matching layer names and shapes. After learning, the next generation batch copies weights and batch-normalization buffers into an independent native model. They stay fixed during search. Frozen tournament models stay fixed for the whole comparison. The original residual model-creation C interface is preserved; variants use a separate entry point.

All models take the same eight game-feature planes. `residual` is the default; `pooled` changes the value head; `attention` also adds one four-head board-attention block. Missing architecture in a format-2 checkpoint means residual. Changing architecture needs a new run. See [model choices](model-options.md).

Native inference uses **FP32** (32-bit floats). CUDA learning uses **BF16** through PyTorch autocast; CPU learning stays FP32. There is no TorchScript, AOTInductor, CUDA graph, or custom GPU kernel in this path.

## Settings and learning

[`configs/experiments.toml`](../configs/experiments.toml) holds defaults and presets. Explicit flags override new-run settings. Resume restores saved learning/search settings; workers and concurrency can be changed explicitly without discarding unfinished games. Model/rule changes need a new run. Timed runs are not promised to repeat exactly.

| Control | Allowed range / default |
| --- | --- |
| Board edge / connect length | 2–25 / 2–edge; square boards only |
| Stones per opening / later turn | 1 or 2 each |
| Channels / residual blocks | 4–256 / 0–32; default 64 / 6 |
| Concurrent games / CPU workers | 1–128 / 1–12; default 64 / 6 |
| Learning batch | 2–4096; default 128 |
| Recent replay | 1–200,000 positions; new-run default 50,000; resume keeps its saved limit |

The learner uses AdamW at a fixed default rate of **0.001**, with weight decay **0.0001**. Normally, eight completed games trigger 32 updates; larger completion batches scale the work upward. The default first 16 games use heuristic move selection. Learning samples recent completed games, rotates/reflects square boards, and minimizes policy cross-entropy plus value mean-square error. [Learning-rate controls](learning-rates.md) apply to new runs, not resume overrides.

## Safety stays in place

- **Proofs are checked independently.** Attack discovery may use a bounded shortlist. The verifier reconstructs blocking replies separately, including legal second-stone fillers, and checks whether the opponent can win first. Invalid proofs cannot guide moves. A time/node limit means unknown, so normal search continues.
- **Game results are real.** Tactical proofs guide placements, but value targets wait for actual terminal wins, draws, or losses.
- **Recovery comes before exports.** Format-2 `latest.pt` saves weights, optimizer, bounded replay, unfinished games/plans, pending work, counters, and random state before publishing summaries/metrics. Failed publication can retry from it. Older formats are rejected; there is no old-run importer.
- **Save about once a minute, not after every learning cycle.** Safe generation/update boundaries allow autosaves, including within long learning cycles. Every successful save resets the timer at completion. Startup, milestone, and normal-stop saves remain; long operations can delay a save.
- **Best is a playing model, not a recovery file.** `incumbent.json` chooses the frozen milestone; `best.pt` mirrors it atomically. Resume repairs missing/interrupted exports. Later acceptance requires at least 55 points in 100 verified paired games and at most 0.1 seconds of turn overtime. The first baseline is unvalidated.
- **No silent deletion.** Atomic replacement, exclusive run locks, disk caps, immutable milestones, and verified tournament histories remain. No permanent finished-self-play move archive is created.

Refactoring code can invalidate an unfinished tournament's source checksum. If resume reports that mismatch, use the explicit [restart procedure](cheatsheet.md). Do not mix old-code and new-code results or restart a gate automatically.

## Read timings carefully

Search timing includes tactics and feature preparation. Inference timing includes transfers and waiting for results—not just GPU kernels. Learning and saves have separate timers. Weight refresh/loading are outside warm search medians but inside whole-process benchmark budgets.

A recent save-cadence check reduced checkpoint calls from 12 to 2 and improved short-run throughput by about 38%. That is not a new search-speed or strength result. See [training measurements and regression guards](selfplay-training-guidance-2026.md). More workers can help big batches and hurt small ones; use [measured results](migration-validation.md), not utilization alone.
