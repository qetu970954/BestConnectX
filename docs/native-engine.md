# Native engine and experiment config

[繁體中文](native-engine.zh-TW.md) · [Cheatsheet](cheatsheet.md)

A **checkpoint** is saved state needed to resume. **Replay** means recent training samples. An **optimizer** changes weights using loss gradients: values that show which direction can reduce loss. **Inference** predicts scores without changing weights.

## Language split

| File | Role |
| --- | --- |
| `engine/runtime.h`, `engine/runtime.cpp` | Rules, cached line windows, boards, threats, legal moves, heuristic, TSS, independent proof checking, PUCT, sampling, workers, and whole bot turns |
| `engine/inference.cpp` | C++ residual, pooled, and attention models, weight buffers, and CPU/CUDA inference through LibTorch |
| `engine/native.cpp` | PUCT selection used by native search, plus feature/count kernels checked against Python |
| `engine/native.py`, `engine/runtime.py` | Build/loading and typed state transport through standard-library `ctypes` |
| `engine/network.py`, `engine/selfplay.py` | Python model definition, sample preparation, optimizer updates, and test-only reference playing |
| `engine/game.py`, `engine/search.py`, `engine/tactics.py`, `engine/tss.py` | Shared rule/state types and independent Python correctness references |
| `engine/training.py`, `engine/generation.py` | Request work, count games, manage learning/evaluation, and save state |
| `engine/config.py`, `engine/cli.py`, `engine/web.py` | TOML settings, commands, and local dashboard |

Python holds serializable game state for management. Production rule calculations and decisions use C++. Search copies and plays its own leaf boards without Python calls. Workers prepare leaves/features across roots, then one native call evaluates the batch. The stop callback is a management check, not a game or model call.

The JSON bot command runs its full one-/two-stone turn inside C++. Training and gates yield at placement-safe points so active games can be saved. The C++ engine is required for production; there is no Python playing fallback. There is no separate actor service or concurrent learner.

## Build and model agreement

```sh
uv sync --locked
uv run python -m engine.native
```

The binary cache key covers all native source/header content and the PyTorch version. Windows uses Release-style MSVC `/MD` and the installed wheel's libraries. Importing PyTorch loads its matching runtime libraries before the native client. CPU and CUDA were checked on Windows; the Unix build path remains unchecked.

The Python and C++ models keep board shape and have matching layer names and shapes. The config has channel count, residual block count, and board edge. Python copies weights and batch-normalization running buffers into an independent native inference model. After optimizer work, the next generation call refreshes those buffers before search starts. Frozen gate models are not refreshed by the learner.

Config also selects `residual`, `pooled`, or `attention`. The experimental variants use a globally pooled value head; `attention` appends one four-head Transformer block with learned 2D relative bias. All use the same eight game features and C++ inference. Missing architecture in an older checkpoint means `residual`. Changing architecture needs a new run. See [models and checks](model-options.md) and the [ADR](adr/0002-shared-board-model-experiments.md).

**FP32** is 32-bit floating-point math. Native inference uses FP32 on CPU/CUDA. **BF16** is a smaller floating-point format; CUDA learning uses it through PyTorch autocast. This release does not use TorchScript, model export, AOTInductor, CUDA graphs, or a custom GPU kernel.

## Config and limits

[`configs/experiments.toml`](../configs/experiments.toml) has `[defaults]` and `[presets.NAME]` tables. Keys match CLI destinations, such as `board_size`, `channels`, `blocks`, `workers`, and `parallel`. Bad keys, types, and value ranges fail before training. Explicit flags override file values for new runs.

- Board edge: 2..25; connect length: 2..edge; opening/later stones: 1 or 2.
- Model: 4..256 channels, 0..32 residual blocks. The shipped large preset is 128/10, not the maximum allowed model. Check memory before large runs.
- Concurrent roots: 1..128. Native workers: 1..12. Learning batch: 2..4096.
- Recent replay: 1..200,000 samples; default 20,000. Active-game samples also consume RAM/checkpoint space.
- Default learning rate: constant AdamW at 0.001, with weight decay 0.0001. Set `learning_rate` or `--learning-rate` for a new run. Default bootstrap: 16 games. Every eight completed games earns 32 update steps; larger completed batches retain proportional work.

Resume restores saved learning/search settings. An explicit worker/concurrency change keeps unfinished games and rotates them through the new batch size. Model/rule changes need a new run. The config system does not promise exact timed repeats.

An explicit `--learning-rate` does not override the saved rate on resume. There is no automatic learning-rate schedule or separate policy/value rate. See [learning-rate settings and research](learning-rates.md).

## Proof and save safety

Attack candidates may be ranked and bounded. Proof checking reconstructs defensive replies independently of discovery's cover generator. It includes legal second-stone fillers and earlier opponent wins. Invalid proofs cannot guide moves; a time/node limit means unknown and normal search continues. Games still reach real terminal states before value labels are assigned.

Checkpoint format 2 holds bounded replay and active games with plans/strategies, plus weights, optimizer, pending metrics/updates, counters, and RNG state. Older formats are rejected. No old-run importer or permanent finished-game archive exists.

The checkpoint is saved before summary/metric publication. A failed publication can retry from that checkpoint. Atomic writes, exclusive run locks, disk caps, immutable models, and checked gate histories remain. No silent file pruning is added.

## Measurement limits

Native timings split search/tactics/features wall time from inference wall time. Inference time includes required CPU/GPU transfers and output waits, not just GPU kernel activity. Learner and checkpoint times are measured separately with a high-resolution clock. Setup and weight refresh are included in the outer check-process budget, but not in warm steady-search medians.

More workers can help large batches and hurt small ones. Use the [measured results](migration-validation.md), not CPU/GPU usage alone, to select settings. Long strength comparisons remain user-run work.

Game-theory context: [Gomoku's first-player win and its meaning for value loss](gomoku-first-player.md).
