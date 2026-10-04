# BestConnectX

[繁體中文](README.zh-TW.md) · [Cheatsheet](docs/cheatsheet.md) · [Native engine](docs/native-engine.md) · [Checks and timings](docs/migration-validation.md)

A local, from-scratch connection-game bot. **C++ runs playing and model inference. Python/PyTorch updates the model and manages experiments.** The browser is for results and play, not training controls.

Default: **15×15 freestyle Gomoku**, with **64 channels and 6 residual blocks**. A residual block adds its input to its learned result. Width and depth can change through config. No trained bot weights are shipped.

Optional `pooled` and `attention` models are available for both Gomoku and Connect6. They keep all eight game features, native inference, and safe resume. See [model presets and checks](docs/model-options.md) and the [architecture decision](docs/adr/0002-shared-board-model-experiments.md).

## Setup

```sh
uv sync --locked
uv run python -m engine.native
uv run python dashboard.py --open-browser
```

The native build is required. On Windows it uses installed MSVC x64 C++ tools and the headers/libraries in the installed PyTorch wheel. No separate CUDA SDK, CMake, or model export is needed for this route. Windows CPU/CUDA use was checked with PyTorch 2.11.0+cu128. The Unix build path needs a C++17 compiler; it has not been checked here.

The dashboard opens at **http://127.0.0.1:8765**. Close it with Ctrl+C. Use `--port 8766` for another dashboard. Run all commands from the repo root. Rebuild after changing native sources or PyTorch.

## Train, stop, resume

These commands start training **only when you run them**:

```sh
uv run python train.py
uv run python -m engine status --data data/connect5-15x15-s1-o1-v1
uv run python -m engine stop --data data/connect5-15x15-s1-o1-v1
```

Ctrl+C or `stop` asks for a safe checkpoint. Wait for the saved message. An abrupt kill can lose work since the last save.

Repeat the training command to resume. With a custom run, `train.py --data PATH` restores its saved rules, model, and learning settings. Explicit runtime limits, device, worker count, and game concurrency may change. A different model shape or rule set needs a new run directory.

```sh
uv run python train.py --preset gomoku-large --data data/larger-model
uv run python train.py --data data/larger-model --hours 0.25 --workers 2
uv run python dashboard.py --data data/larger-model --open-browser
```

See [`configs/experiments.toml`](configs/experiments.toml). `--config FILE` selects another file; `--preset NAME` selects a named preset. Explicit CLI flags override the file for a new run. There is no exact-repeat promise or old-checkpoint importer.

## Other game presets

```sh
# First single-game check: no model updates or saved game records
uv run python -m engine selfplay --preset tictactoe --games 1 --device cpu

# Request complete native games; summary only, no files of moves
uv run python -m engine selfplay --preset gomoku --games 10 --model heuristic --device cpu

# Later Connect6 experiment
uv run python train.py --preset connect6
uv run python dashboard.py --preset connect6 --open-browser
```

Boards are square, edge 2..25. `--connect` is 2..the edge. A line of that length **or longer** wins in any of four directions. No forbidden moves or swap opening. Black's first turn and later turns each allow one or two stones. A win stops play at once, including a partial two-stone turn. Different rules use separate default run directories.

## Playing and learning

```text
C++: rules → tactics and checked proofs → batched PUCT/inference → real terminal games
Python: recent training samples → model updates → safe checkpoint and frozen models
C++: updated weights received at safe points → next playing batch
```

**MCTS** (Monte Carlo tree search) explores future positions and backs up their values. **PUCT** is its branch-selection rule: it balances estimated value, model move priors, and visit counts. **TSS** is threat-space search: it looks for forced winning plans. Attack discovery is bounded. Independent checking must cover every reply that can block the attack and rule out an earlier opponent win. An unfinished proof means unknown, not loss. Proofs guide moves; training labels come only from actual terminal results.

- C++ inference uses FP32. CUDA learning uses BF16; CPU learning uses FP32.
- Model updates and generation alternate. Weights do not change during a search batch.
- Loss is policy cross-entropy plus value mean-square error. Eight square-board rotations/reflections are sampled during learning, not saved eight times.
- Default: 64 concurrent games, 6 CPU workers, 64 simulations per placement, and optimizer batches of 128. These are separate controls. More workers are not always faster.
- The active-time goal remains 80% training / 20% evaluation. Evaluation earns one second per four training seconds. With no pending gate, all active time goes to training.
- Save a frozen model every 1,000 completed games. The first is an initial baseline, not proven strength.
- Later models play 100 games from 50 paired openings with colors exchanged. Promotion needs score above 50% and full-turn overtime no greater than 0.1 seconds. Draws score 0.5. Default thinking time is 0.25 seconds per full turn.
- Gates resume and never enter training replay. Frozen weights, rules, and code checks protect each comparison. Use `--restart-gate` only to archive/restart an unfinished gate after changing code or limits.

The dashboard shows run/model details, measured search/inference/learning/save times, loss and gate charts, and summaries for the latest 1,000 completed training games. These summaries have winner, length, turns, and move-source counts—not moves or board histories. Self-play win rates describe data, not strength. Pause training before bot play. Both colors, whole bot turns, and best/latest/frozen model choices are supported. Arrows move keyboard focus; Enter/Space places a stone. Requests stay local and use a same-origin token.

## Storage

| Path inside a run | Purpose |
| --- | --- |
| `run.json` | Saved rules, model shape, settings, and environment facts |
| `latest.pt` | Weights, optimizer, bounded replay, active games/proof plans, pending updates/metrics, counters, and RNG state |
| `selfplay-stats.json` | At most 1,000 game summaries, published after checkpoints |
| `models/`, `incumbent.json` | Frozen models and the current best |
| `metrics/`, `gate-model-*.json`, `gate-archive/` | Loss data, resumable evaluation histories, and archived gate reports |

No permanent `selfplay/` move archive, per-game `replay/` exports, or source ZIP is created. Evaluation histories remain for gate checks. Recent replay samples and unfinished games are allowed inside the checkpoint.

Saves are atomic. The default disk cap is 20 GiB. On exhaustion, the previous checkpoint remains; there is no silent pruning. Dropping old samples from the bounded replay window is normal learning behavior, not deletion of existing files. Generated runs and `.native-cache/` are ignored by Git. Deleting them is not reversible through Git.

## Checks

```sh
uv run python -m unittest discover -s tests -v
uv run --group browser python tests/ui_smoke.py
uv run python -m experiments.benchmark_native --device cpu --output .native-cache/checks.json
```

Browser checks use installed Chrome, CPU, and temporary runs. No browser is downloaded. The [validation report](docs/migration-validation.md) records staged checks and paired search timings. Short tests do **not** establish improved playing strength per training hour.

Current requirements: [approved agreement](docs/migration-requirements.md). Model designs: [survey](docs/model-options.md). [Terms](CONTEXT.md) define the game's shared language. Superseded Python-engine reports and benchmarks have been removed; independent Python references remain for correctness tests.
