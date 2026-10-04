# BestConnectX cheatsheet

[繁體中文](cheatsheet.zh-TW.md) · [README](../README.md)

Run from the repo root. Training starts only when you run a training command.

## Setup and first game

```sh
uv sync --locked
uv run python -m engine.native
uv run python -m engine selfplay --preset tictactoe --games 1 --device cpu
```

Windows needs installed MSVC x64 C++ build tools. The build uses the matching PyTorch wheel. Rebuild after native-code or PyTorch changes.

## Default: 15×15 Gomoku

```sh
# Start or resume; two-hour limit, 20 GiB cap
uv run python train.py

# Results and play, not training controls
uv run python dashboard.py --open-browser

# Inspect or request a safe stop
uv run python -m engine status --data data/connect5-15x15-s1-o1-v1
uv run python -m engine stop --data data/connect5-15x15-s1-o1-v1
```

Ctrl+C also requests a safe save. Wait for the saved message before closing. The browser is at **http://127.0.0.1:8765**. Pause training before bot play. Arrows move focus; Enter/Space places a stone.

## Select game and model size

| Preset | Rules | Channels / blocks |
| --- | --- | --- |
| `tictactoe` | 3×3, connect 3, one stone | 8 / 1 |
| `gomoku` | 15×15, connect 5, one stone | 64 / 6 |
| `gomoku-small` | Same Gomoku rules | 32 / 2 |
| `gomoku-large` | Same Gomoku rules | 128 / 10 |
| `connect6` | 19×19, connect 6; one opening stone, then two per turn | 64 / 6 |

```sh
uv run python train.py --preset gomoku-large --data data/large
uv run python train.py --preset connect6
uv run python dashboard.py --preset connect6 --open-browser
```

Different model sizes on the same rules share the default rule directory. **Use `--data` for a separate model-size experiment.** A win ends play at once, even midway through a two-stone turn.

## Simple config

Use [`configs/experiments.toml`](../configs/experiments.toml), or copy it to your own file. Edit `[defaults]` or a `[presets.NAME]` table:

```toml
[presets.my-model]
channels = 96
blocks = 8
workers = 2
parallel = 32
```

```sh
uv run python train.py --config configs/experiments.toml --preset gomoku-small --data data/small --simulations 32
uv run python train.py --channels 96 --blocks 8 --data data/model-96
```

For your own file and added preset, use `--config FILE --preset my-model`. Explicit CLI flags override the file for new runs. Unknown keys/types are rejected. Square board flags remain available, for example `--board-size "15*15" --connect 5`.

## Resume and useful limits

```sh
uv run python train.py --data data/large --hours 0.25 --workers 2 --parallel 32
uv run python dashboard.py --data data/large --port 8766 --open-browser
```

| Flag | Meaning / default |
| --- | --- |
| `--hours 0.25` | 15-minute session; default 2 hours |
| `--device cpu` | CPU checks; `auto` selects CUDA when available |
| `--workers 6` | Native CPU workers, not concurrent games |
| `--parallel 64` | Concurrent self-play games, not optimizer batch size |
| `--batch 128` | Learning minibatch |
| `--simulations 64` | Search simulations per placement |
| `--channels 64 --blocks 6` | Model width and residual depth |
| `--replay-limit 20000` | Recent training-sample bound |
| `--max-games 1000` | Stop at this **total** game count, not 1,000 extra games |
| `--seconds 0.25` | Gate limit per full turn, not per stone |
| `--disk-gib 20` | Disk cap; stop rather than delete files |

Resume restores model, rules, and learning settings. Session limits, device, workers, and concurrency may change. New rules or model shapes require a new run. `--seconds` changes future gates; changing a pending gate requires explicit `--restart-gate`. Do not use that flag for normal resume.

## Request X complete games, without learning

```sh
uv run python -m engine selfplay --preset gomoku --games 10 --parallel 8 --workers 2 --model heuristic --device cpu
uv run python -m engine selfplay --data data/connect5-15x15-s1-o1-v1 --games 10 --model latest
```

`selfplay` prints summary JSON and saves no game records or training state. `--parallel`, `--workers`, and `--tactical-ms` also override generation settings. Use `train.py` for checkpointed, resumable learning. For a non-default saved game, pass its matching preset or rule flags to `selfplay`.

## Files and checks

`latest.pt` holds bounded replay, active games, weights, optimizer, and resume state. `selfplay-stats.json` holds only the latest 1,000 game summaries. Frozen models and evaluation histories remain. No finished self-play move archive or permanent per-game replay exports are created.

```sh
uv run python -m unittest discover -s tests -v
uv run --group browser python tests/ui_smoke.py
uv run python -m experiments.benchmark_native --device cpu --output .native-cache/checks.json

# User-started GPU checks, with a process time limit
uv run python -m experiments.benchmark_native --device cuda --batch 64 --simulations 64 --wall-seconds 90 --output .native-cache/cuda-checks.json

# Finish an existing frozen gate without learning; bypasses the 80/20 scheduler
uv run python -m engine evaluate --data data/connect5-15x15-s1-o1-v1 --report gate-model-00002000.json
```

Browser checks need installed Chrome, not a downloaded browser. `CHROME_EXECUTABLE` may point to your installed browser. Tests and self-play win rates do not prove strength. See [checks and timings](migration-validation.md).
