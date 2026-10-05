# BestConnectX cheatsheet

[繁體中文](cheatsheet.zh-TW.md) · [README](../README.md)

Run these commands from the project folder. Training only starts when you run `train.py`.

## Build and check one game

```sh
uv sync --locked
uv run python -m engine.native
uv run python -m engine selfplay --preset tictactoe --games 1 --device cpu
```

Windows needs MSVC x64 C++ build tools. The build uses PyTorch's matching libraries. Rebuild after native-code or PyTorch changes.

## 15×15 Connect6: train, stop, resume

**The `connect6` preset is 19×19. Add `--board-size "15*15"` for 15×15.** The other rules stay the same: connect six, one opening stone, then two stones per turn.

```sh
# Start training; use a new directory if you want a fresh model
uv run python train.py --preset connect6 --board-size "15*15" --data data/connect6-15x15 --device cuda --hours 2

# Open the matching dashboard in another terminal
uv run python dashboard.py --data data/connect6-15x15 --open-browser

# Check progress, or request a safe stop
uv run python -m engine status --data data/connect6-15x15
uv run python -m engine stop --data data/connect6-15x15

# Resume: saved rules/model/learning settings are restored
uv run python train.py --data data/connect6-15x15 --device cuda --hours 2
```

Ctrl+C also requests a safe stop. **Wait for Saved before closing the terminal.** The two-hour limit applies to the training loop; startup and final saving add time. An existing checkpoint in that directory is resumed, not replaced with a fresh model. For another experiment, change `--data`.

The dashboard opens at **http://127.0.0.1:8765**. Choose English or 繁體中文; the browser remembers your choice. Its thinking-time menu offers **0.5, 1, 2, 4, 8, and 16 seconds per full turn**. Pause training before bot play. Arrow keys move focus; Enter/Space places a stone. Use `--port 8766` for a second dashboard. No CUDA? Replace `--device cuda` with `--device cpu`; small CPU checks can also use `--workers 1`.

For the larger Connect6 setup with the same **256-sample learning batch**:

```sh
uv run python train.py --preset connect6 --board-size "15*15" --data data/connect6-15x15-b256-p128 --device cuda --batch 256 --parallel 128 --workers 6 --replay-limit 400000 --hours 2
```

### If resume reports a changed unfinished gate

Use this **only when the CLI asks for it**, not on every resume:

```sh
uv run python train.py --data data/connect6-15x15 --device cuda --hours 2 --restart-gate
```

It archives the old tournament report and starts that comparison again with current code. It keeps the learner's weights, optimizer, and training data. Ordinary resume does not need this flag.

## Other presets and model experiments

| Preset | Game / model |
| --- | --- |
| `tictactoe` | 3×3 connect-three, 8 channels / 1 block |
| `gomoku` | 15×15 freestyle Gomoku, 64 / 6; the default |
| `gomoku-small` | Same Gomoku rules, 32 / 2 |
| `gomoku-large` | Same Gomoku rules, 128 / 10 |
| `connect6` | 19×19 Connect6, 64 / 6 |
| `gomoku-pooled`, `connect6-pooled` | Same rules, pooled value head, 64 / 6 |
| `gomoku-attention`, `connect6-attention` | Same rules, pooled value plus one four-head attention block, 64 / 6 |

```sh
uv run python train.py
uv run python dashboard.py --open-browser
uv run python train.py --preset gomoku-large --data data/large
uv run python train.py --preset connect6 --data data/connect6-19x19
uv run python train.py --preset gomoku-attention --data data/gomoku-attention
uv run python train.py --preset connect6-attention --board-size "15*15" --data data/connect6-15x15-attention
```

**Same rules share the default directory, even with a different model.** Always give model experiments a separate `--data`. A new architecture, model size, or board size needs a fresh run. Attention channels must be divisible by four. These variants work, but better strength has not been established. See [model choices](model-options.md).

## Settings: what can I change?

The starting values were checked on a Windows desktop with a multicore CPU and a CUDA-capable GPU. Exact hardware specifications are omitted for privacy. They are a starting point, not proven optimal settings. `auto` uses CUDA when available.

| Flag | What it does |
| --- | --- |
| `--hours 0.25` | 15-minute training-loop allowance; default 2 hours |
| `--device cpu` | Choose CPU; other choices are `auto` and `cuda` |
| `--workers 6` | Native CPU threads; can change on resume |
| `--parallel 64` | Concurrent self-play games; can change on resume |
| `--batch 128` | Samples per learning update; restored from checkpoint on resume |
| `--learning-rate 0.001` | Fixed AdamW rate for a new run; restored on resume |
| `--simulations 64` | Search simulations per placement; restored on resume |
| `--channels 64 --blocks 6` | Model width and residual depth; a change needs a new run |
| `--architecture attention` | Experimental model design; default `residual` |
| `--replay-limit 400000` | Stored symmetry views; default 400,000 = 50,000 original positions; restored on resume |
| `--max-games 1000` | Stop at **1,000 total games**, not 1,000 extra |
| `--snapshot-every 1000` | Games between frozen milestones; restored on resume |
| `--seconds 0.25` | Tournament time per full turn, not per stone |
| `--disk-gib 20` | Disk cap; stops instead of deleting files |

Completed positions enter replay as **eight views each**, with their board and policy rotated/reflected together and the terminal result unchanged. The **400,000-entry** default retains 50,000 original positions; learning still samples the configured batch size. Older entries are dropped when full. Existing checkpoints keep their saved limit—even if you pass another `--replay-limit`.

**`--parallel` and `--batch` do different jobs.** Changing batch size also changes how often old samples are reused. Passing `--batch 256` or a new learning rate on resume does not override the checkpoint. Use a fresh directory for those trials. See [learning rates](learning-rates.md).

For new runs, explicit flags override [`configs/experiments.toml`](../configs/experiments.toml). Use `--config FILE --preset NAME` for your own file. Unknown keys/types and invalid ranges are rejected. To resume without reapplying preset defaults:

```sh
uv run python train.py --data data/large --hours 0.25 --workers 2 --parallel 32
uv run python dashboard.py --data data/large --port 8766 --open-browser
```

`--seconds` changes future tournaments. Changing an unfinished tournament's limit requires `--restart-gate`, so old and new results are not mixed.

## Generate games without learning

```sh
uv run python -m engine selfplay --preset gomoku --games 10 --parallel 8 --workers 2 --model heuristic --device cpu
uv run python -m engine selfplay --preset connect6 --board-size "15*15" --data data/connect6-15x15 --games 10 --model latest
```

This prints summary JSON. It does not update weights, save game records, or provide resumable training. `--model` can be `heuristic`, `latest`, or `best`. For saved non-default games, include the matching preset/rules. Pause other GPU work first.

## Saves and tournaments, in plain terms

- **`latest.pt`: continue learning.** Full state, including optimizer, recent samples, and unfinished games. Saves once with each model milestone (normally every 1,000 completed games) and on safe stop. No startup/resume rewrite or timed saves; forced shutdown loses work since the last save.
- **Numbered milestones: keep a fixed opponent.** `models/model-00001000.pt` means 1,000 completed self-play games, not 1,000 updates. It cannot recover the full learner.
- **`best.pt`: play with the accepted model.** `incumbent.json` decides which milestone it mirrors. Resume repairs interrupted exports. The first milestone is an unvalidated baseline.
- **Later promotions: at least 55 points in 100 games.** Fifty paired openings, swapped colors, checked histories, and maximum full-turn overtime of 0.1 seconds. A win earns 1 and a draw 0.5. This is a provisional filter, not proof of strength.
- **Finish evaluation, then keep learning.** After each milestone after the initial baseline, the full 100-game tournament runs with self-play and optimization paused. A rejected candidate does not reset the learner. An interrupted tournament finishes before learning resumes.
- **Keep files; stop at the cap.** No automatic deletion of old milestones. `selfplay-stats.json` keeps the latest 1,000 summaries, not move histories. Recent replay lives in the checkpoint. Tournament histories remain for verification.

## Run the checks

```sh
uv run python -m unittest discover -s tests -v
uv run --frozen --group browser python tests/ui_smoke.py
uv run python -m experiments.benchmark_native --device cpu --output .native-cache/checks.json

# Optional user-run GPU check with a process deadline
uv run python -m experiments.benchmark_native --device cuda --batch 64 --simulations 64 --wall-seconds 90 --output .native-cache/cuda-checks.json

# Continue an existing tournament without starting training
uv run python -m engine evaluate --data data/connect6-15x15 --report gate-model-00002000.json
```

Browser checks use installed Chrome and temporary CPU runs; no browser download. `CHROME_EXECUTABLE` can select your installed browser. The `evaluate` command requires that report to exist. Tests, lower loss, and self-play win rates are not strength results. See [validation history](migration-validation.md) and [training measurements](selfplay-training-guidance-2026.md).
