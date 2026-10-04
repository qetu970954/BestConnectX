# BestConnect6 cheatsheet

Run commands from the project root. The repo starts fresh: no saved games or trained models. Commands below start work only when **you** run them.

## Setup

```sh
uv sync --locked
```

## 9×9 Gomoku

Five or more in a row wins. Black starts; each turn places one stone. No forbidden moves or swap opening.

```sh
# Start training, or resume the same run; stop after two hours
uv run python train.py --connect 5 --board_size "9*9" --stones_per_turn 1 --starter-stones 1 --hours 2

# Open play/results in another terminal
uv run python dashboard.py --connect 5 --board_size "9*9" --stones_per_turn 1 --starter-stones 1 --open-browser

# Inspect progress / request a safe stop
uv run python -m engine status --data data/connect5-9x9-s1-o1-v1
uv run python -m engine stop --data data/connect5-9x9-s1-o1-v1
```

These are the default rules, so the short forms are:

```sh
uv run python train.py
uv run python dashboard.py --open-browser
```

## 19×19 Connect6

Six or more in a row wins. Black's opening places one stone; every later turn places two. A win ends the game immediately, even after the first stone of a two-stone turn.

```sh
# Start training, or resume the same run; stop after two hours
uv run python train.py --connect 6 --board_size "19*19" --stones_per_turn 2 --starter-stones 1 --hours 2

# Open the matching play/results dashboard
uv run python dashboard.py --connect 6 --board_size "19*19" --stones_per_turn 2 --starter-stones 1 --open-browser

# Inspect progress / request a safe stop
uv run python -m engine status --data data/connect6-19x19-s2-o1-v1
uv run python -m engine stop --data data/connect6-19x19-s2-o1-v1
```

**Resume by repeating the same training command, including the Connect6 rule flags.** Do not use the default Gomoku training command for a Connect6 run.

## Everyday controls

- **Stop:** Ctrl+C or the `stop` command. Wait for the checkpoint save before closing the terminal.
- **Resume:** repeat the original training command. No `--restart-gate` for a fresh start or normal resume.
- **Browser:** http://127.0.0.1:8765. It displays progress, loss/gate charts and statistics for the latest 1,000 saved self-play games (lengths, black/white wins, draws and decision sources). Statistics update after checkpoints; they do not measure strength. The browser does not start training. Pause training/evaluation before bot play.
- **Separate experiment:** append `--data data/my-run` to both training and dashboard commands, keeping the same rule flags.
- **Square boards only:** quote `"9*9"` / `"19*19"`; rectangles such as `"9*13"` are rejected.

### Useful training flags

| Flag | Purpose / default |
| --- | --- |
| `--hours 0.25` | 15-minute session; default is 2 hours |
| `--device cpu` | Force CPU; default `auto` uses CUDA when available |
| `--parallel 64` | Concurrent self-play games; default 64, adjustable on resume |
| `--batch 128` | Optimizer minibatch; default 128 |
| `--simulations 64` | MCTS simulations per placement; default 64 |
| `--seconds 0.25` | Gate time per **complete turn**, not per stone |
| `--seed 5070` | Initialization seed; default 5070 |
| `--disk-gib 20` | Artifact cap; pauses safely rather than deleting data |

Seed, batch size and learning/search settings are restored from a saved run, not overwritten by new CLI values. Session duration/device/disk cap and explicit `--parallel` can change. `--seconds` can change future gates; changing an unfinished gate requires an explicit `--restart-gate`, which archives its old report. See [README.md](README.md) for gate recovery and evaluation-only commands.

## Overall design

```text
Square-board rules → batched MCTS + verified TSS → terminal games
    → replay targets → policy/value updates → latest.pt (resume)
    → snapshot every 1,000 games → frozen comparison against current best
```

- One **`engine/`** package; Gomoku and Connect6 are rule configurations.
- Eight square-board symmetries augment board/policy samples on the fly.
- Approximately **80% training / 20% evaluation** of active time; no pending gate means all training.
- First snapshot is the comparison baseline, not proven strength. Later snapshots play 100 paired-opening/color-swapped games against the best; promotion needs more than 50 points and passes legality/timing checks. Evaluation never enters replay.
- Continuing training uses the latest state; promotion changes the best model. Models and datasets are not shared across rule configurations.

## Where games and models are saved

Default run directories:

- Gomoku: `data/connect5-9x9-s1-o1-v1/`
- Connect6: `data/connect6-19x19-s2-o1-v1/`

Inside each run:

| Path | Contents |
| --- | --- |
| `selfplay/game-*.json` | Actual terminal moves, without policy probabilities |
| `replay/game-*.pt` | Internal training targets |
| `latest.pt` | Weights, optimizer, pending exports/work, unfinished games and RNG state; fresh runs restore replay from recent `replay/` exports |
| `models/model-*.pt` | Immutable model snapshots |
| `incumbent.json` | Current best and previous best references |
| `run.json` | Rules, seed, settings and environment |
| `metrics/`, `gate-model-*.json` | Loss history and evaluation records |
| `source-*.zip` | Source used by the run |

Generated artifacts are ignored by Git. Deleting a run directory deletes its games/models/resume state; Git cannot restore ignored data.

## Framework checks

```sh
uv run python -m unittest discover -s tests -v
uv run --group browser python tests/ui_smoke.py
```

Checks use CPU and disposable run directories. Browser checks require installed Chrome/Chromium. Passing tests validates the framework, not playing strength.
