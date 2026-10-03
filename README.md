# BestConnect6

Original AlphaZero-style self-play + verified TSS for **square-board connection games**. Train from the CLI; play and inspect results in the browser.

**Status:** CPU-tested pipeline, not a proven strong learned bot. No GPU training was run for this delivery. Existing Connect6 data is preserved and is not Gomoku training data.

## Start

```sh
uv sync --locked

# 9×9 Gomoku
uv run python train.py --connect 5 --board_size "9*9" --stones_per_turn 1 --starter-stones 1

# Standard Connect6
uv run python train.py --connect 6 --board_size "19*19" --stones_per_turn 2 --starter-stones 1

# Play/results dashboard
uv run python dashboard.py --open-browser
```

Windows: `start.cmd` opens the dashboard. Default URL: **http://127.0.0.1:8765**.

- Boards are **N×N**, N = 2..25; `9*13` is rejected. Quote the `*` in shell commands.
- `--connect` is 2..N; a run of this length or longer wins horizontally, vertically or diagonally. No forbidden moves or swap opening.
- Black's first turn uses `--starter-stones`; all later turns use `--stones_per_turn`. Both accept 1 or 2. A win ends play immediately, even midway through a turn.
- Rules select separate default directories, e.g. `data/connect5-9x9-s1-o1-v1`. Use `--data PATH` for another run. Different square sizes work, but weights/data are not shared across configurations.

## Stop, resume, inspect

Run the **same training command** to resume. Sessions default to two hours; `--hours` changes the limit. Ctrl+C requests a safe checkpoint—wait for the save message before closing. An abrupt kill can lose work since the last save.

```sh
uv run python -m gomoku status --data data/connect5-9x9-s1-o1-v1
uv run python -m gomoku stop --data data/connect5-9x9-s1-o1-v1
uv run python dashboard.py --data data/connect6-19x19-s2-o1-v1 --open-browser
```

Seed and learning settings are fixed at run creation; resume uses the saved settings. `--parallel N` explicitly changes self-play concurrency on resume without dropping unfinished games. `--seconds` sets future gate limits; changing a pending gate requires `--restart-gate`. Session duration, device and disk cap may change. `--device auto` selects CUDA when available; use `--device cpu` for checks. These are user-started commands, not authorization for agent-run GPU work.

The browser has **no training controls**. Pause CLI work before bot play. It supports either color, complete bot turns, latest/best/milestone model selection, loss curves and candidate win-rate/score charts. Keyboard: arrows navigate; Enter/Space places a stone. Requests stay local and mutations require a same-origin token.

## Learning and model replacement

- Small from-scratch policy/value network, batched PUCT self-play and independently verified TSS. Every training/evaluation game plays to an actual terminal result; unknown search results are not win/loss labels.
- Target **80% training / 20% evaluation** of measured active wall time. Evaluation earns one second per four training seconds and yields between placements; unfinished gates resume while training continues. Counters survive safe stop/resume. No pending gate means 100% training; cooperative operations may briefly overshoot a slice.
- New runs use **64 concurrent self-play games** (`--parallel`), batched inference across unresolved roots and cached live boards. `--batch` is the separate optimizer minibatch size (default 128). Larger game batches retain proportional optimizer work.
- Loss = policy cross-entropy + value MSE. Four rotations × optional reflection provide **eight symmetries**, sampled on the fly for boards and matching policy targets. Value labels stay unchanged; no eightfold storage duplication.
- Save immutable models every **1,000 completed self-play games**. The first is an initial comparison baseline, not a validated strength claim.
- Later snapshots challenge the current best over **100 games / 50 color-swapped legal opening pairs**. New gates cap each engine at **0.25 seconds per complete turn** and the saved self-play simulation count per placement (default 64), rather than searching up to 100,000 simulations. This compares short-search play, not five-second strength.
- Snapshots produced while a gate is pending stay queued on disk. Each queued gate freezes the then-current best when it starts.
- Gate slices reuse their two frozen networks while still rechecking file checksums; model loading counts toward evaluation time. Whole-turn overtime survives pauses between stones.
- Replace the incumbent only above **50 points** (win 1, draw 0.5, loss 0), with verified terminal histories and no clock overrun above 0.1 seconds. Ties retain the old model; failed candidates and previous bests stay saved.
- Gates resume across sessions and never enter replay. Models/settings/source are frozen; mismatched rules, altered weights or changed gate code are rejected. Restore the recorded source to continue an older-code gate, or explicitly restart it as below.

This is an internal raw-score comparison, **not Elo, a confidence guarantee or proof of external strength**. Timed search is cooperative; GPU kernels/OS stalls cannot be preempted. Model loading is outside thinking time.

To adopt the faster framework for an **older pending gate**, after stopping the old process:

```sh
uv run python train.py --data data/connect5-9x9-s1-o1-v1 --parallel 64 --seconds 0.25 --restart-gate
```

Use `--restart-gate` **once**. It archives the previous gate byte-for-byte under `gate-archive/` and restarts the same frozen models without mixing old results. Weights, optimizer, replay and training games remain intact. Subsequent runs omit `--restart-gate`.

To finish a pending gate without training (**explicit evaluation-only mode bypasses the 80/20 scheduler**):

```sh
uv run python -m gomoku evaluate --data data/connect5-9x9-s1-o1-v1 --report gate-model-00002000.json
```

## Saved data

| File/directory | Purpose |
| --- | --- |
| `run.json` | Rules, seed, settings and environment |
| `latest.pt` | Weights, optimizer, replay, unfinished games/work and RNG states |
| `models/`, `incumbent.json` | Immutable snapshots, current best and previous best |
| `selfplay/` | Terminal move records without probabilities |
| `replay/` | Permanent internal training targets; distinct from the rolling replay window |
| `metrics/`, `gate-model-*.json`, `gate-archive/` | Loss history, resumable matches and preserved superseded reports |
| `source-*.zip` | Recorded game/search/training code |

Old `rules.json` metadata remains readable; existing files are not deleted. Stored data supports replay, loading existing models and continued training without starting over. Seeds do not promise bit-identical timed training across runtimes.

The default artifact cap is **20 GiB** (`--disk-gib` adjusts it). Atomic writes reserve the old file, incoming file and a metadata margin. On exhaustion, stop and retain the previous checkpoint; never silently prune. Pending dataset exports are retained in the checkpoint and retried. Software environments/caches are separate.

## Checks and code

```sh
uv run python -m unittest discover -s tests -v
uv run --group browser python tests/connection_ui_smoke.py
```

Browser checks use installed Chrome, CPU and disposable data directories. Set `CHROME_EXECUTABLE` if needed; no browser is downloaded.

`gomoku/game.py` owns rules; `gomoku/training.py` owns learning/storage/gates; `gomoku/web.py` serves play/results. Thin CLI adapters connect them. No plugin framework or new frontend dependencies.

[Requirements](docs/alignment.md) · [Validation](docs/connection-validation.md) · [Recent arXiv research](docs/training-research.md)

Legacy fixed-19×19 commands remain under `python -m connect6`; their `data/` weights/replay/reports are unchanged. Do not point the new trainer at legacy `data/latest.pt`. Historical strength results are in [tactical-results.md](docs/tactical-results.md), not evidence of a strong new Gomoku model.
