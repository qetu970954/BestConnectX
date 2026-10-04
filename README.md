# BestConnectX

[繁體中文](README.zh-TW.md) · [Commands](docs/cheatsheet.md) · [How the bot thinks](figures/how-bot-thinks.html)

Train a connection-game bot on your own machine, then play against it in the browser. **C++ handles the game and model inference. Python/PyTorch handles learning and saves.** No trained weights are included.

The default is **15×15 freestyle Gomoku**, using a **64-channel, 6-block residual model**. Connect6 and two experimental model designs are also available.

## Get it running

Run these from the project folder:

```sh
uv sync --locked
uv run python -m engine.native
uv run python dashboard.py --open-browser
```

The dashboard opens at **http://127.0.0.1:8765**. Before you train a model, the bot uses an untrained rule-based heuristic. The dashboard lets you play and inspect results; training runs in a terminal.

On Windows, the build needs MSVC x64 C++ tools. It uses the libraries included with PyTorch, so you do not need a separate CUDA SDK or CMake. Windows CPU/CUDA were checked with PyTorch 2.11.0+cu128. The Unix build path needs a C++17 compiler and has not been checked here. Rebuild after changing native code or PyTorch.

## Train 15×15 Connect6

The `connect6` preset uses **19×19** by default. Keep the board override for 15×15:

```sh
uv run python train.py --preset connect6 --board-size "15*15" --data data/connect6-15x15 --device cuda --hours 2
uv run python dashboard.py --data data/connect6-15x15 --open-browser
```

Black opens with one stone. After that, each turn has two placements. Six or more stones in a line wins, and a win ends the turn immediately.

If that directory already has a checkpoint, training resumes it. For a fresh experiment, choose a new `--data` directory. Use `--device cpu` if you do not have CUDA.

For default Gomoku, just run:

```sh
uv run python train.py
```

## Stop and pick up where you left off

Press **Ctrl+C** in the training terminal and wait for **Saved**. You can also request a stop from another terminal:

```sh
uv run python -m engine status --data data/connect6-15x15
uv run python -m engine stop --data data/connect6-15x15
uv run python train.py --data data/connect6-15x15 --device cuda --hours 2
```

Resume restores the saved rules, model, optimizer, and learning settings. You can change session time, device, workers, and concurrency. **Passing a new `--batch` or `--learning-rate` does not change those saved values.** New rules or a different model shape need a new directory.

If resume reports that an unfinished tournament's code changed, see [`--restart-gate`](docs/cheatsheet.md). Do not add that flag to ordinary resume commands.

## What happens during training?

1. The newest model plays games against itself. C++ combines tactical checks with neural MCTS search.
2. Completed games supply recent training samples, called **replay**. Results come from actual wins, draws, and losses—not unfinished tactical proofs.
3. Python updates the model using those samples. The CLI calls this phase **optimizing**. It is learning, not a tournament.
4. Every **1,000 completed games**, the trainer saves a frozen playing model. The filename counts games, not learning updates.
5. The first milestone becomes the initial best. Later milestones challenge the accepted best in an automatic tournament.

A tournament uses **100 games**, with 50 shared openings and swapped colors. A candidate needs **at least 55 points**, verified game histories, and at most **0.1 seconds** of full-turn overtime. A win earns 1 point and a draw 0.5. Default thinking time is **0.25 seconds per full turn**, including both Connect6 placements, with the same simulation cap for both bots (default 64).

This is a practical regression filter, **not proof of stronger play**. Losing the tournament does not reset the learner. Self-play keeps using its newest weights; only the accepted best stays unchanged.

Training and evaluation take turns on the GPU. The scheduler aims for roughly **80% training / 20% evaluation**. Without a pending tournament, it spends the time training.

Want the search explained visually? Open [How the bot thinks](figures/how-bot-thinks.html). **MCTS** explores future positions; **PUCT** chooses which branches to explore; **TSS** looks for forced wins that must pass independent checking.

## Which model file should I use?

| File inside your run | Use it for |
| --- | --- |
| `latest.pt` | Continue training: weights, optimizer, recent samples, unfinished games, pending work, and random state |
| `best.pt` | Play with the accepted model; the first baseline is not strength-validated |
| `models/model-00001000.pt`, etc. | Play with a frozen milestone; these cannot restore the full learner |
| `incumbent.json` | The authoritative record of which milestone is best |
| `run.json`, `status.json` | Saved configuration and current progress |
| `selfplay-stats.json`, `metrics/` | Recent game summaries and loss history |
| `gate-model-*.json`, `gate-archive/` | Resumable tournament histories and archived comparisons |

`latest.pt` autosaves about **every 60 seconds at safe boundaries**, plus startup, milestone, and normal-stop saves. Long operations can delay it. A power failure or forced kill loses work since the last complete save.

Saves use atomic replacement. The default disk cap is **20 GiB**: if it fills, training stops rather than silently deleting old models. Finished self-play move histories are not archived; recent samples and unfinished games stay in the checkpoint. Tournament histories are retained for verification. If a `best.pt` export is interrupted, resume repairs it from `incumbent.json`.

## Play, tune, and check

Pause training before asking the dashboard bot to move. Choose best, latest, or a numbered milestone. Connect6 bot moves complete the whole turn. Arrow keys move focus; Enter/Space places a stone. Close the dashboard with Ctrl+C, or use `--port 8766` for another dashboard. Requests stay local and use a same-origin token.

The starting settings are **6 CPU workers, 64 concurrent games, 64 simulations per placement, and learning batch 128**. These are different controls—not interchangeable batch sizes. More workers or a busier GPU does not automatically mean a stronger bot. The default learning rate is fixed at **0.001**; there is no automatic schedule.

Use [`configs/experiments.toml`](configs/experiments.toml) or explicit flags for new runs. The optional `pooled` and `attention` models need separate directories. Boards must be square, with edge 2–25. Gomoku has no forbidden moves or swap opening; overlines win.

```sh
uv run python -m unittest discover -s tests -v
uv run --frozen --group browser python tests/ui_smoke.py
```

Browser checks use installed Chrome, CPU, and temporary runs. No browser download is needed. Python reference implementations remain because the tests use them to check C++ correctness; they are not a production playing fallback.

- [Cheatsheet](docs/cheatsheet.md): copy-paste commands, presets, and resume rules.
- [Native engine](docs/native-engine.md): code layout and safety checks.
- [Model choices](docs/model-options.md): residual, pooled, and attention.
- [Learning rates](docs/learning-rates.md): what you can change and when.
- [Training decisions and measurements](docs/selfplay-training-guidance-2026.md): saving overhead and the short speed comparison.
- [Validation history](docs/migration-validation.md), [approved migration scope](docs/migration-requirements.md), and [terms](CONTEXT.md).
