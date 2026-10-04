# Which model should I use?

[繁體中文](model-options.zh-TW.md) · [Commands](cheatsheet.md) · [Visual guide](../figures/how-bot-thinks.html)

**Start with `residual`.** `pooled` and `attention` also work with C++ inference, but neither has shown better playing strength here. All three use the same eight input planes and work with Gomoku or Connect6.

## What changes between them?

| Architecture | What it changes | Presets |
| --- | --- | --- |
| `residual` | Spatial convolutions and residual blocks; the value head flattens the board features | `gomoku`, `connect6` |
| `pooled` | Same body/policy head, but the value head averages features over the board | `gomoku-pooled`, `connect6-pooled` |
| `attention` | Pooled value head plus one four-head attention block after the residual body | `gomoku-attention`, `connect6-attention` |

A **residual block** adds its input back to its learned result. **Channels** set the body's width; **blocks** set its residual depth. The **policy head** scores where to place a stone. The **value head** predicts the outcome from the current player's perspective.

The pooled head projects to 16 channels, averages each channel over the board, then uses a 64-unit hidden layer. The attention block uses pre-normalization, learned 2D relative-position bias, a two-times-width GELU feedforward path, and no dropout. Attention channels must be divisible by four. This is our small experiment, not a KataGo or ResTNet implementation.

Pooling reduces direct spatial detail in the value head. Attention lets distant cells interact, but adds compute. Neither is a free upgrade, and pooling does not make saved weights portable across board sizes or rules.

## Commands

Give each experiment a **new directory**:

```sh
uv run python train.py --preset gomoku-pooled --data data/gomoku-pooled
uv run python train.py --preset gomoku-attention --data data/gomoku-attention
uv run python train.py --preset connect6-pooled --board-size "15*15" --data data/connect6-15x15-pooled
uv run python train.py --preset connect6-attention --board-size "15*15" --data data/connect6-15x15-attention
```

The Connect6 presets are 19×19 unless overridden. Variants with the same rules share the default directory, so do not omit `--data` for model experiments. You can also set `--architecture residual|pooled|attention`, `--channels`, and `--blocks` directly.

To resume, use `train.py --data PATH` without choosing another preset. It restores the saved architecture, weights, and optimizer. Changing architecture, width, depth, or board size needs a new run. A format-2 checkpoint without an architecture field means residual. There is no weight-transfer, pretrained-image initialization, automatic growth, or architecture-search feature.

## How big are they?

Measured from [`Network`](../engine/network.py), at 64 channels / 6 residual blocks:

| Architecture | 15×15 parameters | 19×19 parameters |
| --- | ---: | ---: |
| `residual` | 477,764 | 495,172 |
| `pooled` | 450,898 | 450,898 |
| `attention` | 487,734 | 489,846 |

Other residual sizes:

| Board | Channels / blocks | Parameters | Use |
| --- | --- | ---: | --- |
| 3×3 | 8 / 1 | 3,084 | Quick checks |
| 15×15 | 32 / 2 | 68,516 | Small baseline |
| 15×15 | 96 / 8 | 1,366,500 | Larger experiment |
| 15×15 | 128 / 10 | 2,993,028 | `gomoku-large`, not the maximum allowed size |

A small weight file does not guarantee fast inference or low training memory. Training also keeps activations, gradients, optimizer state, and temporary buffers. Dense convolution cost grows roughly with width squared; adding blocks grows body cost roughly linearly.

## What was checked?

The model-variant stage passed **72 CPU tests** on 2026-10-04. Checks covered 3×3, 15×15, and 19×19; both colors; Connect6 placement phases; nonzero position bias; real weight updates; full turns; frozen models; and resume. The original C model-creation interface was kept after reproducing and fixing a DLL access violation. Saved Gomoku best/latest models then returned legal CPU turns.

These **historical** CPU search medians used two roots, four simulations, one worker, and three warm repeats. Startup, loading, and learning were excluded:

| Game | Residual | Pooled | Attention |
| --- | ---: | ---: | ---: |
| 15×15 Gomoku | 8.60 ms | 9.72 ms | 16.98 ms |
| 19×19 Connect6 | 11.60 ms | 8.95 ms | 16.86 ms |

Pooling was not consistently faster. Attention cost more in this small CPU batch. These are not completed-game throughput or strength results. Each architecture also completed legal games and temporary tic-tac-toe train/resume checks.

An attention CUDA check passed FP32 learner/native output agreement, complete games, BF16 learning, refreshed-weight agreement, and resume on all three boards. It used eight roots / eight simulations and 32 updates per rule set, with batch 128 on larger boards. Temporary runs resumed from eight to nine games and kept unfinished games. The successful process took 23.874 seconds; a failed attempt added 4.772 seconds. With earlier migration checks, that approval used roughly 133 seconds of its 300-second GPU budget. Allocation peaks of about 254/437 MiB were cumulative PyTorch process peaks, not total VRAM or isolated model costs.

Raw local records are in `.native-cache/model-{residual,pooled,attention}-cpu.json` and `.native-cache/model-attention-cuda.json`. They are ignored by Git. [Validation history](migration-validation.md) records other checks; it does not rerun these timings for every revision.

```sh
uv run python -m experiments.benchmark_native --architecture attention --device cpu --repeats 3 --batch 2 --simulations 4 --wall-seconds 60 --output .native-cache/model-attention-cpu.json
uv run python -m experiments.benchmark_native --architecture attention --device cuda --repeats 1 --batch 8 --simulations 8 --training-presets tictactoe,gomoku,connect6 --wall-seconds 60 --output .native-cache/model-attention-cuda.json
```

The GPU command is a user-run example, not permission for extra agent GPU work. Test-process time includes startup and warmup. A long strength comparison needs separate authorization.

## What would justify changing the default?

Check matching learner/native outputs before and after updates, full games, and safe resume first. Then compare both color-swapped play and strength after equal total training time under each game's rules. Bigger models and lower loss do not settle that question.

SE blocks, bottleneck residual blocks, and board-adapted ConvNeXt remain research ideas, not implemented options. Use papers to choose designs; do not import an existing game engine or trained bot weights. The [network survey](network-research-2026.md), [learning-rate guide](learning-rates.md), and [architecture decision](adr/0002-shared-board-model-experiments.md) give the background.

## Sources

1. [Silver et al., AlphaZero](https://discovery.ucl.ac.uk/id/eprint/10069050/1/alphazero_preprint.pdf): residual board model; its 256-channel / 19-block body is not our local default.
2. [He et al., residual networks](https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf): residual and bottleneck designs.
3. [Hu et al., squeeze-and-excitation](https://openaccess.thecvf.com/content_cvpr_2018/papers/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.pdf): channel gating, measured on image tasks.
4. [Liu et al., ConvNeXt](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.pdf): image-model results, not Connect6 strength evidence.
