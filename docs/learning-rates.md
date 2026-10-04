# Learning rates and resume

[繁體中文](learning-rates.zh-TW.md) · [Cheatsheet](cheatsheet.md) · [Network research](network-research-2026.md)

Reviewed 2026-10-04. **Learning rates still matter. This engine lets you set the rate for a new run; resume restores the saved rate. It currently has no learning-rate schedule.**

## Current behavior

| Setting | Implemented behavior |
| --- | --- |
| Optimizer | AdamW, with weight decay `0.0001` |
| Default base rate | `0.001` |
| New-run controls | TOML `learning_rate` or CLI `--learning-rate`; the explicit flag wins |
| Allowed rate | A finite number greater than zero and at most `0.1`; this is a validation limit, not a useful-rate recommendation |
| Policy/value heads | One optimizer and one shared base rate |
| Automatic changes | None: no warmup, decay, plateau rule, or separate head rates |
| Resume | Restores learning settings and AdamW state, including its rate and moment estimates |

These facts come from [config loading](../engine/config.py), [CLI validation](../engine/cli.py), [optimizer creation and restore](../engine/training.py), and [weight updates](../engine/selfplay.py).

The base rate scales the weight update. AdamW adjusts updates using gradient history, but that adjustment still uses the base `lr`; it does not remove the need to choose one. [Current PyTorch AdamW documentation](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html).

## Set a rate for a new run

Use a fresh directory for each experiment. These commands start training only when you run them:

```sh
uv run python train.py --data data/gomoku-lr3e4 --learning-rate 0.0003
uv run python train.py --preset connect6 --data data/connect6-lr3e4 --learning-rate 0.0003
```

Or set the value in your chosen TOML preset:

```toml
[presets.gomoku]
learning_rate = 0.0003
```

`0.0003` is an example, not a measured best rate for either game. Gomoku and Connect6 require separate results even when they use the same optimizer.

For an existing run, even an explicit `--learning-rate` is ignored in favor of the checkpoint's saved learning settings and optimizer state. Changing TOML or `run.json` does not change that checkpoint rate. There is no supported rate override on resume in this version. Ordinary resume is:

```sh
uv run python train.py --data data/gomoku-lr3e4
```

A disposable CPU check used a TOML rate of `0.0002`, overrode it with CLI `0.0003`, completed eight tic-tac-toe games and two optimizer steps, then resumed to nine games with CLI `0.0001`. Both saved settings and optimizer groups retained `0.0003`. Existing training runs were not changed. This checks configuration and resume, not learning quality.

## What has changed in newer methods?

A **schedule** changes the base rate over training. PyTorch provides cosine decay and a rule that reduces the rate when a chosen metric stops improving. They remain supported options, though this engine does not use either. [CosineAnnealingLR](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html), [ReduceLROnPlateau](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html).

Schedule-free optimization removes the requirement for a decay schedule. Its authors still require learning-rate tuning; their implementation also needs optimizer train/eval handling and special care with batch-normalization buffers. It would affect our native weight refresh, snapshots, and resume, so it is not a drop-in upgrade. [Authors' implementation and caveats](https://github.com/facebookresearch/schedule_free).

## Advice for this engine

Keep constant-rate AdamW as the comparison baseline. First compare, for example, `0.001` and `0.0003` in separate new runs, with the same model, search budget, update ratio, and total training time. Check held-out value errors and color-swapped playing results under each game's rules. This is an experiment proposal, not evidence that a lower rate will improve strength.

Low value loss alone is not a reason to reduce the rate: self-play positions and results change as the bot learns. A plateau rule needs a stable validation target; a low training loss can still coexist with poor predictions elsewhere. See [Gomoku's first-player result and its limits](gomoku-first-player.md).

If a schedule is later added, count lifetime optimizer updates across resumes and save its state alongside AdamW. The per-session `--hours` limit is not the full training horizon. Rate tuning, new representations, and larger models should be tested separately so their effects can be measured.
