# Changing the learning rate

[繁體中文](learning-rates.zh-TW.md) · [Commands](cheatsheet.md) · [Network research](network-research-2026.md)

**For a new run, you can choose the rate. Resume keeps the saved rate.** The default is fixed-rate AdamW at **0.001**, with weight decay **0.0001**. There is no automatic learning-rate schedule.

The rate controls how large weight updates are. AdamW also uses gradient history to adjust them, but you still need to choose a base rate. Both policy and value heads share one optimizer and rate. See [PyTorch's AdamW docs](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html).

## Start a rate experiment

Use separate new directories:

```sh
uv run python train.py --data data/gomoku-lr3e4 --learning-rate 0.0003
uv run python train.py --preset connect6 --board-size "15*15" --data data/connect6-15x15-lr3e4 --learning-rate 0.0003
```

Or set it in your preset:

```toml
[presets.gomoku]
learning_rate = 0.0003
```

An explicit flag overrides TOML for a new run. The accepted range is a finite number greater than zero and at most 0.1; that is an input limit, **not a recommended tuning range**. `0.0003` is an example, not a measured best rate for either game.

## Resume does not replace the rate

```sh
uv run python train.py --data data/gomoku-lr3e4
```

This restores the saved learning settings and AdamW state, including its rate and gradient-history estimates. Adding `--learning-rate 0.0001` does not override them. Editing TOML or `run.json` does not change the checkpoint either. There is no supported rate override on resume.

A disposable CPU check confirmed this: TOML 0.0002 was overridden by CLI 0.0003 for a new tic-tac-toe run. After eight games/two updates, resume with CLI 0.0001 kept 0.0003 in both settings and optimizer groups. Original runs were untouched. This checks behavior, not learning quality. Implementation: [config](../engine/config.py), [CLI](../engine/cli.py), [training](../engine/training.py), and [updates](../engine/selfplay.py).

## Do I need a newer optimizer or schedule?

Not just because loss is low. The bot's self-play data changes as it learns; low training loss can still go with poor predictions or weak play. See [Gomoku's first-player result and its limits](gomoku-first-player.md).

PyTorch supports [cosine decay](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html) and [ReduceLROnPlateau](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html), but this engine uses neither. A plateau rule needs a stable validation target, not just changing self-play loss.

[Schedule-free optimization](https://github.com/facebookresearch/schedule_free) removes the decay-schedule requirement, not rate tuning. It also needs optimizer train/eval handling and care with batch-normalization buffers. That affects native weight refresh, frozen models, and resume; it is not a drop-in upgrade.

Keep constant-rate AdamW as the baseline. To compare 0.001 and 0.0003, use separate runs with the same model, search budget, update ratio, and total training time. Compare held-out value errors and color-swapped play separately for Gomoku and Connect6. This is a test proposal, not evidence that the lower rate wins.

If a schedule is added later, save its state and count lifetime updates across resumes. A session's `--hours` limit is not the full training horizon. Change rate, inputs, and model size in separate experiments so their effects stay clear.
