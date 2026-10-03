# Fresh framework validation

CPU-only checks use `CUDA_VISIBLE_DEVICES=''` and disposable directories. **All 49 unit tests and the browser smoke check passed after cleanup.** They validate framework behavior, not a trained bot's strength. No real training or GPU job was started for this cleanup.

## Regression checks

```sh
uv run python -m unittest discover -s tests -v
```

The suite covers:

- One configurable `Game`: 9×9 Gomoku, 13×13 and 19×19 Connect6-style rules, square-only validation, both opening/turn sizes, overlines and immediate termination.
- Matching board/policy transforms for all eight square symmetries, finite losses and actual CPU optimizer updates in temporary tests.
- Batched unresolved-root inference, exact tactical plans, independent TSS verification, malformed-proof rejection, cancellation and plan continuation.
- Actual terminal self-play; no proof adjudication or premature value labels.
- Checkpoint/resume, immutable milestones, move-only public records, internal replay targets, persisted RNG and recovery from interrupted exports.
- Frozen 100-game paired gates, terminal-history verification, tie retention, promotion safeguards and previous-best publication.
- Persisted 80/20 time accounting, concurrent training with pending gates and explicit concurrency changes without dropping games.
- Whole-turn overtime across a pause between stones: 0.31 + 0.06 seconds against 0.25 records the full 0.12-second overrun.
- Slow-loading gate progress, reuse of frozen networks, continued checksum enforcement and inclusion of setup time in evaluation credit.
- Pre-consolidation configurable checkpoint compatibility without either old package; explicit gate restart preserves archived report bytes. Retired fixed-format checkpoints are rejected without overwriting them.

Synthetic promotion fixtures validate publication logic, **not measured candidate superiority**. Short test budgets do not establish production playing strength.

## Browser checks

```sh
uv run --group browser python tests/ui_smoke.py
```

Installed Chrome/Chromium is used; no browser is downloaded. Checks cover 9×9 and 13×13 boards, full bot turns, model selection without incumbent mutation, loss/score charts, keyboard/mobile use, same-origin tokens, run locking, no browser training endpoints and no JavaScript errors.

At 1920×1080 and 1920×960, 100% zoom, the dashboard fits without document scrolling. Long history remains keyboard-scrollable. Empty success notices are hidden; thinking/errors remain visible.

## Fresh checkout and generated data

At the user's explicit request, all local `data/` runs, weights, replay, game records, old benchmark/gate reports and screenshots were removed. The repository contains source, documentation and curated regression fixtures—not a training dataset or pretrained model. `.gitignore` excludes new generated artifacts, including custom run paths.

Tests clean up temporary checkpoints and games. The optional browser check writes an ignored `docs/connection-dashboard.png`; it is not a shipped result. New user-started runs initialize from scratch. The old fixed-rule trainer/UI and forwarding packages are removed; historical source remains in Git.

CUDA utilization, GPU throughput, playing strength and external-engine performance remain unmeasured for the current framework.
