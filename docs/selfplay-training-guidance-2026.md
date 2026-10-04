# Self-play training: decisions and measurements

[繁體中文](selfplay-training-guidance-2026.zh-TW.md) · [Commands](cheatsheet.md) · [Engine](native-engine.md)

Reviewed 2026-10-04. **Keep learning with the newest model; use tournaments to choose best.** The saving fix reduced overhead in one short comparison. It has not established stronger play or the best batch/concurrency settings.

## Training policy

The implemented policy is:

| Choice | Current behavior |
| --- | --- |
| Main goal | Playing strength comes before CPU/GPU utilization |
| Self-play actor | Newest learner; a failed promotion does not roll it back |
| Recovery | Full `latest.pt` about every 60 seconds at safe boundaries, plus startup, milestones, and normal stop |
| Milestones | Frozen model-only exports every 1,000 completed games; numbers count games, not updates |
| Best | `incumbent.json` selects the accepted milestone; `best.pt` mirrors it atomically |
| First milestone | Initial baseline, not a strength-validated model |
| Promotion | 100 games / 50 shared-opening color-swapped pairs; at least 55 points; all histories verified; maximum turn overtime 0.1 seconds |
| Default gate limits | 0.25 seconds per full turn, including both Connect6 placements; equal simulation caps, default 64 |
| Evaluation schedule | One evaluation second per four measured training seconds; GPU work alternates, unfinished gates can wait |
| Retention | Keep every numbered milestone; stop at the disk cap instead of deleting old files |
| Tuning | Keep defaults at batch 128 / concurrency 64 / six workers until a separate decision |
| Replay capacity | Later user request sets new-run default to 50,000 positions; existing checkpoints retain their saved limit |

The later 50,000-position default is a user-selected setting, not a measured strength improvement. Historical profiling and trials below used 20,000 positions; those measurements have not been repeated at the larger capacity.

The 55-point rule is a quick, provisional regression filter—not statistical proof of improvement. A rejected model can still generate training samples. Protecting best does not remove those samples or prove they are better data.

The agent's measurement approval was separate: disposable copies only, at most **300 seconds of combined GPU-program duration**, including startup, loading, and warmup. No persistent training, long strength tournament, or modification of original run files was authorized. The user-run batch-256 trial was also a separate approval. Prior runs do not constrain the new design, but they must not be deleted or modified.

## What the research supports—and what it doesn't

KataGo's official guidance allows both gated and ungated acceptance. Gated self-play uses the newest accepted model; without a gatekeeper, every export can become an actor. The configuration notes say ungated runs often work fine and can be more efficient, while early gating helps debugging and checking improvement. This supports having a choice, not a universal answer for Connect6.

Our policy is a hybrid: latest drives learning, tournaments select the playing best. It avoids actor delays on one GPU, but has not proved better strength per hour. Repeated substantial regressions would justify testing accepted-best actors as a separate experiment.

KataGo also bounds learning exposure relative to fresh samples and warns about excessive reuse. Its sequential single-machine loop suggests reducing cycle/export overhead as training matures. Those example settings are lightly tested and are not ready-made Connect6 defaults.

Sources reviewed on 2026-10-04: [official training guide](https://github.com/lightvector/KataGo/blob/master/SelfplayTraining.md), [configuration notes](https://github.com/lightvector/KataGo/blob/master/cpp/configs/training/README.md), and [single-machine loop](https://github.com/lightvector/KataGo/blob/master/python/selfplay/synchronous_loop.sh).

## Why saving was the first thing to fix

Before the change, the inspected 15×15 Connect6 run was paused at 2,000 games / 10,944 updates. Its first baseline remained best, and its next tournament had completed only one game. That temporary 100% score said nothing about strength. The observed 5% GPU use / approximately 2,002 MiB VRAM were idle readings, not proof of training headroom.

The run's cumulative timers showed:

| Work | Seconds |
| --- | ---: |
| CPU search | 177.0 |
| Native inference | 127.5 |
| Learning | 99.0 |
| Checkpointing and exports | 244.0 |

These are accumulated phase times, not utilization percentages. The old code saved full replay after every learning cycle. That made saving worth profiling before adding asynchronous machinery.

A disposable CPU profile of the 20,000-position checkpoint measured these medians after warmup:

| Save work | Seconds |
| --- | ---: |
| Serialization | 1.2001 |
| Durable disk write | 0.0383 |
| Complete save | 1.2383 |

Serialization dominated. We removed redundant save calls, **not** `fsync`, atomic replacement, optimizer/replay state, disk caps, or recovery checks. Safe boundaries still allow timed saves within long learning cycles. The next interval starts after a successful save finishes. Milestone publication keeps its recovery checkpoints.

## The short CUDA comparison

Old and new save cadence ran sequentially from **identical temporary copies** of the 358-game checkpoint. Both used 15×15 Connect6, batch 256, concurrency 64, six workers, and a 36-second training-loop allowance. Neither reached a milestone tournament.

| Result | Old cadence | New cadence |
| --- | ---: | ---: |
| Session timer, including final save | 36.9 s | 37.1 s |
| Newly completed games | 92 | 128 |
| Games per session minute | 149.6 | 207.0 |
| New learning updates | 448 | 672 |
| Full checkpoint calls | 12 | 2 |
| Checkpoint/export time | 13.00 s | 2.17 s |

That is about **38.4% more games/minute in this one short comparison**. It is not a long-run guarantee, a strength result, or a faster search kernel. Game lengths and timed tactical work vary. The new run kept finite loss, optimizer state, and 64 resumable unfinished games.

Separate warmed component checks alternated case order:

| Check | Smaller case | Larger case | What it shows |
| --- | --- | --- | --- |
| Native placements | 64 roots: median 0.2009 s | 128 roots: median 0.3518 s | About 14% more placements/s, not complete-game throughput |
| Learning update | Batch 128: median 0.0124 s | Batch 256: median 0.0203 s | About 22% more sampled positions/s, but a slower individual update |

At the same update count, batch 256 samples twice as many rows as batch 128. Faster row processing does not establish better learning or equal exposure. Allocator peaks do not include all driver/context VRAM. A trained-model CUDA turn also returned two legal Connect6 placements and handed control back to Black.

The three GPU test programs used **105.498 seconds total**, including startup/loading/warmup, within their 300-second approval. Watchdogs limited each process. Temporary run copies were removed; SHA-256 checks confirmed original `data/` files were unchanged. Raw local records: `.native-cache/training-save-profile.json` and `.native-cache/training-tuning-cuda.json` (ignored by Git).

## The batch-256 trial

Before the save fix, a separate batch-256 trial directory paused cleanly at **358 games, 1,984 updates, and 20,000 replay positions**. Total/policy/value losses were finite: 3.0435126 / 2.7973495 / 0.2461631. Timers recorded 65.01 s search, 40.48 s inference, 27.94 s learning, 43.04 s checkpointing/exports, and 181.0 s session time. Setup was outside that session timer.

This proves the batch-256 configuration ran; it does not show a speedup over a matched fresh batch-128 trial or better strength. To try it yourself in PowerShell:

```powershell
$trial = "data/connect6-15x15-b256-" + (Get-Date -Format "yyyyMMdd-HHmmss")
uv run python train.py --preset connect6 --board-size "15*15" --data $trial --device cuda --batch 256 --parallel 64 --workers 6 --hours 0.05
```

Use a fresh directory: resume restores the old batch even if you pass `--batch 256`. This is a three-minute **training-loop** allowance, not a whole-process limit. Run one GPU trial at a time. Test concurrency separately, and account for changed sample reuse before changing permanent defaults.

## What stops the saving regression from returning?

Tests check the cause, not a hardware-dependent 207 games/minute threshold:

- A short CPU run performs several learning cycles without a due autosave/milestone. It must write **two** full checkpoints: startup and shutdown. The guard passed with the fix and failed against old code with five writes.
- Slow-update tests check timed autosaves, saved pending updates, and resume. Saving resets the timer at completion rather than triggering duplicate cycle saves.
- Other tests keep immutable milestones, full latest recovery, model-only best exports, interrupted-export repair, the exact 55-point boundary, overtime/history checks, disk caps, and failed-publication recovery.
- CLI checks require useful loss/milestone/time fields, no invented pre-update loss, and retained gate/error/save messages. Chrome checks cover the 55% chart reference and stable layout.

```sh
uv run python -m unittest discover -s tests -p test_connections.py -k short_run_saves -v
uv run python -m unittest discover -s tests -v
uv run --frozen --group browser python tests/ui_smoke.py
```

The save-fix stage passed 74 CPU tests in 29.678 seconds and the installed-Chrome smoke check. Later cleanup validation is recorded in [validation history](migration-validation.md), rather than treating those old timings as new results.

Code refactoring can invalidate an unfinished gate's checksum. Use `--restart-gate` only when resume reports the mismatch; it archives the old comparison. Do not restart automatically or mix results. Ordinary latest/best/milestone roles remain unchanged. See [the cheatsheet](cheatsheet.md) for normal 15×15 Connect6 training and resume commands.
