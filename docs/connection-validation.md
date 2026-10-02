# Configurable square-board training validation

2026-10-02. Native project Python environment; **CPU only**, with `CUDA_VISIBLE_DEVICES=''`. All new training/checkpoint/gate data used disposable directories. No GPU experiment, persistent trained Gomoku checkpoint, or external-strength result was produced.

## Regression checks

```sh
uv run python -m unittest discover -s tests -v
```

**47 CPU tests passed in 53.748 seconds**, including the 80/20 scheduler and faster batched self-play. This includes the existing Connect6 core/tactics/TSS tests and the revised connection-game checks.

The new checks exercise:

- Single-stone Gomoku, configurable square sizes, Connect6-style turns, a two-stone opening, and immediate termination on the first winning placement.
- Horizontal/vertical/both diagonal wins, overlines, invalid placements and histories.
- Rejection of rectangular rules and CLI `9*13`, without changing an existing checkpoint.
- All **eight distinct symmetry transforms** of an asymmetric sample, matching board/policy locations; rejection of rectangular augmentation inputs.
- Real finite policy/value losses and actual neural weight changes.
- Verified Gomoku TSS moves followed by actual terminal play, not proof adjudication. Multi-stone opening plans survive the intermediate same-player placement with both TSS and fork-proof paths.
- Direct CLI self-play, immutable milestone publication, recorded configuration/seed/source, frozen settings on resume, complete move-only public records, and permanent internal targets.
- Preview metadata expands into one `run.json`; older `rules.json` and extra checkpoint fields remain readable/preserved. Fresh runs no longer generate duplicate rule metadata or an unused `initial.pt`.
- An injected export failure: the durable checkpoint retains all pending game records/targets, and the next run completes exports without losing or duplicating games.
- A paused/resumed **100-game gate**: legal paired openings, swapped engine colors, real terminal histories, unchanged frozen weights, tie retention, strict above-50 promotion, timing-overrun rejection, previous-best preservation and idempotent manifest publication.
- A controlled-clock scheduler check: training publishes later snapshots while an earlier evaluation remains incomplete, stays within the 20% allowance apart from one cooperative operation, and retains time counters on resume. Smaller concurrency takes effect without dropping unfinished games.
- Gates use a frozen simulation cap; 64 unresolved roots actually arrive together at network inference.
- Explicit gate restart archives old report bytes and preserves both immutable models; changed model bytes are rejected before restart. An additional upgrade check used a disposable copy of the older 2,000-game checkpoint, retaining all 7,744 optimizer updates and 19,657 replay positions. Real run files were unchanged.

The promotion-positive fixture substitutes an actually legal played win into the completed test report. It validates decision/publication logic, **not a measured trained candidate's superiority**. The gate's 0.02-second test setting deliberately uses the quick fallback; new production gates default to 0.25 seconds per full turn and the saved self-play simulation cap (normally 64).

## Browser checks

```sh
uv run --group browser python tests/connection_ui_smoke.py
```

Passed with installed Chrome, no browser download, CPU-only subprocesses and disposable run directories:

- **9×9 Gomoku and 13×13 Connect6-style** boards with correct placement counts and complete bot turns.
- At **1920×1080 and 1920×960**, 100% zoom, the board, controls, summaries, both charts and footer fit without document scrolling. Long evaluation history scrolls inside its keyboard-accessible region; mobile remains scrollable.
- Successful actions clear transient messages rather than leaving an `OK` banner; thinking/error messages retain the live status region.
- Actual loss component curves and win-rate/draw-adjusted-score plots from the temporary CLI run.
- Selecting latest unvalidated weights without changing the accepted model.
- Keyboard navigation, mobile overflow checks and no JavaScript errors.
- Missing-token rejection, invalid human-color rejection, and authenticated `/api/train` returning 404.
- Data-directory process locking blocks a bot request while another process owns the run.

The temporary dashboard run used 8-game milestones, 16 self-play games, 64 optimizer updates and a 100-game short-budget gate, completed separately through the explicit evaluation-only command after normal training stopped at its quota. These are smoke settings, **not** the production 1,000-game milestone or evidence of strong play. Screenshot: [connection-dashboard.png](connection-dashboard.png).

## CPU decision-generation measurement

Four runs per setting (first warmup excluded), frozen saved weights, the same retained 9×9 positions repeated across roots, 64 simulations and 2 ms tactical budgets. This is inference/decision work only, not a training run.

| Concurrent games | Median seconds/batch | Decisions/second | Median inference batch |
| --- | ---: | ---: | ---: |
| 8 | 0.1118 | 71.6 | 7 |
| 64 | 0.5570 | 114.9 | 55 |

About **1.6× CPU decision throughput**, with much larger inference batches. An earlier CPU profile found tree selection around 8% of total time, not the dominant measured cost. These results do **not** measure CUDA utilization or justify a C++ port by themselves. Native search remains conditional on a representative GPU profile; no C++ rewrite or GPU benchmark was performed.

## Preserved history and limits

Legacy `data/latest.pt`, its replay, incumbent and evaluation records remain separate. No tests resumed the real legacy training run. The original Connect6 frontend remains available for that legacy interface; Windows `start.cmd` now launches the new play/results-only dashboard.

The old CUDA validation is documented in [validation.md](validation.md); it is not a new GPU check. GPU throughput/utilization, faster-gate playing strength and external-engine results remain **unmeasured**. CPU tests used disposable data only; no real training or GPU job was started by the assistant.
