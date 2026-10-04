# Historical validation

> Results before the full C++ migration. Old archive/fallback checks and test counts are not current acceptance results. See [migration checks and timings](migration-validation.md).

Framework checks establish operation and recovery, not playing strength or optimal settings. Architecture and commands are in the [project diagram](project-architecture.html) and [README](../README.md).

## CPU and browser, 2026-10-04

- **C++ enabled:** 59 CPU tests passed (29.365 seconds).
- **Python fallback:** 59 discovered; 56 passed, three native-only checks skipped (28.817 seconds).
- **Browser:** 3×3, 9×9, 13×13 and 19×19 passed with browser GPU acceleration disabled. The 3×3 check read the active user's dashboard without bot/training requests; other boards used disposable CPU runs.

Coverage includes rules/turns/overlines; eight board-policy symmetries; terminal labels; batched inference; native/reference agreement; proof validation and cancellation; checkpoints/RNG/replay/export recovery; disk limits; immutable models; complete paired gates; promotion/rejection safeguards; persisted 80/20 scheduling; and concurrency changes without lost unfinished games.

The statistics check selects the latest 1,000 completed records by completion number, excludes unfinished/evaluation games, checks color/draw denominators and partial turns, and verifies cache invalidation on new exports. Real CLI resume and browser checks cover saved-window counts, distinct placement/turn lengths, empty states and keyboard-accessible details.

The dashboard fits 1920×1080 and 1920×960 without page scrolling; mobile/keyboard play, charts, model selection, local tokens and run locking passed. No browser training endpoints exist. A regression check verifies saved 3×3 rules, rejection of a second dashboard on an occupied port, and immediate restart after closing the first. Windows address reuse previously let an old dashboard silently share the new run's URL.

## CUDA evidence

- Temporary 19×19 test: 64 terminal games, 288 finite updates, four immutable models and exact recovery of 20,000 replay positions. Startup restored weights, counters and NumPy/CPU/CUDA RNG states.
- Its comparison resumed to 42/100 games. GPU processes used 177.014 seconds; total harness attempts used 191.668 seconds under the separate 300-second grant. Temporary data was removed. Test-only 16-game snapshots and mostly bootstrap samples prevent throughput/strength conclusions.
- The original end-of-session CPU-RNG assertion was invalid: frozen-network initialization consumes CPU RNG. Startup restoration passed without an engine change.
- A subsequent normal **9×9 CUDA run** stopped safely at **3,004 games / 15,936 updates**, with 20,000 replay positions. Its 2,000-game candidate completed a 100-game gate, scored 47%, and was not promoted. The 3,000-game candidate paused at 3/100 games. Actual GPU rejection is verified; accepted promotion remains CPU-tested, not exercised by this run.

Local raw receipts are in `.native-cache/framework-readiness-results.json` and `data/connect5-9x9-s1-o1-v1/`. They are ignored and not shipped. Historical reports are cached in `.native-cache/docs-archive/`. Long-session stability and external playing strength remain unmeasured; see [performance](training-performance.md).

## Run checks

```sh
uv run python -m unittest discover -s tests -v
uv run --group browser python tests/ui_smoke.py
```

Checks use disposable data and clean up their games/models. Browser checks require installed Chrome/Chromium (`CHROME_EXECUTABLE` can override its path); no download occurs. Screenshots go to ignored `.native-cache/ui-dashboard.png`, not `docs/`. User training data is never removed by these checks.
