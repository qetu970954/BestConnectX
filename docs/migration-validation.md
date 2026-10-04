# Validation history and measured speed

[繁體中文](migration-validation.zh-TW.md)

These are recorded checks, not new timings for every revision. The original migration numbers date from 2026-10-04, before model variants and later cleanup. [Model results](model-options.md) and the [save-cadence comparison](selfplay-training-guidance-2026.md) are separate. For commands you can run now, use the [cheatsheet](cheatsheet.md).

## Machine and method

Windows desktop with a multicore CPU and a CUDA-capable GPU; PyTorch 2.11.0+cu128; installed MSVC x64 tools. Exact hardware specifications are omitted for privacy, so the timings below are illustrative rather than hardware-reproducible.

Search pairs use the same weights, legal opening boards, rules, batch, simulations, and **FP32** on Python and C++ paths. Values below are medians of three warm runs, measured with `time.perf_counter`. Native inference includes required transfers and waits. Build, model loading, and warmup are outside these steady-search medians, but inside the full GPU check-process time budget.

These are paired **search** timings, not a measured ratio of completed games or training strength. Startup, learning, and file I/O have separate costs.

## CUDA: default-size batches and search

64 roots, 64 simulations. C++ uses the default six workers. The 3×3 model is 8 channels / 1 block; the larger boards use 64 / 6.

| Game | Python reference | C++ / LibTorch | Ratio: Python / native |
| --- | ---: | ---: | ---: |
| 3×3 tic-tac-toe | 378.60 ms | 30.69 ms | 12.34× |
| 15×15 Gomoku | 582.44 ms | 224.82 ms | 2.59× |
| 19×19 Connect6 | 664.77 ms | 249.23 ms | 2.67× |

The model sizes are 3,084, 477,764, and 495,172 parameters respectively. Larger numbers above do not prove better strength.

## CPU and worker trade-off

Eight roots, 16 simulations, same models, three warm runs:

| Game | Python CPU | Native CPU, 1 worker | Native CPU, 6 workers |
| --- | ---: | ---: | ---: |
| 3×3 | 23.97 ms | 5.00 ms | 10.74 ms |
| 15×15 | 89.83 ms | 59.82 ms | 58.23 ms |
| 19×19 | 89.19 ms | 70.64 ms | 81.21 ms |

Six workers were slower than one on the small CPU tic-tac-toe and Connect6 batches. More threads are not always better. On the larger CUDA Connect6 batch, six workers beat one (249.23 vs 267.43 ms). Worker creation and coordination have a cost. Keep the default six-worker preset, but tune `--workers` for the chosen device and batch.

An early native profile found unnecessary allocation while scanning non-threatening lines. The scanner now stops once a line has too many holes or an enemy stone. Full rule/feature/proof checks were rerun after this change.

## Correctness and framework checks

- Exhaustive native/reference rules, actions, and features for **all 5,478 legal tic-tac-toe states**.
- Complete native games in order: 3×3, 15×15, then 19×19. Smoke games use four simulations at most; they are operation checks, not strength tests.
- Matching CPU search policies/backups with the reference, including same-player Connect6 placements.
- CPU/CUDA model output agreement. CPU checks cover 32/2, 64/6, and 128/10 Gomoku models. After real updates, CPU/CUDA checks compare refreshed weights and running buffers again.
- Native multi-turn TSS proof also accepted by the Python verifier. Missing replies are rejected. Partial turns and a saved defensive branch resume correctly. Limits remain unknown.
- Temporary CPU/CUDA learning: 32 update steps, then stop/resume from eight to nine completed games. CUDA checks include the default 64/6 models and learning batch 128 on 15×15 and 19×19, retaining unfinished games.
- Checkpoint-owned bounded replay, summary-only statistics, no self-play move archives or permanent replay exports. A failed summary write recovers from the durable checkpoint.
- Atomic-save/disk-cap/run-lock checks; frozen 100-game gates, promotion, code/checksum changes, and restart archival checks.
- Chrome checks: 15×15/13×13/19×19 boards; whole bot turns; charts; keyboard; mobile; local token protection; no browser training controls; no JavaScript errors.

The migration-stage suite passed **68 unit tests** (27.516 seconds). The unit suite is CPU-only. Browser checks passed using temporary CPU runs. No persistent training experiment was started. Passing tests is not proof that no bugs remain.

During CUDA learning checks, reported PyTorch peak allocation was about 139.2 MiB for Gomoku and 205.6 MiB for Connect6. These are process peaks, not isolated per-model memory or total VRAM use. They exclude CUDA context/driver and memory outside the PyTorch allocator. They do not establish that every allowed model/batch fits memory.

## Test-time budget

The temporary GPU check processes used **about 104 seconds total**, including startup and warmup, within the approved five-minute cap. CPU checks, native builds, and CPU browser checks do not use that GPU cap. All repeated checks are included in the total. The final search measurements above ran separately from the unit/browser checks. Temporary run directories were removed.

Warm search timings do not include the build time. The native build succeeded with the installed compiler. One first-load attempt was briefly blocked by Windows application control after a build; later loads and all checks passed. No security setting was disabled.

## Housekeeping checks on 2026-10-04

After removing the retired half-native forest, three Python-engine benchmarks, and superseded reports, the native build succeeded and **all 72 CPU tests passed in 28.379 seconds**. The search comparison now checks the production C++ engine against independent Python selection/backups, rather than comparing the retired bridge with its fallback. Tests also check the shipped machine defaults. The CPU-only Chrome smoke check passed for 15×15 Gomoku, 13×13 and 19×19 Connect6, saved best/latest selection, complete turns, responsive layout, keyboard controls, CSRF protection, and run locking.

All three architectures passed additional CPU learner/native agreement, complete-game checks on 3×3, 15×15, and 19×19, and temporary tic-tac-toe learning/resume checks. Each benchmark used two roots, four simulations, and one warm repeat; these small checks do not establish new speed or strength rankings. Raw records are `.native-cache/housekeeping-{residual,pooled,attention}-cpu.json`. No new CUDA benchmark or persistent training was run. Existing data and session files were not cleaned or migrated.

Source changes invalidate an unfinished gate's code checksum. Use `--restart-gate` only if resume reports a pending-gate code change; this archives the previous comparison rather than mixing results. Ordinary checkpoints and model weights are retained.

## Latest cleanup check: 2026-10-04

The Python cleanup removed unused config/source return values and a one-call opening wrapper, moved replay restoration to one bounded path, and reused the native batch's timing output. It did not change model architectures, game rules, or permanent batch/concurrency defaults. Python references and storage/proof protections remain.

**All 75 CPU tests passed in 29.842 seconds**, including the save-cadence guard and a new check that parses both cheatsheets' published command as 15×15 Connect6 without starting training. The installed-Chrome CPU smoke check also passed: boards, full turns, model choices, charts, stable layout, keyboard/mobile access, local token protection, and run locks.

Model parameter counts, Python/JavaScript syntax, UTF-8, local links, and bilingual command parity passed. Source-digest calculation still matches the old algorithm. SHA-256 checks confirmed that **all 275 original `data/` files were unchanged**. No new GPU benchmark or persistent training was started. Local logs are `.native-cache/refactor-all-tests.log` and `.native-cache/refactor-ui-tests.log`.

This validates the cleanup, not a new speed or strength gain. The earlier [save-cadence measurement](selfplay-training-guidance-2026.md) remains a separate, short comparison. Guides are now more conversational; migration decisions and research evidence remain historical records.

## Later replay-default change

The user subsequently requested **50,000 replay positions** as the new-run default. TOML, CLI, and training fallbacks now agree. CPU tests cover the CLI fallback without a TOML replay entry, the bounded 50,000-position window, and saved-limit preservation on resume. The separate feature cache remains bounded at 20,000 entries; it is not the replay buffer.

All 75 CPU tests passed in 27.871 seconds, and the Chrome CPU smoke check passed. This is functional validation, not a full-capacity speed or strength benchmark. Historical 20,000-position measurements above remain unchanged. The earlier 275-file integrity check applies to the cleanup stage; those old run folders were no longer present for this later check, so it could not be repeated.

## Run the checks

```sh
uv run python -m engine.native
uv run python -m unittest discover -s tests -v
uv run --group browser python tests/ui_smoke.py
uv run python -m experiments.benchmark_native --device cpu --output .native-cache/checks.json
uv run python -m experiments.benchmark_native --device cuda --batch 64 --simulations 64 --wall-seconds 90 --output .native-cache/cuda-checks.json
uv run python -m experiments.benchmark_native --device cuda --repeats 1 --simulations 8 --training-presets tictactoe,gomoku,connect6 --wall-seconds 90 --output .native-cache/training-checks.json
```

The benchmark controller limits the child process's wall time and uses temporary training directories. CUDA commands are user-started; they are not permission for more agent GPU work. Raw local JSON results remain in ignored `.native-cache/`.

The project goal remains more playing strength per training hour. A long, controlled user-run comparison is still needed. This report does not claim improved strength, full training throughput, or optimal hardware use.
