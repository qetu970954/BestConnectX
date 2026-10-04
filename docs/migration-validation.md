# Migration checks and measured search speed

[繁體中文](migration-validation.zh-TW.md)

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

All **68 unit tests passed** (27.516 seconds). The unit suite is CPU-only. Browser checks passed using temporary CPU runs. No persistent training experiment was started. Passing tests is not proof that no bugs remain.

During CUDA learning checks, reported PyTorch peak allocation was about 139.2 MiB for Gomoku and 205.6 MiB for Connect6. These are process peaks, not isolated per-model memory or total VRAM use. They exclude CUDA context/driver and memory outside the PyTorch allocator. They do not establish that every allowed model/batch fits memory.

## Test-time budget

The temporary GPU check processes used **about 104 seconds total**, including startup and warmup, within the approved five-minute cap. CPU checks, native builds, and CPU browser checks do not use that GPU cap. All repeated checks are included in the total. The final search measurements above ran separately from the unit/browser checks. Temporary run directories were removed.

Warm search timings do not include the build time. The native build succeeded with the installed compiler. One first-load attempt was briefly blocked by Windows application control after a build; later loads and all checks passed. No security setting was disabled.

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
