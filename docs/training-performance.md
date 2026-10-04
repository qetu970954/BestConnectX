# Historical training performance

> Historical Python-engine report. Archive layout, defaults, and next steps below are not the current design. See [native engine](native-engine.md) and [current checks/timings](migration-validation.md).

**The framework runs on CUDA; optimal settings and playing strength are not established.** Optimize strength gained per training hour, not GPU utilization alone. See [validation](connection-validation.md) and the [project diagram](project-architecture.html).

## Implemented

- Cached threats and line counts, plus batched feature construction.
- Original C++17 feature/count and PUCT-selection kernels, loaded by source hash through stdlib ctypes. Missing binaries use Python; builds are explicit.
- A 20,000-position uint8 replay-feature LRU outside checkpoints.
- Recent-export replay recovery, pending-export deduplication and small fresh checkpoints. Legacy embedded replay remains supported.

Trees, expansion, backups and tactical proof verification remain Python. Training, learning, storage and evaluation run sequentially; `--parallel` batches games, not CPU workers.

## Measurements on CUDA-capable GPU, 2026-10-03/04

Opening-position CUDA search, 64 roots and 64 simulations, frozen weights/seeds, tactics disabled. Python already included cached threats and batched NumPy features.

| Rules | Python / C++ wall time | Speedup |
| --- | --- | --- |
| 9×9 Gomoku | 274 / 219 ms | 1.25× |
| 15×15 Gomoku | 372 / 312 ms | 1.19× |
| 19×19 Connect6 | 444 / 252 ms | 1.76× |

Compared policies matched. One warmup and two repetitions give limited evidence, not complete games/hour or learning quality. At 128 roots, neither 9×9 nor 19×19 was uniformly faster; keep the default 64.

A separate 19×19 trace with 64 roots and 16 simulations measured **81.292 ms wall time versus 3.984 ms CUDA device activity** with C++. Device-event time is not NVIDIA utilization. A full unresolved 64-simulation search commonly makes 65 dependent inference calls; serial CPU tree work and synchronous results leave GPU idle gaps.

The first normal 9×9 CUDA run completed **3,004 games and 15,936 updates** before a user-requested safe stop. Its 290.9-second session included 233.187 seconds training and 57.484 seconds evaluation (19.8%). This startup session is not a steady-state or playing-strength benchmark.

## Populated-archive CPU profiles, 2026-10-04

Disposable copies used actual saved 3×3/9×9 weights, optimizer, unfinished games and 20,000-position replay, with C++ enabled. Both completed 64 more games and 352 updates at 64 roots/64 simulations/batch 128, normal 1,000-game snapshots and 2 ms tactics. Only the copy's old evaluation credit was reset. Originals were not trained, stopped or rewritten.

The copies contained about 22,259 files (3×3, 10,657 saved games) and 17,690 files (9×9, 8,467 games). Below are wall-clock hooks **without cProfile**, excluding copy preparation but including startup. cProfile was run separately to identify calls; its overhead increased the same short runs to 13.44/18.80 seconds.

| Measured component | 3×3 Connect3 | 9×9 Gomoku |
| --- | --- | --- |
| Whole sample, including startup | 9.04 s | 13.38 s |
| Replay recovery at startup | 2.12 s | 2.06 s |
| All archive-usage scans | 2.53 s / 143 scans | 2.76 s / 187 scans |
| Learner updates, including CPU neural work | 1.96 s | 4.50 s |
| Self-play, including tactics/search/inference | 1.71 s | 2.34 s |
| Evaluation slices | none | 1.77 s |

These rows are **inclusive and overlap**: evaluation contains scans/inference, and the scan row spans every phase. Do not sum them or treat them as live CUDA percentages. CPU learner timings include compiled PyTorch kernels, not just Python.

A separate **complete 100-game Connect3 CPU gate** on the populated copy took **12.01 s**. Its 501 capped-write scans consumed **10.16 s (84.6%)**; atomic writes including those scans took 10.95 s. All 1,198 inference calls used batch one and totaled 0.45 s. This is a timing check, not an external strength measurement.

### What consumes time

1. **Repeated whole-archive scans.** `storage.atomic_bytes` calls `usage(root)` before every capped write. `training._flush_exports` writes one JSON and one tensor file per game; `run_gate` saves after every placement. The profiled 3×3 training copy inspected 3.19 million file entries in 143 scans. Existing scandir is faster than pathlib, but repeating it scales with archive size.
2. **Python tree work and ctypes crossings.** In the profiled 3×3 sample, 60,635 `Node.select` calls took 0.97 s inclusively, including 0.77 s in `native.select`. NumPy ctypes argument validation/casts still occur for each native call. C++ selection does not move traversal, child creation or expansion out of Python.
3. **Dependent inference.** Separate fixed-seed 9×9/19×19 CPU searches still made 65 inference calls for 64 simulations. Serial gates infer batch one. The earlier CUDA trace establishes host/device gaps; this new CPU work does not measure their current CUDA-capable GPU share.
4. **Replay startup overhead.** Restoring 20,000 positions loads thousands of small tensor files. Roughly two seconds here is a restart cost, not a continual per-game cost.

## Next improvement

First remove repeated disk-accounting work, not the disk cap: checked byte accounting under the run lock, updated only after successful atomic writes, with recovery validation and existing reservation safety. Then measure coarser native traversal/expansion to reduce per-node Python/ctypes calls. Keep the independent proof verifier and Python oracle. Larger networks, actor pools and shared inference are not justified by these CPU profiles; a separately authorized CUDA profile is needed to rank current GPU/host synchronization costs.

## Reproduce

```sh
uv run python -m experiments.benchmark_pipeline --size 9 15 --connect 5
uv run python -m experiments.benchmark_pipeline --size 19 --connect 6 --stones 2
```

These commands default to CPU, synthetic replay and disposable networks; no training artifacts are created. GPU experiments require explicit authorization. Local populated-copy profiling is reproducible with `.venv/Scripts/python.exe .native-cache/profile_training.py --data data/connect3-3x3-s1-o1-v1 --no-cprofile`; add `--gate-only` for comparisons. Results and cProfile dumps are in `.native-cache/cpu-training-profile-*` and `.native-cache/cpu-search-profile-*`. These ignored local tools/receipts and historical `.native-cache/docs-archive/` are not shipped in a fresh clone. Native build instructions are in [README](../README.md).
