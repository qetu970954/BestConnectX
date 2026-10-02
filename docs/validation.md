# Initial implementation validation

Date: 2026-09-28. This records the initial implementation checks, not a strength claim.

**Later updates:** the tactical heuristic passed its separate 400-game gate; a recursive Python TSS reference and integration followed. See [tactical results](tactical-results.md) and [TSS notes](tss-python.md). The initial-run figures below are preserved as historical measurements. No neural model has passed a gate.

## Machine and environment

- Native Windows 11, Python 3.12.13, PyTorch 2.11.0+cu128.
- CUDA-capable GPU; existing driver retained. Exact model and memory capacity omitted.
- Original network: 347,204 parameters, initialized from a local seed. No imported bot weights or engine source.
- Local `.venv` occupies approximately 4.4 GiB including optional browser-test tooling. This excludes uv's shared cache. Generated training/test artifacts occupied approximately 47 MiB when inspected.
- Dependencies are locked in `uv.lock`. No paid service, cloud GPU, or driver/system configuration change was used.

## Executed checks

### GPU

`python -m connect6 doctor --device cuda` passed a real policy/value forward pass and gradient backward pass on the CUDA-capable GPU.

### Rules, search and persistence

`python -m unittest discover -s tests -v`: **10 tests passed** in approximately 10 seconds on the latest run.

Coverage includes:

- Opening and same-player second-placement sequencing; legal record replay and invalid inputs.
- Six/overline wins in all four directions, immediate termination, and a full-board draw fixture.
- Exact immediate tactical actions and 1,562 defensive hypergraphs checked against brute force.
- Same-player value backup and a two-placement winning search fixture.
- A real optimizer update and consistent board/policy symmetry transformations.
- Atomic checkpoint round-trip, failed-cap preservation, and cross-process lock behavior.
- Promotion-statistic rejection/acceptance examples and held-out opening generation.
- Actual bounded CPU training followed by a second process resuming its checkpoint: 10 completed games to 12, preserving learned state and run settings.

### GPU training smoke test

Separate smoke sessions confirmed checkpoint resume from 10 games / 32 optimizer updates to 34 games / 128 updates.

The main local run used the default configuration with an approximately one-minute session limit. It stopped cleanly at **48 completed games, 160 optimizer updates, and 12,399 replay positions**. This includes heuristic bootstrap games; it is not 48 exclusively neural self-play games. Its checkpoint remains in `data/latest.pt` for manual continuation. Reported final loss was approximately 2.992; this is not a playing-strength metric.

No long training job remains active from these checks. The validation runs were well below the agreed per-experiment and cumulative GPU-hour limits.

### Browser

`python tests/ui_smoke.py` passed in real headless Chrome:

- All 361 intersections render.
- Human opening, automatic two-stone bot replies, and subsequent human two-stone turns work.
- A POST without the local request token is rejected.
- The layout fits a 390-pixel viewport without horizontal overflow.
- Arrow-key board navigation works; no JavaScript page errors occurred.
- Dashboard training start, immediate stop request, and checkpoint persistence work, including the startup/stop boundary.

Screenshot: [dashboard.png](dashboard.png). This screenshot uses the untrained heuristic incumbent, not an evaluated neural champion. `node --check connect6/static/app.js` also passed.

### Search throughput

`python -m experiments.benchmark_search --checkpoint data/latest.pt` measured eight concurrent game roots with 64 simulations each. The latest recorded run took **0.2559 seconds**, about **2,001 simulations/second**, including Python tree work, feature extraction and inference after warmup. See [raw result and checkpoint hash](search-benchmark.json).

This small opening-position benchmark is not a self-play games/hour estimate or a general hardware guarantee. A previous repeat differed, as expected; no statistically established performance improvement is claimed.

## What has NOT been established

- No neural candidate has passed the 400-game promotion gate. The playable incumbent remains the accepted tactical heuristic.
- No match against MiniZero, NCTU6, CLAP, Nebula, or another established external engine has been run.
- No superiority, championship strength, or convergence/training-time estimate has been demonstrated.
- The Python TSS is bounded and incomplete; RZOP/DBS and a native Rust/C++ search backend do not exist yet.

Next strength milestone: profile sustained neural self-play, complete a held-out gate within manually authorized sessions, then benchmark against accessible external opponents. Retain the original baseline unless the evidence supports replacement.
