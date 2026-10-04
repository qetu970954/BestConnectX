# Approved C++ migration agreement

[繁體中文](migration-requirements.zh-TW.md)

**Status: approved. The user confirmed: “This scope is correct—start implementation.” The agreed scope is implemented; checks and limits are recorded in the [validation report](migration-validation.md).**

This document records the current discussion. Recommendations are not approved unless marked as agreed.

The older [requirements](alignment.md) and [ADR 0001](adr/0001-python-reference-and-native-search.md) describe earlier work. This approved agreement replaces their conflicting requirements.

## Agreed direction: rounds 1–3

| Question | User decision | Effect |
| --- | --- | --- |
| Q1 | Stronger bots after the same training time. | Strength gained per hour is the goal. More games or higher GPU use alone does not prove success. |
| Q2 | Native game engine with Python tools. | Q15 confirms that Python also runs model learning. All production playing and inference use C++. |
| Q3 | Preserve the current game, learning, and evaluation rules. | Keep real terminal results, checked proofs, PUCT, the 80/20 time policy, snapshots, and candidate comparisons. Later model-size requirements allow model changes. |
| Q4 | Remove old-run compatibility from the requirements. | Fresh runs are allowed. Do not add an old-run importer. This does not permit deleting existing files. |
| Q5 | Remove the experiment-repeat requirement. | No exact-repeat tools or promises. Keep the state needed for safe resume under Q11. |
| Q6 | Include GPU tests. | The five-minute cap is approved under Q12. |
| Q7 | English by default, with Traditional Chinese versions and a technical tone. | Both versions must contain the same claims and commands. |
| Q8 | Use C++ inference. Make the model easy to change. Survey other model types, but keep size reasonable. | C++ inference is required, not an optional later port. Q16 selects a plain residual network with configurable width and depth. See the [model survey](model-options.md). |
| Q9 | Move the entire playing path to C++. Python handles management. | Native rules, board generation, placements, actions, features, PUCT, tactics, TSS discovery, independent proof checks, move choice, self-play, and bot play. Python references may remain for tests, not production game decisions. |
| Q10 | Focus on 15×15 Gomoku. Start with one 3×3 tic-tac-toe game, then 15×15, then 19×19 Connect6 if prior checks pass. | Use this order for runtime checks and benchmarks. Do not claim that passing tests proves no bugs exist. |
| Q11 | Resume is a must. Do not save self-play move records to local disk. | No finished-game kifu archive. Q17 allows bounded replay samples and unfinished games in the checkpoint. Q18 keeps small game summaries. |
| Q12 | Approve the proposed test budget. | At most five minutes of total GPU-process wall time, including warmup. Temporary inference/search and short training/resume tests only. No persistent training. Ask before more GPU work. |
| Q13 | Approve common English, short sentences, and defined technical terms. | Product names, code, commands, and paths are allowed. Do not claim a strict check against an unspecified 4,000-word list. |
| Q14 | Approve the proposed doc layout. | Root English/Traditional Chinese READMEs. Active guides and cheatsheets in `docs/`, with `.zh-TW.md` partners. Old reports stay clearly marked as history. |
| Q15 | A: keep model updates in Python; self-play stays in C++. | Python/PyTorch runs loss, gradients, and optimizer updates. C++ runs inference and all playing. |
| Q16 | A: configurable plain residual network. | Start 15×15 with 64 channels and 6 blocks. Provide smaller test and larger experiment presets. Other model families stay in the survey. |
| Q17 | Accept checkpoint replay and active-game state. | Save a bounded recent replay window, weights, optimizer, active games, counters, pending work, and RNG state. No permanent per-game replay exports. |
| Q18 | Keep summary stats and implement a useful dashboard and framework. | Retain the latest 1,000 self-play summaries without move lists or board histories. Deliver working run views and bot play, not only design notes. |
| Q19 | Yes: native workers and batched C++ inference. | Native CPU workers feed one batched C++ GPU inference path. Alternate generation and Python learning at safe points. Do not update weights during active search batches. No separate actor/learner service in the first release. |
| Q20 | Accept simple config/resume rules; document use in the cheatsheet. | One TOML config per experiment, named presets, and explicit CLI overrides. Resume restores model/rules/learning settings. Runtime limits, device, workers, and game concurrency may change. New model shapes or rules need a new run. |
| Q21 | Accept staged checks and a speed report. | Short tests check correctness, model agreement, updates, resume, and speed. A long user-run comparison is needed to measure strength gained per training hour. |

## What remains requested

1. Use C++ to use the hardware well, especially for self-play board generation.
2. Separate CPU work, GPU work, and file I/O when choosing what to change.
3. Refactor configs so experiments are easy to set up and change.
4. Organize the README and cheatsheet.
5. Use an interview to settle the requirements before code changes.
6. Support deeper/wider models through config, with C++ inference.
7. Support safe resume without a finished self-play game archive.

The user removed the repeat-experiment goal. Config work and safe resume remain required.

## Checked baseline before migration

- CPU: multicore desktop CPU; exact model and core counts omitted.
- RAM: exact capacity omitted.
- GPU: CUDA-capable GPU; exact model and memory capacity omitted.
- One Python game implementation supports square boards and one-/two-stone turns. See [`engine/game.py`](../engine/game.py).
- C++ already builds batched features, counts lines, selects PUCT edges, walks trees, and updates search statistics. See [`engine/native.cpp`](../engine/native.cpp).
- Native search still copies and plays new leaf boards in Python. It also expands their legal actions there. See [`engine/search.py`](../engine/search.py).
- TSS discovery and independent proof checking remain in Python. Proofs guide moves. Games still reach real terminal results.
- PyTorch already uses compiled CPU/CUDA math. C++ model calls can reduce Python costs, but they do not promise faster GPU math.
- `--parallel` means game roots in a batch. It does not mean CPU worker threads.
- Search, learner updates, evaluation, and saves run in sequence.
- Disk accounting already uses a cache under the run lock. Atomic writes, disk caps, and recovery checks exist.
- Settings are split between CLI flags and code constants. There is no experiment-config file interface.
- This checkout has no saved training games or weights.

The [performance report](training-performance.md), [diagram](project-architecture.html), and ADR contain stale code descriptions. Their old timings are not results for this migration.

## Decision tree

```text
Migration agreement
├── Stronger bots per training hour [agreed]
│   ├── 3×3 → 15×15 Gomoku → 19×19 Connect6 [agreed]
│   └── Checks and speed report now; long strength comparison later [agreed]
├── Entire playing path and inference in C++ [agreed]
│   ├── Python/PyTorch learner [agreed]
│   ├── Plain residual network; 64 channels / 6 blocks for 15×15 [agreed]
│   └── Native workers and batched C++ inference [agreed]
├── Preserve game/learning/evaluation rules [agreed]
├── No old-run compatibility requirement [agreed]
├── Safe resume, no exact-repeat feature, no self-play kifu archive [agreed]
│   ├── Bounded replay and checkpoints [agreed]
│   ├── Summary-only stats and working dashboard [agreed]
│   └── Simple TOML configs and resume rules; cheatsheet examples [agreed]
├── Five-minute temporary GPU test budget [agreed]
└── English and Traditional Chinese docs and locations [agreed]
```

## Interview result

All Q1–Q21 answers are settled. The user confirmed Q19 with: “Yes. I agree.” Native CPU workers and batched C++ inference are agreed. Measure worker counts rather than assuming that more threads are faster.

The user confirmed the complete delivery scope below. No feature choices remain open in this interview.

## Approved delivery scope

- **Native playing:** C++ owns rules, board generation, legal moves, features, PUCT, tactics, TSS discovery, independent proof checks, complete self-play, evaluation play, and human-versus-bot play. Python game references remain test tools only.
- **Model and learning:** C++ runs model inference. Python/PyTorch defines and updates the configurable plain residual network. Updated weights reach the native inference model at safe points. Tests compare outputs from both sides.
- **Resume and storage:** checkpoints retain bounded replay samples and unfinished work, with model and optimizer state. Keep atomic saves, disk limits, immutable model snapshots, and safe recovery. Do not create finished self-play move archives or permanent per-game replay files.
- **Dashboard:** adapt the existing local dashboard to the native path. Keep run status, learning/evaluation results, recent-game summary stats, and playable bot games. Show actual data, clear errors, and usable controls. Do not add browser training controls, cloud services, or saved self-play replay views.
- **Simple setup:** provide TOML presets for the staged games and model sizes. The default is 15×15 freestyle Gomoku. Document build, preset selection, config overrides, self-play requests, training, resume, dashboard use, and checks in both cheatsheets. Publish commands only after checking them.
- **Checks and reports:** follow the stages below. Report CPU search, C++ inference, learning, and file I/O timings separately where measured. State the game rules, model, batch size, and worker count used for speed reports. Do not present old timings or parameter counts as new speed results.

The dashboard and framework are implementation work, not documentation-only work. No promise of improved strength is made from short tests alone.

## Agreed validation stages

1. **3×3:** start with one complete game. Then check rules against legal-state/minimax references and test C++ inference, learning, and resume.
2. **15×15 freestyle Gomoku:** check native/reference rules and tactical cases. Check model updates, resume, completed games, timing, and GPU memory.
3. **19×19 Connect6:** proceed only after the prior checks pass. Add partial turns, two-stone defenses, early wins, proof checks, and resume cases.

CPU checks do not consume the GPU cap. Count the full wall time of GPU test processes, including startup and warmup. Stop or ask before the cap is exceeded.

## Checks that must stay

Do not remove legal-move checks, real game results, independent checking of claimed tactical wins, safe saves, disk caps, or protection of existing files.

Removing the repeat-experiment goal does not remove these safety checks.

## Current work state

- Native Windows build and CPU/CUDA inference are checked.
- Production playing, TSS/proof checks, PUCT, and model inference use C++; learning remains Python.
- TOML presets, bounded checkpoint replay, safe resume, summary-only stats, and the dashboard are implemented.
- CPU unit and browser checks passed. Temporary GPU checks stayed below the five-minute cap; no persistent training was started.
- Active docs have English/Traditional Chinese versions. See the [native guide](native-engine.md), [cheatsheet](cheatsheet.md), and [validation report](migration-validation.md).

## Related docs

- [README](../README.md) and [cheatsheet](cheatsheet.md).
- [Earlier requirements](alignment.md).
- [Earlier native-search decision](adr/0001-python-reference-and-native-search.md).
- [Domain glossary](../CONTEXT.md).
- [PyTorch C++ research](pytorch-cpp.md).
- [Model options and parameter counts](model-options.md).
