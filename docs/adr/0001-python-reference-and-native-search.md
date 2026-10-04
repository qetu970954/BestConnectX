---
status: superseded
---

# Python correctness references before native playing

[繁體中文](0001-python-reference-and-native-search.zh-TW.md)

The original decision was to implement bounded Python threat-space search and an independent proof verifier before moving measured CPU bottlenecks into native code. Keeping a readable reference made it possible to check the native implementation without weakening legal-move, counterwin, or complete-defense requirements. The goal was playing strength per training hour, not throughput alone.

The [approved migration agreement](../migration-requirements.md) supersedes that partial-port plan: production playing and inference now use C++, while Python/PyTorch owns learning and experiment management. Python game, search, tactics, and proof references remain test oracles, not a production fallback. The retired half-native forest and its benchmarks have been removed.

Current design: [native engine](../native-engine.md) and [shared model experiments](0002-shared-board-model-experiments.md). The original detailed decision and pre-migration reports remain in Git history.
