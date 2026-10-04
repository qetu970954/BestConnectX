---
status: accepted
---

# Shared model experiments for Gomoku and Connect6

[繁體中文](0002-shared-board-model-experiments.zh-TW.md) · [Research](../network-research-2026.md) · [Current engine](../native-engine.md)

On 2026-10-04 the user asked to apply newer board-network ideas to Gomoku and Connect6, and delegated implementation choices. We retain the current residual model as the default and add two selectable experiments: a pooled value head, then that same head with one board-attention block. This tests global context in a small model while keeping playing and inference in C++, learning in Python, and all shared turn features.

## Decisions from the design discussion

```text
Try newer models in the existing engine [user requested]
├── Support both Gomoku and Connect6 [user required]
│   ├── Retain all eight input planes [agreed constraint]
│   └── Change value sign only when the player changes [rule requirement]
├── Compare rather than replace the default [implementation choice]
│   ├── residual: existing 64-channel / 6-block baseline
│   ├── pooled: same body and policy; pooled value head
│   └── attention: pooled variant plus one attention block
├── Keep native inference and Python learning [existing agreement]
│   ├── Matching layers, weights, and normalization buffers
│   └── Preserve the original native model-creation entry point
└── Resume saved experiments safely [user required]
    ├── Save architecture with weights and optimizer
    ├── New architecture requires a new run directory
    └── Keep the saved learning rate; no schedule added
```

The original migration's Q1–Q21 agreement remains in [migration requirements](../migration-requirements.md). The specific layer widths and placement below are engineering choices within the requested experiment, not additional user approvals. Persistent training and a long strength comparison remain user-started work.

## Design and alternatives

The pooled value path projects to 16 channels with a 1×1 convolution, applies ReLU, averages each channel over the board, then uses a 64-unit hidden layer and a scalar tanh output. The attention variant appends one pre-normalized Transformer block after the residual body: four heads, learned 2D relative-position bias, a two-times-width GELU feedforward path, and no dropout. It uses PyTorch/LibTorch scaled-dot-product attention. The policy remains one score per cell. Attention width must be divisible by four.

Two separate variants let the pooled-head effect be compared before adding attention. Pooling loses absolute layout; relative-position attention adds board-wide interaction but also computation and memory. These are hypotheses for stronger play, not a reproduction of ResTNet or KataGo and not a proven remedy for low value loss. We defer larger attention stacks, Rapfi codebook/export/incremental state, imported bot weights, new input planes, and optimizer replacement. Their costs would obscure the first comparison.

## Consequences and checks

The eight-plane order and one-/two-stone turn rules stay shared. In Connect6, the first placement can leave the same player to act; labels and search backups keep that player's value sign. The pair-threat planes are retained even though one-stone Gomoku leaves them inactive. Rules and strength results remain separate per run.

Architecture is saved in model/checkpoint config; its absence means the existing residual model. Explicit architecture changes on resume are rejected. The original `model_create(size, channels, blocks, device)` C interface remains available; experimental models use `model_create_variant`. Changing arguments under the old symbol caused a reproduced null-pointer access violation against a loaded older DLL, which this separation prevents. Source hashes still require rebuilding native code after source changes. Pending gates retain their existing code-hash checks; source changes require the documented `--restart-gate` to archive and restart a pending comparison.

Before delivery, check CPU learner/native agreement before and after real updates, both colors and Connect6 placement phases, nonzero relative-position weights, complete bot turns, saved/frozen models, resume, and unchanged default behavior. Measure model latency separately from playing strength. Promotion to a default requires separate color-swapped matches under each game's rules and strength after equal total training time; functional checks alone do not settle that decision. See [tests](../../tests/test_model_variants.py) and [benchmark command](../../experiments/benchmark_native.py).
