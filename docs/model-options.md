# Model choices for the C++ engine

[繁體中文](model-options.zh-TW.md)

**Status: implemented baseline and experimental variants, all with C++ inference. The default remains a plain residual network with 64 channels / 6 blocks for 15×15. Pooled-value and attention variants are selectable for Gomoku and Connect6; improved strength is not established.**

For newer board-game evidence, shared Gomoku/Connect6 inputs, and the next experiments, see the [2026-10-04 network survey](network-research-2026.md). [Learning-rate controls and newer optimizer methods](learning-rates.md) are covered separately.

## Goal

Make model size easy to change without changing game rules or search code.

A larger model can learn richer patterns. It also costs more per search step. It is useful only if it improves strength gained per hour.

No paper below proves that its model will improve this Gomoku engine. The image-model papers report image tasks, not Gomoku strength.

## Terms

- **Residual block:** a group of model layers that adds its input to its result. This helps train deeper models. [2]
- **Channels:** the number of feature maps at each board cell. More channels make the model wider.
- **Policy head:** the part that gives a score for each possible placement.
- **Value head:** the part that predicts the game result from a board position.
- **SE block:** a small part that uses a whole-board summary to change the weight of each feature channel. [3]
- **Bottleneck block:** a residual block that reduces channel width for its costly spatial operation, then restores it. [2]

## Short survey

| Model type | Main idea | Fit for this engine | Status / advice |
| --- | --- | --- | --- |
| Plain residual network | Keep board shape. Stack residual blocks with two 3×3 convolutions. | Closest to the current model and the AlphaZero design. Depth and width are simple controls. [1][2] | Recommended baseline. |
| Residual network with SE | Add a whole-board channel summary and a small learned gate to residual blocks. | A small change that could help global board context. Extra pooling and model calls still have a cost. [3] | Small trunk variant after the pooled-value-head check in the newer survey. Not a proven strength gain. |
| Bottleneck residual network | Use 1×1, 3×3, and 1×1 layers. Keep the 3×3 layer narrow. | Could make wide models cheaper in arithmetic. More layers can add launch cost for small batches. [2] | Consider if wider models become costly. Measure wall time, not operation count alone. |
| ConvNeXt-style network | Use depthwise spatial layers, channel mixing, and other modern ConvNet changes. | Worth surveying, but the published image model reduces spatial size. A board policy needs an adapted design. [4] | Do not import the image model or its weights. Defer a board-specific version. |

The AlphaZero paper describes 19 residual blocks with 256 channels in its main body. It used far more hardware than this project. That size is not the proposed local default. [1]

## Model sizes grounded in this repo

The parameter counts below were measured from [`Network`](../engine/network.py) on CPU before inference checks. Later CPU/CUDA output and learning checks are recorded in the [validation report](migration-validation.md).

These counts use the `residual` architecture's policy/value heads. The implemented pooled and attention variants have separate counts below.

| Board | Channels | Residual blocks | Trainable parameters | Proposed use |
| --- | --- | --- | --- | --- |
| 3×3 | 8 | 1 | 3,084 | First single-game and bridge checks |
| 15×15 | 32 | 2 | 68,516 | Existing small baseline |
| 15×15 | 64 | 6 | 477,764 | Default Gomoku model |
| 15×15 | 96 | 8 | 1,366,500 | Larger experiment |
| 15×15 | 128 | 10 | 2,993,028 | Upper example, not a required default |
| 19×19 | 64 | 6 | 495,172 | Connect6 baseline |

Implemented preset in [`configs/experiments.toml`](../configs/experiments.toml):

```toml
[presets.gomoku]
channels = 64
blocks = 6
```

Keep the board's spatial shape in the model body. Produce one policy score per cell and one value per position.

## Implemented experiments

All three architectures use the same eight input planes and one score per cell. `pooled` changes the value head to a 16-channel projection, global average pooling, and a 64-unit hidden layer. `attention` uses that same value head and adds one four-head Transformer block after the residual body, with learned 2D relative-position bias, layer normalization, and a two-times-width GELU feedforward path. There is no dropout. This is a small local experiment inspired by the survey, not an implementation of KataGo or ResTNet.

| Architecture | 15×15 parameters | 19×19 parameters | Presets |
| --- | ---: | ---: | --- |
| `residual` | 477,764 | 495,172 | `gomoku`, `connect6` |
| `pooled` | 450,898 | 450,898 | `gomoku-pooled`, `connect6-pooled` |
| `attention` | 487,734 | 489,846 | `gomoku-attention`, `connect6-attention` |

Counts were measured from the implemented 64-channel / 6-block models on CPU. Pooling loses absolute layout, while attention adds global interaction and compute. The pooled head does not make a checkpoint portable between board sizes or rules. Attention requires channels divisible by four.

```sh
uv run python train.py --preset gomoku-pooled --data data/gomoku-pooled
uv run python train.py --preset gomoku-attention --data data/gomoku-attention
uv run python train.py --preset connect6-pooled --data data/connect6-pooled
uv run python train.py --preset connect6-attention --data data/connect6-attention
```

These commands start training only when run. Use a new directory for each architecture; the rule-based default directory is shared by variants of the same game. `--architecture residual|pooled|attention` is also available. Resume without the preset restores the saved architecture, weights, and optimizer; an explicit architecture change is rejected. Existing checkpoints without an architecture field still mean `residual`.

See the [decision tree and tradeoffs](adr/0002-shared-board-model-experiments.md). Functional checks and timing results below do not show improved playing strength.

The parameter count does not prove that a model fits GPU memory. Training also stores activations, gradients, optimizer state, and temporary buffers.

Wider dense convolution layers have roughly square width cost. More blocks add roughly linear body cost at fixed width and board size. A small weight file can still be slow.

## Available model controls

- `--architecture residual|pooled|attention`: the body/head design.
- `--channels`: feature-map width.
- `--blocks`: residual depth; the attention variant still has one attention block.

Training and C++ inference share the model config, layer shapes, weights, and buffers. A different architecture, width, depth, or board size requires a new run. There is no model-transfer feature, pretrained image initialization, automatic growth, or architecture-search framework.

## Checks for a model size

1. Compare C++ and learner outputs with the same weights and inputs.
2. Test a real weight update, then check that C++ inference uses the new weights and buffers.
3. Measure startup, batch-one inference, batched search, learner updates, and peak GPU memory.
4. Test stop/resume with model, optimizer, replay, active games, and RNG state.
5. Report speed separately from strength. A short test cannot establish better Gomoku strength per hour.

The GPU test budget is five minutes of total GPU-process wall time. A long model-strength comparison needs a separate user-run experiment or more explicit authorization.

## Local checks on 2026-10-04

The native build passed and all **72 CPU unit tests passed in 28.864 seconds**. The new cases check both variants on 3×3, 15×15, and 19×19, before/after real optimizer updates, with nonzero relative-position bias, both colors, Connect6's two placement phases, full bot turns, frozen models, and resume. The original model-creation C interface is preserved. The saved best and latest Gomoku models also completed CPU bot turns after reproducing and fixing the access violation.

CPU search medians below use **two roots, four simulations, one native worker, and three warm repeats**; FP32 inference is compared with the learner using identical inputs and weights within each architecture. Large boards use 64 channels / 6 residual blocks. These are small search timings on the machine in the [migration validation report](migration-validation.md), with startup, model loading, and learning excluded.

| Game | Residual | Pooled value | Pooled value + attention |
| --- | ---: | ---: | ---: |
| 15×15 Gomoku | 8.60 ms | 9.72 ms | 16.98 ms |
| 19×19 Connect6 | 11.60 ms | 8.95 ms | 16.86 ms |

Pooling was not consistently faster; attention added CPU cost in this small batch. This is not a comparison of completed-game throughput, training quality, or equal-time strength. Each CPU architecture check also finished legal games on all three boards and performed a temporary tic-tac-toe train/resume check.

The **attention CUDA check passed** FP32 learner/native agreement, complete native games, and BF16 learning followed by updated-weight agreement and resume on all three boards. It used eight search roots / eight simulations, 32 optimizer updates per rule set, and batch 128 for the larger boards. Each temporary run resumed from eight to nine completed games; the large-board runs retained 63 unfinished games. This remains a functional check, with no persistent training or strength claim.

The successful CUDA process took **23.874 seconds**, including startup and warmup. A failed sandbox attempt used another **4.772 seconds**, for **28.646 seconds** of new CUDA-process time. With the earlier migration's roughly 104 seconds, the combined recorded time is about 133 seconds, below the existing five-minute cap. Reported PyTorch allocation peaks were about 254 MiB through Gomoku and 437 MiB through Connect6; these are cumulative process peaks, excluding driver/context and non-PyTorch memory, not isolated model memory.

Raw local records: `.native-cache/model-{residual,pooled,attention}-cpu.json` and `.native-cache/model-attention-cuda.json`. Re-run the same CPU check, or request a bounded CUDA check yourself:

```sh
uv run python -m experiments.benchmark_native --architecture attention --device cpu --repeats 3 --batch 2 --simulations 4 --wall-seconds 60 --output .native-cache/model-attention-cpu.json
uv run python -m experiments.benchmark_native --architecture attention --device cuda --repeats 1 --batch 8 --simulations 8 --training-presets tictactoe,gomoku,connect6 --wall-seconds 60 --output .native-cache/model-attention-cuda.json
```

## Original work requirement

Use papers to choose designs. Implement the board model here. Do not copy a game engine or import trained bot weights.

## Primary sources

1. [Silver et al., AlphaZero paper and supplement](https://discovery.ucl.ac.uk/id/eprint/10069050/1/alphazero_preprint.pdf), architecture and training details.
2. [He et al., Deep Residual Learning for Image Recognition, CVPR 2016](https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf), residual and bottleneck designs.
3. [Hu et al., Squeeze-and-Excitation Networks, CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/papers/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.pdf), channel gates and their measured costs.
4. [Liu et al., A ConvNet for the 2020s, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.pdf), ConvNeXt design and image-task results.
