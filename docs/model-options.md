# Model choices for the C++ engine

[繁體中文](model-options.zh-TW.md)

**Status: selected and implemented. The first release uses a plain residual network with configurable width/depth and C++ inference. Default: 64 channels / 6 blocks for 15×15. Other families remain research options.**

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

| Model type | Main idea | Fit for this engine | First-release advice |
| --- | --- | --- | --- |
| Plain residual network | Keep board shape. Stack residual blocks with two 3×3 convolutions. | Closest to the current model and the AlphaZero design. Depth and width are simple controls. [1][2] | Recommended baseline. |
| Residual network with SE | Add a whole-board channel summary and a small learned gate to residual blocks. | A small change that could help global board context. Extra pooling and model calls still have a cost. [3] | Best small variant to test next. Not a proven strength gain. |
| Bottleneck residual network | Use 1×1, 3×3, and 1×1 layers. Keep the 3×3 layer narrow. | Could make wide models cheaper in arithmetic. More layers can add launch cost for small batches. [2] | Consider if wider models become costly. Measure wall time, not operation count alone. |
| ConvNeXt-style network | Use depthwise spatial layers, channel mixing, and other modern ConvNet changes. | Worth surveying, but the published image model reduces spatial size. A board policy needs an adapted design. [4] | Do not import the image model or its weights. Defer a board-specific version. |

The AlphaZero paper describes 19 residual blocks with 256 channels in its main body. It used far more hardware than this project. That size is not the proposed local default. [1]

## Model sizes grounded in this repo

The parameter counts below were measured from [`Network`](../engine/network.py) on CPU before inference checks. Later CPU/CUDA output and learning checks are recorded in the [validation report](migration-validation.md).

These counts use the current model and its current policy/value heads. New heads or SE blocks would change the counts.

| Board | Channels | Residual blocks | Trainable parameters | Proposed use |
| --- | --- | --- | --- | --- |
| 3×3 | 8 | 1 | 3,084 | First single-game and bridge checks |
| 15×15 | 32 | 2 | 68,516 | Existing small baseline |
| 15×15 | 64 | 6 | 477,764 | Default Gomoku model |
| 15×15 | 96 | 8 | 1,366,500 | Larger experiment |
| 15×15 | 128 | 10 | 2,993,028 | Upper example, not a required default |
| 19×19 | 64 | 6 | 495,172 | Later Connect6 test |

Implemented preset in [`configs/experiments.toml`](../configs/experiments.toml):

```toml
[presets.gomoku]
channels = 64
blocks = 6
```

Keep the board's spatial shape in the model body. Produce one policy score per cell and one value per position.

The parameter count does not prove that a model fits GPU memory. Training also stores activations, gradients, optimizer state, and temporary buffers.

Wider dense convolution layers have roughly square width cost. More blocks add roughly linear body cost at fixed width and board size. A small weight file can still be slow.

## Controls to expose

For a plain residual model, expose at least:

- Model type, if more than one type is shipped.
- Channel count.
- Residual block count.

Keep one shared model config for training and C++ inference. Both paths must agree on layer shapes, weights, and buffers.

Do not allow a resume to silently change model shape. A wider or deeper model needs a new run unless an explicit model-transfer feature is requested.

Do not add pretrained image weights, automatic model growth, an architecture-search system, or a plugin framework.

## Checks for a model size

1. Compare C++ and learner outputs with the same weights and inputs.
2. Test a real weight update, then check that C++ inference uses the new weights and buffers.
3. Measure startup, batch-one inference, batched search, learner updates, and peak GPU memory.
4. Test stop/resume with model, optimizer, replay, active games, and RNG state.
5. Report speed separately from strength. A short test cannot establish better Gomoku strength per hour.

The GPU test budget is five minutes of total GPU-process wall time. A long model-strength comparison needs a separate user-run experiment or more explicit authorization.

## Original work requirement

Use papers to choose designs. Implement the board model here. Do not copy a game engine or import trained bot weights.

## Primary sources

1. [Silver et al., AlphaZero paper and supplement](https://discovery.ucl.ac.uk/id/eprint/10069050/1/alphazero_preprint.pdf), architecture and training details.
2. [He et al., Deep Residual Learning for Image Recognition, CVPR 2016](https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf), residual and bottleneck designs.
3. [Hu et al., Squeeze-and-Excitation Networks, CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/papers/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.pdf), channel gates and their measured costs.
4. [Liu et al., A ConvNet for the 2020s, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.pdf), ConvNeXt design and image-task results.
