# Network and input research

[繁體中文](network-research-2026.zh-TW.md) · [Learning rates](learning-rates.md)

Reviewed 2026-10-04. **Keep the residual baseline and all eight inputs.** Pooled value and one-block attention are now implemented experiments, not automatic upgrades. Start with [model choices and local checks](model-options.md) if you want commands; this page explains the papers behind the ideas.

None of the sources below proves better strength per hour for this repo's Gomoku or Connect6. Published results and our local checks stay separate. This survey itself did not run new training or benchmarks. The [architecture decision](adr/0002-shared-board-model-experiments.md) records what we chose to implement.

## Repository baseline and shared inputs

The default is 64 channels and six ordinary residual blocks, a spatial policy head, and a value head that flattens two spatial channels into a 64-unit MLP. Training uses Python/PyTorch; C++ inference explicitly reconstructs the layers with LibTorch. The model starts from scratch. Any architecture change therefore requires matching learner and inference changes and explicit checkpoint compatibility. See [network](../engine/network.py), [C++ inference](../engine/inference.cpp), and [model options](model-options.md).

Keep the shared input order implemented by [Python features](../engine/game.py) and [C++ features](../engine/runtime.h):

| Plane | Meaning |
| --- | --- |
| 0 | Current player's stones |
| 1 | Opponent's stones |
| 2 | Remaining-stones signal: one everywhere when two placements remain, zero otherwise |
| 3 | Current player is black |
| 4 | Current player's immediate one-cell winning placements |
| 5 | Cells belonging to a current-player two-cell winning pair |
| 6 | Opponent's immediate one-cell winning placements |
| 7 | Cells belonging to an opponent two-cell winning pair |

Planes 2, 5, and 7 are inactive for one-stone Gomoku but meaningful for Connect6. Pair masks record membership, not partner identities. In Connect6, the same player can place again after a placement, so the own/opponent interpretation stays the same and value signs change only when the actual player changes. Preserve this in labels and backups; [Python backup](../engine/search.py) and [C++ backup](../engine/runtime.cpp) already use player identity.

For a later input experiment, directional line counts and gaps could add useful structure without replacing these planes. Compute them using the configured winning length and one-/two-stone rules, rather than fixed five-stone patterns. This is a hypothesis: extra features cost preparation time and need matching native/reference checks. Our rules are known for each separate run; omitting move history does not hide a rule state in these games. Adding planes changes the first layer and needs a new model/run; it is not a compatible edit to an existing checkpoint.

## Verified primary evidence

**Rapfi, 2025: directly relevant Gomoku evidence.** MixNet learns directional 11-cell patterns, exports them into a codebook, and updates affected features incrementally. Its policy uses a global summary; its value combines global and regional summaries. The paper reports strong CPU-limited Gomoku results, especially with alpha-beta search. However, it uses approximately 30.8 million Katagomo positions and teacher distillation; exported model storage is roughly 51–111 MiB. That evidence combines representation, data, inference, and search, rather than proving a cheap replacement for this GPU-batched residual learner. [Rapfi paper](https://arxiv.org/html/2503.13178). The [official engine repository](https://github.com/dhbloo/rapfi) confirms the Gomoku/Renju and alpha-beta/NNUE implementation context. Global summaries are a practical idea to adapt; reproducing the whole engine is a much larger project.

**ResTNet, IJCAI 2025: hybrid convolution/attention evidence in Go and Hex.** It preserves one token per board cell and interleaves residual and Transformer blocks with relative positions. In the paper's 9×9 Go comparison, RRTRRT achieved 60.80% against the selected KataGo opponents versus 54.60% for six residual blocks, while reported inference time increased from 3.067 to 4.566 ms. Each architecture's training used about 200 GTX 1080 Ti GPU hours. Pure attention was weaker in that comparison; more attention did not consistently help. These are their opponent comparisons and hardware timings, not measurements of our engine. Gomoku and Connect6 were not evaluated. [IJCAI paper](https://www.ijcai.org/proceedings/2025/0828.pdf), [official code and experiment configurations](https://github.com/rlglab/restnet).

**KataGo, verified current 2026 implementation.** Its architecture document describes convolutional global-pooling blocks and pooled policy/value paths, plus nested bottleneck Transformers using learned 2D rotary positions, RMSNorm, and SwiGLU. The document lists Transformer networks entering the main run on 2026-08-25, with roughly 10.5–70.4 million parameters. Its Elo table uses equal visits, and explicitly says this excludes inference speed; its relative cost estimates vary with backend, hardware, and batch size. [NetworkArchitectures.md](https://github.com/lightvector/KataGo/blob/master/docs/NetworkArchitectures.md). [v1.17.0 release notes](https://github.com/lightvector/KataGo/releases/tag/v1.17.0) document C++ Transformer support and new export/backend requirements; [v1.18.2](https://github.com/lightvector/KataGo/releases/tag/v1.18.2) adds attention acceleration for specific older NVIDIA GPUs. This is strong Go engineering evidence, with substantial inference work behind it, rather than a Connect6 architecture ranking.

**Connect6-specific evidence has a different emphasis.** A 2024 study compares plain and residual CNNs for sudden-death prediction using about 13 million positions; the available abstract and section excerpts support residual CNNs for that supervised task. The full experimental detail was not available here, so it cannot establish a policy/value self-play winner. [Deep learning approaches to the game of Connect6](https://doi.org/10.1016/j.entcom.2024.100707). The JSAI 2021 AlphaZero/JL-UCT opening-book study reports 65% against its own version without the book over 100 color-balanced games, counting draws as half wins. This demonstrates an opening/search contribution, not a newer trunk benefit. [JSAI paper](https://www.jstage.jst.go.jp/article/pjsai/JSAI2021/0/JSAI2021_4N4IS1c05/_pdf/-char/ja).

## Practical experiment order

These are hypotheses for both games, not published gains in this repo:

1. **Pooled value head first.** Keep the eight planes, 64/6 trunk, spatial policy, and scalar value target. Compare the baseline against a modest 1×1 value projection, global mean pooling, and the 64-unit MLP. This removes the dense input proportional to board area, but reduces direct spatial detail in the value head; regional pooling is a later option if that hurts tactics. Global summaries may help assess separated threats, including Connect6 pair threats.
2. **Then test the implemented attention variant.** It keeps the pooled value head and adds one board-attention block. Full attention has quadratic cell-pair work and extra memory cost. Compare it against both `residual` and `pooled` before changing the default.
3. **Only then consider another change.** A 64/8 trunk, SE-style block, regional pooling, or directional inputs are separate hypotheses, not implemented improvements. Rapfi-style codebooks also need export, incremental state, and rule-specific adaptation. Add one change at a time only if the measured strength benefit justifies its cost.

Before promotion, verify Python/C++ policy and value agreement with identical weights, normalization buffers, both colors, Gomoku positions, and both Connect6 placement phases. Check all eight feature planes, immediate and paired wins/defenses, value signs, a real weight update, and stop/resume. Compare batch-one and search-batch latency, learner throughput, and peak memory. Run separate paired, color-swapped matches for each rule set with common openings and equal move time; report uncertainty and strength after equal total training hours, including self-play and learning. Fixed-position losses or faster inference alone do not establish stronger play. Continue to initialize local models without importing pretrained bot weights.
