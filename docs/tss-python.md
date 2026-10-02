# Python threat-space search (TSS)

Status: implemented as the first Python reference in [ADR 0001](adr/0001-python-reference-and-native-search.md). Checked on 2026-09-29. The accepted heuristic and its gate result are unchanged; this is not a playing-strength promotion.

## Contract

`connect6.tss.search_tss(game, max_turns, deadline, max_nodes, ...)` returns a candidate proof tree or `None`. Its absolute `deadline` uses `time.perf_counter()` units. Candidate search can omit attack moves; omission can only miss a proof. **Callers must run `verify_tss` before using any returned tree.** The verifier returns `True`, `False`, or `None` (verification did not finish). Only `True` can adjudicate a game or dictate candidate moves.

`max_turns` counts the attacker's turns needed to finish the proof, including the current turn. A nonterminal threat therefore needs at least two turns. The recursive solver supports up to 20 turns, but current integrations use a much smaller horizon.

## Proof and defense handling

At each attacker node, the solver ranks a bounded set of possible attack turns by the repository's existing line heuristic. It keeps only the top 24 candidate cells and at most 1,000 turns. A move is useful only if it wins now or leaves immediate completion threats.

For a nonterminal attack, the solver:

1. Rejects the line if the defender can win during its next turn.
2. Finds every minimal set of up to two cells that hits all attacker's immediate winning completions.
3. Expands those sets into **all** complete legal defensive turns. If one stone blocks the threats, the defender's other stone is allowed to be any empty cell; it is never treated as irrelevant.
4. Searches every such reply recursively. A draw, defender win, unresolved continuation, or work limit prevents a proof. Replies that fail to block all threats lose to the attacker's immediate completion.

The verifier does not trust candidate ordering or the solver's list of responses. It independently enumerates legal defender turns and checks that the proof contains every reply that blocks all current threats, exactly once. It rejects counterwins, draws, omitted/extra branches, illegal placements, and malformed trees. The work limit can interrupt this enumeration; that returns `None`, not a partial proof.

The board is fixed at 19×19. Full-turn reply enumeration is intentionally simple and may be expensive. A `ponytail:` note in `connect6/tss.py` records the shortlist ceiling and its evidence-based widening path. This Python version is the correctness reference, not the final high-throughput implementation.

## Integration

- **Self-play:** the current hook uses a two-attacker-turn proof horizon and a 2 ms total per-root tactical budget; up to 90% is reserved for TSS discovery and the rest for independent verification. Unresolved positions continue through the existing tactical fallback or neural/heuristic policy. An adjudicated TSS win trains the current turn's one-hot placement targets and assigns the proven result to prior samples. Replay retains the root moves and `tss_win` source, not the potentially large proof tree; verification occurs before the result is committed.
- **Candidate play/evaluation:** new candidates enable TSS with a maximum three-attacker-turn horizon, 20,000 search nodes, and up to 50 ms per placement within the existing thinking budget. The verified root-turn plan is carried across both placements and checkpointed during held-out matches. The incumbent's TSS setting remains unchanged unless a candidate passes its gate.
- **Unknown:** no proof, deadline, node limit, candidate shortlist miss, or verifier timeout means normal search. None of these conditions creates a loss target.

RZOP and DBS are not implemented. The attack shortlist is heuristic, so the solver is incomplete: it can miss forced wins. Only verified returned proofs are sound under the game's current threat model.

## Checks and measured cost

- **26 unit tests pass**, including recursive proof verification, a defender with every second-stone choice enumerated, opponent counterwins, deadline/node cutoffs, and carrying a verified plan across both placements.
- The nested verifier fixture has a root attack with four complete defensive replies. Each reply has its own checked continuation. Removing one reply makes verification fail.
- A short CUDA frozen-network comparison used the existing 64-simulation network on the CUDA-capable GPU, eight roots per batch, five measured alternating-order runs, and a 2 ms self-play tactical budget. Full data and checkpoint hash: [`selfplay-benchmark-tss.json`](selfplay-benchmark-tss.json).

| Workload | Search-all-roots median | TSS-assisted median | Network positions |
|---|---:|---:|---:|
| Eight quiet openings | 211.578 ms | 210.913 ms | 520 → 520 |
| Selected tactical mix | 218.538 ms | 178.630 ms | 490 → 260 |

The selected mix is deliberately tactical and correlated. It was about **18% faster** and used **47% fewer** network-position evaluations; quiet openings were effectively unchanged. One selected fork root received a verified `tss_win` in three of the five tactical-mix runs; three immediate-loss roots were value-adjudicated in each run. These measurements cover decision generation only, not complete training throughput, convergence, or strength. They do not establish a general speedup.

## Next

Keep Python as the executable rules/proof reference. Profile sustained self-play and candidate gates, then move the proven CPU-heavy board/TSS/search work to Rust or C++ while preserving differential proof tests. Select the native language after profiling; do not add RZOP/DBS or a cross-language layer before the simpler implementation has measured limits.
