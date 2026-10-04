# Python threat-space search (TSS)

`engine/tss.py` is the bounded Python reference described in [ADR 0001](adr/0001-python-reference-and-native-search.md). It works with the single configurable `Game`, for both one- and two-stone turns on square boards. It is not evidence of playing strength.

## Contract

`search_tss(game, max_turns, deadline, max_nodes, ...)` returns a candidate proof tree or `None`. Its absolute deadline uses `time.perf_counter()` units. **Callers must independently run `verify_tss` before using a tree.** The verifier returns `True`, `False`, or `None` if verification did not finish. Only `True` permits proof-guided placements.

`max_turns` counts the attacker's turns, including the current turn. A nonterminal threat needs at least two turns. The solver supports up to 20; current integrations search at most three.

## Proof and defense handling

The solver ranks a bounded set of attack turns using the original line heuristic. A nonterminal attack must leave immediate winning completions and must not allow the defender to win first.

The solver expands defensive hitting sets into **every complete legal defensive turn**. If one stone blocks the threat, the defender's other stone can be any empty cell. Every reply that blocks all current threats must have a checked continuation; a draw, counterwin, unresolved continuation or work limit prevents a proof.

The verifier independently reconstructs these replies. It rejects missing/extra branches, illegal placements, malformed trees and counterwins. A work limit returns `None`, never a partial proof. Attack shortlists can miss wins; a failed search is not a loss label.

## Integration

- `engine/selfplay.py` batches unresolved roots for neural inference. Its default tactical budget is 2 ms per root, with time reserved for independent verification.
- Verified TSS/fork plans guide actual placements. **Every game still plays to a terminal result.** Value targets come only from that result; the legacy proof-adjudication path is removed.
- Play/evaluation carries verified plans and response strategies across placements. Gates persist partial turns; callers remove a planned stone only after committing it.
- Unknown or expired proofs fall back to normal neural/heuristic search.

## Checks and limits

The CPU regression suite covers recursive verification, every second-stone filler reply, counterwins, malformed certificates, deadline/node cutoffs, forged flags, and plan continuation across restarts. See [framework validation](connection-validation.md).

RZOP and DBS are not implemented. Keep Python as the correctness reference; consider native search only after representative profiling establishes a CPU bottleneck. TSS itself has no native port. Feature/selection acceleration and measured limits are in [performance notes](training-performance.md).
