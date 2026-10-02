# Tactical implementation and measured promotion

Date: 2026-09-28. This is an internal engine improvement, not a world-strength claim.

## Implemented

- `connect6/tactics.py`: bounded complete-turn fork discovery with exact verification.
- Checks our immediate win first. Otherwise verifies that the opponent cannot win before defending and cannot cover all our next-turn winning completions with its remaining stones.
- Candidate discovery uses a heuristic shortlist (24 cells, at most 256 candidate turns, up to 50 ms within the playing budget). These limits can miss wins, so failure returns **unknown**, never a proved loss.
- A successful two-stone plan is retained until its placements are committed. Pausing after thinking but before placement does not lose or advance the plan.
- Accepted and candidate tactical configurations are explicit. Old pending neural gates do not silently change algorithm mid-match.
- The dashboard can preview the latest unvalidated neural candidate without promoting it. `start.cmd` opens the dashboard from a normal Windows console.

These are established tactical-search and hitting-set ideas implemented originally here. We do not claim algorithmic novelty.

## Correctness and selected regressions

A seeded discovery run inspected 32 eligible positions and found three where the original greedy pair lacks a one-turn winning certificate but the new search finds one. All three are from the same generated game; they were deliberately selected as regressions, not sampled as an unbiased strength estimate.

`tests/fixtures/forks.json` preserves the full legal move histories. Tests replay them, verify all eight board symmetries, check player-perspective reversal, reject opponent counter-wins, exercise deadline/cancellation behavior, and check interrupted/resumed plans.

An independent line scanner plus exhaustive enumeration verifies all **157,691** possible two-stone defender replies across these three post-certificate boards. No reply wins first or blocks every attacking completion.

The 20-repeat microbenchmark found a certificate every time for each fixture. Median discovery time was 1.267 ms, 1.089 ms, and 0.669 ms respectively on this machine. See `tactical-benchmark.json`. These selected positions do not represent average tactical-search cost.

## Development screen

A fixed eight-opening, color-swapped screen compared:

- Candidate: original heuristic + bounded fork certificates.
- Opponent: unchanged original heuristic.
- Same five-second full-turn time limit; both often finish much earlier.

Result: **8 wins, 6 draws, 2 losses**, score **68.75%**. No promotion was based on this small screen. Data: `tactical-screen.json`.

## Predeclared final gate

After the development screen, a separate opening suite (778 rather than 777) was used for the agreed final test:

- **200 opening pairs / 400 games**.
- Each opening played with candidate colors swapped.
- Five seconds per full turn for both engines.
- Paired bootstrap with 10,000 resamples, treating opening pairs as the statistical units.
- Required lower endpoint of the 95% score interval above 50%, no rules failures, and no turn overrun above 0.1 seconds.
- No model training or parameter tuning during this gate.

Results:

| Metric | Observed |
|---|---:|
| Wins / draws / losses | **217 / 144 / 39** |
| Score | **72.25%** |
| Paired-bootstrap 95% lower endpoint | **69.25%** |
| Rules failures | **0** |
| Largest recorded turn overrun | **0 seconds** |
| Gate wall time | **120.906 seconds** |

Data and every game record: `tactical-gate.json`. This comparison is CPU-only; it does not train or benchmark a neural model. The clocks are equal upper limits, not a claim that both algorithms consume identical CPU time.

## Promotion

`experiments/promote_tactics.py` independently replayed all 400 games, checked opening/color assignments, terminal scores, timing bounds, and recomputed the confidence criterion. Under the run lock it also checked that the incumbent was still the tested original heuristic and no neural gate was pending.

The accepted opponent was then atomically changed to the tactical heuristic. The previous configuration remains in `data/incumbent.json`, and `data/gate-tactics-1.json` records the result and engine source hashes. No learned weights were promoted or overwritten.

The neural candidate remains unvalidated. The next neural gate must beat this improved accepted opponent, not the old weaker baseline.

## Reproduction

```sh
uv run python -m unittest discover -s tests -v
uv run python -m experiments.benchmark_tactics
uv run python -m experiments.screen_tactics --pairs 8 --suite 777
uv run python -m experiments.screen_tactics --pairs 200 --suite 778
uv run --group browser python tests/ui_smoke.py
```

Fourteen tests and the real-browser test passed, including candidate preview without promotion. Timing-bounded discovery may differ under machine load, so new match runs need their own results.

The one-off promotion command refuses to overwrite an already upgraded incumbent. Replaying this suite later is replication, not a fresh held-out gate for a newly tuned algorithm.

## Still unproven

No MiniZero, NCTU6, CLAP, Nebula, or other established external bot was tested. The improvement is measured only against our own original baseline on the specified opening distribution. This historical gate did not include recursive forcing search. A bounded Python TSS follow-up now exists, but it has not been strength-gated; see [TSS reference notes](tss-python.md).
