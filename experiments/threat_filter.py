"""CPU-only research probe, NOT a competitive engine. Python 3.10+, stdlib only.
Run: uv run --no-project experiments/threat_filter.py > experiments/threat_filter_results.json
"""
import itertools as it
import json
import platform
import random
import statistics
import time
from functools import lru_cache


@lru_cache(None)
def windows(n):
    return tuple(tuple((r + k * dr) * n + c + k * dc for k in range(6))
                 for r in range(n) for c in range(n)
                 for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1))
                 if 0 <= r + 5 * dr < n and 0 <= c + 5 * dc < n)


def threats(board, player, n=19):
    """Empty sets that player can fill to win with one or two stones."""
    return frozenset(frozenset(i for i in line if board[i] == 0)
                     for line in windows(n)
                     if all(board[i] != -player for i in line)
                     and 4 <= sum(board[i] == player for i in line) <= 5)


def won(board, player, n=19):
    return any(all(board[i] == player for i in line) for line in windows(n))


def covers(edges, budget=2):
    """All inclusion-minimal hitting sets of size <= budget.

    An opponent completion must be hit at least once. Branch on an uncovered
    edge, NOT on all board pairs. Each edge has at most two cells.
    Empty result means no defense; {frozenset()} means no threat.
    """
    if not edges:
        return {frozenset()}
    if budget == 0:
        return set()
    edge = min(edges, key=lambda e: (len(e), tuple(sorted(e))))
    found = {rest | {cell} for cell in sorted(edge)
             for rest in covers(frozenset(e for e in edges if cell not in e), budget - 1)}
    return {s for s in found if not any(t < s for t in found)}


def ranking(board, player):
    # ponytail: hand-scored policy proxy, replace with a frozen neural policy for engine ablations.
    scores = [0.0] * len(board)
    for line in windows(19):
        own = sum(board[i] == player for i in line)
        opp = sum(board[i] == -player for i in line)
        value = (5 ** own if not opp else 0) + (4 ** opp if not own else 0)
        for i in line:
            if not board[i]:
                scores[i] += value
    return sorted((i for i, x in enumerate(board) if not x), key=lambda i: (-scores[i], i))


def inject(base, order, own, enemy):
    """Keep baseline plus one representative completion per tactical constraint.

    Only tests existence of an immediate win/defense, NOT optimal second-stone
    placement. Production search must allow every legal filler for a singleton.
    """
    required = own if own else covers(enemy)
    added = set(base)
    for cells in required:
        if not cells:
            continue
        pair = set(cells)
        for cell in order:
            if len(pair) == 2:
                break
            pair.add(cell)
        if len(pair) == 2:
            added.add(frozenset(pair))
    return added


def self_check():
    # Exhaustively compare all <=3-edge hypergraphs on six cells with brute force.
    universe = range(6)
    edges = [frozenset(e) for size in (1, 2) for e in it.combinations(universe, size)]
    choices = [frozenset(c) for size in (0, 1, 2) for c in it.combinations(universe, size)]
    checked = 0
    for count in range(4):
        for family in it.combinations(edges, count):
            valid = [c for c in choices if all(c & e for e in family)]
            expected = {c for c in valid if not any(d < c for d in valid)}
            assert covers(frozenset(family)) == expected
            checked += 1
    assert covers(frozenset({frozenset({1}), frozenset({2}), frozenset({3})})) == set()
    assert len(windows(19)) == 924

    # Independent board oracle: place every opponent pair and check actual six.
    rng = random.Random(318)
    boards = 0
    pair_checks = 0
    while boards < 30:
        board = rng.choices((0, 1, -1), weights=(4, 2, 4), k=36)
        if won(board, 1, 6) or won(board, -1, 6):
            continue
        empty = [i for i, x in enumerate(board) if not x]
        predicted = threats(board, -1, 6)
        for a, b in it.combinations(empty, 2):
            board[a] = board[b] = -1
            assert won(board, -1, 6) == any(e <= {a, b} for e in predicted)
            board[a] = board[b] = 0
            pair_checks += 1
        boards += 1
    # Explicit overline and both diagonal directions.
    for dr, dc, start in ((0, 1, 0), (1, 0, 0), (1, 1, 0), (1, -1, 18)):
        board = [0] * 361
        for k in range(7):
            board[start + k * (19 * dr + dc)] = 1
        assert won(board, 1)
    return {"exhaustive_hypergraphs": checked, "synthetic_oracle_boards": boards,
            "independent_pair_checks": pair_checks}


def positions(seed, count):
    """Legal random games concentrated in central 9x9; NOT expert self-play."""
    rng = random.Random(seed)
    pool = [r * 19 + c for r in range(5, 14) for c in range(5, 14)]
    emitted = 0
    games = 0
    while emitted < count and games < 1000:
        games += 1
        board = [0] * 361
        board[180] = 1
        player = -1
        while True:
            own, enemy = threats(board, player), threats(board, -player)
            if own or enemy:
                yield board.copy(), player, own, enemy
                emitted += 1
                if emitted == count:
                    return
            empty = [i for i in pool if not board[i]]
            if len(empty) < 2:
                break
            terminal = False
            for cell in rng.sample(empty, 2):
                board[cell] = player
                if won(board, player):
                    terminal = True
                    break
            if terminal:
                break
            player = -player
    raise RuntimeError("Insufficient tactical positions")


def benchmark(seed, count=200):
    result = {"seed": seed, "positions": count, "own_win": 0, "defensible": 0,
              "unavoidable_next_turn_loss": 0, "baseline_success": 0, "injected_success": 0}
    overhead, timings = [], []
    for board, player, own, enemy in positions(seed, count):
        order = ranking(board, player)
        base = {frozenset(pair) for pair in it.combinations(order[:16], 2)}
        start = time.perf_counter_ns()
        augmented = inject(base, order, own, enemy)
        timings.append((time.perf_counter_ns() - start) / 1000)
        overhead.append(len(augmented) - len(base))
        if own:
            result["own_win"] += 1
            good = lambda pair: any(e <= pair for e in own)
        elif covers(enemy):
            result["defensible"] += 1
            good = lambda pair: all(e & pair for e in enemy)
        else:
            result["unavoidable_next_turn_loss"] += 1
            continue
        result["baseline_success"] += int(any(good(pair) for pair in base))
        result["injected_success"] += int(any(good(pair) for pair in augmented))
    possible = result["own_win"] + result["defensible"]
    assert result["injected_success"] == possible
    result.update(baseline_candidates=120, mean_extra_candidates=statistics.mean(overhead),
                  max_extra_candidates=max(overhead),
                  median_injection_us_excluding_scan=round(statistics.median(timings), 2))
    return result


if __name__ == "__main__":
    start = time.perf_counter()
    checks = self_check()
    results = [benchmark(seed) for seed in (6, 5070, 2026)]
    print(json.dumps({"python": platform.python_version(), "platform": platform.platform(),
                      "checks": checks, "experiments": results,
                      "elapsed_seconds": round(time.perf_counter() - start, 3),
                      "scope": "Synthetic tactical candidate recall; not neural training or match strength."}, indent=2))
