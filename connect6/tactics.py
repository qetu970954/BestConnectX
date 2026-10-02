"""Bounded one-turn fork search with exact certificates, not a full TSS solver.

A certificate wins now, or leaves the opponent unable to prevent our next-turn
win AND unable to win first. None always means UNKNOWN, never proven loss.
"""
from itertools import combinations
import time
import numpy as np
from .game import covers, heuristic_scores


def verifies(game, moves):
    """Verify a complete-turn certificate independently of candidate ranking."""
    if game.done or not moves or len(moves) > game.left:
        return False
    attacker = game.player
    child = game.copy()
    try:
        for move in moves:
            child.play(move)
    except ValueError:
        return False
    if child.done:
        return child.winner == attacker
    if child.player == attacker or child.threats(child.player, child.left):
        return False  # incomplete turn, or opponent wins before needing to defend
    return not covers(child.threats(attacker), child.left)


def forcing_win(game, *, deadline, max_candidates=256, width=24, stopped=lambda: False):
    """Return verified placements, or None on timeout/exhaustion/candidate miss.

    Limits bound discovery only. A returned certificate checks all one/two-stone
    opponent wins and all defensive covers, not just the opponent's shortlist.
    """
    if game.done or max_candidates < 1 or width < 2 or stopped() or time.monotonic() >= deadline:
        return None
    immediate = game.threats(game.player, game.left)
    if immediate:
        return tuple(sorted(min(immediate, key=lambda e: (len(e), tuple(sorted(e))))))
    cells = game.board[game.lines]
    potential = game.lines[(np.sum(cells == -game.player, axis=1) == 0)
                      & (np.sum(cells == game.player, axis=1) >= game.win_length - game.turn_stones - game.left)]
    if len(potential) <= game.turn_stones:
        return None  # Not enough future completion windows to exceed the defender's turn.
    pool = {int(i) for i in potential.ravel() if not game.board[i]}
    enemy = game.threats(-game.player)
    for threat in enemy:
        pool.update(threat)
    scores = heuristic_scores(game)
    ranked = sorted(pool, key=lambda i: (-scores[i], i))[:width]
    if game.left == 1:
        candidates = [(a,) for a in ranked]
    else:
        candidates = sorted(combinations(ranked, 2), key=lambda p: -(scores[p[0]] + scores[p[1]]))
    # ponytail: bounded heuristic shortlist can miss forks; widen only with measured benefit.
    for moves in candidates[:max_candidates]:
        if stopped() or time.monotonic() >= deadline:
            return None
        # Necessary defense test avoids expensive scans of plainly losing turns.
        if enemy and any(not set(moves).intersection(e) for e in enemy):
            continue
        if verifies(game, moves):
            return tuple(moves)
    return None
