"""Bounded threat-space search. Deadlines are absolute time.perf_counter values."""
from itertools import combinations
import time
import numpy as np
from .game import covers, heuristic_scores


class _Limit(Exception):
    pass


class _Budget:
    def __init__(self, deadline, max_nodes, stopped):
        self.deadline, self.max_nodes, self.stopped = deadline, max_nodes, stopped
        self.nodes = 0

    def tick(self):
        if self.nodes >= self.max_nodes or self.stopped() or time.perf_counter() >= self.deadline:
            raise _Limit
        self.nodes += 1


def _valid_limits(max_turns, max_nodes):
    return (isinstance(max_turns, int) and not isinstance(max_turns, bool) and max_turns > 0
            and isinstance(max_nodes, int) and not isinstance(max_nodes, bool) and max_nodes > 0)


def _apply_turn(game, moves):
    if len(moves) > game.left:
        raise ValueError("Placement sequence crosses a turn boundary.")
    child = game.copy()
    played = []
    for move in moves:
        child.play(move)
        played.append(int(move))
        if child.done:
            break
    return child, tuple(played)


def _defenses(game, threats, budget):
    """Every complete legal reply that hits every immediate winning line."""
    minimal = covers(threats, game.left)
    budget.tick()
    if not minimal:
        return []
    empty = np.flatnonzero(game.board == 0).tolist()
    count = min(game.left, len(empty))
    result = set()
    for cover in minimal:
        if len(cover) > count:
            continue
        rest = count - len(cover)
        available = [cell for cell in empty if cell not in cover]
        for extra in combinations(available, rest):
            budget.tick()
            result.add(tuple(sorted((*cover, *extra))))
    return sorted(result)


def _verified_defenses(game, threats, budget):
    """Reconstruct every complete reply without trusting the search-side cover generator."""
    if not threats:
        return []
    empty = np.flatnonzero(game.board == 0).tolist()
    count = min(game.left, len(empty))
    if count == 1:
        replies = []
        for cell in empty:
            budget.tick()
            if all(cell in edge for edge in threats):
                replies.append((cell,))
        return replies
    if count == 2:
        replies = set()
        for first in empty:
            budget.tick()
            uncovered = [edge for edge in threats if first not in edge]
            if not uncovered:
                for second in empty:
                    budget.tick()
                    if second != first:
                        replies.add(tuple(sorted((first, second))))
                continue
            possible = set(uncovered[0])
            for edge in uncovered[1:]:
                possible.intersection_update(edge)
                if not possible:
                    break
            for second in possible:
                if second != first:
                    budget.tick()
                    replies.add(tuple(sorted((first, second))))
        return sorted(replies)
    # Nonstandard turn sizes retain exact exhaustive behavior.
    replies = []
    for reply in combinations(empty, count):
        budget.tick()
        if all(any(cell in edge for cell in reply) for edge in threats):
            replies.append(tuple(reply))
    return replies


def _candidates(game, width, max_candidates, budget):
    own = game.threats(game.player, game.left)
    if own:
        return [tuple(sorted(min(own, key=lambda edge: (len(edge), tuple(sorted(edge))))))]
    cells = game.board[game.lines]
    needed = min(game.left, int(np.count_nonzero(game.board == 0)))
    threshold = game.win_length - game.turn_stones - needed
    potential = game.lines[(np.sum(cells == -game.player, axis=1) == 0)
                      & (np.sum(cells == game.player, axis=1) >= threshold)]
    pool = {int(cell) for cell in potential.ravel() if not game.board[cell]}
    if not pool or not needed:
        return []
    scores = heuristic_scores(game)
    # ponytail: top-width attack candidates may miss proofs; widen only with benchmark evidence.
    ranked = sorted(pool, key=lambda cell: (-scores[cell], cell))[:width]
    ordered = []
    for turn in combinations(ranked, needed):
        budget.tick()
        ordered.append(tuple(turn))
    ordered.sort(key=lambda turn: (-sum(scores[cell] for cell in turn), turn))
    return ordered[:max_candidates]


def _search_node(game, attacker, turns_left, width, max_candidates, budget):
    budget.tick()
    if game.done or game.player != attacker or turns_left < 1:
        return None
    own = game.threats(attacker, game.left)
    if own:
        child, played = _apply_turn(game, min(own, key=lambda edge: (len(edge), tuple(sorted(edge)))))
        if child.done and child.winner == attacker:
            return {"moves": list(played), "responses": []}
    if turns_left < 2:
        return None
    for moves in _candidates(game, width, max_candidates, budget):
        child, played = _apply_turn(game, moves)
        if child.done:
            if child.winner == attacker:
                return {"moves": list(played), "responses": []}
            continue
        if child.player == attacker:
            continue
        threats = child.threats(attacker)
        if not threats or child.threats(child.player, child.left):
            continue
        defenses = _defenses(child, threats, budget)
        if not defenses:
            return {"moves": list(played), "responses": []}
        responses = []
        for reply in defenses:
            budget.tick()
            position, _ = _apply_turn(child, reply)
            if position.done or position.player != attacker:
                break
            proof = _search_node(position, attacker, turns_left - 1, width,
                                 max_candidates, budget)
            if proof is None:
                break
            responses.append({"moves": list(reply), "proof": proof})
        if len(responses) == len(defenses):
            return {"moves": list(played), "responses": responses}
    return None


def _verify_node(game, attacker, turns_left, node, budget):
    budget.tick()
    if (not isinstance(node, dict) or set(node) != {"moves", "responses"}
            or not isinstance(node["moves"], list) or not node["moves"]
            or not isinstance(node["responses"], list)):
        return False
    moves = node["moves"]
    if any(isinstance(move, bool) or not isinstance(move, int) for move in moves):
        return False
    try:
        child, played = _apply_turn(game, moves)
    except (ValueError, TypeError):
        return False
    if played != tuple(moves):
        return False
    if child.done:
        return child.winner == attacker and not node["responses"]
    if child.player == attacker or turns_left < 2:
        return False
    threats = child.threats(attacker)
    if not threats or child.threats(child.player, child.left):
        return False
    defenses = _verified_defenses(child, threats, budget)
    if not defenses:
        return not node["responses"]
    if len(node["responses"]) != len(defenses):
        return False
    seen = set()
    for row, expected in zip(node["responses"], defenses):
        budget.tick()
        if (not isinstance(row, dict) or set(row) != {"moves", "proof"}
                or not isinstance(row["moves"], list)
                or any(isinstance(move, bool) or not isinstance(move, int)
                       for move in row["moves"])):
            return False
        reply = tuple(row["moves"])
        if reply != expected or reply in seen:
            return False
        seen.add(reply)
        position, played_reply = _apply_turn(child, reply)
        if played_reply != reply or position.done or position.player != attacker:
            return False
        if not _verify_node(position, attacker, turns_left - 1, row["proof"], budget):
            return False
    return True


def verify_tss(game, proof, *, deadline, max_nodes=100_000, stopped=lambda: False):
    """Return True/False, or None if the complete proof could not be checked."""
    if not _valid_limits(1, max_nodes):
        raise ValueError("max_nodes must be a positive integer.")
    budget = _Budget(deadline, max_nodes, stopped)
    try:
        budget.tick()
        if (not isinstance(proof, dict) or set(proof) != {"version", "attacker", "max_turns", "tree"}
                or type(proof["version"]) is not int or proof["version"] != 1
                or type(proof["attacker"]) is not int or proof["attacker"] != game.player
                or not _valid_limits(proof["max_turns"], max_nodes)
                or proof["max_turns"] > 20):
            return False
        return _verify_node(game, proof["attacker"], proof["max_turns"], proof["tree"], budget)
    except _Limit:
        return None


def search_tss(game, *, max_turns=3, deadline, max_nodes=100_000,
               width=24, max_candidates=256, stopped=lambda: False):
    """Find a bounded proof candidate; callers must verify it before use."""
    if not _valid_limits(max_turns, max_nodes) or max_turns > 20:
        raise ValueError("max_turns must be 1–20 and max_nodes must be a positive integer.")
    if (isinstance(width, bool) or not isinstance(width, int) or not 2 <= width <= 64
            or isinstance(max_candidates, bool) or not isinstance(max_candidates, int)
            or max_candidates < 1):
        raise ValueError("width must be at least 2 and max_candidates must be positive.")
    if game.done:
        return None
    attacker = game.player
    budget = _Budget(deadline, max_nodes, stopped)
    try:
        tree = _search_node(game, attacker, max_turns, width, max_candidates, budget)
    except _Limit:
        return None
    if tree is None:
        return None
    if stopped() or time.perf_counter() >= deadline:
        return None
    return {"version": 1, "attacker": attacker, "max_turns": max_turns, "tree": tree}
