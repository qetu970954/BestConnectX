"""Independent Python PUCT reference for tests; production search uses runtime.cpp."""
import time
import numpy as np


def _expansion(game, logits):
    actions = game.actions()
    x = logits[actions].astype(np.float64)
    x = np.exp(x - x.max())
    return actions, x / x.sum()


class Node:
    def __init__(self, game):
        self.game = game
        self.actions = None
        self.prior = None
        self.visits = None
        self.total = None
        self.children = {}

    def expand(self, logits):
        self.actions, self.prior = _expansion(self.game, logits)
        self.visits = np.zeros(len(self.actions), dtype=np.int32)
        self.total = np.zeros(len(self.actions), dtype=np.float64)

    def select(self):
        q = np.divide(self.total, self.visits, out=np.zeros_like(self.total), where=self.visits > 0)
        u = 1.5 * self.prior * np.sqrt(1 + self.visits.sum()) / (1 + self.visits)
        index = int(np.argmax(q + u))
        if index not in self.children:
            game = self.game.copy()
            game.play(int(self.actions[index]))
            self.children[index] = Node(game)
        return index, self.children[index]


def backup(path, leaf_player, value):
    # A turn has two placements. Negate only when the actual player changes.
    for parent, index in path:
        parent.visits[index] += 1
        parent.total[index] += value if parent.game.player == leaf_player else -value


def search(games, network, simulations=64, rng=None, deadline=None, stopped=lambda: False):
    if not games or any(g.done for g in games) or simulations < 1:
        raise ValueError("Search needs live games and at least one simulation.")
    roots = [Node(g) for g in games]
    logits, _ = network.evaluate(games)
    for root, p in zip(roots, logits):
        root.expand(p)
        if rng is not None:
            root.prior = .75 * root.prior + .25 * rng.dirichlet(np.full(len(root.actions), .1))
    completed = 0
    # Correctness oracle, not a production fallback.
    for _ in range(simulations):
        if stopped() or (deadline is not None and time.monotonic() >= deadline):
            break
        pending = []
        for root in roots:
            node, path = root, []
            while node.actions is not None and not node.game.done:
                index, child = node.select()
                path.append((node, index))
                node = child
            if node.game.done:
                backup(path, node.game.player, float(node.game.winner * node.game.player))
            else:
                pending.append((node, path))
        if pending:
            logits, values = network.evaluate([node.game for node, _ in pending])
            for (node, path), p, v in zip(pending, logits, values):
                node.expand(p)
                backup(path, node.game.player, float(v))
        completed += 1
    policies = []
    for root in roots:
        p = np.zeros(root.game.board.size, dtype=np.float32)
        counts = root.visits.astype(np.float32)
        p[root.actions] = counts / counts.sum() if counts.sum() else root.prior
        policies.append(p)
    return policies, completed


def choose(policy, rng=None, temperature=1.0):
    if rng is None or temperature == 0:
        return int(np.argmax(policy))
    p = policy.astype(np.float64) ** (1 / temperature)
    p /= p.sum()
    return int(rng.choice(len(p), p=p))
