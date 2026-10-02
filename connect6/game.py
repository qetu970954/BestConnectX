"""Standard 19x19 Connect6. Actions are single placements, not whole turns."""
from dataclasses import dataclass, field
import numpy as np

SIZE = 19
CELLS = SIZE * SIZE
def winning_lines(size, length):
    return np.array([[(r + k * dr) * size + c + k * dc for k in range(length)]
                     for r in range(size) for c in range(size)
                     for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1))
                     if 0 <= r + (length - 1) * dr < size
                     and 0 <= c + (length - 1) * dc < size], dtype=np.int32).reshape(-1, length)


LINES = winning_lines(SIZE, 6)


@dataclass
class Game:
    size = SIZE
    win_length = 6
    turn_stones = 2
    lines = LINES
    board: np.ndarray = field(default_factory=lambda: np.zeros(CELLS, dtype=np.int8))
    player: int = 1
    left: int = 1
    winner: int = 0
    done: bool = False
    moves: list[int] = field(default_factory=list)

    def copy(self):
        return type(self)(self.board.copy(), self.player, self.left, self.winner,
                    self.done, self.moves.copy())

    @property
    def shape(self):
        return self.size, self.size

    def empty(self):
        return type(self)()

    def play(self, action):
        if isinstance(action, bool) or not isinstance(action, (int, np.integer)):
            raise ValueError("Placement must be an integer cell index.")
        if self.done or not 0 <= action < self.board.size or self.board[action]:
            raise ValueError("Illegal placement: game ended, occupied cell, or out of bounds.")
        action = int(action)
        self.board[action] = self.player
        self.moves.append(action)
        height, width = self.shape
        r, c = divmod(action, width)
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            length = 1
            for sign in (-1, 1):
                rr, cc = r + sign * dr, c + sign * dc
                while (0 <= rr < height and 0 <= cc < width
                       and self.board[rr * width + cc] == self.player):
                    length += 1
                    rr, cc = rr + sign * dr, cc + sign * dc
            if length >= self.win_length:
                self.winner, self.done = self.player, True
                return
        if not np.any(self.board == 0):
            self.done = True
            return
        self.left -= 1
        if self.left == 0:
            self.player = -self.player
            self.left = self.turn_stones

    @classmethod
    def from_moves(cls, moves):
        if not isinstance(moves, list) or len(moves) > cls.size ** 2:
            raise ValueError(f"Expected a list of at most {cls.size ** 2} placements.")
        game = cls()
        for action in moves:
            game.play(action)
        return game

    def threats(self, player, budget=None):
        budget = self.turn_stones if budget is None else budget
        cells = self.board[self.lines]
        selected = self.lines[(np.sum(cells == player, axis=1) >= self.win_length - budget)
                         & (np.sum(cells == -player, axis=1) == 0)
                         & (np.sum(cells == 0, axis=1) > 0)]
        return frozenset(frozenset(int(i) for i in line if self.board[i] == 0) for line in selected)

    def features(self):
        planes = np.zeros((8, *self.shape), dtype=np.float32)
        planes[0] = (self.board == self.player).reshape(self.shape)
        planes[1] = (self.board == -self.player).reshape(self.shape)
        planes[2].fill(self.left == 2)
        planes[3].fill(self.player == 1)
        for offset, player in ((4, self.player), (6, -self.player)):
            for threat in self.threats(player):
                for cell in threat:
                    planes[offset + len(threat) - 1].flat[cell] = 1
        return planes

    def actions(self):
        """Exact immediate tactics, otherwise all empties (no permanent top-k)."""
        if self.done:
            return np.array([], dtype=np.int32)
        own = self.threats(self.player, self.left)
        if own:
            shortest = min(map(len, own))
            return np.array(sorted(set().union(*(e for e in own if len(e) == shortest))), dtype=np.int32)
        enemy = self.threats(-self.player)
        if enemy:
            required = covers(enemy, self.left)
            if required:
                # Any viable first defense can be completed within the remaining turn.
                return np.array(sorted(set().union(*required)), dtype=np.int32)
        return np.flatnonzero(self.board == 0)


def covers(edges, budget):
    """Minimal defensive hitting sets; each edge contains one or two cells."""
    if not edges:
        return {frozenset()}
    if budget == 0:
        return set()
    edge = min(edges, key=lambda e: (len(e), tuple(sorted(e))))
    found = {rest | {cell} for cell in sorted(edge)
             for rest in covers(frozenset(e for e in edges if cell not in e), budget - 1)}
    return {s for s in found if not any(t < s for t in found)}


def heuristic_scores(game):
    """Shared original line scores; no learned parameters."""
    scores = np.zeros(game.board.size, dtype=np.float64)
    cells = game.board[game.lines]
    own = np.sum(cells == game.player, axis=1)
    enemy = np.sum(cells == -game.player, axis=1)
    weights = np.where(enemy == 0, 5.0 ** own, 0) + np.where(own == 0, 4.0 ** enemy, 0)
    np.add.at(scores, game.lines.ravel(), np.repeat(weights, game.win_length))
    # Prefer a central opening when all line scores tie.
    height, width = game.shape
    y, x = np.indices((height, width))
    scores -= ((x - width // 2) ** 2 + (y - height // 2) ** 2).ravel() * .001
    return scores


def heuristic(game):
    """Original, untrained fallback and baseline. Not a competitive-strength claim."""
    scores = heuristic_scores(game)
    legal = game.actions()
    if not len(legal):
        raise ValueError("Cannot choose a placement after game end.")
    return int(legal[np.argmax(scores[legal])])
