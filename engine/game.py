"""Square-board connection games. Rules vary; the placement engine does not."""
from dataclasses import asdict, dataclass, field
from functools import lru_cache
import re
import numpy as np


@dataclass(frozen=True)
class Rules:
    height: int = 9
    width: int = 9
    connect: int = 5
    stones_per_turn: int = 1
    starter_stones: int = 1

    def __post_init__(self):
        if any(type(value) is not int for value in asdict(self).values()):
            raise ValueError("Rule values must be integers.")
        if self.height != self.width:
            raise ValueError('Only square boards are supported; use N*N, for example "9*9".')
        if not (2 <= self.height <= 25 and 2 <= self.connect <= self.height):
            raise ValueError("Board size must be 2..25; connect must be 2..the board size.")
        if self.stones_per_turn not in (1, 2) or self.starter_stones not in (1, 2):
            raise ValueError("This engine supports one- or two-stone turns and openings.")

    @property
    def id(self):
        return f"connect{self.connect}-{self.height}x{self.width}-s{self.stones_per_turn}-o{self.starter_stones}-v1"

    def to_dict(self):
        return asdict(self)


DEFAULT_RULES = Rules()


def parse_board(text):
    match = re.fullmatch(r"(\d+)[*xX×](\d+)", text)
    if not match:
        raise ValueError('Board size must be N*N, for example "9*9" or "19*19".')
    height, width = map(int, match.groups())
    if height != width:
        raise ValueError('Only square boards are supported; use N*N, for example "9*9".')
    return height, width


@lru_cache(maxsize=32)
def _lines(rules):
    size, length = rules.height, rules.connect
    return np.array([[(r + k * dr) * size + c + k * dc for k in range(length)]
                     for r in range(size) for c in range(size)
                     for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1))
                     if 0 <= r + (length - 1) * dr < size
                     and 0 <= c + (length - 1) * dc < size], dtype=np.int32).reshape(-1, length)


@dataclass
class Game:
    board: np.ndarray | None = None
    player: int = 1
    left: int | None = None
    winner: int = 0
    done: bool = False
    moves: list[int] = field(default_factory=list)
    rules: Rules = field(default_factory=Rules)

    def __post_init__(self):
        if self.board is None:
            self.board = np.zeros(self.rules.height * self.rules.width, dtype=np.int8)
        if self.board.shape != (self.rules.height * self.rules.width,):
            raise ValueError("Board data does not match the configured rules.")
        if self.left is None:
            self.left = self.rules.starter_stones

    @property
    def size(self):
        return self.rules.height

    @property
    def shape(self):
        return self.size, self.size

    @property
    def win_length(self):
        return self.rules.connect

    @property
    def turn_stones(self):
        return self.rules.stones_per_turn

    @property
    def lines(self):
        return _lines(self.rules)

    def empty(self):
        return Game(rules=self.rules)

    def copy(self):
        return Game(self.board.copy(), self.player, self.left, self.winner,
                    self.done, self.moves.copy(), self.rules)

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
    def from_moves(cls, moves, *, rules=DEFAULT_RULES):
        if not isinstance(moves, list) or len(moves) > rules.height * rules.width:
            raise ValueError("Expected a move list no longer than the board's cell count.")
        game = cls(rules=rules)
        for move in moves:
            game.play(move)
        return game

    def threats(self, player, budget=None):
        budget = self.turn_stones if budget is None else budget
        # Features, legal actions and proofs repeatedly inspect the same position.
        # Include board contents and rules: callers can also edit the board directly.
        key = (self.rules, self.board.tobytes())
        if getattr(self, "_threat_key", None) != key:
            cells = self.board[self.lines]
            self._line_counts = {color: np.count_nonzero(cells == color, axis=1).astype(np.uint8)
                                 for color in (-1, 0, 1)}
            self._threat_cache = {}
            self._threat_key = key
        query = (player, budget)
        if query in self._threat_cache:
            return self._threat_cache[query]
        selected = self.lines[(self._line_counts[player] >= self.win_length - budget)
                             & (self._line_counts[-player] == 0)
                             & (self._line_counts[0] > 0)]
        result = frozenset(frozenset(int(i) for i in line if self.board[i] == 0) for line in selected)
        self._threat_cache[query] = result
        return result

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
                return np.array(sorted(set().union(*required)), dtype=np.int32)
        return np.flatnonzero(self.board == 0)


def batch_features(games):
    """Build identical planes with optional native kernels; reuse leaf line counts."""
    from . import native
    if not games:
        raise ValueError("Features need at least one game.")
    rules = games[0].rules
    if any(game.rules != rules for game in games):
        return np.stack([game.features() for game in games])
    boards = np.stack([game.board for game in games])
    players = np.array([game.player for game in games], dtype=np.int32)
    lines = games[0].lines
    keys = [(game.rules, game.board.tobytes()) for game in games]
    result = native.features(boards, players, np.array([game.left for game in games], dtype=np.int32),
                             lines, rules.stones_per_turn)
    if result is not None:
        planes, counts, threats = result
        for index, (game, key) in enumerate(zip(games, keys)):
            game._line_counts = {color: counts[index, slot] for slot, color in enumerate((-1, 0, 1))}
            game._threat_cache = {}
            game._threat_key = key
            for slot, color in enumerate((-1, 1)):
                mask = 0
                for budget in range(1, rules.stones_per_turn + 1):
                    mask |= 1 << (2 * slot + budget - 1)
                    if not int(threats[index]) & mask:
                        game._threat_cache[(color, budget)] = frozenset()
        return planes.reshape(len(games), 8, rules.height, rules.width)
    if len(games) == 1:
        return np.stack([games[0].features()])
    cells = np.take(boards, lines, axis=1)
    if all(getattr(game, "_threat_key", None) == key for game, key in zip(games, keys)):
        counts = {color: np.stack([game._line_counts[color] for game in games])
                  for color in (-1, 0, 1)}
    else:
        counts = {color: np.count_nonzero(cells == color, axis=2).astype(np.uint8)
                  for color in (-1, 0, 1)}
        for index, (game, key) in enumerate(zip(games, keys)):
            game._line_counts = {color: counts[color][index] for color in counts}
            game._threat_cache = {}
            game._threat_key = key
    planes = np.zeros((len(games), 8, boards.shape[1]), dtype=np.float32)
    planes[:, 0] = boards == players[:, None]
    planes[:, 1] = boards == -players[:, None]
    planes[:, 2] = np.array([game.left == 2 for game in games])[:, None]
    planes[:, 3] = (players == 1)[:, None]
    for offset, colors in ((4, players), (6, -players)):
        own = np.where((colors == 1)[:, None], counts[1], counts[-1])
        enemy = np.where((colors == 1)[:, None], counts[-1], counts[1])
        valid = (enemy == 0) & (own >= rules.connect - rules.stones_per_turn)
        has_threat = np.zeros(len(games), dtype=bool)
        for gap in range(1, rules.stones_per_turn + 1):
            eligible = valid & (counts[0] == gap)
            has_threat |= eligible.any(axis=1)
            for index in np.flatnonzero(~has_threat):
                games[index]._threat_cache[(int(colors[index]), gap)] = frozenset()
            rows, windows = np.nonzero(eligible)
            edges, inside = np.nonzero(boards[rows[:, None], lines[windows]] == 0)
            planes[rows[edges], offset + gap - 1, lines[windows[edges], inside]] = 1
    return planes.reshape(len(games), 8, rules.height, rules.width)


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
    """Original line scores; no learned parameters."""
    scores = np.zeros(game.board.size, dtype=np.float64)
    cells = game.board[game.lines]
    own = np.sum(cells == game.player, axis=1)
    enemy = np.sum(cells == -game.player, axis=1)
    weights = np.where(enemy == 0, 5.0 ** own, 0) + np.where(own == 0, 4.0 ** enemy, 0)
    np.add.at(scores, game.lines.ravel(), np.repeat(weights, game.win_length))
    y, x = np.indices(game.shape)
    scores -= ((x - game.size // 2) ** 2 + (y - game.size // 2) ** 2).ravel() * .001
    return scores


def heuristic(game):
    """Untrained fallback, not a competitive-strength claim."""
    scores = heuristic_scores(game)
    legal = game.actions()
    if not len(legal):
        raise ValueError("Cannot choose a placement after game end.")
    return int(legal[np.argmax(scores[legal])])
