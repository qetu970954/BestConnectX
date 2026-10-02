"""Configurable freestyle connection games; shared placement rules and tactical search."""
from dataclasses import asdict, dataclass, field
from functools import lru_cache
import re
import numpy as np
from connect6.game import Game as PlacementGame, winning_lines


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
    return winning_lines(rules.height, rules.connect)


@dataclass
class Game(PlacementGame):
    board: np.ndarray | None = None
    left: int | None = None
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
    def win_length(self):
        return self.rules.connect

    @property
    def turn_stones(self):
        return self.rules.stones_per_turn

    @property
    def lines(self):
        return _lines(self.rules)

    def empty(self):
        return type(self)(rules=self.rules)

    def copy(self):
        return type(self)(self.board.copy(), self.player, self.left, self.winner,
                          self.done, self.moves.copy(), self.rules)

    @classmethod
    def from_moves(cls, moves, *, rules=DEFAULT_RULES):
        if not isinstance(moves, list) or len(moves) > rules.height * rules.width:
            raise ValueError("Expected a move list no longer than the board's cell count.")
        game = cls(rules=rules)
        for move in moves:
            game.play(move)
        return game
