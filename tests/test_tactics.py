"""Selected tactical regressions, not an unbiased playing-strength tournament."""
import itertools
import json
from pathlib import Path
import time
import unittest
import numpy as np
from connect6.game import Game, heuristic
from connect6.tactics import forcing_win, verifies
from connect6.training import turn_action

FIXTURES = json.loads((Path(__file__).parent / 'fixtures/forks.json').read_text())['selected_fixtures']


def independent_completions(game, player):
    # Separate scanner: do not reuse LINES, threats(), or covers().
    edges = set()
    for r in range(19):
        for c in range(19):
            for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1)):
                if not (0 <= r + 5*dr < 19 and 0 <= c + 5*dc < 19):
                    continue
                indices = [(r+k*dr)*19+c+k*dc for k in range(6)]
                if any(game.board[i] == -player for i in indices):
                    continue
                empty = [i for i in indices if not game.board[i]]
                if 0 < len(empty) <= 2:
                    edges.add(sum(1 << i for i in empty))
    return edges


class TacticalTests(unittest.TestCase):
    def test_selected_forks_and_every_defensive_pair(self):
        for fixture in FIXTURES:
            game = Game.from_moves(fixture['moves'])
            self.assertFalse(game.done)
            self.assertFalse(game.threats(game.player, game.left))
            self.assertFalse(verifies(game, fixture['baseline']))
            certificate = forcing_win(game, deadline=time.monotonic() + 5)
            self.assertIsNotNone(certificate)
            self.assertTrue(verifies(game, certificate))
            player = game.player
            child = game.copy()
            for move in certificate:
                child.play(move)
            self.assertFalse(child.done)  # this is a fork, not an immediate six
            attack = independent_completions(child, player)
            counterwins = independent_completions(child, -player)
            for a, b in itertools.combinations(np.flatnonzero(child.board == 0), 2):
                reply = (1 << int(a)) | (1 << int(b))
                self.assertFalse(any(reply & e == e for e in counterwins))
                self.assertTrue(any(reply & e == 0 for e in attack))

    def test_symmetries_and_player_perspective(self):
        base = Game.from_moves(FIXTURES[0]['moves'])
        for k in range(4):
            for flip in (False, True):
                indices = np.arange(361).reshape(19, 19)
                transformed = np.rot90(indices, k)
                if flip:
                    transformed = transformed[:, ::-1]
                mapping = np.argsort(transformed.ravel())
                game = Game.from_moves([int(mapping[m]) for m in base.moves])
                proof = forcing_win(game, deadline=time.monotonic() + 5)
                self.assertIsNotNone(proof)
                self.assertTrue(verifies(game, proof))
                swapped = game.copy(); swapped.board *= -1; swapped.player *= -1
                self.assertTrue(verifies(swapped, proof))

    def test_counterwin_unknown_and_partial_turn(self):
        game = Game(); game.left = 2
        game.board[[9*19+c for c in (5,6,7)] + [r*19+8 for r in (5,6,7)]] = 1
        self.assertTrue(verifies(game, (179, 160)))
        game.board[:5] = -1
        self.assertFalse(verifies(game, (179, 160)))  # opponent wins before blocking
        self.assertIsNone(forcing_win(game, deadline=0))
        self.assertIsNone(forcing_win(game, deadline=time.monotonic()+5, stopped=lambda: True))
        self.assertIsNone(forcing_win(Game(), deadline=time.monotonic()+5))
        self.assertFalse(verifies(game, (179,)))  # not a complete turn
        self.assertFalse(verifies(game, (179, 179)))
        game = Game.from_moves(FIXTURES[0]['moves'])
        proof = FIXTURES[0]['certificate']
        game.play(proof[0])
        found = forcing_win(game, deadline=time.monotonic()+5)
        self.assertEqual(len(found), 1)
        self.assertTrue(verifies(game, found))

    def test_plan_is_committed_only_with_placement(self):
        game = Game.from_moves(FIXTURES[0]['moves']); original = game.copy()
        plan = []
        first = turn_action(game, None, 5, 1, lambda: False, tactics=True, plan=plan)
        self.assertEqual(len(plan), 2)
        # Simulate session stop after thinking but before committing the placement.
        restored = json.loads(json.dumps(plan))
        self.assertEqual(turn_action(game, None, 0, 1, lambda: False, tactics=True, plan=restored), first)
        game.play(first); restored.pop(0)
        second = turn_action(game, None, 0, 1, lambda: False, tactics=True, plan=restored)
        game.play(second); restored.pop(0)
        self.assertEqual(restored, [])
        self.assertTrue(verifies(original, (first, second)))
        # Legacy incumbent stays unchanged unless tactics are explicitly enabled.
        self.assertEqual(turn_action(original, None, 5, 1, lambda: False), heuristic(original))


if __name__ == '__main__':
    unittest.main()
