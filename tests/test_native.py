"""Differential checks for CPU kernels and the production C++ search."""
import unittest
from unittest.mock import patch
import numpy as np
import torch
from engine import native
from engine.game import Game, Rules, batch_features
from engine.network import Network
from engine.search import search
from engine.runtime import search as native_search
from engine.training import opening
from tests.test_performance import completions


class FallbackTests(unittest.TestCase):
    def test_python_batch_cache_matches_independent_threat_scanner(self):
        rng = np.random.default_rng(5070)
        with patch('engine.native.library', return_value=None):
            for rules in (Rules(2, 2, 2, 2, 1), Rules(15, 15, 5),
                          Rules(19, 19, 6, 2, 1), Rules(25, 25, 25, 2, 2)):
                games = [Game(rng.choice(np.array([-1, 0, 1], dtype=np.int8), rules.height ** 2),
                              player, rules.stones_per_turn, rules=rules) for player in (-1, 1)]
                expected = np.stack([game.features() for game in games])
                np.testing.assert_array_equal(batch_features(games), expected)
                for game in games:
                    for player in (-1, 1):
                        for budget in (1, 2):
                            self.assertEqual(game.threats(player, budget), completions(game, player, budget))


@unittest.skipIf(native.library() is None, 'Optional C++ kernels have not been built/enabled.')
class NativeTests(unittest.TestCase):
    def test_features_counts_and_threat_cache_match_python_across_supported_limits(self):
        rng = np.random.default_rng(5070)
        for rules in (Rules(), Rules(15, 15, 5), Rules(19, 19, 6, 2, 1),
                      Rules(2, 2, 2, 2, 1), Rules(25, 25, 25, 2, 2)):
            games = [Game(rng.choice(np.array([-1, 0, 1], dtype=np.int8), rules.height ** 2),
                          int(rng.choice((-1, 1))), int(rng.integers(1, 3)), rules=rules)
                     for _ in range(12)]
            # Expected features are computed before native code populates any caches.
            expected = np.stack([game.features() for game in games])
            boards = np.stack([game.board for game in games])
            result = native.features(boards, np.array([g.player for g in games], dtype=np.int32),
                np.array([g.left for g in games], dtype=np.int32), games[0].lines, rules.stones_per_turn)
            planes, counts, _ = result
            np.testing.assert_array_equal(planes.reshape(expected.shape), expected)
            for slot, color in enumerate((-1, 0, 1)):
                np.testing.assert_array_equal(counts[:, slot],
                    np.count_nonzero(boards[:, games[0].lines] == color, axis=2))
            for game in games:
                game._threat_key = None
            np.testing.assert_array_equal(batch_features(games), expected)
            for game in games:
                for player in (-1, 1):
                    for budget in (1, 2):
                        self.assertEqual(game.threats(player, budget), completions(game, player, budget))

    def test_puct_matches_numpy_including_first_tie_and_nan(self):
        rng = np.random.default_rng(5070)
        for size in (1, 2, 81, 225, 361, 625):
            for _ in range(100):
                prior = rng.dirichlet(np.ones(size))
                visits = rng.integers(0, 500, size, dtype=np.int32)
                total = rng.uniform(-1, 1, size) * visits
                q = np.divide(total, visits, out=np.zeros_like(total), where=visits > 0)
                u = 1.5 * prior * np.sqrt(1 + visits.sum()) / (1 + visits)
                self.assertEqual(native.select(prior, visits, total), int(np.argmax(q + u)))
        prior, visits, total = np.ones(4), np.zeros(4, dtype=np.int32), np.zeros(4)
        self.assertEqual(native.select(prior, visits, total), 0)
        prior[1:3] = np.nan
        self.assertEqual(native.select(prior, visits, total), 1)
        with self.assertRaises(ValueError):
            native.select(prior, visits[:1], total)
        with self.assertRaises(ValueError):
            native.features(np.zeros((1, 4), dtype=np.int8), np.ones(1, dtype=np.int32),
                            np.ones(1, dtype=np.int32), np.array([[0, 4]], dtype=np.int32), 1)

    def test_reference_search_uses_independent_python_selection(self):
        class Uniform:
            def evaluate(self, games):
                return (np.zeros((len(games), games[0].board.size), dtype=np.float32),
                        np.full(len(games), .375, dtype=np.float32))
        rules = Rules(3, 3, 3)
        games = [opening(i, 123456, rules) for i in range(4)]
        with patch('engine.native.select', side_effect=AssertionError('per-node ctypes call')):
            policies, count = search(games, Uniform(), 64)
        self.assertEqual(count, 64)
        for game, policy in zip(games, policies):
            self.assertAlmostEqual(float(policy.sum()), 1)
            self.assertFalse(np.any(policy[game.board != 0]))

    def test_actual_search_policies_and_backups_are_identical(self):
        torch.set_num_threads(2)
        torch.manual_seed(5070)
        for rules in (Rules(), Rules(15, 15, 5), Rules(19, 19, 6, 2, 1)):
            network = Network(size=rules.height, channels=8, blocks=1)
            games = [opening(i, 123456, rules) for i in range(8)]
            references = [Game.from_moves(game.moves, rules=rules) for game in games]
            actual, count = native_search(games, network, simulations=32, workers=2)
            with patch('engine.native.library', return_value=None):
                expected, expected_count = search(references, network, simulations=32)
            self.assertEqual(count, expected_count)
            np.testing.assert_array_equal(actual, expected)


if __name__ == '__main__':
    unittest.main()
