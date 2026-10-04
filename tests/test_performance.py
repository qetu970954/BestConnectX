"""Cache correctness across rule variants, mutations and real optimizer updates."""
import copy
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import torch
from engine.game import Game, Rules, batch_features
from engine.network import Network
from engine.selfplay import _replay_features, observation, train_step
from engine.training import _replay_window


def completions(game, player, budget):
    edges = set()
    size, length = game.size, game.win_length
    for row in range(size):
        for col in range(size):
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                if not (0 <= row + (length - 1) * dr < size
                        and 0 <= col + (length - 1) * dc < size):
                    continue
                cells = [(row + i * dr) * size + col + i * dc for i in range(length)]
                empty = frozenset(cell for cell in cells if game.board[cell] == 0)
                if (0 < len(empty) <= budget
                        and all(game.board[cell] in (0, player) for cell in cells)):
                    edges.add(empty)
    return frozenset(edges)


class PerformanceTests(unittest.TestCase):
    def tearDown(self):
        _replay_features.cache_clear()

    def test_threat_cache_matches_scanner_after_direct_edits_and_rule_changes(self):
        rng = np.random.default_rng(5070)
        for rules in (Rules(), Rules(15, 15, 5), Rules(19, 19, 6, 2, 1)):
            game = Game(rules=rules)
            for _ in range(8):
                game.board[:] = rng.choice((-1, 0, 1), game.board.size)
                for player in (-1, 1):
                    for budget in (1, 2):
                        expected = completions(game, player, budget)
                        self.assertEqual(game.threats(player, budget), expected)
                        self.assertEqual(game.threats(player, budget), expected)
            # Changing the win length on an existing board must also invalidate it.
            game.rules = Rules(rules.height, rules.width, rules.connect - 1,
                               rules.stones_per_turn, rules.starter_stones)
            self.assertEqual(game.threats(1, 2), completions(game, 1, 2))
            child = game.copy()
            cell = int(np.flatnonzero(child.board == 0)[0])
            child.play(cell)
            self.assertEqual(child.threats(child.player),
                             completions(child, child.player, child.turn_stones))
            self.assertEqual(game.threats(1, 2), completions(game, 1, 2))

    def test_replay_cache_preserves_updates_and_tracks_board_and_perspective(self):
        torch.set_num_threads(2)
        for rules in (Rules(), Rules(15, 15, 5), Rules(19, 19, 6, 2, 1)):
            game = Game.from_moves([rules.height ** 2 // 2], rules=rules)
            policy = np.zeros(game.board.size, dtype=np.float32)
            actions = game.actions()
            policy[actions] = 1 / len(actions)
            row = observation(game, policy)
            row['result'] = 1.
            cached = Network(channels=8, blocks=1, size=game.size)
            uncached = copy.deepcopy(cached)
            first = torch.optim.AdamW(cached.parameters())
            second = torch.optim.AdamW(uncached.parameters())
            for seed in range(3):
                loss = train_step(cached, first, [row], 4, np.random.default_rng(seed),
                                  torch.device('cpu'), rules=rules)
                with patch('engine.selfplay._replay_features', _replay_features.__wrapped__):
                    expected = train_step(uncached, second, [row], 4, np.random.default_rng(seed),
                                          torch.device('cpu'), rules=rules)
                self.assertEqual(loss, expected)
                for key, value in cached.state_dict().items():
                    torch.testing.assert_close(value, uncached.state_dict()[key], rtol=0, atol=0)
                row['board'][seed] = -1
                row['player'] = -row['player']
                row['left'] = 3 - row['left']
                board = row['board'].numpy()
                features = _replay_features(board.tobytes(), board.dtype.str,
                                            row['player'], row['left'], rules)
                np.testing.assert_array_equal(features,
                    Game(board, row['player'], row['left'], rules=rules).features())

    def test_resume_reads_only_recent_games_and_deduplicates_pending_exports(self):
        root = Path('unused-run')
        paths = [root / 'replay' / f'game-{number:09d}.pt' for number in range(1, 6)]
        state = {'settings': {'replay_limit': 5},
                 'pending_games': [{'record': {'game': 5}, 'samples': [50, 51]}]}
        def load(path):
            number = int(path.stem.split('-')[1])
            return {'samples': [10 * number, 10 * number + 1]}
        with patch.object(Path, 'glob', return_value=iter(paths)), \
                patch('engine.training.load_checkpoint', side_effect=load) as loaded:
            self.assertEqual(_replay_window(root, state), [31, 40, 41, 50, 51])
            self.assertEqual([call.args[0] for call in loaded.call_args_list], [paths[3], paths[2]])
        with patch.object(Path, 'glob', return_value=iter([])):
            self.assertEqual(_replay_window(root, state), [50, 51])

    def test_batched_features_match_scalar_features_across_rules_and_cache_edits(self):
        rng = np.random.default_rng(5070)
        groups = []
        for rules in (Rules(), Rules(15, 15, 5), Rules(19, 19, 6, 2, 1), Rules(5, 5, 3, 2, 2)):
            games = [Game(rng.choice(np.array([-1, 0, 1], dtype=np.int8), rules.height ** 2),
                          int(rng.choice((-1, 1))), int(rng.integers(1, 3)), rules=rules)
                     for _ in range(16)]
            groups.append(games)
            for warm in (False, True):
                if not warm:
                    for game in games:
                        game._threat_key = None
                np.testing.assert_array_equal(batch_features(games), np.stack([g.features() for g in games]))
            for game in games:
                game.board[0] = -game.board[0]
                game.player *= -1
                game.left = 3 - game.left
            np.testing.assert_array_equal(batch_features(games), np.stack([g.features() for g in games]))
        # Same board dimensions can still have different winning lengths/turn rules.
        mixed = [groups[0][0], Game(rules=Rules(9, 9, 6, 2, 1))]
        np.testing.assert_array_equal(batch_features(mixed), np.stack([g.features() for g in mixed]))


if __name__ == '__main__':
    unittest.main()
