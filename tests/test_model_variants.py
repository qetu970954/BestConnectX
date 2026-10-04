"""Experimental models must agree across the learner/native boundary and resume safely."""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from engine.cli import main
from engine.config import load_options
from engine.game import Game as Reference, Rules
from engine.network import Network
from engine.runtime import Game, backend, full_turn, model_for, search
from engine.selfplay import observation, train_step
from engine.training import load_state, network


class ModelVariantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_original_native_creation_interface(self):
        # Existing clients pass three dimensions and a device string, without an architecture argument.
        lib = backend()
        handle = lib.model_create(3, 8, 1, b'cpu')
        self.assertTrue(handle, lib.runtime_error().decode())
        lib.model_destroy(handle)
        self.assertFalse(lib.model_create(3, 8, 1, None))

    def test_outputs_learning_and_turns_for_both_games(self):
        for rules in (Rules(3, 3, 3), Rules(15, 15, 5), Rules(19, 19, 6, 2, 1)):
            center = rules.height ** 2 // 2
            histories = [[], [center], [center, center + 1]]
            if rules.stones_per_turn == 2:
                histories.extend([[center, center + 1, center + 2],
                                  [center, center + 1, center + 2, center - rules.width]])
            games = [Game.from_moves(moves, rules=rules) for moves in histories]
            x = np.stack([game.features() for game in games])
            for game, features in zip(games, x):
                np.testing.assert_array_equal(features, Reference.from_moves(game.moves, rules=rules).features())
            if rules.stones_per_turn == 2:
                self.assertEqual([(g.player, g.left) for g in games[1:]],
                                 [(-1, 2), (-1, 1), (1, 2), (1, 1)])
                pair = Game(rules=rules)
                pair.board[:4] = 1
                pair.board[rules.width:rules.width + 4] = -1
                self.assertTrue(pair.features()[5].any())
                self.assertTrue(pair.features()[7].any())
                np.testing.assert_array_equal(pair.features(), Reference(pair.board.copy(), rules=rules).features())
            for architecture in ('pooled', 'attention'):
                with self.subTest(rules=rules.id, architecture=architecture):
                    torch.manual_seed(37)
                    net = Network(8, 1, rules.height, architecture=architecture).eval()
                    if net.attention is not None:
                        # Nonzero positional weights make incorrect native row/column indexing observable.
                        with torch.no_grad():
                            net.attention.relative_bias.normal_(std=.1)
                    before, values = model_for(net).evaluate_features(x)
                    with torch.inference_mode():
                        expected, expected_values = net(torch.from_numpy(x))
                    np.testing.assert_allclose(before, expected.numpy(), atol=2e-5, rtol=2e-4)
                    np.testing.assert_allclose(values, expected_values.numpy(), atol=2e-5, rtol=2e-4)
                    replay = []
                    for i, game in enumerate(games):
                        policy = (game.board == 0).astype(np.float32)
                        policy /= policy.sum()
                        sample = observation(game, policy)
                        sample['result'] = float((1, -1, 0)[i % 3])
                        replay.append(sample)
                    train_step(net, torch.optim.AdamW(net.parameters(), lr=.0003), replay, 4,
                               np.random.default_rng(42), torch.device('cpu'), rules=rules)
                    net.eval()
                    after, values = model_for(net).evaluate_features(x)
                    with torch.inference_mode():
                        expected, expected_values = net(torch.from_numpy(x))
                    self.assertFalse(np.array_equal(before, after))
                    np.testing.assert_allclose(after, expected.numpy(), atol=2e-5, rtol=2e-4)
                    np.testing.assert_allclose(values, expected_values.numpy(), atol=2e-5, rtol=2e-4)
                    if net.attention is not None:
                        self.assertTrue(torch.isfinite(net.attention.relative_bias.grad).all())
                        self.assertGreater(float(net.attention.relative_bias.grad.abs().sum()), 0)
                    policies, _ = search(games, net, 4, workers=1)
                    for game, policy in zip(games, policies):
                        self.assertAlmostEqual(float(policy.sum()), 1., places=5)
                        self.assertTrue(np.all(policy[game.board != 0] == 0))
                    for game in games[1:3]:
                        child = game.copy()
                        color, left = child.player, child.left
                        moves, _, _ = full_turn(child, net, .02)
                        self.assertGreater(len(moves), 0)
                        for move in moves:
                            self.assertEqual(child.player, color)
                            child.play(move)
                        self.assertTrue(child.done or child.player != color)
                        self.assertLessEqual(len(moves), left)

    def test_architecture_is_saved_and_cannot_change_on_resume(self):
        for architecture in ('pooled', 'attention'):
            with self.subTest(architecture=architecture), tempfile.TemporaryDirectory() as name:
                root = Path(name)
                with redirect_stdout(StringIO()):
                    main(['train', '--preset', 'tictactoe', '--architecture', architecture, '--data', name,
                          '--device', 'cpu', '--max-games', '8', '--updates-per-cycle', '1',
                          '--simulations', '2', '--snapshot-every', '8', '--tactical-ms', '0', '--hours', '.01'])
                    first = load_state(root / 'latest.pt')
                    main(['train', '--data', name, '--device', 'cpu', '--max-games', '9', '--hours', '.01'])
                second = load_state(root / 'latest.pt')
                self.assertEqual(first['config']['architecture'], architecture)
                self.assertEqual(first['config'], second['config'])
                self.assertEqual(second['games'], 9)
                self.assertGreater(first['step'], 0)
                self.assertTrue(first['optimizer']['state'])
                restored = network(root, {'file': 'models/model-00000008.pt'}, torch.device('cpu'),
                                   Rules(**first['rule_config']))
                self.assertEqual(restored.config, first['config'])
                policy, value = model_for(restored).evaluate_features(np.zeros((1, 8, 3, 3), dtype=np.float32))
                self.assertTrue(np.isfinite(policy).all() and np.isfinite(value).all())
                original = (root / 'latest.pt').read_bytes()
                with self.assertRaises(SystemExit):
                    main(['train', '--data', name, '--device', 'cpu', '--architecture', 'residual'])
                self.assertEqual((root / 'latest.pt').read_bytes(), original)
                self.assertFalse((root / 'selfplay').exists())
                self.assertFalse((root / 'replay').exists())

    def test_presets_and_attention_width_validation(self):
        _, defaults, _ = load_options([])
        self.assertEqual({key: defaults[key] for key in ('architecture', 'device', 'workers', 'parallel', 'batch')},
                         {'architecture': 'residual', 'device': 'auto', 'workers': 6, 'parallel': 64, 'batch': 128})
        self.assertEqual((defaults['channels'], defaults['blocks'], defaults['replay_limit']), (64, 6, 20_000))
        for game in ('gomoku', 'connect6'):
            for architecture in ('pooled', 'attention'):
                _, options, _ = load_options(['--preset', f'{game}-{architecture}'])
                self.assertEqual(options['architecture'], architecture)
                self.assertEqual(options['stones_per_turn'], 2 if game == 'connect6' else 1)
        with self.assertRaises(ValueError):
            Network(6, 1, 3, architecture='attention')
        with self.assertRaises(ValueError):
            Network(8, 1, 3, architecture='unknown')


if __name__ == '__main__':
    unittest.main()
