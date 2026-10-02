import itertools
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from connect6.game import Game, LINES, covers, heuristic
from connect6.network import Network, augment
from connect6.search import Node, backup, search
from connect6.storage import atomic_bytes, run_lock, busy, save_checkpoint, load_checkpoint
from connect6.training import (confidence, train_step, observation, opening, turn_action,
                               verify_tss_sample)
from connect6.tactics import verifies


class Uniform:
    def evaluate(self, games):
        return np.zeros((len(games), 361), dtype=np.float32), np.zeros(len(games), dtype=np.float32)


class CoreTests(unittest.TestCase):
    def test_turns_and_replay(self):
        g = Game()
        expected = [(-1, 2), (-1, 1), (1, 2), (1, 1), (-1, 2)]
        for action, phase in enumerate(expected):
            g.play(action)
            self.assertEqual((g.player, g.left), phase)
        replay = Game.from_moves(g.moves)
        np.testing.assert_array_equal(g.board, replay.board)
        for invalid in (True, -1, 361, "4", 1.0, None, 0):
            with self.assertRaises(ValueError):
                g.play(invalid)

    def test_wins_and_immediate_stop(self):
        for stride, start in ((1, 0), (19, 0), (20, 0), (18, 18)):
            for length in (6, 7):
                g = Game()
                g.left = 2
                indexes = [start + i * stride for i in range(length)]
                hole = indexes.pop(3)
                g.board[indexes] = 1
                g.play(hole)
                self.assertTrue(g.done)
                self.assertEqual(g.winner, 1)
                self.assertEqual(g.player, 1)
                with self.assertRaises(ValueError):
                    g.play(300)
        g = Game.from_moves([0, 19, 20, 1, 2, 21, 22, 3, 4, 23, 40, 5])
        self.assertEqual(g.winner, 1)
        with self.assertRaises(ValueError):
            Game.from_moves(g.moves + [100])

    def test_draw(self):
        g = Game()
        g.board = np.array([1 if (r + 2*c) % 4 < 2 else -1 for r in range(19) for c in range(19)], dtype=np.int8)
        self.assertFalse(np.any(np.all(g.board[LINES] == 1, axis=1)))
        self.assertFalse(np.any(np.all(g.board[LINES] == -1, axis=1)))
        player = int(g.board[360]); g.board[360] = 0; g.player = player
        g.play(360)
        self.assertTrue(g.done)
        self.assertEqual(g.winner, 0)

    def test_tactics(self):
        self.assertEqual(len(LINES), 924)
        g = Game(); g.board[0:4] = 1; g.left = 2
        self.assertEqual(set(g.actions()), {4, 5})
        # A proven immediate win must not spend the neural thinking budget.
        self.assertIn(turn_action(g, object(), 5, 100000, lambda: False), (4, 5))
        g.play(4)
        self.assertEqual(list(g.actions()), [5])
        g.play(5); self.assertEqual(g.winner, 1)
        g = Game(); g.board[0:5] = -1; g.left = 2
        self.assertEqual(heuristic(g), 5)
        g.play(5)
        self.assertGreater(len(g.actions()), 1)  # second defensive stone stays flexible

    def test_exhaustive_cover_oracle(self):
        edges = [frozenset(e) for n in (1, 2) for e in itertools.combinations(range(6), n)]
        choices = [frozenset(e) for n in (0, 1, 2) for e in itertools.combinations(range(6), n)]
        for n in range(4):
            for family in itertools.combinations(edges, n):
                valid = [c for c in choices if all(c & e for e in family)]
                expected = {c for c in valid if not any(d < c for d in valid)}
                self.assertEqual(covers(frozenset(family), 2), expected)

    def test_same_player_backup_and_search(self):
        g = Game(); g.play(180)
        a = Node(g); a.expand(np.zeros(361))
        g = g.copy(); g.play(0)
        b = Node(g); b.expand(np.zeros(361))
        backup([(a, 0), (b, 0)], -1, .75)
        self.assertEqual(a.total[0], .75); self.assertEqual(b.total[0], .75)
        backup([(a, 1)], 1, 1)
        self.assertEqual(a.total[1], -1)
        g = Game(); g.board[0:4] = 1; g.left = 2
        policies, count = search([g], Uniform(), 12)
        self.assertEqual(count, 12)
        self.assertAlmostEqual(float(policies[0].sum()), 1)
        self.assertEqual(set(np.flatnonzero(policies[0])), {4, 5})

    def test_learning_and_symmetry(self):
        torch.set_num_threads(2); torch.manual_seed(42)
        net = Network(channels=8, blocks=1)
        g = Game(); g.play(180)
        p = np.zeros(361, dtype=np.float32); p[0] = 1
        sample = observation(g, p); sample['result'] = 1.0
        before = net.policy.weight.detach().clone()
        loss = train_step(net, torch.optim.AdamW(net.parameters()), [sample], 2,
                          np.random.default_rng(0), torch.device('cpu'))
        self.assertTrue(np.isfinite(loss))
        self.assertFalse(torch.equal(before, net.policy.weight))
        x = np.zeros((1, 8, 19, 19), dtype=np.float32); x[0, 0, 0, 0] = 1
        for seed in range(16):
            xx, pp = augment(x, p[None], np.random.default_rng(seed))
            self.assertEqual(xx[0, 0].argmax(), pp.argmax())

    def test_storage_lock_and_cap(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); path = root / 'state.pt'
            save_checkpoint(path, {'test': torch.tensor([1])}, 10 * 1024**2)
            self.assertEqual(load_checkpoint(path)['test'].item(), 1)
            original = path.read_bytes()
            with self.assertRaises(OSError):
                atomic_bytes(path, b'bad', cap=1)
            self.assertEqual(path.read_bytes(), original)
            with run_lock(root):
                self.assertTrue(busy(root))
            self.assertFalse(busy(root))

    def test_training_session_resume(self):
        with tempfile.TemporaryDirectory() as root:
            # Use a bounded test-only budget so this persistence test reliably records proof targets;
            # the production/default training budget remains 2 ms.
            command = [sys.executable, '-m', 'connect6', 'train', '--data', root,
                       '--device', 'cpu', '--hours', '0.02', '--parallel', '2',
                       '--batch', '2', '--simulations', '2', '--tactical-ms', '50']
            subprocess.run(command + ['--max-games', '10'], check=True, timeout=100,
                           capture_output=True, text=True)
            first = load_checkpoint(Path(root) / 'latest.pt')
            self.assertEqual(first['games'], 10)
            self.assertGreater(first['step'], 0)
            self.assertGreater(first['selfplay_counts'].get('proof_win', 0) + first['selfplay_counts'].get('tss_win', 0), 0)
            for sample in first['replay']:
                if sample.get('source') == 'proof_win':
                    position = Game(sample['board'].numpy(), sample['player'], sample['left'])
                    self.assertTrue(verifies(position, sample['proof']))
                    self.assertEqual(sample['result'], 1)
                elif sample.get('source') == 'tss_win':
                    self.assertEqual(sample['policy'].sum().item(), 1)
                    self.assertEqual(sample['result'], 1)
                    self.assertTrue(verify_tss_sample(sample))
                elif sample.get('source') == 'proof_loss':
                    self.assertEqual(sample['result'], -1)
                    self.assertEqual(sample['policy'].sum().item(), 0)
            # Simulate the pre-integration checkpoint schema without replacing its learned state.
            legacy = {**first, 'settings': dict(first['settings'])}
            legacy['settings'].pop('tactical_ms')
            legacy.pop('selfplay_counts')
            legacy.pop('counted_from_game')
            save_checkpoint(Path(root) / 'latest.pt', legacy, 100 * 1024**2)
            subprocess.run(command + ['--max-games', '12'], check=True, timeout=100,
                           capture_output=True, text=True)
            second = load_checkpoint(Path(root) / 'latest.pt')
            self.assertEqual(second['games'], 12)
            self.assertGreaterEqual(second['step'], first['step'])
            self.assertGreaterEqual(len(second['replay']), len(first['replay']))
            self.assertEqual(second['settings'], first['settings'])
            self.assertEqual(second['counted_from_game'], 10)
            self.assertTrue(second['selfplay_counts'])

    def test_promotion_statistics_and_openings(self):
        self.assertEqual(confidence([1.0] * 199), 0)
        self.assertEqual(confidence([.5] * 200), .5)
        self.assertGreater(confidence([1.0] * 170 + [0.0] * 30), .5)
        self.assertLess(confidence([0.0] * 200), .5)
        self.assertEqual(opening(10, 1).moves, opening(10, 1).moves)
        self.assertNotEqual(opening(10, 1).moves, opening(10, 2).moves)
        self.assertEqual(opening(10, 1).player, -1)


if __name__ == '__main__':
    unittest.main()
