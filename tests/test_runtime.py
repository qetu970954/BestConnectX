"""Production-interface checks, in the approved 3x3 -> Gomoku -> Connect6 order."""
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
import torch
from engine.cli import main
from engine.game import Game as Reference, Rules
from engine.network import Network
from engine.runtime import Game, model_for, search, selfplay_batch, search_tss, verify_tss, turn_action, full_turn
from engine.search import search as reference_search
from engine.training import load_state
from engine.web import history


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_01_tictactoe_all_legal_states_and_complete_game(self):
        rules = Rules(3, 3, 3)
        seen = set()
        def visit(reference):
            key = reference.board.tobytes()
            if key in seen:
                return
            seen.add(key)
            actual = Game.from_moves(reference.moves, rules=rules)
            self.assertEqual((actual.player, actual.left, actual.winner, actual.done),
                             (reference.player, reference.left, reference.winner, reference.done))
            np.testing.assert_array_equal(actual.board, reference.board)
            np.testing.assert_array_equal(actual.actions(), reference.actions())
            np.testing.assert_array_equal(actual.features(), reference.features())
            if not reference.done:
                for action in np.flatnonzero(reference.board == 0):
                    child = reference.copy(); child.play(int(action)); visit(child)
        visit(Reference(rules=rules))
        self.assertEqual(len(seen), 5478)
        game = Game(rules=rules); net = Network(8, 1, 3); rng = np.random.default_rng(1)
        while not game.done:
            decision = selfplay_batch([game], net, 8, rng, workers=1)[0]
            self.assertNotIn('result', decision)
            game.play(decision['action'])
        self.assertTrue(Reference.from_moves(game.moves, rules=rules).done)

    def test_02_gomoku_inference_puct_and_native_only_decisions(self):
        rules = Rules(15, 15, 5)
        references = [Reference.from_moves([112, 113, 97, 98], rules=rules), Reference(rules=rules)]
        games = [Game.from_moves(g.moves, rules=rules) for g in references]
        net = Network(8, 1, 15); net.eval(); model = model_for(net)
        x = np.stack([g.features() for g in references])
        logits, values = model.evaluate_features(x)
        with torch.inference_mode():
            expected, value = net(torch.from_numpy(x))
        np.testing.assert_allclose(logits, expected.numpy(), atol=1e-6, rtol=1e-5)
        np.testing.assert_allclose(values, value.numpy(), atol=1e-6, rtol=1e-5)
        with patch('engine.native.library', return_value=None):
            expected, expected_count = reference_search(references, net, 32)
        with (patch.object(Reference, 'play', side_effect=AssertionError('Python placement')),
              patch.object(Reference, 'actions', side_effect=AssertionError('Python action filtering')),
              patch.object(Network, 'evaluate', side_effect=AssertionError('Python inference'))):
            actual, count = search(games, net, 32, workers=2)
            decisions = selfplay_batch(games, net, 8, np.random.default_rng(42), tactical_ms=0, workers=2)
        self.assertEqual(count, expected_count)
        np.testing.assert_array_equal(actual, expected)
        for game, decision in zip(games, decisions):
            self.assertEqual(float(decision['policy'].sum()), 1.)
            game.play(decision['action'])
        for action in range(4):
            games[0].board[action] = 1
        np.testing.assert_array_equal(games[0].features(), Reference(games[0].board, games[0].player, games[0].left, rules=rules).features())
        for invalid in (True, -1, 225, 1.5, '1'):
            with self.assertRaises(ValueError):
                games[0].play(invalid)

    def test_03_connect6_tactics_proofs_and_two_stone_backups(self):
        rules = Rules(19, 19, 6, 2, 1)
        game = Game(rules=rules); game.board[:4] = 1; game.left = 2
        game.play(4); self.assertEqual(game.player, 1); game.play(5)
        self.assertTrue(game.done); self.assertEqual(game.winner, 1)
        fixture = json.loads((Path(__file__).parent/'fixtures/tss-multiturn.json').read_text())
        game = Game.from_moves(fixture['moves'], rules=rules)
        proof = search_tss(game, max_turns=3, deadline=time.perf_counter()+5,
                           max_nodes=2_000_000, width=64, max_candidates=5000)
        self.assertIsNotNone(proof)
        self.assertTrue(verify_tss(game, proof, deadline=time.perf_counter()+5))
        from engine.tss import verify_tss as reference_verify
        reference = Reference.from_moves(game.moves, rules=rules)
        self.assertTrue(reference_verify(reference, proof, deadline=time.perf_counter()+5))
        broken = json.loads(json.dumps(proof)); broken['tree']['responses'].pop()
        self.assertFalse(verify_tss(game, broken, deadline=time.perf_counter()+5))
        self.assertIsNone(verify_tss(game, proof, deadline=0))
        # A native proof follows partial attacker turns and a saved defensive branch.
        strategy = [{'history': game.moves.copy(), 'proof': proof}]
        plan, checked = [], [None]
        first = turn_action(game, None, 5, 1, lambda: False, tactics=True, tss=True,
                            plan=plan, proof_plan=checked, strategy=strategy)
        self.assertEqual(first, proof['tree']['moves'][0]); game.play(first)
        strategy = [json.loads(json.dumps(strategy[0]))]; plan.clear()
        second = turn_action(game, None, 5, 1, lambda: False, tactics=True, tss=True,
                             plan=plan, proof_plan=checked, strategy=strategy)
        self.assertEqual(second, proof['tree']['moves'][1]); game.play(second)
        reply = proof['tree']['responses'][0]
        for action in reply['moves']:
            game.play(action)
        strategy = [json.loads(json.dumps(strategy[0]))]; plan.clear()
        resumed = turn_action(game, None, 5, 1, lambda: False, tactics=True, tss=True,
                              plan=plan, proof_plan=checked, strategy=strategy)
        self.assertEqual(resumed, reply['proof']['moves'][0])
        moves, _, _ = full_turn(Game.from_moves([180], rules=rules), None, .02)
        self.assertEqual(len(moves), 2)
        net = Network(8, 1, 19)
        for moves in ([180], [180, 0]):
            reference = Reference.from_moves(moves, rules=rules)
            with patch('engine.native.library', return_value=None):
                expected, _ = reference_search([reference], net, 32)
            actual, _ = search([Game.from_moves(moves, rules=rules)], net, 32, workers=1)
            np.testing.assert_array_equal(actual, expected)

    def test_04_supported_rule_limits_match_reference(self):
        from engine import Game as PublicGame
        self.assertIs(PublicGame, Game)
        self.assertEqual(Network().config, {'channels': 64, 'blocks': 6, 'size': 15})
        game = Game()
        with self.assertRaises(ValueError):
            game.threats(1, 3)
        with self.assertRaises(RuntimeError):
            game.query(3, 3)
        game.board[0] = 2
        with self.assertRaises(ValueError):
            game.features()
        game.board = np.zeros(1, dtype=np.int8)
        with self.assertRaises(ValueError):
            game.features()
        rules = Rules(25, 25, 2, 2, 2)
        self.assertEqual(Game(rules=rules).threats(1, 2), Reference(rules=rules).threats(1, 2))
        rng = np.random.default_rng(3)
        for rules in (Rules(2, 2, 2, 1, 2), Rules(5, 5, 3, 2, 2), Rules(25, 25, 25, 2, 1)):
            for _ in range(20):
                board = rng.choice(np.array([-1, 0, 1], dtype=np.int8), rules.height**2)
                if np.all(board != 0):
                    board[0] = 0
                actual = Game(board.copy(), left=2, rules=rules)
                expected = Reference(board.copy(), left=2, rules=rules)
                np.testing.assert_array_equal(actual.features(), expected.features())
                np.testing.assert_array_equal(actual.actions(), expected.actions())
                for player in (-1, 1):
                    for budget in (0, 1, 2):
                        self.assertEqual(actual.threats(player, budget), expected.threats(player, budget))

    def test_05_model_sizes_and_updated_native_weights_match(self):
        from engine.selfplay import observation, train_step
        for channels, blocks in ((32, 2), (64, 6), (128, 10)):
            net = Network(channels, blocks, 15).eval()
            game = Game(rules=Rules(15, 15, 5))
            x = np.stack([game.features()])
            actual, values = model_for(net).evaluate_features(x)
            with torch.inference_mode():
                expected, targets = net(torch.from_numpy(x))
            np.testing.assert_allclose(actual, expected.numpy(), atol=1e-6, rtol=1e-5)
            np.testing.assert_allclose(values, targets.numpy(), atol=1e-6, rtol=1e-5)
        net = Network(8, 1, 3); game = Game(rules=Rules(3, 3, 3))
        policy = np.zeros(9, dtype=np.float32); policy[0] = 1
        sample = observation(game, policy); sample['result'] = 1.
        x = np.stack([game.features()]); before = model_for(net).evaluate_features(x)[0]
        train_step(net, torch.optim.AdamW(net.parameters()), [sample], 2,
                   np.random.default_rng(1), torch.device('cpu'), rules=game.rules)
        after, value = model_for(net).evaluate_features(x); net.eval()
        with torch.inference_mode():
            expected, targets = net(torch.from_numpy(x))
        self.assertFalse(np.array_equal(before, after))
        np.testing.assert_allclose(after, expected.numpy(), atol=1e-6, rtol=1e-5)
        np.testing.assert_allclose(value, targets.numpy(), atol=1e-6, rtol=1e-5)

    def test_06_checkpoint_learning_resume_configs_and_no_archives(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            main(['train', '--preset', 'tictactoe', '--data', name, '--device', 'cpu',
                  '--max-games', '8', '--tactical-ms', '0', '--hours', '.01'])
            first = load_state(root/'latest.pt')
            self.assertEqual(first['games'], 8); self.assertEqual(first['step'], 32)
            self.assertTrue(first['replay'])
            self.assertTrue(all(row['result'] in (-1., 0., 1.) for row in first['replay']))
            self.assertTrue(all('moves' not in row and 'board' not in row for row in first['summaries']))
            main(['train', '--data', name, '--device', 'cpu', '--max-games', '9', '--hours', '.01', '--workers', '2'])
            second = load_state(root/'latest.pt')
            self.assertEqual(second['games'], 9); self.assertEqual(second['config'], first['config'])
            self.assertEqual(second['settings']['workers'], 2)
            self.assertFalse((root/'selfplay').exists()); self.assertFalse((root/'replay').exists())
            self.assertEqual(history(root)['selfplay']['games'], 9)
            original = (root/'latest.pt').read_bytes()
            from contextlib import redirect_stdout
            from io import StringIO
            output = StringIO()
            with redirect_stdout(output):
                main(['selfplay', '--preset', 'tictactoe', '--data', name, '--device', 'cpu',
                      '--games', '3', '--parallel', '2', '--tactical-ms', '0'])
            summary = json.loads(output.getvalue())
            self.assertEqual(summary['games'], 3)
            self.assertEqual(summary['parallel'], 2)
            self.assertEqual(summary['black_wins'] + summary['white_wins'] + summary['draws'], 3)
            self.assertFalse(summary['saved_game_records'])
            with self.assertRaises(SystemExit):
                main(['selfplay', '--preset', 'tictactoe', '--data', name, '--device', 'cpu', '--tactical-ms', 'nan'])
            self.assertEqual((root/'latest.pt').read_bytes(), original)
            with self.assertRaises(SystemExit):
                main(['train', '--data', name, '--device', 'cpu', '--channels', '16'])
            self.assertEqual((root/'latest.pt').read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
