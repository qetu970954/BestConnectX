"""Verified tactics guide actual placements; value targets wait for the terminal."""
import json
from pathlib import Path
import time
import unittest
from unittest.mock import patch
import numpy as np
import torch
from engine.game import Game, Rules, covers, heuristic
from engine.network import Network
from engine.tactics import verifies
from engine.tss import search_tss
from engine.selfplay import selfplay_batch, selfplay_samples, train_step

CONNECT6 = Rules(19, 19, 6, 2, 1)
FIXTURE = json.loads((Path(__file__).parent / 'fixtures/forks.json').read_text())['selected_fixtures'][0]
MULTITURN = json.loads((Path(__file__).parent / 'fixtures/tss-multiturn.json').read_text())


class RecordingNetwork:
    def __init__(self):
        self.batches = []

    def evaluate(self, games):
        self.batches.append([g.board.copy() for g in games])
        return np.zeros((len(games), 361), dtype=np.float32), np.zeros(len(games), dtype=np.float32)


def lost_position():
    game = Game(rules=CONNECT6); game.left = 2
    for row in (0, 3, 6):
        game.board[row*19:row*19+5] = -1
    return game


class SelfplayTests(unittest.TestCase):
    def test_only_unresolved_positions_reach_network(self):
        win = Game.from_moves(FIXTURE['moves'], rules=CONNECT6)
        loss = lost_position()
        forced = Game(rules=CONNECT6); forced.left = 2; forced.board[:5] = -1
        quiet = Game.from_moves([180], rules=CONNECT6)
        boards = [win, loss, forced, quiet]
        originals = [g.board.copy() for g in boards]
        network = RecordingNetwork()
        decisions = selfplay_batch(boards, network, 2, np.random.default_rng(5), tactical_ms=50)
        self.assertEqual([d['source'] for d in decisions], ['tss_move', 'mcts', 'forced', 'mcts'])
        self.assertEqual(len(network.batches), 3)
        self.assertTrue(all(len(batch) == 2 for batch in network.batches))
        np.testing.assert_array_equal(network.batches[0][0], loss.board)
        np.testing.assert_array_equal(network.batches[0][1], quiet.board)
        self.assertEqual(decisions[2]['policy'][5], 1)
        for g, original in zip(boards, originals):
            np.testing.assert_array_equal(g.board, original)
        self.assertTrue(all('winner' not in decision for decision in decisions))

    def test_both_committed_stones_teach_policy_and_terminal_value(self):
        game = Game.from_moves(FIXTURE['moves'], rules=CONNECT6)
        attacker, plans, network, samples = game.player, [[]], RecordingNetwork(), []
        for _ in range(2):
            decision = selfplay_batch([game], network, 2, np.random.default_rng(0), tactical_ms=50,
                                      tss_enabled=False, plans=plans)[0]
            self.assertEqual(decision['source'], 'proof_move')
            self.assertTrue(verifies(game, decision['proof']))
            sample = selfplay_samples(game, decision)[0]
            self.assertNotIn('result', sample)
            samples.append(sample)
            game.play(int(decision['policy'].argmax()))
            plans[0].pop(0)
        self.assertEqual(network.batches, [])
        self.assertFalse(game.done)
        self.assertEqual(samples[0]['player'], samples[1]['player'])
        self.assertEqual(samples[1]['left'], 1)
        while not game.done:
            game.play(heuristic(game))
        self.assertEqual(game.winner, attacker)
        for sample in samples:
            sample['result'] = float(game.winner * sample['player'])
        net = Network(channels=8, blocks=1, size=19)
        before = net.policy.weight.detach().clone()
        value = train_step(net, torch.optim.AdamW(net.parameters()), samples, 2,
                           np.random.default_rng(2), torch.device('cpu'), rules=CONNECT6)
        self.assertTrue(np.isfinite(value))
        self.assertFalse(torch.equal(before, net.policy.weight))

    def test_verified_tss_result_guides_a_placement_not_a_value_label(self):
        game = Game.from_moves(FIXTURE['moves'], rules=CONNECT6)
        proof = search_tss(game, max_turns=2, deadline=time.perf_counter() + 5,
                           max_nodes=100_000, width=64, max_candidates=5_000)
        self.assertIsNotNone(proof)
        strategies = [None]
        with patch('engine.selfplay.search_tss', return_value=proof):
            decision = selfplay_batch([game], None, 2, np.random.default_rng(0),
                                      tactical_ms=50, strategies=strategies)[0]
        self.assertEqual(decision['source'], 'tss_move')
        self.assertEqual(int(decision['policy'].argmax()), proof['tree']['moves'][0])
        self.assertEqual(strategies[0]['proof'], proof)
        self.assertNotIn('winner', decision)
        self.assertNotIn('result', selfplay_samples(game, decision)[0])

    def test_three_turn_proof_guides_play_without_adjudication(self):
        game = Game.from_moves(MULTITURN['moves'], rules=CONNECT6)
        proof = search_tss(game, max_turns=3, deadline=time.perf_counter() + 5,
                           max_nodes=2_000_000, width=64, max_candidates=5_000)
        self.assertIsNotNone(proof)
        self.assertTrue(proof['tree']['responses'])
        strategies = [None]
        with patch('engine.selfplay.search_tss', return_value=proof):
            decision = selfplay_batch([game], None, 2, np.random.default_rng(1),
                                      tactical_ms=50, strategies=strategies)[0]
        self.assertEqual(decision['source'], 'tss_move')
        self.assertEqual(strategies[0]['proof'], proof)
        samples = selfplay_samples(game, decision)
        self.assertEqual(len(samples), 1)
        self.assertNotIn('result', samples[0])
        game.play(int(decision['policy'].argmax()))
        self.assertFalse(game.done)
        self.assertEqual(game.left, 1)

    def test_forged_verified_flag_cannot_guide_play(self):
        game = Game.from_moves(MULTITURN['moves'], rules=CONNECT6)
        fabricated = {'version': 1, 'attacker': game.player, 'max_turns': 3, 'verified': True,
                      'tree': {'moves': [True], 'responses': []}}
        with patch('engine.selfplay.search_tss', return_value=fabricated):
            with self.assertRaisesRegex(RuntimeError, 'invalid self-play TSS proof'):
                selfplay_batch([game], None, 2, np.random.default_rng(0), tactical_ms=50)

    def test_unavoidable_loss_still_plays_and_own_win_has_priority(self):
        game = lost_position()
        self.assertFalse(covers(game.threats(-game.player), game.left))
        decision = selfplay_batch([game], None, 2, np.random.default_rng(0), bootstrap=True)[0]
        self.assertEqual(decision['source'], 'heuristic')
        self.assertNotIn('winner', decision)
        self.assertEqual(selfplay_samples(game, decision)[0]['policy'].sum().item(), 1)
        game.board[190:195] = 1
        decision = selfplay_batch([game], None, 2, np.random.default_rng(0), tactical_ms=0)[0]
        self.assertEqual(decision['source'], 'proof_move')
        game.play(int(decision['policy'].argmax()))
        self.assertTrue(game.done)
        self.assertEqual(game.winner, 1)

    def test_unknown_falls_back_and_cancellation_discards_batch(self):
        game = Game.from_moves(FIXTURE['moves'], rules=CONNECT6)
        net = RecordingNetwork()
        with (patch('engine.selfplay.forcing_win', return_value=None),
              patch('engine.selfplay.search_tss', return_value=None)):
            decisions = selfplay_batch([game], net, 2, np.random.default_rng(0))
        self.assertEqual(decisions[0]['source'], 'mcts')
        self.assertNotIn('winner', decisions[0])
        self.assertTrue(net.batches)
        net = RecordingNetwork()
        self.assertIsNone(selfplay_batch([game], net, 2, np.random.default_rng(0), deadline=0))
        self.assertIsNone(selfplay_batch([game], net, 2, np.random.default_rng(0), stopped=lambda: True))
        self.assertEqual(net.batches, [])
        cancelled = [False]
        class CancellingNetwork(RecordingNetwork):
            def evaluate(self, games):
                cancelled[0] = True
                return super().evaluate(games)
        self.assertIsNone(selfplay_batch([Game.from_moves([180], rules=CONNECT6)], CancellingNetwork(), 2,
                                        np.random.default_rng(0), stopped=lambda: cancelled[0]))
        with (patch('engine.selfplay.forcing_win', return_value=None),
              patch('engine.selfplay.search_tss', return_value=None)):
            decision = selfplay_batch([game], net, 1, np.random.default_rng(0), tactical_ms=.001)[0]
        self.assertEqual(decision['source'], 'mcts')

    def test_bootstrap_also_uses_proofs_without_neural_calls(self):
        game = Game.from_moves(FIXTURE['moves'], rules=CONNECT6)
        quiet = Game.from_moves([180], rules=CONNECT6)
        decisions = selfplay_batch([game, quiet], None, 1, np.random.default_rng(0),
                                   bootstrap=True, tactical_ms=50)
        self.assertEqual([d['source'] for d in decisions], ['tss_move', 'heuristic'])
        self.assertEqual(len(selfplay_samples(game, decisions[0])), 1)
        self.assertFalse(game.done)


if __name__ == '__main__':
    unittest.main()
