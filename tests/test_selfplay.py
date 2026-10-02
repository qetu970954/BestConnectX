"""Tactical teaching must skip neural work only on exact, verified evidence."""
import json
from pathlib import Path
import time
import unittest
from unittest.mock import patch
import numpy as np
import torch
from connect6.game import Game, covers
from connect6.network import Network
from connect6.tactics import verifies
from connect6.tss import search_tss
from connect6.training import (selfplay_batch, selfplay_samples, train_step,
                               verify_tss_sample)

FIXTURE = json.loads((Path(__file__).parent / 'fixtures/forks.json').read_text())['selected_fixtures'][0]
MULTITURN = json.loads((Path(__file__).parent / 'fixtures/tss-multiturn.json').read_text())


class RecordingNetwork:
    def __init__(self):
        self.batches = []

    def evaluate(self, games):
        self.batches.append([g.board.copy() for g in games])
        return np.zeros((len(games), 361), dtype=np.float32), np.zeros(len(games), dtype=np.float32)


def lost_position():
    game = Game(); game.left = 2
    for row in (0, 3, 6):
        game.board[row*19:row*19+5] = -1
    return game


class SelfplayTests(unittest.TestCase):
    def test_only_unresolved_positions_reach_network(self):
        win = Game.from_moves(FIXTURE['moves'])
        loss = lost_position()
        forced = Game(); forced.left = 2; forced.board[:5] = -1
        quiet = Game.from_moves([180])
        boards = [win, loss, forced, quiet]
        originals = [g.board.copy() for g in boards]
        network = RecordingNetwork()
        decisions = selfplay_batch(boards, network, 2, np.random.default_rng(5), tactical_ms=50)
        self.assertEqual([d['source'] for d in decisions], ['tss_win', 'proof_loss', 'forced', 'mcts'])
        self.assertEqual(len(network.batches), 3)  # one root + two simulations, quiet game only
        self.assertTrue(all(len(batch) == 1 for batch in network.batches))
        np.testing.assert_array_equal(network.batches[0][0], quiet.board)
        self.assertEqual(decisions[2]['policy'][5], 1)
        for g, original in zip(boards, originals):
            np.testing.assert_array_equal(g.board, original)
        self.assertEqual(decisions[0]['winner'], win.player)
        self.assertEqual(decisions[1]['winner'], -loss.player)

    def test_both_stones_teach_policy_and_value(self):
        game = Game.from_moves(FIXTURE['moves'])
        network = RecordingNetwork()
        decision = selfplay_batch([game], network, 64, np.random.default_rng(0), tactical_ms=50)[0]
        self.assertEqual(network.batches, [])
        samples = selfplay_samples(game, decision)
        self.assertEqual(len(samples), 2)
        for sample in samples:
            state = Game(sample['board'].numpy(), sample['player'], sample['left'])
            self.assertTrue(verifies(state, sample['proof']))
            self.assertEqual(sample['policy'].sum().item(), 1)
            self.assertEqual(sample['policy'].argmax().item(), sample['proof'][0])
            sample['result'] = float(decision['winner'] * sample['player'])
            self.assertEqual(sample['result'], 1)
        first = decision['proof'][0]
        self.assertEqual(samples[0]['board'][first].item(), 0)
        self.assertEqual(samples[1]['board'][first].item(), game.player)
        self.assertEqual(samples[1]['left'], 1)
        net = Network(channels=8, blocks=1)
        before = net.policy.weight.detach().clone()
        value = train_step(net, torch.optim.AdamW(net.parameters()), samples, 2,
                           np.random.default_rng(2), torch.device('cpu'))
        self.assertTrue(np.isfinite(value))
        self.assertFalse(torch.equal(before, net.policy.weight))

    def test_verified_tss_result_teaches_root_turn(self):
        game = Game.from_moves(FIXTURE['moves'])
        proof = search_tss(game, max_turns=2, deadline=time.perf_counter() + 5,
                           max_nodes=100_000, width=64, max_candidates=5_000)
        self.assertIsNotNone(proof)
        with (patch('connect6.training.search_tss', return_value=proof),
              patch('connect6.training.forcing_win', return_value=None)):
            decision = selfplay_batch([game], None, 64, np.random.default_rng(0),
                                      tactical_ms=50)[0]
        self.assertEqual(decision['source'], 'tss_win')
        self.assertNotIn('verified', decision)
        samples = selfplay_samples(game, decision)
        self.assertEqual(len(samples), len(proof['tree']['moves']))
        for sample, move in zip(samples, proof['tree']['moves']):
            self.assertEqual(sample['source'], 'tss_win')
            self.assertEqual(sample['policy'].argmax().item(), move)
            self.assertTrue(verify_tss_sample(sample))

    def test_three_turn_proof_is_audited_and_teaches_both_current_turn_placements(self):
        game = Game.from_moves(MULTITURN['moves'])
        proof = search_tss(game, max_turns=3, deadline=time.perf_counter() + 5,
                           max_nodes=2_000_000, width=64, max_candidates=5_000)
        self.assertIsNotNone(proof)
        self.assertTrue(proof['tree']['responses'])
        with (patch('connect6.training.search_tss', return_value=proof),
              patch('connect6.training.forcing_win', return_value=None)):
            decision = selfplay_batch([game], None, 64, np.random.default_rng(1),
                                      tactical_ms=50)[0]
        self.assertEqual(decision['source'], 'tss_win')
        samples = decision['samples']
        self.assertEqual(len(samples), len(proof['tree']['moves']))
        self.assertEqual(len(samples), 2)
        self.assertTrue(all(verify_tss_sample(sample) for sample in samples))
        self.assertEqual(samples[0]['player'], samples[1]['player'])
        self.assertEqual(samples[0]['left'], samples[1]['left'] + 1)
        self.assertEqual(samples[0]['tss_certificate'], proof)
        self.assertEqual(samples[1]['tss_certificate'], proof)

    def test_forged_verified_flag_cannot_create_tss_labels(self):
        game = Game.from_moves(MULTITURN['moves'])
        fabricated = {'source': 'tss_win', 'winner': game.player, 'verified': True,
                      'proof': [180, 198]}
        with self.assertRaises(ValueError):
            selfplay_samples(game, fabricated)

    def test_loss_is_value_only_and_own_win_has_priority(self):
        loss = lost_position()
        decision = selfplay_batch([loss], None, 64, np.random.default_rng(0))[0]
        sample = selfplay_samples(loss, decision)[0]
        self.assertEqual(sample['policy'].sum().item(), 0)
        sample['result'] = float(decision['winner'] * sample['player'])
        self.assertEqual(sample['result'], -1)
        self.assertFalse(covers(loss.threats(-loss.player), loss.left))
        net = Network(channels=8, blocks=1)
        value = train_step(net, torch.optim.AdamW(net.parameters(), weight_decay=0), [sample], 2,
                           np.random.default_rng(2), torch.device('cpu'))
        self.assertTrue(np.isfinite(value))
        self.assertEqual(net.policy.weight.grad.abs().sum().item(), 0)
        self.assertGreater(net.value[-2].weight.grad.abs().sum().item(), 0)
        loss.board[190:195] = 1
        decision = selfplay_batch([loss], None, 64, np.random.default_rng(0), tactical_ms=0)[0]
        self.assertEqual(decision['source'], 'proof_win')
        self.assertEqual(decision['winner'], loss.player)

    def test_unknown_falls_back_and_cancellation_discards_batch(self):
        game = Game.from_moves(FIXTURE['moves'])
        net = RecordingNetwork()
        with (patch('connect6.training.forcing_win', return_value=None),
              patch('connect6.training.search_tss', return_value=None)):
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

        self.assertIsNone(selfplay_batch([Game.from_moves([180])], CancellingNetwork(), 64,
                                        np.random.default_rng(0), stopped=lambda: cancelled[0]))
        # Expiry of the discovery budget is not cancellation of the whole training session.
        with (patch('connect6.training.forcing_win', return_value=None),
              patch('connect6.training.search_tss', return_value=None)):
            decision = selfplay_batch([game], net, 1, np.random.default_rng(0), tactical_ms=.001)[0]
        self.assertEqual(decision['source'], 'mcts')

    def test_bootstrap_also_uses_proofs_without_neural_calls(self):
        game = Game.from_moves(FIXTURE['moves'])
        quiet = Game.from_moves([180])
        decisions = selfplay_batch([game, quiet], None, 1, np.random.default_rng(0),
                                   bootstrap=True, tactical_ms=50)
        self.assertEqual([d['source'] for d in decisions], ['tss_win', 'heuristic'])
        self.assertEqual(len(selfplay_samples(game, decisions[0])), 2)


if __name__ == '__main__':
    unittest.main()
