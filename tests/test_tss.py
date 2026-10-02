"""Public-contract tests for bounded threat-space search."""
import json
from itertools import combinations
from pathlib import Path
import time
import unittest
from unittest.mock import patch

from connect6.game import Game
from connect6.tss import _Budget, _verified_defenses, search_tss, verify_tss
from connect6.training import turn_action

FIXTURE = json.loads((Path(__file__).parent / "fixtures/forks.json").read_text())["selected_fixtures"][0]
MULTITURN = json.loads((Path(__file__).parent / "fixtures/tss-multiturn.json").read_text())


class TSSPublicContractTests(unittest.TestCase):
    def test_search_returns_independently_verifiable_forced_win(self):
        game = Game.from_moves(FIXTURE["moves"])
        proof = search_tss(game, max_turns=2, deadline=time.perf_counter() + 10,
                           max_nodes=100_000, width=64, max_candidates=5_000)
        self.assertIsNotNone(proof)
        self.assertEqual(proof["attacker"], game.player)
        self.assertTrue(verify_tss(game, proof, deadline=time.perf_counter() + 10,
                                   max_nodes=100_000))

    def test_candidate_play_keeps_a_verified_turn_plan_across_stones(self):
        game = Game.from_moves(FIXTURE["moves"])
        proof = search_tss(game, max_turns=2, deadline=time.perf_counter() + 5,
                           max_nodes=100_000, width=64, max_candidates=5_000)
        self.assertIsNotNone(proof)
        plan, proof_plan = [], [None]
        with patch("connect6.training.search_tss", return_value=proof):
            first = turn_action(game, None, 5, 1, lambda: False, tactics=True, tss=True,
                                plan=plan, proof_plan=proof_plan)
        self.assertEqual(plan, proof["tree"]["moves"])
        self.assertEqual(proof_plan[0], proof)
        game.play(first)
        plan.pop(0)
        second = turn_action(game, None, 5, 1, lambda: False, tactics=True, tss=True,
                             plan=plan, proof_plan=proof_plan)
        self.assertEqual(second, plan[0])
        game.play(second)
        self.assertNotEqual(game.player, 1)

    def test_verifier_requires_every_filler_reply_to_a_single_blocker(self):
        game = Game()
        game.board[:2] = 1
        game.left = 2
        after_attack = game.copy()
        after_attack.play(2)
        after_attack.play(3)
        self.assertEqual(after_attack.threats(1), {frozenset((4, 5))})
        proof = {"version": 1, "attacker": 1, "max_turns": 2,
                 "tree": {"moves": [2, 3], "responses": []}}
        self.assertFalse(verify_tss(game, proof, deadline=time.perf_counter() + 5,
                                    max_nodes=10_000))

    def test_verifier_recursively_checks_all_defensive_continuations(self):
        fixture_game = Game.from_moves(FIXTURE["moves"])
        root = Game(fixture_game.board.copy(), fixture_game.player, 2)
        root.board[[0, 1, 2, 19, 20, 21]] = root.player
        after_attack = root.copy()
        after_attack.play(3)
        after_attack.play(22)
        replies = []
        for reply in ((4, 23), (4, 24), (5, 23), (5, 24)):
            position = after_attack.copy()
            for move in reply:
                position.play(move)
            continuation = search_tss(position, max_turns=2,
                                      deadline=time.perf_counter() + 5, max_nodes=100_000,
                                      width=64, max_candidates=5_000)
            self.assertIsNotNone(continuation)
            replies.append({"moves": list(reply), "proof": continuation["tree"]})
        proof = {"version": 1, "attacker": root.player, "max_turns": 3,
                 "tree": {"moves": [3, 22], "responses": replies}}
        self.assertTrue(verify_tss(root, proof, deadline=time.perf_counter() + 10,
                                   max_nodes=100_000))
        proof["tree"]["responses"].pop()
        self.assertFalse(verify_tss(root, proof, deadline=time.perf_counter() + 10,
                                    max_nodes=100_000))

    def test_verifier_rejects_opponent_counterwin(self):
        game = Game()
        game.board[:2] = 1
        game.left = 2
        game.board[19:24] = -1
        proof = {"version": 1, "attacker": 1, "max_turns": 2,
                 "tree": {"moves": [2, 3], "responses": []}}
        self.assertFalse(verify_tss(game, proof, deadline=time.perf_counter() + 5,
                                    max_nodes=10_000))

    def test_limits_are_unknown_not_loss(self):
        game = Game.from_moves(FIXTURE["moves"])
        self.assertIsNone(search_tss(game, max_turns=3, deadline=time.perf_counter() - 1,
                                     max_nodes=100_000))
        self.assertIsNone(search_tss(game, max_turns=2, deadline=time.perf_counter() + 5,
                                     max_nodes=1))
        self.assertIsNone(verify_tss(game, {}, deadline=time.perf_counter() - 1,
                                     max_nodes=100_000))

    def test_legal_history_multiturn_discovery_needs_three_turns(self):
        game = Game.from_moves(MULTITURN["moves"])
        self.assertFalse(game.done)
        self.assertIsNone(search_tss(game, max_turns=2, deadline=time.perf_counter() + 5,
                                     max_nodes=2_000_000, width=64, max_candidates=5_000))
        proof = search_tss(game, max_turns=MULTITURN["max_turns"],
                           deadline=time.perf_counter() + 5, max_nodes=2_000_000,
                           width=64, max_candidates=5_000)
        self.assertIsNotNone(proof)
        self.assertTrue(proof["tree"]["responses"])
        self.assertTrue(verify_tss(game, proof, deadline=time.perf_counter() + 5,
                                   max_nodes=10_000))

    def test_verifier_reply_coverage_matches_exhaustive_oracle(self):
        game = Game()
        empty = list(range(9))
        game.board.fill(1)
        game.board[empty] = 0
        game.left = 2
        cases = (
            {frozenset((0, 1))},
            {frozenset((0,)), frozenset((1, 2))},
            {frozenset((0, 1)), frozenset((2, 3))},
            {frozenset((0, 1)), frozenset((2, 3)), frozenset((4, 5))},
        )
        for threats in cases:
            with self.subTest(threats=threats):
                expected = [reply for reply in combinations(empty, 2)
                            if all(any(cell in edge for cell in reply) for edge in threats)]
                actual = _verified_defenses(
                    game, threats, _Budget(time.perf_counter() + 5, 100_000, lambda: False))
                self.assertEqual(actual, expected)

    def test_verifier_rejects_attack_moves_crossing_turn_boundary(self):
        game = Game()
        game.left = 1
        game.board[[0, 1, 2, 3, 19, 20, 21, 22]] = 1
        proof = {"version": 1, "attacker": 1, "max_turns": 2,
                 "tree": {"moves": [4, 100], "responses": []}}
        self.assertFalse(verify_tss(game, proof, deadline=time.perf_counter() + 5,
                                    max_nodes=100_000))

    def test_terminal_and_draw_positions_do_not_produce_wins(self):
        terminal = Game()
        terminal.done, terminal.winner = True, 1
        self.assertIsNone(search_tss(terminal, max_turns=2, deadline=time.perf_counter() + 1))
        proof = {"version": 1, "attacker": 1, "max_turns": 2,
                 "tree": {"moves": [0], "responses": []}}
        self.assertFalse(verify_tss(terminal, proof, deadline=time.perf_counter() + 1))

        sequence = (1, -1, 1, -1, 1, -1, 1)
        draw = Game()
        draw.board[:] = [sequence[(row + 2 * col) % len(sequence)]
                         for row in range(19) for col in range(19)]
        draw.board[[0, 1]] = 0
        draw.left = 2
        after = draw.copy()
        after.play(0)
        after.play(1)
        self.assertTrue(after.done)
        self.assertEqual(after.winner, 0)
        self.assertIsNone(search_tss(draw, max_turns=2, deadline=time.perf_counter() + 5,
                                     max_nodes=100_000, width=64, max_candidates=5_000))

    def test_cancellation_is_unknown_and_malformed_certificates_are_rejected(self):
        game = Game()
        game.left = 1
        game.board[:5] = 1
        proof = {"version": 1, "attacker": 1, "max_turns": 1,
                 "tree": {"moves": [5], "responses": []}}
        stopped = lambda: True
        self.assertIsNone(search_tss(game, max_turns=1, deadline=time.perf_counter() + 5,
                                     stopped=stopped))
        self.assertIsNone(verify_tss(game, proof, deadline=time.perf_counter() + 5,
                                     stopped=stopped))
        malformed = (
            {**proof, "max_turns": True},
            {**proof, "tree": {"moves": [True], "responses": []}},
            {**proof, "tree": {"moves": [5], "responses": [None]}},
        )
        for candidate in malformed:
            with self.subTest(candidate=candidate):
                self.assertFalse(verify_tss(game, candidate, deadline=time.perf_counter() + 5))


if __name__ == "__main__":
    unittest.main()
