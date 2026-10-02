"""Real recursive TSS plans survive committed turns and process restarts."""
import json
from pathlib import Path
import time
import unittest
from unittest.mock import patch

from connect6.game import Game
from connect6.tss import search_tss, verify_tss
from connect6.training import turn_action

MULTITURN = json.loads((Path(__file__).parent / "fixtures/tss-multiturn.json").read_text())


class TSSIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Game.from_moves(MULTITURN["moves"])
        cls.proof = search_tss(cls.root, max_turns=3,
                               deadline=time.perf_counter() + 5, max_nodes=2_000_000,
                               width=64, max_candidates=5_000)
        if cls.proof is None:
            raise AssertionError("The legal-history three-turn fixture must produce a proof.")

    def test_verified_contingency_survives_partial_turns_and_restart(self):
        game = self.root.copy()
        plan, proof_plan, strategy = [], [None], [None]
        # Inject the actual independently checked three-turn fixture proof into play.
        with patch("connect6.training.search_tss", return_value=self.proof):
            first = turn_action(game, None, 5, 1, lambda: False, tactics=True, tss=True,
                                plan=plan, proof_plan=proof_plan, strategy=strategy)
        game.play(first)
        plan.pop(0)
        self.assertEqual(len(plan), 1)

        # A worker restart after the first attacker placement reconstructs only the remainder.
        strategy = [json.loads(json.dumps(strategy[0]))]
        plan, proof_plan = [], [None]
        second = turn_action(game, None, 5, 1, lambda: False, tactics=True, tss=True,
                             plan=plan, proof_plan=proof_plan, strategy=strategy)
        self.assertEqual(second, self.proof["tree"]["moves"][1])
        game.play(second)
        self.assertNotEqual(game.player, self.proof["attacker"])

        response = self.proof["tree"]["responses"][0]
        self.assertEqual(len(response["moves"]), 2)
        game.play(response["moves"][0])
        self.assertEqual(game.left, 1)  # opponent has only committed half its turn
        strategy = [json.loads(json.dumps(strategy[0]))]
        game.play(response["moves"][1])
        self.assertEqual(game.player, self.proof["attacker"])

        # The restarted worker follows this exact verified response branch, not a root-only pair.
        plan, proof_plan = [], [None]
        move = turn_action(game, None, 5, 1, lambda: False, tactics=True, tss=True,
                           plan=plan, proof_plan=proof_plan, strategy=strategy)
        continuation = {"version": self.proof["version"],
                        "attacker": self.proof["attacker"],
                        "max_turns": self.proof["max_turns"] - 1,
                        "tree": response["proof"]}
        self.assertTrue(verify_tss(game, continuation,
                                   deadline=time.perf_counter() + 5, max_nodes=20_000))
        self.assertEqual(move, continuation["tree"]["moves"][0])
        self.assertEqual(strategy[0], {"history": list(game.moves), "proof": continuation})

        game.play(move)
        if plan:
            plan.pop(0)
        if not game.done and game.player == self.proof["attacker"] and plan:
            # A second restart in the same player's partial turn replays the remaining proof.
            strategy = [json.loads(json.dumps(strategy[0]))]
            plan, proof_plan = [], [None]
            move = turn_action(game, None, 5, 1, lambda: False, tactics=True, tss=True,
                               plan=plan, proof_plan=proof_plan, strategy=strategy)
            game.play(move)
        self.assertTrue(game.done or game.player != self.proof["attacker"])

    def test_stop_before_commit_leaves_saved_gate_plan_and_proof_untouched(self):
        gate = {"moves": list(self.root.moves),
                "plan": list(self.proof["tree"]["moves"]),
                "tss_proof": self.proof,
                "tss_strategy": {"history": list(self.root.moves), "proof": self.proof}}
        game = Game.from_moves(gate["moves"])
        plan = list(gate["plan"])
        proof_plan = [gate["tss_proof"]]
        strategy = [gate["tss_strategy"]]
        turn_action(game, None, 5, 1, lambda: True, tactics=True, tss=True,
                    plan=plan, proof_plan=proof_plan, strategy=strategy)
        # The gate loop exits on stopped() before play or copy-back; cached state remains resumable.
        self.assertEqual(gate["plan"], self.proof["tree"]["moves"])
        self.assertEqual(gate["tss_proof"], self.proof)
        self.assertEqual(gate["tss_strategy"]["history"], self.root.moves)


if __name__ == "__main__":
    unittest.main()
