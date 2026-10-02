"""Deterministic fixture discovery against the frozen original heuristic.
Run: uv run python -m experiments.find_forks > /tmp/forks.json
These selected examples are regression fixtures, NOT an unbiased match benchmark.
"""
import json
import time
import numpy as np
from connect6.game import Game, heuristic
from connect6.tactics import forcing_win, verifies

rng = np.random.default_rng(65070)
found = []
checked = 0
for game_id in range(30):
    game = Game(); game.play(180)
    pool = [r * 19 + c for r in range(5, 14) for c in range(5, 14) if r * 19 + c != 180]
    for action in rng.choice(pool, 4, replace=False):
        game.play(int(action))
    while not game.done and len(game.moves) < 160:
        if game.left == 2 and not game.threats(game.player, 2):
            old = game.copy(); baseline = []
            while not old.done and old.player == game.player:
                action = heuristic(old); baseline.append(action); old.play(action)
            if not verifies(game, baseline):
                checked += 1
                proof = forcing_win(game, deadline=time.monotonic() + 5, max_candidates=256)
                if proof:
                    found.append({'moves': game.moves.copy(), 'certificate': list(proof),
                                  'baseline': baseline, 'game_id': game_id})
                    if len(found) == 3:
                        break
        # A fixed mix of original heuristic play and nearby random exploration.
        empty = [i for i in pool if not game.board[i]]
        action = int(rng.choice(empty)) if empty and rng.random() < .12 else heuristic(game)
        game.play(action)
    if len(found) == 3:
        break
print(json.dumps({'seed': 65070, 'checked': checked, 'selected_fixtures': found}, indent=2))
