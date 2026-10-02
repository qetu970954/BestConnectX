"""Selected tactical-regression benchmark. No match-strength inference.
Run: uv run python -m experiments.benchmark_tactics
"""
import json
import math
from pathlib import Path
import statistics
import time
from connect6.game import Game, heuristic
from connect6.tactics import forcing_win, verifies

fixtures = json.loads((Path(__file__).resolve().parent.parent / 'tests/fixtures/forks.json').read_text())
results = []
for case in fixtures['selected_fixtures']:
    game = Game.from_moves(case['moves'])
    before = game.copy(); baseline = []
    while not before.done and before.player == game.player:
        move = heuristic(before); baseline.append(move); before.play(move)
    times, certificates = [], []
    for _ in range(20):
        started = time.perf_counter()
        proof = forcing_win(game, deadline=time.monotonic()+.05)
        times.append((time.perf_counter()-started)*1000)
        certificates.append(proof)
    results.append({'placements_before': len(game.moves), 'baseline': baseline,
                    'baseline_has_one_turn_certificate': verifies(game, baseline),
                    'certificates_found': sum(p is not None for p in certificates), 'attempts': 20,
                    'certificate': certificates[0],
                    'all_returned_certificates_valid': all(verifies(game, p) for p in certificates if p),
                    'defender_pairs_checked_by_tests': math.comb(361-len(game.moves)-game.left, 2),
                    'median_ms': round(statistics.median(times), 3), 'max_ms': round(max(times), 3)})
print(json.dumps({'scope': 'Selected, correlated regression positions from one generated game; not a tournament.',
                  'discovery_seed': fixtures['seed'], 'discovery_positions_checked': fixtures['checked'],
                  'candidate_cap': 256, 'width': 24, 'deadline_ms': 50, 'results': results}, indent=2))
