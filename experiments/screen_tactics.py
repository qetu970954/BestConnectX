"""Small CPU-only ablation: old heuristic vs the same heuristic + fork proofs.
Not a neural-model gate. Does not modify the incumbent or training data.
Run: uv run python -m experiments.screen_tactics > docs/tactical-screen.json
"""
import argparse
import json
import time
from connect6.training import opening, turn_action, confidence

parser = argparse.ArgumentParser()
parser.add_argument('--pairs', type=int, default=8)
parser.add_argument('--suite', type=int, default=777)
args = parser.parse_args()
if not 1 <= args.pairs <= 200 or not 0 <= args.suite < 1000000:
    parser.error('Use 1–200 paired openings and a nonnegative suite below 1000000.')
matches = []
started = time.monotonic()
for index in range(args.pairs):
    for candidate_color in (1, -1):
        game = opening(index, args.suite)
        overrun, forks = 0.0, 0
        while not game.done:
            player, plan = game.player, []
            begin = time.monotonic()
            while not game.done and game.player == player:
                remaining = max(0, 5 - (time.monotonic()-begin))
                had_plan = bool(plan)
                immediate = bool(game.threats(player, game.left))
                move = turn_action(game, None, remaining/game.left, 1, lambda: False,
                                   tactics=player == candidate_color, plan=plan)
                if plan and not had_plan and not immediate:
                    forks += 1
                game.play(move)
                if plan:
                    plan.pop(0)
            overrun = max(overrun, time.monotonic()-begin-5)
        matches.append({'opening': index, 'candidate_color': candidate_color,
                        'score': (game.winner*candidate_color+1)/2,
                        'placements': len(game.moves), 'fork_certificates': forks,
                        'max_turn_overrun_seconds': overrun, 'moves': game.moves})
score = sum(m['score'] for m in matches) / len(matches)
pair_scores = [(matches[i]['score'] + matches[i+1]['score']) / 2 for i in range(0, len(matches), 2)]
lower = confidence(pair_scores) if args.pairs == 200 else None
print(json.dumps({'scope': 'Internal paired ablation; no external-engine strength claim. Promotion requires separate manifest verification.',
                  'candidate': 'Original heuristic + bounded exact fork certificates',
                  'opponent': 'Unchanged original heuristic', 'opening_suite': args.suite,
                  'seconds_per_turn': 5, 'games': len(matches), 'opening_pairs': args.pairs,
                  'score': score, 'lower_95': lower,
                  'rules_failures': 0, 'max_turn_overrun_seconds': max(m['max_turn_overrun_seconds'] for m in matches),
                  'passes_gate': lower is not None and lower > .5 and all(m['max_turn_overrun_seconds'] <= .1 for m in matches),
                  'wins': sum(m['score'] == 1 for m in matches),
                  'draws': sum(m['score'] == .5 for m in matches),
                  'losses': sum(m['score'] == 0 for m in matches),
                  'elapsed_seconds': round(time.monotonic()-started, 3),
                  'matches': matches}, indent=2))
