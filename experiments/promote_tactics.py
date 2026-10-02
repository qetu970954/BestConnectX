"""Verify this tactical gate and atomically promote only the original baseline.
Run: uv run python -m experiments.promote_tactics
Refuses to replace an already upgraded incumbent or interrupt a pending gate.
"""
import hashlib
import json
from pathlib import Path
from connect6.game import Game
from connect6.training import confidence, opening
from connect6.storage import run_lock, load_json, load_checkpoint, atomic_bytes, save_json, GIB

def require(condition):
    if not condition:
        raise ValueError('Gate validation failed; incumbent was not changed.')


root = Path('data')
report = load_json('docs/tactical-gate.json')
require(report['opening_suite'] == 778 and report['seconds_per_turn'] == 5)
require(report['games'] == 400 and len(report['matches']) == 400)
require(report['rules_failures'] == 0 and report['max_turn_overrun_seconds'] <= .1)
scores = []
for i, match in enumerate(report['matches']):
    require(match['opening'] == i // 2)
    require(match['candidate_color'] == (1 if i % 2 == 0 else -1))
    start = opening(i // 2, 778).moves
    require(match['moves'][:len(start)] == start)
    game = Game.from_moves(match['moves'])  # independent replay of every completed game
    require(game.done)
    score = (game.winner * match['candidate_color'] + 1) / 2
    require(score == match['score'])
    require(match['max_turn_overrun_seconds'] <= .1)
    scores.append(score)
lower = confidence([(scores[i]+scores[i+1])/2 for i in range(0, 400, 2)])
require(lower > .5 and lower == report['lower_95'])
require(sum(scores)/400 == report['score'])
with run_lock(root):
    previous = load_json(root / 'incumbent.json')
    if previous['kind'] != 'heuristic' or previous.get('tactics', False):
        raise RuntimeError('Incumbent is no longer the tested original heuristic; refusing promotion.')
    if (root / 'latest.pt').exists() and load_checkpoint(root / 'latest.pt')['gate'] is not None:
        raise RuntimeError('A neural gate is pending; its opponent must not change mid-evaluation.')
    report['promoted'] = True
    report['engine_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in map(Path, ('connect6/game.py', 'connect6/tactics.py', 'connect6/training.py'))}
    status = load_json(root / 'status.json', {})
    atomic_bytes(root / 'gate-tactics-1.json', json.dumps(report).encode(), status.get('cap_bytes', 20*GIB))
    incumbent = {'kind': 'heuristic', 'tactics': True, 'label': 'Tactical heuristic · paired gate passed',
                 'gate': 'gate-tactics-1.json', 'previous': previous}
    save_json(root / 'incumbent.json', incumbent)
print(json.dumps({'promoted': True, 'score': report['score'], 'lower_95': lower,
                  'opponent': 'Original heuristic only; no external-engine claim.'}, indent=2))
