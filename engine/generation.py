"""Management command: request X complete native games, return summaries, save no kifus."""
import json
from pathlib import Path
import time
import numpy as np
from .network import device_for
from .runtime import selfplay_batch, opening
from .storage import load_json, run_lock
from .training import network


def generate(args):
    root = Path(args.data).resolve()
    rng = np.random.default_rng(args.seed)
    net = None
    if args.model != 'heuristic':
        with run_lock(root):
            entry = {'file': 'latest.pt'} if args.model == 'latest' else load_json(root / 'incumbent.json', {})
            if not entry.get('file'):
                raise ValueError('No selected model exists. Train first, or use --model heuristic.')
            net = network(root, entry, device_for(args.device), args.rules)
    began = time.perf_counter()
    live, contexts = [], []
    completed = black = white = draws = placements = 0
    timing = {'cpu_search_seconds': 0., 'inference_seconds': 0.}
    parallel = min(getattr(args, 'parallel', 64), args.games)
    if not 1 <= parallel <= 128:
        raise ValueError('Self-play concurrency must be 1..128.')
    while completed < args.games:
        while len(live) < parallel and completed + len(live) < args.games:
            live.append(opening(0, int(rng.integers(0, 2**63)), args.rules, random_start=True))
            contexts.append({'strategies': [None, None], 'plans': [[], []]})
        slots = [0 if game.player == 1 else 1 for game in live]
        strategies = [row['strategies'][slot] for row, slot in zip(contexts, slots)]
        plans = [row['plans'][slot] for row, slot in zip(contexts, slots)]
        measured = {}
        decisions = selfplay_batch(live, net, args.simulations, rng, bootstrap=net is None,
                                  strategies=strategies, plans=plans, workers=args.workers,
                                  tactical_ms=getattr(args, 'tactical_ms', 2), metrics=measured)
        for key, value in measured.items():
            timing[key] += value
        survivors, next_contexts = [], []
        for game, row, decision, slot, strategy, plan in zip(live, contexts, decisions, slots, strategies, plans):
            game.play(decision['action'])
            row['strategies'][slot] = strategy
            row['plans'][slot] = plan[1:] if plan and plan[0] == decision['action'] else []
            if game.done:
                completed += 1; placements += len(game.moves)
                black += game.winner == 1; white += game.winner == -1; draws += game.winner == 0
            else:
                survivors.append(game); next_contexts.append(row)
        live, contexts = survivors, next_contexts
    elapsed = time.perf_counter() - began
    print(json.dumps({'games': completed, 'black_wins': black, 'white_wins': white, 'draws': draws,
                      'placements': placements, 'seconds': elapsed, 'games_per_second': completed/elapsed,
                      'rules': args.rules.to_dict(), 'model': net.config if net else 'heuristic',
                      'workers': args.workers, 'parallel': parallel, 'timings': timing,
                      'saved_game_records': False}, indent=2))
