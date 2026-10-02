"""Frozen-network decision-work benchmark, not training convergence or strength.
Run: uv run python -m experiments.benchmark_selfplay > docs/selfplay-benchmark-tss.json
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import time
import numpy as np
import torch
from connect6.game import Game
from connect6.network import device_for
from connect6.search import search
from connect6.storage import run_lock
from connect6.training import opening, read_network, selfplay_batch


class CountedNetwork:
    def __init__(self, net):
        self.net, self.positions = net, 0

    def evaluate(self, games):
        self.positions += len(games)
        return self.net.evaluate(games)


parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default='data/latest.pt')
args = parser.parse_args()
checkpoint = Path(args.checkpoint)
fixtures = json.loads(Path('tests/fixtures/forks.json').read_text())['selected_fixtures']
forks = [Game.from_moves(row['moves']) for row in fixtures]
losses = [g.copy() for g in forks]
for game, row in zip(losses, fixtures):
    for move in row['certificate']:
        game.play(move)
workloads = {'quiet_openings': [opening(i, 999) for i in range(8)],
             'selected_tactical_mix': forks + losses + [opening(i, 999) for i in range(2)]}
with run_lock(checkpoint.parent):
    device = device_for()
    net = read_network(checkpoint, device)
    results = {}
    modes = ('search_only', 'fork_only', 'combined')
    for name, boards in workloads.items():
        runs = {mode: [] for mode in modes}
        # One warmup per path, then alternate measurement order to reduce order bias.
        for repeat in range(6):
            order = modes if repeat % 2 else tuple(reversed(modes))
            for mode in order:
                counted = CountedNetwork(net)
                rng = np.random.default_rng(5070)
                start = time.perf_counter()
                if mode == 'search_only':
                    search(boards, counted, 64, rng=rng)
                    sources = {'mcts': len(boards)}
                else:
                    decisions = selfplay_batch(boards, counted, 64, rng, tactical_ms=2,
                                               tss_enabled=(mode == 'combined'))
                    sources = dict(Counter(d['source'] for d in decisions))
                seconds = time.perf_counter() - start
                if repeat:
                    runs[mode].append({'seconds': seconds,
                                       'network_positions': counted.positions,
                                       'sources': sources})
        results[name] = {
            mode: {'median_seconds': round(statistics.median(r['seconds'] for r in rows), 6),
                   'median_network_positions': statistics.median(r['network_positions'] for r in rows),
                   'runs': rows}
            for mode, rows in runs.items()}
    print(json.dumps({'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                      'torch': str(torch.__version__), 'device': str(device),
                      'gpu': torch.cuda.get_device_name() if device.type == 'cuda' else None,
                      'simulations_per_unresolved_root': 64, 'tactical_ms_per_root': 2,
                      'selfplay_tss_max_turns': 3,
                      'scope': 'Decision generation only; selected tactical mix is correlated and not representative self-play. Efficiency is not strength evidence.',
                      'results': results}, indent=2))
