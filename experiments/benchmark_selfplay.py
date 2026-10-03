"""Frozen-checkpoint decision throughput. No training, exports, or promotion.
Run: uv run python -m experiments.benchmark_selfplay --device cpu
GPU benchmarking requires separate authorization.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import time
import numpy as np
from engine.game import Game, Rules
from engine.network import device_for
from engine.storage import run_lock
from engine.training import digest, load_state, network, opening
from engine.selfplay import selfplay_batch


parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default='data/connect5-9x9-s1-o1-v1/latest.pt')
parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
parser.add_argument('--parallel', type=int, nargs='+', default=[8, 64])
parser.add_argument('--simulations', type=int, default=64)
args = parser.parse_args()
if any(not 1 <= count <= 128 for count in args.parallel) or not 1 <= args.simulations <= 100000:
    parser.error('Use 1..128 parallel games and 1..100000 simulations.')
checkpoint = Path(args.checkpoint).resolve()
with run_lock(checkpoint.parent):
    state = load_state(checkpoint)
    rules = Rules(**state['rule_config'])
    net = network(checkpoint.parent, {'file': checkpoint.name}, device_for(args.device), rules)
    positions = [Game.from_moves(row['moves'], rules=rules) for row in state.get('active', [])]
    positions = positions or [opening(i, 123456, rules) for i in range(8)]
    batches, evaluate = [], net.evaluate
    def counted(games):
        batches.append(len(games))
        return evaluate(games)
    net.evaluate = counted
    results = []
    for parallel in args.parallel:
        boards = [positions[i % len(positions)] for i in range(parallel)]
        timings = []
        for repeat in range(4):
            batches.clear()
            started = time.perf_counter()
            decisions = selfplay_batch(boards, net, args.simulations, np.random.default_rng(5070), tactical_ms=2)
            elapsed = time.perf_counter() - started
            if repeat:  # First run warms each batch size; network transfer is already synchronous.
                timings.append(elapsed)
        seconds = statistics.median(timings)
        results.append({'parallel': parallel, 'median_seconds': seconds,
            'decisions_per_second': parallel / seconds,
            'inference_batch_median': statistics.median(batches) if batches else 0,
            'sources': dict(Counter(row['source'] for row in decisions))})
    print(json.dumps({'rules': rules.id, 'checkpoint_sha256': digest(checkpoint), 'device': args.device,
        'simulations': args.simulations, 'results': results,
        'scope': 'Retained positions repeated across roots; decision throughput is not strength or training convergence.'}, indent=2))
