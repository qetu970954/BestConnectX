"""Disposable search/learner benchmark for arbitrary supported square rules.

Example: python -m experiments.benchmark_pipeline --size 9 15 --connect 5
Connect6: python -m experiments.benchmark_pipeline --size 19 --connect 6 --stones 2
Defaults to CPU. No saved runs, exports or strength claims.
"""
import argparse
import json
import os
import statistics
import time
import numpy as np
import torch
from engine import network as network_module
from engine import native
from engine.game import Rules
from engine.network import Network, device_for
from engine.selfplay import observation, selfplay_batch, train_step
from engine.training import opening


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, nargs='+', default=[9])
    parser.add_argument('--connect', type=int, default=5)
    parser.add_argument('--stones', type=int, default=1)
    parser.add_argument('--starter-stones', type=int, default=1)
    parser.add_argument('--parallel', type=int, default=64)
    parser.add_argument('--simulations', type=int, default=64)
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--features', choices=('batched', 'scalar'), default='batched')
    parser.add_argument('--backend', choices=('auto', 'python'), default='auto')
    args = parser.parse_args()
    if min(args.parallel, args.simulations, args.batch, args.repeats) < 1:
        parser.error('Batch sizes, simulations and repeats must be positive.')
    device = device_for(args.device)
    if args.backend == 'python':
        os.environ['BESTCONNECT6_NATIVE'] = '0'
        native.library.cache_clear()
    if args.features == 'scalar':
        network_module.batch_features = lambda games: np.stack([game.features() for game in games])
    for size in args.size:
        rules = Rules(size, size, args.connect, args.stones, args.starter_stones)
        torch.manual_seed(5070)
        net = Network(size=size).to(device)
        search_times, update_times = [], []
        replay = []
        for i in range(args.parallel):
            game = opening(i, 123456, rules)
            policy = np.zeros(game.board.size, dtype=np.float32)
            actions = game.actions()
            policy[actions] = 1 / len(actions)
            sample = observation(game, policy)
            sample['result'] = 0.
            replay.append(sample)
        learner = Network(size=size).to(device)
        optimizer = torch.optim.AdamW(learner.parameters(), lr=.001)
        for repeat in range(args.repeats + 1):
            boards = [opening(i, 123456, rules) for i in range(args.parallel)]
            start = time.perf_counter()
            decisions = selfplay_batch(boards, net, args.simulations,
                                       np.random.default_rng(5070), tactical_ms=0)
            seconds = time.perf_counter() - start
            assert decisions is not None and len(decisions) == args.parallel
            if repeat:
                search_times.append(seconds)
            start = time.perf_counter()
            train_step(learner, optimizer, replay, args.batch, np.random.default_rng(repeat),
                       device, rules=rules)
            if repeat:
                update_times.append(time.perf_counter() - start)
        seconds = statistics.median(search_times)
        print(json.dumps({'rules': rules.id, 'device': str(device), 'torch': str(torch.__version__),
            'features': args.features,
            'backend': 'cpp' if native.library() is not None else 'python',
            'parallel': args.parallel, 'simulations': args.simulations, 'batch': args.batch,
            'search_seconds': seconds, 'decisions_per_second': args.parallel / seconds,
            'update_ms': statistics.median(update_times) * 1000,
            'scope': 'Opening-position search and synthetic replay; not end-to-end training or strength.'}),
            flush=True)


if __name__ == '__main__':
    main()
