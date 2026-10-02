"""End-to-end search throughput, not playing strength.
Run: uv run python -m experiments.benchmark_search --checkpoint data/latest.pt
"""
import argparse
import hashlib
from pathlib import Path
import json
import time
import torch
from connect6.network import device_for
from connect6.search import search
from connect6.training import opening, read_network

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', required=True)
args = parser.parse_args()
device = device_for()
net = read_network(args.checkpoint, device)
games = [opening(i, 999) for i in range(8)]
net.evaluate(games)  # warmup excluded; features, tree work and inference included below
if device.type == 'cuda':
    torch.cuda.synchronize()
start = time.perf_counter()
_, simulations = search(games, net, simulations=64)
elapsed = time.perf_counter() - start
print(json.dumps({'checkpoint': args.checkpoint,
                  'checkpoint_sha256': hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
                  'torch': str(torch.__version__), 'device': str(device),
                  'gpu': torch.cuda.get_device_name() if device.type == 'cuda' else None,
                  'games': len(games), 'simulations_per_game': simulations,
                  'seconds': round(elapsed, 4),
                  'simulations_per_second': round(simulations * len(games) / elapsed, 1),
                  'scope': 'Warm-start end-to-end Python search, no strength conclusion.'}, indent=2))
