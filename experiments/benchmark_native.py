"""Paired FP32 search timings and temporary training/resume checks. CPU by default."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch
import numpy as np
import torch
from engine.game import Game as Reference, Rules
from engine.network import Network, device_for
from engine.runtime import Game, model_for, opening, search, selfplay_batch
from engine.search import search as python_search
from engine.cli import main as cli
from engine.training import load_state


def measure(device, repeats, batch, simulations, training_presets):
    torch.manual_seed(5070)
    results = []
    for rules in (Rules(3, 3, 3), Rules(15, 15, 5), Rules(19, 19, 6, 2, 1)):
        channels, blocks = (8, 1) if rules.height == 3 else (64, 6)
        net = Network(channels, blocks, rules.height).to(device).eval()
        native = model_for(net)
        games = [opening(i, 5070, rules) for i in range(batch)]
        reference = [Reference.from_moves(game.moves, rules=rules) for game in games]
        # Use FP32 on both paths; comparing half precision to FP32 would confound language cost.
        def evaluate(games):
            with torch.inference_mode():
                x = torch.from_numpy(np.stack([g.features() for g in games])).to(device)
                policy, value = net(x)
                return policy.cpu().numpy(), value.cpu().numpy()
        net.evaluate = evaluate
        x = np.stack([game.features() for game in reference])
        actual, values = native.evaluate_features(x)
        expected, expected_values = evaluate(reference)
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-4)
        np.testing.assert_allclose(values, expected_values, atol=2e-5, rtol=2e-4)
        timings = []
        with patch('engine.native.library', return_value=None):
            python_search(reference, net, simulations)  # Warmup.
            for _ in range(repeats):
                began = time.perf_counter(); python_search(reference, net, simulations); timings.append(time.perf_counter()-began)
        python_seconds = statistics.median(timings)
        workers = {}
        for count in (1, 2, 6):
            search(games, net, simulations, workers=count)
            timings = []
            for _ in range(repeats):
                began = time.perf_counter(); search(games, net, simulations, workers=count); timings.append(time.perf_counter()-began)
            workers[str(count)] = {'seconds': statistics.median(timings), 'last_native_parts': native.timings.copy()}
        game = Game(rules=rules); rng = np.random.default_rng(5070)
        began = time.perf_counter()
        while not game.done:
            decision = selfplay_batch([game], net, min(4, simulations), rng, tactical_ms=0, workers=1)[0]
            game.play(decision['action'])
        assert Reference.from_moves(game.moves, rules=rules).done
        results.append({'rules': rules.to_dict(), 'model': net.config, 'parameters': sum(p.numel() for p in net.parameters()),
                        'precision': 'float32', 'batch': batch, 'simulations': simulations, 'repeats': repeats,
                        'python_seconds': python_seconds, 'native_workers': workers,
                        'complete_native_game': {'placements': len(game.moves), 'winner': game.winner, 'seconds': time.perf_counter()-began}})
        if device.type == 'cuda':
            results[-1]['peak_allocated_gpu_bytes'] = torch.cuda.max_memory_allocated()
    training = []
    for preset in training_presets:
        with tempfile.TemporaryDirectory() as directory:
            cli(['train', '--preset', preset, '--data', directory, '--device', str(device),
                 '--max-games', '8', '--hours', '.01', '--tactical-ms', '0'])
            first = load_state(Path(directory)/'latest.pt')
            assert first['games'] == 8 and first['step'] == 32 and first['optimizer']['state']
            restored = Network(**first['config']).to(device).eval()
            restored.load_state_dict(first['weights'])
            position = Game(rules=Rules(**first['rule_config']))
            x = np.stack([position.features()])
            updated_policy, updated_value = model_for(restored).evaluate_features(x)
            with torch.inference_mode():
                policy, value = restored(torch.from_numpy(x).to(device))
            np.testing.assert_allclose(updated_policy, policy.cpu().numpy(), atol=2e-5, rtol=2e-4)
            np.testing.assert_allclose(updated_value, value.cpu().numpy(), atol=2e-5, rtol=2e-4)
            cli(['train', '--data', directory, '--device', str(device), '--max-games', '9', '--hours', '.01'])
            last = load_state(Path(directory)/'latest.pt')
            assert last['games'] == 9 and last['step'] == first['step'] and last['replay']
            assert not (Path(directory)/'selfplay').exists() and not (Path(directory)/'replay').exists()
            training.append({'preset': preset, 'model': first['config'], 'batch': first['settings']['batch'],
                             'updates': first['step'], 'resumed_games': last['games'], 'updated_native_outputs': 'passed',
                             'active_games_after_resume': len(last['active']), 'timings': last['timings']})
            if device.type == 'cuda':
                training[-1]['peak_allocated_gpu_bytes'] = torch.cuda.max_memory_allocated()
    return {'device': str(device), 'torch': str(torch.__version__), 'search_checks': results,
            'temporary_learning_resume': 'passed', 'training_checks': training}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--simulations', type=int, default=16)
    parser.add_argument('--wall-seconds', type=float, default=90)
    parser.add_argument('--training-presets', default='tictactoe', help='Comma-separated temporary learning/resume checks')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 10 or not 1 <= args.batch <= 128 or not 1 <= args.simulations <= 64 or not 1 <= args.wall_seconds <= 300:
        parser.error('Invalid benchmark or process time limits.')
    training_presets = args.training_presets.split(',')
    if not training_presets or set(training_presets) - {'tictactoe', 'gomoku', 'connect6'}:
        parser.error('Training checks support tictactoe,gomoku,connect6.')
    if not args.worker:
        command = [sys.executable, '-m', 'experiments.benchmark_native', *sys.argv[1:], '--worker']
        began = time.perf_counter()
        try:
            subprocess.run(command, timeout=args.wall_seconds, check=True)
        finally:
            print(f'Check process wall time, including startup/warmup: {time.perf_counter()-began:.3f}s', flush=True)
        return
    result = measure(device_for(args.device), args.repeats, args.batch, args.simulations, training_presets)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(f'Checks saved to {args.output}', flush=True)


if __name__ == '__main__':
    main()
