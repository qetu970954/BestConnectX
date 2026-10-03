"""Read-only usage scan benchmark; no training or CUDA. Run with --data RUN_DIR."""
import argparse
from pathlib import Path
import statistics
import time
from engine.storage import usage


def reference(root):
    total = 0
    for path in root.rglob('*'):
        try:
            if path.is_file():
                total += path.stat().st_size
        except FileNotFoundError:
            pass
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    args = parser.parse_args()
    if not args.data.is_dir():
        parser.error('--data must be an existing, paused run directory')
    timings = {reference: [], usage: []}
    for repeat in range(6):
        totals = []
        for scan in ((reference, usage) if repeat % 2 == 0 else (usage, reference)):
            start = time.perf_counter()
            totals.append(scan(args.data))
            elapsed = time.perf_counter() - start
            if repeat:
                timings[scan].append(elapsed)
        assert totals[0] == totals[1], 'Run changed during measurement, or scan totals differ.'
    old, new = (statistics.median(timings[fn]) for fn in (reference, usage))
    print(f'Bytes: {totals[0]} | old scan: {old:.6f}s | current scan: {new:.6f}s | ratio: {old/new:.2f}x')
    print('Directory-scan speed only; not end-to-end training throughput or GPU utilization.')


if __name__ == '__main__':
    main()
