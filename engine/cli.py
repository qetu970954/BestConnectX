"""Small command-line interface for configurable connection-game experiments."""
import argparse
import json
import math
from pathlib import Path
import sys
from .storage import busy, load_json
from .game import Rules, parse_board
from .runtime import DEFAULT_RULES
from .config import DEFAULT_CONFIG, load_options


def add_rules(parser):
    parser.add_argument("--connect", type=int, default=5, help="Consecutive stones needed to win (overlines also win)")
    parser.add_argument("--board_size", "--board-size", default="15*15", help='Square N*N; quote it in shells, e.g. "9*9" or "19*19"')
    parser.add_argument("--stones_per_turn", "--stones-per-turn", type=int, default=1)
    parser.add_argument("--starter-stones", "--starter_stones", type=int, default=1,
                        help="Number of stones in Black's first turn")
    parser.add_argument("--data", default=None, help="Default: a separate directory per rule configuration")


def common(parser):
    parser.add_argument("--hours", type=float, default=2)
    parser.add_argument("--disk-gib", type=float, default=20)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")


def resolve_rules(args):
    height, width = parse_board(args.board_size)
    requested = Rules(height, width, args.connect, args.stones_per_turn, args.starter_stones)
    recorded = load_json(Path(args.data) / 'run.json') if args.data else None
    if recorded and args.command == 'train':
        args.rules = Rules(**recorded['rule_config'])
        explicit = getattr(args, 'explicit', set())
        if explicit & {'connect', 'board_size', 'stones_per_turn', 'starter_stones'} and requested != args.rules:
            raise ValueError('Changing rules requires a new --data directory.')
        config = recorded.get('network', {})
        if any(key in explicit and config.get(key, getattr(args, key)) != getattr(args, key) for key in ('channels', 'blocks')):
            raise ValueError('Changing model shape requires a new --data directory.')
    else:
        args.rules = requested
    args.data = args.data or str(Path('data') / args.rules.id)


def validate_session(args):
    if (not math.isfinite(args.hours) or not 0 < args.hours <= 168
            or not math.isfinite(args.disk_gib) or not .1 <= args.disk_gib <= 10000):
        raise ValueError("Session must be 0..168 hours (exclusive zero); artifact cap must be 0.1..10000 GiB.")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="Original C++ connection-game self-play with Python learning")
    try:
        selected, options, explicit = load_options(argv)
    except (ValueError, OSError) as exc:
        parser.exit(1, f'Config error: {exc}\n')
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="Start/resume bounded training",
                                   prog=parser.prog if Path(sys.argv[0]).name == "train.py" else f"{parser.prog} train")
    add_rules(training)
    common(training)
    training.add_argument("--seed", type=int, default=5070)
    training.add_argument("--simulations", type=int, default=64)
    training.add_argument("--parallel", type=int, default=None,
                          help="Concurrent self-play games (new-run default: 64); explicit values also apply on resume")
    training.add_argument("--batch", type=int, default=128)
    training.add_argument("--tactical-ms", type=float, default=2)
    training.add_argument("--snapshot-every", type=int, default=1000, help="Self-play games between immutable model snapshots")
    training.add_argument("--seconds", type=float, default=None,
                          help="Equal gate full-turn limit (new-run default: 0.25s); frozen in each gate")
    training.add_argument("--restart-gate", action="store_true",
                          help="Archive the pending gate and restart it with current code/limits; never mix old results")
    training.add_argument('--max-games', type=int, default=0, help='Stop at this total completed-game count; 0 means time limit only')
    training.add_argument('--channels', type=int, default=64)
    training.add_argument('--blocks', type=int, default=6)
    training.add_argument('--workers', type=int, default=None, help='Native CPU threads; explicit values also apply on resume')
    training.add_argument('--learning-rate', type=float, default=.001)
    training.add_argument('--replay-limit', type=int, default=20_000)
    training.add_argument('--bootstrap-games', type=int, default=16)
    training.add_argument('--updates-per-cycle', type=int, default=32)
    generation = commands.add_parser('selfplay', help='Generate a bounded number of native self-play games without model updates')
    add_rules(generation)
    generation.add_argument('--games', type=int, default=1)
    generation.add_argument('--model', default='heuristic', choices=('heuristic', 'latest', 'best'))
    generation.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    generation.add_argument('--simulations', type=int, default=64)
    generation.add_argument('--workers', type=int, default=6)
    generation.add_argument('--parallel', type=int, default=64, help='Concurrent native self-play games')
    generation.add_argument('--tactical-ms', type=float, default=2)
    generation.add_argument('--seed', type=int, default=5070)
    evaluation = commands.add_parser("evaluate", help="Resume an existing frozen 100-game gate")
    common(evaluation)
    evaluation.add_argument("--data", required=True)
    evaluation.add_argument("--report", required=True, help="gate-model-00002000.json, relative to --data")
    play = commands.add_parser("play", help="JSON stdin/stdout bot-turn interface")
    play.add_argument("--data", default=str(Path("data") / DEFAULT_RULES.id))
    play.add_argument("--model", default="best", help="best, latest, heuristic, or model-NNNNNNNN.pt")
    play.add_argument("--seconds", type=float, default=5)
    play.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    web = commands.add_parser("web", help="Local play/results dashboard; no browser training controls",
                              prog=parser.prog if Path(sys.argv[0]).name == "dashboard.py" else f"{parser.prog} web")
    add_rules(web)
    web.add_argument("--port", type=int, default=8765)
    web.add_argument("--open-browser", action="store_true")
    for action in ("status", "stop"):
        command = commands.add_parser(action, help="Show JSON status" if action == "status" else "Request a safe checkpoint and stop")
        command.add_argument("--data", default=str(Path("data") / DEFAULT_RULES.id))
    for command in (training, web, generation):
        command.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
        command.add_argument('--preset', default='gomoku')
        command.set_defaults(**options)
    args = parser.parse_args(argv)
    args.explicit = explicit
    # File defaults set new-run values; only explicit runtime values override a saved run.
    if args.command == 'train' and args.data and (Path(args.data) / 'latest.pt').exists():
        for key in ('parallel', 'workers', 'seconds'):
            if key not in explicit:
                setattr(args, key, None)
    try:
        if args.command == "train":
            resolve_rules(args)
            validate_session(args)
            if (not 1 <= args.simulations <= 100000 or (args.parallel is not None and not 1 <= args.parallel <= 128)
                    or not 2 <= args.batch <= 4096 or not 0 <= args.seed < 2**32
                    or args.max_games < 0 or args.snapshot_every < 1
                    or not 4 <= args.channels <= 256 or not 0 <= args.blocks <= 32
                    or (args.workers is not None and not 1 <= args.workers <= 12)
                    or not 1 <= args.replay_limit <= 200_000 or not 0 <= args.bootstrap_games <= 10_000
                    or not 1 <= args.updates_per_cycle <= 10_000
                    or not math.isfinite(args.learning_rate) or not 0 < args.learning_rate <= .1
                    or not math.isfinite(args.tactical_ms) or not 0 <= args.tactical_ms <= 50
                    or (args.seconds is not None and (not math.isfinite(args.seconds) or not .02 <= args.seconds <= 30))):
                raise ValueError("Invalid search, seed, milestone, tactical, or gate setting.")
            from .training import train
            train(args)
        elif args.command == 'selfplay':
            resolve_rules(args)
            if (not 1 <= args.games <= 100000 or not 1 <= args.workers <= 12 or not 1 <= args.parallel <= 128
                    or not 1 <= args.simulations <= 100000 or not 0 <= args.seed < 2**32
                    or not math.isfinite(args.tactical_ms) or not 0 <= args.tactical_ms <= 50):
                raise ValueError('Invalid self-play count, workers, concurrency, simulations, tactics, or seed.')
            from .generation import generate
            generate(args)
        elif args.command == "evaluate":
            validate_session(args)
            from .training import evaluate
            evaluate(args)
        elif args.command == "play":
            if not math.isfinite(args.seconds) or not .02 <= args.seconds <= 30:
                raise ValueError("Thinking time must be 0.02..30 seconds.")
            from .play import play
            play(args)
        elif args.command == "web":
            resolve_rules(args)
            if not 1024 <= args.port <= 65535:
                raise ValueError("Port must be 1024..65535.")
            from .web import serve
            serve(args.port, args.data, args.open_browser, args.rules)
        else:
            root = Path(args.data)
            if args.command == "stop":
                if not root.is_dir():
                    raise ValueError("No such data directory.")
                (root / "stop").touch()
                print("Stop requested. Wait for the CLI to save its checkpoint.")
            else:
                status = load_json(root / "status.json", {"phase": "not_started"})
                status["running"] = busy(root) if root.exists() else False
                print(json.dumps(status, indent=2))
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"{type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    main()
