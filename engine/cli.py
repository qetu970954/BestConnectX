"""Small command-line interface for configurable connection-game experiments."""
import argparse
import json
import math
from pathlib import Path
import sys
from .storage import busy, load_json
from .game import DEFAULT_RULES, Rules, parse_board


def add_rules(parser):
    parser.add_argument("--connect", type=int, default=5, help="Consecutive stones needed to win (overlines also win)")
    parser.add_argument("--board_size", "--board-size", default="9*9", help='Square N*N; quote it in shells, e.g. "9*9" or "19*19"')
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
    args.rules = Rules(height, width, args.connect, args.stones_per_turn, args.starter_stones)
    args.data = args.data or str(Path("data") / args.rules.id)


def validate_session(args):
    if (not math.isfinite(args.hours) or not 0 < args.hours <= 168
            or not math.isfinite(args.disk_gib) or not .1 <= args.disk_gib <= 10000):
        raise ValueError("Session must be 0..168 hours (exclusive zero); artifact cap must be 0.1..10000 GiB.")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Original, local connection-game self-play")
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
    training.add_argument("--max-games", type=int, default=0, help=argparse.SUPPRESS)
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
    args = parser.parse_args(argv)
    try:
        if args.command == "train":
            resolve_rules(args)
            validate_session(args)
            if (not 1 <= args.simulations <= 100000 or (args.parallel is not None and not 1 <= args.parallel <= 128)
                    or not 2 <= args.batch <= 4096 or not 0 <= args.seed < 2**32
                    or args.max_games < 0 or args.snapshot_every < 1
                    or not math.isfinite(args.tactical_ms) or not 0 <= args.tactical_ms <= 50
                    or (args.seconds is not None and (not math.isfinite(args.seconds) or not .02 <= args.seconds <= 30))):
                raise ValueError("Invalid search, seed, milestone, tactical, or gate setting.")
            from .training import train
            train(args)
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
    except (ValueError, OSError) as exc:
        parser.exit(1, f"{type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    main()
