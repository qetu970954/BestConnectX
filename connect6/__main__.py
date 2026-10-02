import argparse
import json
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description="BestConnect6 — original, local, from-scratch training")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Start or resume a bounded training session")
    from .training import add_arguments, validate
    add_arguments(train)
    web = commands.add_parser("web", help="Open the local play/training dashboard")
    web.add_argument("--port", type=int, default=8765)
    web.add_argument("--data", default="data")
    web.add_argument("--open-browser", action="store_true")
    play = commands.add_parser("play", help="Internal JSON bot-turn interface on stdin/stdout")
    play.add_argument("--data", default="data")
    play.add_argument("--seconds", type=float, default=5)
    play.add_argument("--candidate", action="store_true", help="Play the unvalidated latest network with tactical proofs; does not promote it")
    check = commands.add_parser("doctor", help="Validate local PyTorch/GPU forward and backward execution")
    check.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.command == "train":
        from .training import run
        validate(args)
        run(args)
    elif args.command == "web":
        from .web import serve
        if not 1024 <= args.port <= 65535:
            parser.error("Port must be between 1024 and 65535.")
        serve(args.port, args.data, args.open_browser)
    elif args.command == "doctor":
        import torch
        from .network import Network, device_for
        from .game import Game
        device = device_for(args.device)
        net = Network().to(device)
        x = torch.from_numpy(Game().features()).unsqueeze(0).repeat(2, 1, 1, 1).to(device)
        p, v = net(x)
        (p.square().mean() + v.square().mean()).backward()
        if device.type == "cuda":
            torch.cuda.synchronize()
        print(json.dumps({"python": sys.version, "torch": torch.__version__, "device": str(device),
                          "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
                          "parameters": sum(p.numel() for p in net.parameters()),
                          "forward_backward": "passed"}, indent=2))
    else:
        import math
        from .game import Game
        from .storage import run_lock, load_json
        from .training import read_network, turn_action
        from .network import device_for
        if not math.isfinite(args.seconds) or not .05 <= args.seconds <= 30:
            parser.error("Thinking time must be between 0.05 and 30 seconds.")
        root = Path(args.data).resolve()
        message = sys.stdin.read(16385)
        if len(message) > 4 * 1024 * 1024:
            raise ValueError("Game and tactical strategy input too large.")
        request = json.loads(message)
        if isinstance(request, list):
            moves, strategy = request, None
        elif isinstance(request, dict) and set(request) <= {"moves", "strategy"}:
            moves, strategy = request.get("moves"), request.get("strategy")
        else:
            raise ValueError("Expected a move list or an internal move/strategy request.")
        game = Game.from_moves(moves)
        if game.done:
            raise ValueError("Game has ended.")
        with run_lock(root):
            best = load_json(root / "incumbent.json", {"kind": "heuristic"})
            if args.candidate:
                if not (root / "latest.pt").is_file():
                    raise ValueError("No candidate checkpoint yet. Train a session first.")
                best = {"kind": "candidate", "file": "latest.pt", "tactics": True, "tss": True}
            net = read_network(root / best["file"], device_for()) if best["kind"] in ("network", "candidate") else None
            color, start, moves = game.player, time.monotonic(), []
            plan, proof_plan, strategy = [], [None], [strategy]
            while not game.done and game.player == color:
                remaining = max(0, args.seconds - (time.monotonic() - start))
                action = turn_action(game, net, remaining / game.left, 100000, lambda: False,
                                     tactics=best.get("tactics", False), tss=best.get("tss", False),
                                     plan=plan, proof_plan=proof_plan, strategy=strategy)
                game.play(action)
                if plan:
                    plan.pop(0)
                moves.append(action)
            print(json.dumps({"moves": moves, "seconds": time.monotonic() - start,
                              "kind": best["kind"], "strategy": strategy[0]}))


if __name__ == "__main__":
    main()
