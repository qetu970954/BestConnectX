"""JSON full-turn CLI adapter. Loading a model is outside its thinking-time measurement."""
import json
from pathlib import Path
import re
import sys
from .network import device_for
from .storage import load_json, run_lock
from .runtime import DEFAULT_RULES, Game, Rules, full_turn
from .training import network


def play(args):
    request = sys.stdin.read(4 * 1024 * 1024 + 1)
    if len(request) > 4 * 1024 * 1024:
        raise ValueError("Move/strategy request is too large.")
    payload = json.loads(request)
    if not isinstance(payload, dict) or set(payload) - {"moves", "strategy"}:
        raise ValueError("Expected a JSON move/strategy object.")
    root = Path(args.data).resolve()
    identity = load_json(root / "run.json") or load_json(root / "rules.json")
    rules = Rules(**identity["rule_config"]) if identity else DEFAULT_RULES
    game = Game.from_moves(payload.get("moves"), rules=rules)
    if game.done:
        raise ValueError("Game has ended.")
    with run_lock(root):
        entry = load_json(root / "incumbent.json", {"kind": "heuristic"})
        if args.model == "heuristic":
            entry = {"kind": "heuristic"}
        elif args.model == "latest":
            entry = {"kind": "candidate", "file": "latest.pt", "id": "latest · unvalidated"}
        elif args.model != "best":
            if not re.fullmatch(r"model-\d{8,}\.pt", args.model):
                raise ValueError("Choose best, latest, heuristic, or a saved model filename.")
            entry = {"kind": "candidate", "file": f"models/{args.model}", "id": args.model[:-3]}
        net = network(root, entry, device_for(args.device), rules) if entry.get("file") else None
        moves, strategy, duration = full_turn(game, net, args.seconds, payload.get('strategy'))
        print(json.dumps({'moves': moves, 'model': entry.get('id', 'heuristic'),
                          'kind': entry['kind'], 'seconds': duration, 'strategy': strategy}))
