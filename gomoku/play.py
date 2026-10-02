"""JSON full-turn CLI adapter. Loading a model is outside its thinking-time measurement."""
import json
from pathlib import Path
import re
import sys
import time
from connect6.network import device_for
from connect6.storage import load_json, run_lock
from connect6.training import turn_action
from .game import DEFAULT_RULES, Game, Rules
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
        color, started, moves = game.player, time.monotonic(), []
        plan, proof_plan, strategy = [], [None], [payload.get("strategy")]
        while not game.done and game.player == color:
            remaining = max(0, args.seconds - (time.monotonic() - started))
            action = turn_action(game, net, remaining / game.left, 100_000, lambda: False,
                                tactics=True, tss=True, plan=plan, proof_plan=proof_plan, strategy=strategy)
            game.play(action)
            if plan:
                plan.pop(0)
            moves.append(action)
        print(json.dumps({"moves": moves, "model": entry.get("id", "heuristic"),
                          "kind": entry["kind"], "seconds": time.monotonic() - started,
                          "strategy": strategy[0]}))
