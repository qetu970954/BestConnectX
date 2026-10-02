"""Training lifecycle: terminal self-play, durable datasets, milestones, and frozen gates.

The game engine has no training dependency. The dashboard reads the JSON artifacts.
"""
import hashlib
import io
import json
from pathlib import Path
import platform
import signal
import sys
import time
import zipfile
import numpy as np
import torch
from connect6.network import Network, device_for
from connect6.search import choose
from connect6.storage import (GIB, atomic_bytes, load_checkpoint, load_json, run_lock,
                              save_checkpoint, save_json, usage)
from connect6.training import model_state, selfplay_batch, selfplay_samples, train_step, turn_action
from .game import DEFAULT_RULES, Game, Rules

GATE_GAMES = 100
SOURCE_FILES = ["gomoku/game.py", "gomoku/training.py", "connect6/game.py", "connect6/network.py",
                "connect6/search.py", "connect6/training.py", "connect6/tactics.py", "connect6/tss.py"]
PROJECT = Path(__file__).resolve().parent.parent


def artifact(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Artifact must stay inside this run's data directory.")
    return path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_state(path, rules=None):
    state = load_checkpoint(path)
    recorded = Rules(**state["rule_config"]) if "rule_config" in state else None
    if (state.get("format") != 1 or recorded is None or state.get("rules") != recorded.id
            or state["config"]["size"] != recorded.height
            or state["config"].get("width", state["config"]["size"]) != recorded.width
            or (rules is not None and rules != recorded)):
        raise ValueError("Checkpoint rules do not match this run. Use a separate --data directory.")
    state["config"].pop("width", None)  # Old square prototypes stored a redundant width.
    return state


def snapshot(net, step, rules=DEFAULT_RULES, games=0):
    return {**model_state(net, step), "rules": rules.id, "rule_config": rules.to_dict(), "games": games}


def network(root, entry, device, rules):
    path = artifact(root, entry["file"])
    if entry.get("sha256") and digest(path) != entry["sha256"]:
        raise ValueError("A frozen model's checksum changed; refusing to use it.")
    state = load_state(path, rules)
    net = Network(**state["config"]).to(device)
    net.load_state_dict(state["weights"])
    net.eval()
    return net


def _source():
    payloads = {name: (PROJECT / name).read_bytes() for name in SOURCE_FILES}
    sha = hashlib.sha256(b"".join(name.encode() + value for name, value in payloads.items())).hexdigest()
    return payloads, sha


def _environment(root, cap):
    payloads, sha = _source()
    archive = root / f"source-{sha[:16]}.zip"
    if not archive.exists():
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as zipped:
            for name, value in payloads.items():
                zipped.writestr(name, value)
        atomic_bytes(archive, stream.getvalue(), cap, root=root)
    return {"python": sys.version, "platform": platform.platform(), "torch": str(torch.__version__),
            "numpy": np.__version__, "cuda_build": torch.version.cuda, "code_sha256": sha,
            "source_archive": archive.name}


def _immutable_json(root, name, value, cap):
    path = artifact(root, name)
    if path.exists():
        if load_json(path) != value:
            raise ValueError(f"Refusing to overwrite archived record: {name}")
    else:
        save_json(path, value, root=root, cap=cap)


def _flush_exports(root, games, metrics, rules, cap):
    # Checkpoint first, export second. Interrupted exports are idempotently retried on resume.
    for item in games:
        stem = f"game-{item['record']['game']:09d}"
        _immutable_json(root, f"selfplay/{stem}.json", item["record"], cap)
        replay_path = root / "replay" / f"{stem}.pt"
        if not replay_path.exists():
            save_checkpoint(replay_path, {"rules": rules.id, "rule_config": rules.to_dict(),
                            "game": item["record"]["game"], "samples": item["samples"]}, cap, root=root)
    if metrics:
        name = f"metrics/updates-{metrics[0]['step']:09d}-{metrics[-1]['step']:09d}.json"
        _immutable_json(root, name, metrics, cap)
    games.clear()
    metrics.clear()


def _new_game(rules, rng):
    game = Game(rules=rules)
    if rng.random() < .5:
        height, width = game.shape
        pool = [r * width + c for r in range(max(0, height // 2 - 1), min(height, height // 2 + 2))
                for c in range(max(0, width // 2 - 1), min(width, width // 2 + 2))]
        for move in rng.choice(pool, rules.starter_stones, replace=False):
            game.play(int(move))
            if game.done:
                return Game(rules=rules)
    return game


def opening(index, seed, rules=DEFAULT_RULES):
    """Legal, held-out starts. A pair shares the start; engine colors, not stones, swap."""
    rng = np.random.default_rng(seed + index)
    for _ in range(20):
        game = Game(rules=rules)
        count = min(rules.starter_stones + 2 * rules.stones_per_turn, game.board.size - 1)
        pool = np.arange(game.board.size)
        center = (rules.height // 2) * rules.width + rules.width // 2
        moves = [center, *rng.choice(pool[pool != center], count - 1, replace=False).tolist()]
        for move in moves:
            game.play(int(move))
            if game.done:
                break
        if not game.done:
            return game
    return Game(rules=rules)  # Tiny games can have no live opening of the requested length.


def gate_summary(report):
    scores = [row["score"] for row in report["matches"]]
    report.update(games=len(scores), required_games=GATE_GAMES, complete=len(scores) == GATE_GAMES,
                  wins=scores.count(1.0), draws=scores.count(.5), losses=scores.count(0.0),
                  win_rate=scores.count(1.0) / len(scores) if scores else None,
                  score=sum(scores) / len(scores) if scores else None)
    return {key: value for key, value in report.items() if key not in ("matches", "current", "opponent")}


def verify_gate(report, rules):
    """Recompute results from actual terminal histories, including paired opening/color checks."""
    if len(report["matches"]) != GATE_GAMES:
        raise ValueError("Cannot decide an incomplete gate.")
    for i, row in enumerate(report["matches"]):
        initial = opening(i // 2, report["opening_seed"], rules)
        game = Game.from_moves(row["moves"], rules=rules)
        color = 1 if i % 2 == 0 else -1
        if (not game.done or row["opening"] != initial.moves
                or row["moves"][:len(initial.moves)] != initial.moves
                or row["candidate_color"] != color
                or row["score"] != (game.winner * color + 1) / 2):
            raise ValueError("Invalid gate history, opening, color, or terminal score.")
    gate_summary(report)
    return report["score"] > .5 and report["max_turn_overrun_seconds"] <= .1


def _finish_gate(root, path, report, rules, cap):
    accepted = verify_gate(report, rules)
    report["passes_gate"] = accepted
    current = load_json(root / "incumbent.json", {})
    if accepted:
        previous = report["opponent"]
        if current.get("file") == report["candidate"]["file"] and current.get("gate") == path.name:
            pass  # A previous process published the manifest but stopped before clearing the pending gate.
        elif current.get("file") == previous.get("file") and current.get("sha256") == previous.get("sha256"):
            best = {**report["candidate"], "kind": "network", "rules": rules.id,
                    "label": f"Accepted model · {report['candidate']['games']} self-play games",
                    "gate": path.name, "previous": {k: v for k, v in previous.items() if k != "previous"}}
            save_json(root / "incumbent.json", best, root=root, cap=cap)
        else:
            raise ValueError("Incumbent changed during a frozen gate; refusing promotion.")
        report["promoted"] = True
    report["decision_recorded"] = True
    save_json(path, report, root=root, cap=cap)


def eval_allowance(phase_seconds):
    """Evaluation gets one second of credit per four measured training seconds."""
    return max(0., phase_seconds["training"] / 4 - phase_seconds["evaluation"])


def _gate_report(rules, candidate, opponent, settings, code_sha):
    report = {"rules": rules.id, "rule_config": rules.to_dict(), "candidate": candidate,
        "opponent": {k: v for k, v in opponent.items() if k != "previous"},
        "opening_seed": 90_000_000 + candidate["games"], "seconds_per_turn": settings["gate_seconds"],
        "simulations": settings["simulations"], "matches": [], "current": None,
        "max_turn_overrun_seconds": 0., "elapsed_seconds": 0., "promoted": False, "code_sha256": code_sha}
    gate_summary(report)
    return report


def next_gate(root, rules, settings, code_sha, cap):
    """The immutable model directory is the queue; freeze the opponent when a gate starts."""
    opponent = load_json(root / "incumbent.json", {})
    if opponent.get("kind") != "network":
        return None
    for path in sorted((root / "models").glob("model-*.pt")):
        games = int(path.stem.rsplit("-", 1)[-1])
        if games <= opponent["games"]:
            continue
        name = f"gate-{path.stem}.json"
        report = load_json(root / name)
        if report is None:
            state = load_state(path, rules)
            candidate = {"id": path.stem, "file": f"models/{path.name}", "sha256": digest(path),
                         "games": games, "step": state["step"]}
            save_json(root / name, _gate_report(rules, candidate, opponent, settings, code_sha), root=root, cap=cap)
            return name
        if not report.get("decision_recorded"):
            return name
    return None


def restart_gate(root, name, rules, settings, code_sha, cap):
    """Explicit restart: archive old bytes before changing code or search limits. Never mix results."""
    path = artifact(root, name)
    old_bytes = path.read_bytes()
    old = json.loads(old_bytes)
    if old["rules"] != rules.id or old.get("decision_recorded"):
        raise ValueError("Only an unfinished gate with matching rules can be restarted.")
    incumbent = load_json(root / "incumbent.json", {})
    if any(incumbent.get(k) != old["opponent"].get(k) for k in ("file", "sha256")):
        raise ValueError("Incumbent changed; refusing to restart this gate.")
    for entry in (old["candidate"], old["opponent"]):
        model = artifact(root, entry["file"])
        if digest(model) != entry["sha256"]:
            raise ValueError("A frozen model's checksum changed; refusing to restart.")
        load_state(model, rules)
    archive = f"gate-archive/{path.stem}-{hashlib.sha256(old_bytes).hexdigest()[:16]}.json"
    archived = artifact(root, archive)
    if archived.exists():
        if archived.read_bytes() != old_bytes:
            raise ValueError("Archived gate bytes changed.")
    else:
        atomic_bytes(archived, old_bytes, cap, root=root)
    report = _gate_report(rules, old["candidate"], old["opponent"], settings, code_sha)
    report.update(opening_seed=old["opening_seed"], restart_from=archive)
    save_json(path, report, root=root, cap=cap)
    return report


def run_gate(root, name, rules, device, deadline, stopped, cap, status=lambda *_: None):
    path = artifact(root, name)
    report = load_json(path)
    if report["rules"] != rules.id:
        raise ValueError("Gate rules changed.")
    if report.get("decision_recorded"):
        return report
    if report["code_sha256"] != _source()[1]:
        raise ValueError("Search code changed during a gate. Restore the saved source or use train --restart-gate to archive and restart it.")
    candidate = network(root, report["candidate"], device, rules)
    opponent = network(root, report["opponent"], device, rules)
    started = time.monotonic()
    elapsed_before = report.get("elapsed_seconds", 0.0)
    while len(report["matches"]) < GATE_GAMES and not stopped() and time.monotonic() < deadline:
        i = len(report["matches"])
        initial = opening(i // 2, report["opening_seed"], rules)
        current = report.get("current") or {"moves": initial.moves, "strategies": [None, None],
                                            "plan": [], "proof": None, "turn_remaining": report["seconds_per_turn"]}
        game = Game.from_moves(current["moves"], rules=rules)
        color = game.player
        slot = 0 if color == 1 else 1
        candidate_color = 1 if i % 2 == 0 else -1
        engine = candidate if color == candidate_color else opponent
        plan, proof, strategy = list(current["plan"]), [current["proof"]], [current["strategies"][slot]]
        remaining = current["turn_remaining"]
        begin = time.monotonic()
        # Yield between placements, not by giving one engine a shortened thinking budget.
        if deadline - begin < remaining / game.left:
            break
        action = turn_action(game, engine, remaining / game.left,
                             report.get("simulations", 100_000), stopped, tactics=True, tss=True, plan=plan,
                             proof_plan=proof, strategy=strategy)
        if stopped():
            break
        game.play(action)
        duration = time.monotonic() - begin
        if plan:
            plan.pop(0)
        current.update(moves=game.moves, plan=plan, proof=proof[0] if plan else None)
        current["strategies"][slot] = strategy[0]
        current["turn_remaining"] = (report["seconds_per_turn"] if game.player != color
                                     else max(0, remaining - duration))
        report["max_turn_overrun_seconds"] = max(report["max_turn_overrun_seconds"], duration - remaining)
        report["current"] = current
        if game.done:
            report["matches"].append({"moves": game.moves, "opening": initial.moves,
                "candidate_color": candidate_color, "score": (game.winner * candidate_color + 1) / 2})
            report["current"] = None
        report["elapsed_seconds"] = elapsed_before + time.monotonic() - started
        gate_summary(report)
        save_json(path, report, root=root, cap=cap)
        status("evaluating", f"Frozen gate: {report['games']}/{GATE_GAMES} games | game {i + 1}, {len(game.moves)} stones.")
    gate_summary(report)
    if len(report["matches"]) == GATE_GAMES:
        _finish_gate(root, path, report, rules, cap)
    return report


def train(args):
    """Start/resume a bounded run. args is supplied by the CLI, not the dashboard."""
    root, rules = Path(args.data).resolve(), args.rules
    with run_lock(root):
        _train(args, root, rules)


def _train(args, root, rules):
    saved = load_state(root / "latest.pt", rules) if (root / "latest.pt").exists() else None
    manifest = load_json(root / "incumbent.json")
    run_info = load_json(root / "run.json") or load_json(root / "rules.json")
    if ((run_info and run_info["rules"] != rules.id)
            or (manifest and manifest.get("rules") != rules.id)):
        raise ValueError("This data directory belongs to different rules.")
    cap = int(args.disk_gib * GIB)
    settings = saved["settings"].copy() if saved else {
        "simulations": args.simulations, "parallel": args.parallel or 64, "batch": args.batch, "seed": args.seed,
        "bootstrap_games": 16, "updates_per_cycle": 32, "replay_limit": 20_000,
        "tactical_ms": args.tactical_ms, "snapshot_every": args.snapshot_every,
        "gate_seconds": args.seconds if args.seconds is not None else .25}
    gate = saved.get("gate") if saved else None
    restart = getattr(args, "restart_gate", False)
    if args.parallel is not None:
        settings["parallel"] = args.parallel
    if args.seconds is not None:
        if gate and not restart and args.seconds != load_json(root / gate)["seconds_per_turn"]:
            raise ValueError("Changing a pending gate's time requires --restart-gate; old results will be archived.")
        settings["gate_seconds"] = args.seconds
    elif restart:
        settings["gate_seconds"] = .25
    environment = _environment(root, cap)
    if restart:
        if not gate:
            raise ValueError("No pending gate to restart. Omit --restart-gate for normal resume.")
        restart_gate(root, gate, rules, settings, environment["code_sha256"], cap)
    if gate:
        report = load_json(root / gate)
        if not report.get("decision_recorded") and report["code_sha256"] != environment["code_sha256"]:
            raise ValueError("Pending gate code changed. Use --restart-gate to archive/restart it, or restore its source.")
    device = device_for(args.device)
    torch.manual_seed(settings["seed"])
    rng = np.random.default_rng(settings["seed"])
    config = saved["config"] if saved else {"size": rules.height, "channels": 32, "blocks": 2}
    net = Network(**config).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=.001, weight_decay=.0001)
    games = steps = since_update = last_milestone = 0
    replay, active = [], []
    pending_games, pending_metrics = [], []
    loss_metrics = None
    phase_seconds = saved.get("phase_seconds", {"training": 0., "evaluation": 0.}) if saved else {"training": 0., "evaluation": 0.}
    if saved:
        net.load_state_dict(saved["weights"])
        optimizer.load_state_dict(saved["optimizer"])
        rng.bit_generator.state = saved["rng"]
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda" and saved.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        games, steps, since_update = saved["games"], saved["step"], saved["since_update"]
        replay, active = saved["replay"], saved["active"]
        last_milestone, gate = saved["last_milestone"], saved["gate"]
        pending_games, pending_metrics = saved["pending_games"], saved["pending_metrics"]
        loss_metrics = saved["loss_metrics"]
        _flush_exports(root, pending_games, pending_metrics, rules, cap)
    boards = [Game.from_moves(row["moves"], rules=rules) for row in active]
    if not run_info or run_info.get("settings") != settings:
        save_json(root / "run.json", {"rules": rules.id, "rule_config": rules.to_dict(),
            "settings": settings, "network": net.config, "environment": environment}, root=root, cap=cap)
    if manifest is None:
        save_json(root / "incumbent.json", {"kind": "heuristic", "rules": rules.id,
            "label": "No milestone model yet; untrained heuristic for preview"}, root=root, cap=cap)
    stop_file = root / "stop"
    stop_file.unlink(missing_ok=True)
    interrupted = [False]
    old_signals = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        old_signals[sig] = signal.signal(sig, lambda *_: interrupted.__setitem__(0, True))
    start = time.monotonic()
    deadline = start + args.hours * 3600
    stopped = lambda: interrupted[0] or stop_file.exists() or time.monotonic() >= deadline
    last_save = last_status = start

    def checkpoint():
        state = {**(saved or {}), **snapshot(net, steps, rules, games)}  # Preserve older checkpoint fields.
        state.update(optimizer=optimizer.state_dict(), settings=settings, since_update=since_update,
            replay=replay, active=active, last_milestone=last_milestone, gate=gate, phase_seconds=phase_seconds,
            pending_games=pending_games, pending_metrics=pending_metrics, loss_metrics=loss_metrics,
            environment=environment, rng=rng.bit_generator.state,
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else None)
        save_checkpoint(root / "latest.pt", state, cap, root=root)
        _flush_exports(root, pending_games, pending_metrics, rules, cap)

    def status(phase, message="Training is CLI-controlled."):
        nonlocal last_status
        now = time.monotonic()
        if phase not in ("paused", "error") and now - last_status < 3:
            return
        last_status = now
        payload = {"phase": phase, "message": message, "rules": rules.id, "device": str(device),
            "games": games, "updates": steps, "replay_positions": len(replay), "loss_metrics": loss_metrics,
            "next_milestone": last_milestone + settings["snapshot_every"], "phase_seconds": phase_seconds.copy(),
            "evaluation_share": phase_seconds["evaluation"] / max(1e-9, sum(phase_seconds.values())),
            "parallel": settings["parallel"],
            "session_seconds": round(now - start, 1), "remaining_seconds": round(max(0, deadline - now), 1),
            "artifact_bytes": usage(root), "cap_bytes": cap, "updated_at": time.time()}
        # Atomic writers reserve 1 MiB for status/recovery metadata, even when the artifact cap is reached.
        save_json(root / "status.json", payload)
        print(f"{phase}: {games} games | {steps} updates | {len(replay)} replay | {message}", flush=True)

    def milestone():
        nonlocal last_milestone, gate
        filename = f"models/model-{games:08d}.pt"
        path = artifact(root, filename)
        if not path.exists():
            frozen = snapshot(net, steps, rules, games)
            frozen.update(environment=environment, settings=settings.copy())
            save_checkpoint(path, frozen, cap, root=root)
        frozen = load_state(path, rules)
        if frozen["step"] != steps or any(not torch.equal(frozen["weights"][key], value.cpu())
                                         for key, value in net.state_dict().items()):
            raise ValueError("An immutable milestone already exists with different weights.")
        candidate = {"id": path.stem, "file": filename, "sha256": digest(path), "games": games, "step": steps}
        opponent = load_json(root / "incumbent.json", {})
        if opponent.get("kind") != "network" or opponent.get("file") == filename:
            save_json(root / "incumbent.json", {**candidate, "kind": "network", "rules": rules.id,
                "initial_baseline": True, "label": "Initial milestone baseline; strength not validated"}, root=root, cap=cap)
        if gate is None:
            gate = next_gate(root, rules, settings, environment["code_sha256"], cap)
        last_milestone = games
        checkpoint()

    try:
        checkpoint()
        while not stopped():
            began, phase = time.monotonic(), "training"
            credit = eval_allowance(phase_seconds)
            if gate is None and credit >= 1:
                gate = next_gate(root, rules, settings, environment["code_sha256"], cap)
            if gate and credit >= max(1., load_json(root / gate)["seconds_per_turn"] + .05):
                phase = "evaluation"
                # ponytail: short serial gate slices; batch evaluation only if loading dominates measured cost.
                until = min(deadline, began + min(credit, max(2., settings["gate_seconds"] + .05)))
                gate_stopped = lambda: stopped() or time.monotonic() >= until
                result = run_gate(root, gate, rules, device, until, gate_stopped, cap, status)
                if result.get("decision_recorded"):
                    gate = None
            elif since_update >= 8 or (games >= last_milestone + settings["snapshot_every"] and since_update):
                # Preserve learning work when a larger batch finishes more than eight games at once.
                pending = settings.get("pending_updates", settings["updates_per_cycle"] * max(1, (since_update + 7) // 8))
                while pending and not stopped():
                    metric = {}
                    train_step(net, optimizer, replay, settings["batch"], rng, device,
                        game_type=lambda board, player, left: Game(board, player, left, rules=rules), metrics=metric)
                    steps += 1
                    pending -= 1
                    settings["pending_updates"] = pending
                    loss_metrics = metric
                    pending_metrics.append({"step": steps, "games": games, **metric})
                    status("optimizing")
                if not pending:
                    since_update = 0
                    settings.pop("pending_updates", None)
                checkpoint()
            elif games >= last_milestone + settings["snapshot_every"]:
                checkpoint()  # Durable weights and dataset precede publishing an immutable snapshot.
                milestone()
            elif args.max_games and games >= args.max_games:
                break
            else:
                while len(active) < settings["parallel"]:
                    game = _new_game(rules, rng)
                    active.append({"moves": game.moves, "opening": game.moves.copy(), "samples": [],
                                   "turns": [], "strategies": [None, None], "plans": [[], []]})
                    boards.append(game)
                batch_boards = boards[:settings["parallel"]]
                slots = [0 if game.player == 1 else 1 for game in batch_boards]
                strategies = [row["strategies"][slot] for row, slot in zip(active, slots)]
                plans = [list(row["plans"][slot]) for row, slot in zip(active, slots)]
                decisions = selfplay_batch(batch_boards, net, settings["simulations"], rng,
                    bootstrap=games < settings["bootstrap_games"], tactical_ms=settings["tactical_ms"],
                    deadline=deadline, stopped=stopped, strategies=strategies, plans=plans, adjudicate_proofs=False)
                if decisions is None or stopped():
                    phase_seconds[phase] += time.monotonic() - began
                    break
                ceiling = last_milestone + settings["snapshot_every"]
                if args.max_games:
                    ceiling = min(ceiling, args.max_games)
                # Lowering --parallel keeps unfinished games; rotate them through the smaller batch.
                survivors, live = active[len(batch_boards):], boards[len(batch_boards):]
                for row, game, decision, slot, strategy, plan in zip(active, boards, decisions, slots, strategies, plans):
                    if games >= ceiling:
                        survivors.append(row)
                        live.append(game)
                        continue
                    action = choose(decision["policy"], rng, temperature=1 if len(game.moves) < 12 else .25)
                    source = decision["source"]
                    row["samples"].extend(selfplay_samples(game, decision))
                    row["turns"].append({"move": action, "step": steps, "source": source})
                    row["strategies"][slot] = strategy
                    row["plans"][slot] = plan[1:] if plan and plan[0] == action else []
                    game.play(action)
                    row["moves"] = game.moves
                    if game.done:
                        for sample in row["samples"]:
                            sample["result"] = float(game.winner * sample["player"])
                        replay.extend(row["samples"])
                        replay = replay[-settings["replay_limit"]:]
                        games += 1
                        since_update += 1
                        record = {"rules": rules.id, "rule_config": rules.to_dict(), "game": games,
                            "seed": settings["seed"], "opening": row["opening"], "moves": game.moves,
                            "winner": game.winner, "complete": True, "turns": row["turns"],
                            "code_sha256": environment["code_sha256"]}
                        pending_games.append({"record": record, "samples": row["samples"]})
                    else:
                        survivors.append(row)
                        live.append(game)
                active, boards = survivors, live
                status("bootstrap" if games < settings["bootstrap_games"] else "self_play")
            phase_seconds[phase] += time.monotonic() - began
            if time.monotonic() - last_save >= 60:
                checkpoint()
                last_save = time.monotonic()
        checkpoint()
        status("paused", "Saved. Run the same CLI command to resume; Ctrl+C requests a safe stop.")
    except Exception as exc:
        status("error", f"{type(exc).__name__}: {exc}. Previous checkpoint retained.")
        raise
    finally:
        for sig, handler in old_signals.items():
            signal.signal(sig, handler)


def evaluate(args):
    """Resume an existing milestone gate. Never train or put evaluation games into replay."""
    root = Path(args.data).resolve()
    run = load_json(root / "run.json")
    if not run:
        raise ValueError("No training run exists in this data directory.")
    rules = Rules(**run["rule_config"])
    with run_lock(root):
        interrupted = [False]
        old = signal.signal(signal.SIGINT, lambda *_: interrupted.__setitem__(0, True))
        deadline = time.monotonic() + args.hours * 3600
        stop = root / "stop"
        stop.unlink(missing_ok=True)
        stopped = lambda: interrupted[0] or stop.exists() or time.monotonic() >= deadline
        try:
            report = run_gate(root, args.report, rules, device_for(args.device), deadline,
                              stopped, int(args.disk_gib * GIB))
            print(json.dumps(gate_summary(report), indent=2))
        finally:
            signal.signal(signal.SIGINT, old)
