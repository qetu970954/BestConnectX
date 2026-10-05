"""Training lifecycle: native terminal self-play, bounded replay, milestones, and frozen gates.

The game engine has no training dependency. The dashboard reads the JSON artifacts.
"""
import hashlib
import json
from pathlib import Path
import platform
import signal
import sys
import time
import numpy as np
import torch
from .network import Network, device_for
from .storage import (GIB, atomic_bytes, load_checkpoint, load_json, run_lock,
                      save_checkpoint, save_json, unlink, usage)
from .selfplay import selfplay_samples, replay_symmetries, train_step
from .runtime import DEFAULT_RULES, Game, Rules, selfplay_batch, turn_action, opening as native_opening, backend as native_backend
from . import native

GATE_GAMES = 100
PROMOTION_SCORE = .55
SOURCE_FILES = ["engine/game.py", "engine/training.py", "engine/network.py", "engine/search.py",
                "engine/selfplay.py", "engine/tactics.py", "engine/tss.py", "engine/storage.py",
                "engine/native.py", "engine/native.cpp", "engine/runtime.py", "engine/runtime.cpp",
                "engine/runtime.h", "engine/inference.cpp", "engine/config.py"]
PROJECT = Path(__file__).resolve().parent.parent


def artifact(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Artifact must stay inside this run's data directory.")
    return path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _replay_window(state):
    """Replay is checkpoint-owned; never rebuild it by scanning game archives."""
    return state.get('replay', [])[-state['settings'].get('replay_limit', 400_000):]


def load_state(path, rules=None):
    state = load_checkpoint(path)
    recorded = Rules(**state["rule_config"]) if "rule_config" in state else None
    if (state.get("format") != 2 or recorded is None or state.get("rules") != recorded.id
            or state["config"]["size"] != recorded.height
            or state["config"].get("width", state["config"]["size"]) != recorded.width
            or (rules is not None and rules != recorded)):
        raise ValueError("Checkpoint rules do not match this run. Use a separate --data directory.")
    return state


def snapshot(net, step, rules=DEFAULT_RULES, games=0):
    return {"format": 2, "config": net.config, "step": step,
            "weights": {k: v.detach().cpu() for k, v in net.state_dict().items()},
            "rules": rules.id, "rule_config": rules.to_dict(), "games": games}


def network(root, entry, device, rules, *, cache=None):
    path = artifact(root, entry["file"])
    sha = digest(path) if entry.get("sha256") or cache is not None else None
    if entry.get("sha256") and sha != entry["sha256"]:
        raise ValueError("A frozen model's checksum changed; refusing to use it.")
    key = (path, sha, str(device), rules)
    if cache is not None and key in cache:
        return cache[key]
    state = load_state(path, rules)
    with torch.random.fork_rng(devices=[]):
        net = Network(**state["config"]).to(device)
    net.load_state_dict(state["weights"])
    net.eval()
    if cache is not None:
        cache[key] = net
    return net


def _code_sha256():
    return hashlib.sha256(b"".join(name.encode() + (PROJECT / name).read_bytes()
                                  for name in SOURCE_FILES)).hexdigest()


def _environment():
    native_backend()  # Refuse a production run without the native engine.
    return {'python': sys.version, 'platform': platform.platform(), 'torch': str(torch.__version__),
            'numpy': np.__version__, 'cuda_build': torch.version.cuda, 'code_sha256': _code_sha256(),
            'native_backend': {'kind': 'cpp-libtorch', 'binary_sha256': digest(native.library_path())}}


def _immutable_json(root, name, value, cap):
    path = artifact(root, name)
    if path.exists():
        if load_json(path) != value:
            raise ValueError(f"Refusing to overwrite archived record: {name}")
    else:
        save_json(path, value, root=root, cap=cap)


def _flush_exports(root, games, metrics, cap):
    # Checkpoint first. Summary/metric writes can be retried after an interrupted save.
    save_json(root / 'selfplay-stats.json', {'window': 1000, 'summaries': games[-1000:]}, root=root, cap=cap)
    if metrics:
        name = f"metrics/updates-{metrics[0]['step']:09d}-{metrics[-1]['step']:09d}.json"
        _immutable_json(root, name, metrics, cap)
    metrics.clear()


def opening(index, seed, rules=DEFAULT_RULES):
    """Legal, held-out starts. A pair shares the start; engine colors, not stones, swap."""
    return native_opening(index, seed, rules)


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
    return report["score"] >= PROMOTION_SCORE and report["max_turn_overrun_seconds"] <= .1


def _export_best(root, entry, cap):
    """Export the authoritative manifest's frozen model; retry a failed mirror write on resume."""
    if not entry.get("file"):
        return
    payload = artifact(root, entry["file"]).read_bytes()
    sha = hashlib.sha256(payload).hexdigest()
    if entry.get("sha256") and sha != entry["sha256"]:
        raise ValueError("A frozen model's checksum changed; refusing to export best.")
    path = root / "best.pt"
    if not path.exists() or digest(path) != sha:
        atomic_bytes(path, payload, cap, root=root)


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
                    "label": f"Accepted model (provisional) · {report['candidate']['games']} self-play games",
                    "gate": path.name, "previous": {k: v for k, v in previous.items() if k != "previous"}}
            save_json(root / "incumbent.json", best, root=root, cap=cap)
        else:
            raise ValueError("Incumbent changed during a frozen gate; refusing promotion.")
        _export_best(root, report["candidate"], cap)
        report["promoted"] = True
    report["decision_recorded"] = True
    save_json(path, report, root=root, cap=cap)


def _gate_report(rules, candidate, opponent, settings, code_sha):
    report = {"rules": rules.id, "rule_config": rules.to_dict(), "candidate": candidate,
        "opponent": {k: v for k, v in opponent.items() if k != "previous"},
        "opening_seed": 90_000_000 + candidate["games"], "seconds_per_turn": settings["gate_seconds"],
        "simulations": settings["simulations"], "promotion_score_threshold": PROMOTION_SCORE,
        "matches": [], "current": None,
        "max_turn_overrun_seconds": 0., "elapsed_seconds": 0., "promoted": False, "code_sha256": code_sha}
    gate_summary(report)
    return report


def next_gate(root, rules, settings, code_sha, cap):
    """Compare the newest frozen milestone with best; never drain an old backlog."""
    opponent = load_json(root / "incumbent.json", {})
    if opponent.get("kind") != "network":
        return None
    path = max((root / "models").glob("model-*.pt"), default=None,
               key=lambda p: int(p.stem.rsplit("-", 1)[-1]))
    if path is None:
        return None
    games = int(path.stem.rsplit("-", 1)[-1])
    if games <= opponent["games"]:
        return None
    name = f"gate-{path.stem}.json"
    report = load_json(root / name)
    if report is None:
        state = load_state(path, rules)
        candidate = {"id": path.stem, "file": f"models/{path.name}", "sha256": digest(path),
                     "games": games, "step": state["step"]}
        save_json(root / name, _gate_report(rules, candidate, opponent, settings, code_sha), root=root, cap=cap)
        return name
    return name if not report.get("decision_recorded") else None


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


def run_gate(root, name, rules, device, deadline, stopped, cap, status=lambda *_: None, *, model_cache=None):
    path = artifact(root, name)
    report = load_json(path)
    if report["rules"] != rules.id:
        raise ValueError("Gate rules changed.")
    if report.get("decision_recorded"):
        return report
    if report["code_sha256"] != _code_sha256():
        raise ValueError("Search code changed during a gate. Restore the saved source or use train --restart-gate to archive and restart it.")
    candidate = network(root, report["candidate"], device, rules, cache=model_cache)
    opponent = network(root, report["opponent"], device, rules, cache=model_cache)
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
        budget = max(0., remaining) / game.left
        begin = time.monotonic()
        # Yield between placements, not by giving one engine a shortened thinking budget.
        if deadline - begin < budget:
            break
        action = turn_action(game, engine, budget,
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
        # Keep negative time between stones so overtime accumulates across slices/resumes.
        current["turn_remaining"] = (report["seconds_per_turn"] if game.player != color
                                     else remaining - duration)
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
        'bootstrap_games': getattr(args, 'bootstrap_games', 16),
        'updates_per_cycle': getattr(args, 'updates_per_cycle', 32),
        'replay_limit': getattr(args, 'replay_limit', 400_000),
        'learning_rate': getattr(args, 'learning_rate', .001), 'workers': getattr(args, 'workers', None) or 6,
        "tactical_ms": args.tactical_ms, "snapshot_every": args.snapshot_every,
        "gate_seconds": args.seconds if args.seconds is not None else .25}
    gate = saved.get("gate") if saved else None
    restart = getattr(args, "restart_gate", False)
    explicit = getattr(args, 'explicit', {'parallel', 'workers', 'seconds'})
    if args.parallel is not None and (not saved or 'parallel' in explicit):
        settings['parallel'] = args.parallel
    if getattr(args, 'workers', None) is not None and (not saved or 'workers' in explicit):
        settings['workers'] = args.workers
    if args.seconds is not None:
        if gate and not restart and args.seconds != load_json(root / gate)["seconds_per_turn"]:
            raise ValueError("Changing a pending gate's time requires --restart-gate; old results will be archived.")
        settings["gate_seconds"] = args.seconds
    elif restart:
        settings["gate_seconds"] = .25
    environment = _environment()
    if restart:
        if not gate:
            raise ValueError("No pending gate to restart. Omit --restart-gate for normal resume.")
        restart_gate(root, gate, rules, settings, environment["code_sha256"], cap)
    if gate:
        report = load_json(root / gate)
        if not report.get("decision_recorded") and report["code_sha256"] != environment["code_sha256"]:
            raise ValueError("Pending gate code changed. Use --restart-gate to archive/restart it, or restore its source.")
    if manifest and manifest.get("kind") == "network":
        _export_best(root, manifest, cap)
    device = device_for(args.device)
    torch.manual_seed(settings["seed"])
    rng = np.random.default_rng(settings["seed"])
    config = saved['config'] if saved else {'size': rules.height, 'channels': getattr(args, 'channels', 64),
        'blocks': getattr(args, 'blocks', 6), 'architecture': getattr(args, 'architecture', 'residual')}
    net = Network(**config).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=settings.get('learning_rate', .001), weight_decay=.0001)
    games = steps = since_update = last_milestone = 0
    replay, active = [], []
    summaries, pending_metrics = [], []
    loss_metrics = None
    timings = {'cpu_search_seconds': 0., 'inference_seconds': 0., 'learning_seconds': 0., 'checkpoint_seconds': 0.}
    phase_seconds = saved.get("phase_seconds", {"training": 0., "evaluation": 0.}) if saved else {"training": 0., "evaluation": 0.}
    if saved:
        net.load_state_dict(saved["weights"])
        optimizer.load_state_dict(saved["optimizer"])
        rng.bit_generator.state = saved["rng"]
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda" and saved.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        games, steps, since_update = saved["games"], saved["step"], saved["since_update"]
        replay = _replay_window(saved)
        active = saved["active"]
        last_milestone, gate = saved["last_milestone"], saved["gate"]
        summaries, pending_metrics = saved.get('summaries', []), saved['pending_metrics']
        loss_metrics = saved['loss_metrics']
        timings.update(saved.get('timings', {}))
        _flush_exports(root, summaries, pending_metrics, cap)
    boards = [Game.from_moves(row["moves"], rules=rules) for row in active]
    if not run_info or run_info.get("settings") != settings or run_info.get("environment") != environment:
        save_json(root / "run.json", {"rules": rules.id, "rule_config": rules.to_dict(),
            "settings": settings, "network": net.config, "environment": environment}, root=root, cap=cap)
    if manifest is None:
        save_json(root / "incumbent.json", {"kind": "heuristic", "rules": rules.id,
            "label": "No milestone model yet; untrained heuristic for preview"}, root=root, cap=cap)
    stop_file = root / "stop"
    unlink(stop_file, missing_ok=True)
    interrupted = [False]
    old_signals = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        old_signals[sig] = signal.signal(sig, lambda *_: interrupted.__setitem__(0, True))
    start = time.monotonic()
    deadline = start + args.hours * 3600
    stopped = lambda: interrupted[0] or stop_file.exists() or time.monotonic() >= deadline
    last_status = start
    checkpoint_needed = True

    def checkpoint():
        nonlocal checkpoint_needed
        begin = time.perf_counter()
        state = {**(saved or {}), **snapshot(net, steps, rules, games)}  # Preserve older checkpoint fields.
        state.update(replay=replay, summaries=summaries, timings=timings,
            optimizer=optimizer.state_dict(), settings=settings, since_update=since_update,
            active=active, last_milestone=last_milestone, gate=gate, phase_seconds=phase_seconds,
            pending_metrics=pending_metrics, loss_metrics=loss_metrics,
            environment=environment, rng=rng.bit_generator.state,
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else None)
        save_checkpoint(root / "latest.pt", state, cap, root=root)
        _flush_exports(root, summaries, pending_metrics, cap)
        timings['checkpoint_seconds'] += time.perf_counter() - begin
        checkpoint_needed = False

    def status(phase, message=""):
        nonlocal last_status
        now = time.monotonic()
        if phase not in ("paused", "error") and now - last_status < 3:
            return
        last_status = now
        payload = {"phase": phase, "message": message, "rules": rules.id, "device": str(device),
            "games": games, "updates": steps, "replay_positions": len(replay), "loss_metrics": loss_metrics,
            "next_milestone": last_milestone + settings["snapshot_every"], "phase_seconds": phase_seconds.copy(),
            "evaluation_share": phase_seconds["evaluation"] / max(1e-9, sum(phase_seconds.values())),
            'parallel': settings['parallel'], 'workers': settings['workers'], 'network': net.config,
            'timings': timings.copy(), 'backend': 'cpp-libtorch',
            "session_seconds": round(now - start, 1), "remaining_seconds": round(max(0, deadline - now), 1),
            "artifact_bytes": usage(root), "cap_bytes": cap, "updated_at": time.time()}
        # Atomic writers reserve 1 MiB for status/recovery metadata, even when the artifact cap is reached.
        save_json(root / "status.json", payload)
        progress = f"{phase}: {games} games | {steps} updates | {len(replay)} replay"
        if loss_metrics:
            progress += (f" | loss {loss_metrics['loss']:.4f}"
                         f" (policy {loss_metrics['policy_loss']:.4f}, value {loss_metrics['value_loss']:.4f})")
        progress += f" | next {payload['next_milestone']} | {payload['remaining_seconds']:.0f}s left"
        print(progress + (f" | {message}" if message else ""), flush=True)

    def milestone(*, save=True):
        nonlocal last_milestone
        last_milestone = games
        if save:
            checkpoint()  # Full recovery precedes publication; resume retries an interrupted export.
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
        if opponent.get("kind") != "network":
            save_json(root / "incumbent.json", {**candidate, "kind": "network", "rules": rules.id,
                "initial_baseline": True, "label": "Initial milestone baseline; strength not validated"}, root=root, cap=cap)
            _export_best(root, candidate, cap)

    try:
        if (saved and games and games == last_milestone
                and (not (root / f"models/model-{games:08d}.pt").exists()
                     or (manifest or {}).get("kind") != "network")):
            milestone(save=False)
        while not stopped():
            began, phase = time.monotonic(), "training"
            if gate is None:
                gate = next_gate(root, rules, settings, environment["code_sha256"], cap)
            if gate:
                phase = "evaluation"
                result = run_gate(root, gate, rules, device, deadline, stopped, cap, status)
                if result.get("decision_recorded"):
                    gate = None
                else:
                    checkpoint_needed = True
                    phase_seconds[phase] += time.monotonic() - began
                    break
            elif since_update >= 8 or (games >= last_milestone + settings["snapshot_every"] and since_update):
                checkpoint_needed = True
                # Preserve learning work when a larger batch finishes more than eight games at once.
                pending = settings.get("pending_updates", settings["updates_per_cycle"] * max(1, (since_update + 7) // 8))
                while pending and not stopped():
                    metric = {}
                    learn_started = time.perf_counter()
                    train_step(net, optimizer, replay, settings['batch'], rng, device, rules=rules, metrics=metric)
                    timings['learning_seconds'] += time.perf_counter() - learn_started
                    steps += 1
                    pending -= 1
                    settings["pending_updates"] = pending
                    loss_metrics = metric
                    pending_metrics.append({"step": steps, "games": games, **metric})
                    status("optimizing")
                if not pending:
                    since_update = 0
                    settings.pop("pending_updates", None)
            elif games >= last_milestone + settings["snapshot_every"]:
                milestone()
            elif args.max_games and games >= args.max_games:
                break
            else:
                checkpoint_needed = True
                while len(active) < settings["parallel"]:
                    game = native_opening(0, int(rng.integers(0, 2**63)), rules, random_start=True)
                    active.append({'moves': game.moves, 'samples': [], 'source_counts': {},
                                   'strategies': [None, None], 'plans': [[], []]})
                    boards.append(game)
                batch_boards = boards[:settings["parallel"]]
                slots = [0 if game.player == 1 else 1 for game in batch_boards]
                strategies = [row["strategies"][slot] for row, slot in zip(active, slots)]
                plans = [list(row["plans"][slot]) for row, slot in zip(active, slots)]
                measured = {}
                decisions = selfplay_batch(batch_boards, net, settings["simulations"], rng,
                    bootstrap=games < settings["bootstrap_games"], tactical_ms=settings["tactical_ms"],
                    deadline=deadline, stopped=stopped, strategies=strategies, plans=plans,
                    workers=settings['workers'], metrics=measured)
                if decisions is None or stopped():
                    phase_seconds[phase] += time.monotonic() - began
                    break
                for key, value in measured.items():
                    timings[key] += value
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
                    action = decision['action']
                    source = decision['source']
                    row['samples'].extend(selfplay_samples(game, decision))
                    row['source_counts'][source] = row['source_counts'].get(source, 0) + 1
                    row["strategies"][slot] = strategy
                    row["plans"][slot] = plan[1:] if plan and plan[0] == action else []
                    game.play(action)
                    row["moves"] = game.moves
                    if game.done:
                        for sample in row["samples"]:
                            sample["result"] = float(game.winner * sample["player"])
                        replay.extend(replay_symmetries(row['samples'], rules.height))
                        replay = replay[-settings["replay_limit"]:]
                        games += 1
                        since_update += 1
                        length = len(game.moves)
                        turns = 1 + max(0, (length-rules.starter_stones+rules.stones_per_turn-1)//rules.stones_per_turn)
                        summaries.append({'game': games, 'winner': game.winner, 'placements': length,
                                          'turns': turns, 'source_counts': row['source_counts']})
                        summaries = summaries[-1000:]
                    else:
                        survivors.append(row)
                        live.append(game)
                active, boards = survivors, live
                status("bootstrap" if games < settings["bootstrap_games"] else "self_play")
            phase_seconds[phase] += time.monotonic() - began
        if checkpoint_needed:
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
        unlink(stop, missing_ok=True)
        stopped = lambda: interrupted[0] or stop.exists() or time.monotonic() >= deadline
        try:
            report = run_gate(root, args.report, rules, device_for(args.device), deadline,
                              stopped, int(args.disk_gib * GIB))
            print(json.dumps(gate_summary(report), indent=2))
        finally:
            signal.signal(signal.SIGINT, old)
