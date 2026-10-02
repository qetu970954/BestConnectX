"""Bounded, resumable self-play. Every learned parameter originates in this run."""
import argparse
import hashlib
import math
from pathlib import Path
import signal
import time
import numpy as np
import torch
from .game import Game, heuristic, covers
from .network import Network, device_for, augment
from .search import search, choose
from .tactics import forcing_win, verifies
from .tss import search_tss, verify_tss
from .storage import (GIB, usage, save_json, load_json, save_checkpoint,
                      load_checkpoint, run_lock)


def model_state(net, step):
    return {"format": 1, "config": net.config, "step": step,
            "weights": {k: v.detach().cpu() for k, v in net.state_dict().items()}}


def read_network(path, device, *, size=19):
    state = load_checkpoint(path)
    if state["config"].get("size", 19) != size:
        raise ValueError("Checkpoint board size does not match this game.")
    net = Network(**state["config"]).to(device)
    net.load_state_dict(state["weights"])
    net.eval()
    return net


def observation(game, policy):
    return {"board": torch.from_numpy(game.board.copy()), "player": game.player,
            "left": game.left, "policy": torch.from_numpy(policy.astype(np.float16))}


def selfplay_batch(boards, net, simulations, rng, *, bootstrap=False, tactical_ms=2,
                   deadline=None, stopped=lambda: False, tss_enabled=True, strategies=None,
                   adjudicate_proofs=True, plans=None):
    """Use exact root proofs/forced actions; batch only unresolved roots on the GPU.

    With adjudicate_proofs=True, a proof ends self-play with a certified outcome.
    With False, verified proofs select placements; the caller plays to a real terminal.
    A proved loss has a value-only target when adjudication is enabled.
    None means cancellation; callers must not partially commit an interrupted batch.
    """
    decisions, unresolved = [], []
    for i, game in enumerate(boards):
        if stopped() or (deadline is not None and time.monotonic() >= deadline):
            return None
        if game.done:
            raise ValueError("Self-play expects nonterminal positions.")
        own = game.threats(game.player, game.left)
        enemy = game.threats(-game.player)
        proof = tuple(sorted(min(own, key=lambda e: (len(e), tuple(sorted(e)))))) if own else None
        if proof is None and plans is not None and plans[i]:
            if verifies(game, plans[i]):
                proof = tuple(plans[i])
            else:
                plans[i] = []
        if (adjudicate_proofs and not own and enemy
                and not covers(enemy, game.left)):
            decisions.append({"source": "proof_loss", "winner": -game.player})
            continue
        if proof is None and tactical_ms > 0:
            if tss_enabled and strategies is not None and strategies[i] is not None:
                plan, checked_plan, cached = [], [None], [strategies[i]]
                expires = time.perf_counter() + tactical_ms / 1000
                if _resume_tss_strategy(game, cached, plan, checked_plan, expires, stopped):
                    strategies[i] = cached[0]
                    if adjudicate_proofs:
                        decisions.append({"source": "tss_win", "winner": game.player,
                                          "proof": plan, "certificate": checked_plan[0]})
                    else:
                        policy = np.zeros(game.board.size, dtype=np.float32)
                        policy[plan[0]] = 1
                        decisions.append({"source": "tss_move", "policy": policy,
                                          "proof": plan})
                    continue
                strategies[i] = cached[0]
            # Reserve the final 10% of this root's budget to independently check a TSS proof.
            expires = time.perf_counter() + tactical_ms / 1000
            search_expires = time.perf_counter() + tactical_ms * .9 / 1000
            is_stopped = lambda until: lambda: stopped() or time.perf_counter() >= until
            proof_tree = None
            if tss_enabled:
                # Preserve cheap fork proofs before spending the remaining budget on recursion.
                for horizon in (2, 3):
                    if time.perf_counter() >= search_expires:
                        break
                    proof_tree = search_tss(game, max_turns=horizon, deadline=search_expires,
                                            max_nodes=10_000, max_candidates=1_000,
                                            stopped=is_stopped(search_expires))
                    if proof_tree is not None:
                        break
            if proof_tree is not None:
                checked = verify_tss(game, proof_tree, deadline=expires,
                                     max_nodes=10_000, stopped=is_stopped(expires))
                if checked is False:
                    raise RuntimeError("Rejected an invalid self-play TSS proof.")
                if checked is True:
                    if strategies is not None:
                        strategies[i] = {"history": list(game.moves), "proof": proof_tree}
                    if adjudicate_proofs:
                        decisions.append({"source": "tss_win", "winner": game.player,
                                          "proof": proof_tree["tree"]["moves"],
                                          "certificate": proof_tree,
                                          "samples": _tss_samples(game, proof_tree)})
                    else:
                        policy = np.zeros(game.board.size, dtype=np.float32)
                        policy[proof_tree["tree"]["moves"][0]] = 1
                        decisions.append({"source": "tss_move", "policy": policy,
                                          "proof": proof_tree["tree"]["moves"]})
                    continue
            proof = forcing_win(game, deadline=deadline if deadline is not None else math.inf,
                                stopped=is_stopped(expires))
        if proof is not None:
            if not verifies(game, proof):
                raise RuntimeError("Rejected an invalid self-play tactical certificate.")
            if adjudicate_proofs:
                decisions.append({"source": "proof_win", "winner": game.player, "proof": proof})
            else:
                policy = np.zeros(game.board.size, dtype=np.float32)
                policy[proof[0]] = 1
                if plans is not None:
                    plans[i] = list(proof)
                decisions.append({"source": "proof_move", "policy": policy,
                                  "proof": list(proof)})
            continue
        legal = game.actions()
        if len(legal) == 1 or bootstrap:
            policy = np.zeros(game.board.size, dtype=np.float32)
            policy[int(legal[0]) if len(legal) == 1 else heuristic(game)] = 1
            decisions.append({"source": "forced" if len(legal) == 1 else "heuristic", "policy": policy})
        else:
            decisions.append(None)
            unresolved.append(i)
    if stopped() or (deadline is not None and time.monotonic() >= deadline):
        return None
    if unresolved:
        policies, _ = search([boards[i] for i in unresolved], net, simulations, rng=rng,
                             deadline=deadline, stopped=stopped)
        for i, policy in zip(unresolved, policies):
            decisions[i] = {"source": "mcts", "policy": policy}
    if stopped() or (deadline is not None and time.monotonic() >= deadline):
        return None
    return decisions


def _tss_samples(game, certificate):
    proof = certificate["tree"]["moves"]
    state, samples = game.copy(), []
    for index, move in enumerate(proof):
        policy = np.zeros(game.board.size, dtype=np.float32)
        policy[move] = 1
        sample = observation(state, policy)
        sample.update(source="tss_win", proof=list(proof[index:]), tss_certificate=certificate,
                      tss_root={"board": torch.from_numpy(game.board.copy()),
                                "player": game.player, "left": game.left}, tss_index=index)
        samples.append(sample)
        state.play(move)
    return samples


def verify_tss_sample(sample, *, deadline=None, game_type=Game):
    """Recheck retained TSS evidence and its exact same-turn placement target."""
    certificate, root_data = sample.get("tss_certificate"), sample.get("tss_root")
    if not isinstance(certificate, dict) or not isinstance(root_data, dict):
        return False
    try:
        root = game_type(root_data["board"].numpy().astype(np.int8, copy=True),
                    root_data["player"], root_data["left"])
        index = sample["tss_index"]
        moves = certificate["tree"]["moves"]
        if type(index) is not int or not 0 <= index < len(moves) or sample["proof"] != moves[index:]:
            return False
        checked = verify_tss(root, certificate,
                             deadline=time.perf_counter() + 1 if deadline is None else deadline,
                             max_nodes=10_000)
        if checked is not True:
            return False
        target = root.copy()
        for move in moves[:index]:
            target.play(move)
        return (np.array_equal(target.board, sample["board"].numpy())
                and target.player == sample["player"] and target.left == sample["left"]
                and int(sample["policy"].argmax()) == moves[index]
                and float(sample["policy"].sum()) == 1.0)
    except (KeyError, TypeError, ValueError, AttributeError):
        return False


def selfplay_samples(game, decision):
    """Build samples for one decision without mutating its input position."""
    source = decision["source"]
    if source in ("proof_win", "tss_win"):
        proof = decision.get("proof")
        if decision.get("winner") != game.player:
            raise ValueError("Invalid winning supervision.")
        if source == "proof_win":
            if not isinstance(proof, (list, tuple)) or not verifies(game, proof):
                raise ValueError("Invalid winning supervision.")
            state, samples = game.copy(), []
            for index, move in enumerate(proof):
                policy = np.zeros(game.board.size, dtype=np.float32)
                policy[move] = 1
                sample = observation(state, policy)
                sample.update(source=source, proof=list(proof[index:]))
                samples.append(sample)
                state.play(move)
            return samples
        certificate = decision.get("certificate")
        if (not isinstance(certificate, dict) or proof != certificate.get("tree", {}).get("moves")
                or verify_tss(game, certificate, deadline=time.perf_counter() + 1,
                              max_nodes=10_000) is not True):
            raise ValueError("Invalid TSS winning supervision.")
        return _tss_samples(game, certificate)
    policy = np.zeros(game.board.size, dtype=np.float32) if source == "proof_loss" else decision["policy"]
    sample = observation(game, policy)
    sample["source"] = source
    return [sample]


def train_step(net, optimizer, replay, batch_size, rng, device, *, game_type=Game, metrics=None):
    batch = [replay[int(i)] for i in rng.integers(len(replay), size=batch_size)]
    boards = [game_type(row["board"].numpy(), row["player"], row["left"]) for row in batch]
    x = np.stack([g.features() for g in boards])
    p = np.stack([row["policy"].numpy().astype(np.float32) for row in batch])
    x, p = augment(x, p, rng)
    inputs = torch.from_numpy(x).to(device)
    target = torch.from_numpy(p).to(device)
    z = torch.tensor([row["result"] for row in batch], device=device)
    net.train()
    optimizer.zero_grad(set_to_none=True)
    # BF16 avoids FP16 loss-scaling issues on Blackwell; CPU stays float32.
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        logits, value = net(inputs)
        occupied = (inputs[:, 0] + inputs[:, 1]).flatten(1) > 0
        logits = logits.float().masked_fill(occupied, -1e9)
        # Exact losing positions teach value only, not a preference among losing actions.
        policy_rows = -(target * logits.log_softmax(1)).sum(1)
        policy_loss = policy_rows.sum() / (target.sum(1) > 0).sum().clamp(min=1)
        value_loss = (value.float() - z).square().mean()
        loss = policy_loss + value_loss
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite training loss; refusing to save corrupted weights.")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 5, error_if_nonfinite=True)
    optimizer.step()
    if metrics is not None:
        metrics.update(loss=float(loss.detach()), policy_loss=float(policy_loss.detach()),
                       value_loss=float(value_loss.detach()))
    return float(loss.detach())


def confidence(scores):
    """Opening-pair bootstrap. Each score averages two color-swapped games."""
    if len(scores) < 200:
        return 0.0
    rng = np.random.default_rng(421337)
    values = np.asarray(scores, dtype=float)
    means = values[rng.integers(len(values), size=(10000, len(values)))].mean(1)
    return float(np.quantile(means, .025))


def opening(index, generation):
    # Independent of self-play RNG; fresh suite per final gate, never fed to replay.
    rng = np.random.default_rng(90000000 + generation * 1000 + index)
    game = Game()
    game.play(180)
    cells = [r * 19 + c for r in range(6, 13) for c in range(6, 13) if r * 19 + c != 180]
    for action in rng.choice(cells, 4, replace=False):
        game.play(int(action))
    return game


def _resume_tss_strategy(game, strategy, plan, proof_plan, deadline, stopped):
    cached = strategy[0]
    if cached is None:
        return False
    if not isinstance(cached, dict) or set(cached) != {"history", "proof"}:
        strategy[0] = None
        return False
    proof, history = cached["proof"], cached["history"]
    if not isinstance(proof, dict) or not isinstance(history, list):
        strategy[0] = None
        return False
    if game.player != proof.get("attacker"):
        return False
    if (len(game.moves) < len(history) or game.moves[:len(history)] != history
            or any(type(move) is not int for move in history)):
        strategy[0] = None
        return False
    try:
        root = game.empty()
        for move in history:
            root.play(move)
    except ValueError:
        strategy[0] = None
        return False
    if (root.player != game.player
            or verify_tss(root, proof, deadline=deadline, max_nodes=20_000,
                          stopped=stopped) is not True):
        strategy[0] = None
        return False
    attack = proof["tree"]["moves"]
    suffix = game.moves[len(history):]
    if len(suffix) < len(attack):
        if suffix != attack[:len(suffix)]:
            strategy[0] = None
            return False
        remainder = attack[len(suffix):]
        resumed = {**proof, "tree": {**proof["tree"], "moves": remainder}}
        if verify_tss(game, resumed, deadline=deadline, max_nodes=20_000,
                      stopped=stopped) is not True:
            strategy[0] = None
            return False
        plan[:] = remainder
        proof_plan[0] = resumed
        return bool(plan)
    if suffix[:len(attack)] != attack or len(suffix) == len(attack):
        return False
    response_moves = suffix[len(attack):]
    response = next((row for row in proof["tree"]["responses"]
                     if row["moves"] == response_moves), None)
    if response is None or proof["max_turns"] < 2:
        strategy[0] = None
        return False
    continuation = {"version": proof["version"], "attacker": proof["attacker"],
                    "max_turns": proof["max_turns"] - 1, "tree": response["proof"]}
    if verify_tss(game, continuation, deadline=deadline, max_nodes=20_000,
                  stopped=stopped) is not True:
        strategy[0] = None
        return False
    strategy[0] = {"history": list(game.moves), "proof": continuation}
    plan[:] = continuation["tree"]["moves"]
    proof_plan[0] = continuation
    return bool(plan)


def turn_action(game, net, seconds, simulations, stopped, *, tactics=False, tss=False,
                plan=None, proof_plan=None, strategy=None):
    """Caller removes plan[0] only AFTER committing the returned placement."""
    started = time.monotonic()
    highres_started = time.perf_counter()
    if tss and (plan is None or proof_plan is None):
        raise ValueError("TSS play requires caller-owned turn and proof plans.")
    if plan:
        if proof_plan is not None and proof_plan[0] is not None:
            saved = proof_plan[0]
            root = dict(saved["tree"])
            root["moves"] = list(plan)
            resumed = {**saved, "tree": root}
            checked = verify_tss(game, resumed,
                                 deadline=highres_started + min(.05, max(.002, seconds * .1)),
                                 stopped=stopped)
            if checked is True:
                return plan[0]
            plan.clear()
            proof_plan[0] = None
        elif verifies(game, plan):
            return plan[0]
        else:
            raise ValueError("Stored tactical plan is not valid for this position.")
    if tss and strategy is not None and not plan:
        expires = highres_started + min(.05, max(.002, seconds * .1))
        if _resume_tss_strategy(game, strategy, plan, proof_plan, expires, stopped):
            return plan[0]
    if (tactics or tss) and seconds > .02:
        if plan is None:
            raise ValueError("Tactical search requires a caller-owned full-turn plan.")
        budget = min(.05, seconds * .1)
        tactical_deadline = started + budget
        if tss:
            proof = search_tss(game, max_turns=3, deadline=highres_started + budget * .7,
                               max_nodes=20_000, max_candidates=1_000, stopped=stopped)
            if proof is not None:
                checked = verify_tss(game, proof, deadline=highres_started + budget,
                                     stopped=stopped)
                if checked is True:
                    plan[:] = proof["tree"]["moves"]
                    proof_plan[0] = proof
                    if strategy is not None:
                        cached = strategy[0]
                        cached_proof = cached.get("proof") if isinstance(cached, dict) else None
                        if (cached is None or not isinstance(cached_proof, dict)
                                or cached_proof.get("attacker") == game.player):
                            strategy[0] = {"history": list(game.moves), "proof": proof}
                    return plan[0]
        if tactics:
            proof = forcing_win(game, deadline=tactical_deadline, stopped=stopped)
            if proof:
                plan[:] = proof
                return plan[0]
    seconds = max(0, seconds - (time.monotonic() - started))
    if net is None or seconds <= .02 or game.threats(game.player, game.left):
        return heuristic(game)
    legal = game.actions()
    if len(legal) == 1:
        return int(legal[0])
    # Leave a small margin for result handling; a running GPU kernel is not preemptible.
    policies, _ = search([game], net, simulations, deadline=time.monotonic() + seconds - .02,
                         stopped=stopped)
    return choose(policies[0])


def run(args):
    root = Path(args.data).resolve()
    with run_lock(root):
        _run(args, root)


def _run(args, root):
    cap = int(args.disk_gib * GIB)
    stop_file = root / "stop"
    if not args.keep_stop:
        stop_file.unlink(missing_ok=True)
    device = device_for(args.device)
    deadline = time.monotonic() + args.hours * 3600
    interrupted = False

    def interrupt(*_):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    stopped = lambda: interrupted or stop_file.exists() or time.monotonic() >= deadline
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    saved = load_checkpoint(root / "latest.pt") if (root / "latest.pt").exists() else None
    if saved and (saved["format"] != 1 or saved.get("rules", "connect6") != "connect6"):
        raise ValueError("Not a Connect6 checkpoint. Use a separate data directory.")
    settings = saved["settings"] if saved else {
        "simulations": args.simulations, "parallel": args.parallel, "batch": args.batch,
        "bootstrap_games": 32, "gate_every": 256, "replay_limit": 50000,
        "updates_per_cycle": 32, "seed": args.seed}
    # Older checkpoints keep their weights/replay/gates and adopt tactical self-play on resume.
    settings.setdefault("tactical_ms", args.tactical_ms)
    net = Network(**(saved["config"] if saved else {"channels": 64, "blocks": 4})).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=.001, weight_decay=.0001)
    games, steps, generation, since_update = 0, 0, 0, 0
    replay, active, gate = [], [], None
    loss = None
    if saved:
        net.load_state_dict(saved["weights"])
        optimizer.load_state_dict(saved["optimizer"])
        rng.bit_generator.state = saved["rng"]
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda" and saved.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        games, steps, generation = saved["games"], saved["step"], saved["generation"]
        replay, active, gate = saved["replay"], saved["active"], saved["gate"]
        since_update = saved["since_update"]
    selfplay_counts = saved.get("selfplay_counts", {}) if saved else {}
    counted_from_game = saved.get("counted_from_game", games) if saved else 0
    incumbent = load_json(root / "incumbent.json", {"kind": "heuristic", "label": "Original untrained tactical baseline"})
    if not (root / "incumbent.json").exists():
        save_json(root / "incumbent.json", incumbent)
    gate_opponent = None
    if gate and incumbent["kind"] == "network":
        gate_opponent = read_network(root / incumbent["file"], device)
    start = time.monotonic()
    last_save, last_status = start, 0
    last_log, last_phase = 0, None

    def status(phase, message=""):
        nonlocal last_log, last_phase
        now = time.monotonic()
        if phase != last_phase or now - last_log >= 5:
            print(f"{phase}: {games} games | {steps} updates | {len(replay)} replay positions | "
                  f"{max(0, deadline - now):.0f}s remaining. {message}", flush=True)
            last_log, last_phase = now, phase
        report = load_json(root / f"gate-{generation}.json")
        summary = {k: v for k, v in report.items() if k != "matches"} if report else None
        save_json(root / "status.json", {"phase": phase, "message": message, "device": str(device),
            "games": games, "updates": steps, "replay_positions": len(replay), "loss": loss,
            "selfplay_counts": selfplay_counts, "counted_from_game": counted_from_game,
            "tactical_ms": settings["tactical_ms"],
            "session_seconds": round(time.monotonic() - start, 1),
            "remaining_seconds": round(max(0, deadline - time.monotonic()), 1),
            "artifact_bytes": usage(root), "cap_bytes": cap,
            "gate_games": len(gate["matches"]) if gate else 0, "gate_required_games": 400,
            "incumbent": incumbent, "last_evaluation": summary, "updated_at": time.time(),
            "cuda_allocated_bytes": torch.cuda.memory_allocated() if device.type == "cuda" else 0,
            "cuda_reserved_bytes": torch.cuda.memory_reserved() if device.type == "cuda" else 0})

    def checkpoint():
        state = model_state(net, steps)
        state.update(optimizer=optimizer.state_dict(), games=games, generation=generation,
            replay=replay, active=active, gate=gate, since_update=since_update,
            rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(), settings=settings,
            selfplay_counts=selfplay_counts, counted_from_game=counted_from_game,
            cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else None)
        save_checkpoint(root / "latest.pt", state, cap)

    try:
        checkpoint()
        while not stopped():
            if gate is not None:
                status("evaluating", "Candidate is frozen; held-out games never enter training replay.")
                match_index = len(gate["matches"])
                if match_index == 400:
                    pair_scores = [(gate["matches"][i]["score"] + gate["matches"][i + 1]["score"]) / 2
                                   for i in range(0, 400, 2)]
                    lower = confidence(pair_scores)
                    promote = lower > .5 and gate.get("max_overrun", 0) <= .1
                    report = {"generation": generation, "games": 400, "opening_pairs": 200,
                              "score": float(np.mean(pair_scores)), "lower_95": lower,
                              "promoted": promote, "matches": gate["matches"],
                              "seconds_per_turn": 5, "max_simulations_per_placement": 100000,
                              "max_turn_overrun_seconds": gate.get("max_overrun", 0),
                              "candidate_tactics": gate.get("tactics", False),
                              "candidate_tss": gate.get("tss", False),
                              "incumbent_tactics": incumbent.get("tactics", False),
                              "incumbent_tss": incumbent.get("tss", False)}
                    save_json(root / f"gate-{generation}.json", report)
                    if promote:
                        filename = f"incumbent-{generation}.pt"
                        save_checkpoint(root / filename, model_state(net, steps), cap)
                        digest = hashlib.sha256((root / filename).read_bytes()).hexdigest()
                        incumbent = {"kind": "network", "file": filename, "sha256": digest,
                                     "step": steps, "previous": incumbent, "gate": f"gate-{generation}.json",
                                     "tactics": gate.get("tactics", False),
                                     "tss": gate.get("tss", False)}
                        save_json(root / "incumbent.json", incumbent)
                    gate, gate_opponent = None, None
                    checkpoint()
                    continue
                game = Game.from_moves(gate["moves"]) if gate["moves"] is not None else opening(match_index // 2, generation)
                candidate_color = 1 if match_index % 2 == 0 else -1
                remaining = gate.get("turn_remaining", 5.0)
                begin = time.monotonic()
                color = game.player
                engine = net if color == candidate_color else gate_opponent
                candidate_turn = color == candidate_color
                tactics = gate.get("tactics", False) if candidate_turn else incumbent.get("tactics", False)
                tss = gate.get("tss", False) if candidate_turn else incumbent.get("tss", False)
                plan = list(gate.get("plan", []))
                proof_plan = [gate.get("tss_proof")]
                strategy = [gate.get("tss_strategy")]
                action = turn_action(game, engine, min(remaining / game.left, max(0, deadline - begin)),
                                     100000, stopped, tactics=tactics, tss=tss,
                                     plan=plan, proof_plan=proof_plan, strategy=strategy)
                if stopped():
                    break
                game.play(action)
                if plan:
                    plan.pop(0)
                gate["plan"] = plan
                gate["tss_proof"] = proof_plan[0] if plan else None
                gate["tss_strategy"] = strategy[0]
                elapsed = time.monotonic() - begin
                gate["moves"] = game.moves
                gate["turn_remaining"] = 5.0 if game.player != color else max(0, remaining - elapsed)
                gate["max_overrun"] = max(gate.get("max_overrun", 0), elapsed - remaining)
                if game.done:
                    score = (game.winner * candidate_color + 1) / 2
                    gate["matches"].append({"score": score, "moves": game.moves,
                                            "candidate_color": candidate_color})
                    gate["moves"], gate["turn_remaining"] = None, 5.0
                    checkpoint()
            elif since_update >= 8 and replay:
                # Store remaining optimizer work to avoid repeating a completed cycle on resume.
                pending = settings.get("pending_updates", settings["updates_per_cycle"])
                while pending and not stopped():
                    loss = train_step(net, optimizer, replay, settings["batch"], rng, device)
                    steps += 1
                    pending -= 1
                    settings["pending_updates"] = pending
                    status("optimizing")
                if pending == 0:
                    since_update = 0
                    settings.pop("pending_updates", None)
                    if games >= (generation + 1) * settings["gate_every"]:
                        generation += 1
                        gate = {"matches": [], "moves": None, "turn_remaining": 5.0,
                                "tactics": True, "tss": True, "plan": [], "tss_proof": None}
                        gate_opponent = read_network(root / incumbent["file"], device) if incumbent["kind"] == "network" else None
                    checkpoint()
            else:
                while len(active) < settings["parallel"]:
                    game = Game()
                    # Mix central starts with varied legal openings; no external book or weights.
                    game.play(180)
                    if rng.random() < .5:
                        cells = [r * 19 + c for r in range(6, 13) for c in range(6, 13) if r * 19 + c != 180]
                        for action in rng.choice(cells, 4, replace=False):
                            game.play(int(action))
                    active.append({"moves": game.moves, "samples": []})
                boards = [Game.from_moves(row["moves"]) for row in active]
                decisions = selfplay_batch(boards, net, settings["simulations"], rng,
                    bootstrap=games < settings["bootstrap_games"], tactical_ms=settings["tactical_ms"],
                    deadline=deadline, stopped=stopped)
                if decisions is None or stopped():
                    break
                survivors = []
                for row, game, decision in zip(active, boards, decisions):
                    if args.max_games and games >= args.max_games:
                        survivors.append(row)
                        continue
                    source = decision["source"]
                    selfplay_counts[source] = selfplay_counts.get(source, 0) + 1
                    row["samples"].extend(decision.get("samples") or selfplay_samples(game, decision))
                    winner = decision.get("winner")
                    if winner is None:
                        action = choose(decision["policy"], rng, temperature=1 if len(game.moves) < 30 else .25)
                        game.play(action)
                        row["moves"] = game.moves
                        if game.done:
                            winner = game.winner
                    # Certified wins/losses end training games early, but never evaluation games.
                    if winner is not None:
                        for sample in row["samples"]:
                            sample["result"] = float(winner * sample["player"])
                        replay.extend(row["samples"])
                        replay = replay[-settings["replay_limit"]:]
                        games += 1
                        since_update += 1
                    else:
                        survivors.append(row)
                active = survivors
            now = time.monotonic()
            if now - last_save >= 60:
                checkpoint()
                last_save = now
            if now - last_status >= 2:
                phase = "evaluating" if gate else ("bootstrap" if games < settings["bootstrap_games"] else "self_play")
                status(phase)
                last_status = now
            if args.max_games and games >= args.max_games:
                break
        checkpoint()
        status("paused", "Checkpoint saved. Start another session to resume.")
    except Exception as exc:
        # Do not overwrite known-good artifacts after an error/non-finite update.
        status("error", f"{type(exc).__name__}: {exc}. Previous checkpoint retained.")
        raise


def add_arguments(parser):
    parser.add_argument("--data", default="data")
    parser.add_argument("--hours", type=float, default=2)
    parser.add_argument("--disk-gib", type=float, default=20)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--parallel", type=int, default=8)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--seed", type=int, default=5070)
    parser.add_argument("--tactical-ms", type=float, default=2,
                        help="Per-position fork-discovery budget in self-play; 0 disables forks, not exact immediate checks")
    parser.add_argument("--max-games", type=int, default=0, help="Smoke-test stop limit (lifetime completed games).")
    parser.add_argument("--keep-stop", action="store_true", help=argparse.SUPPRESS)


def validate(args):
    if not math.isfinite(args.hours) or not 0 < args.hours <= 168:
        raise ValueError("Session duration must be greater than zero and at most 168 hours.")
    if not math.isfinite(args.disk_gib) or not .1 <= args.disk_gib <= 10000:
        raise ValueError("Artifact cap must be between 0.1 and 10000 GiB.")
    if not 1 <= args.simulations <= 100000 or not 1 <= args.parallel <= 128 or not 2 <= args.batch <= 4096:
        raise ValueError("Invalid simulations, parallel games, or batch size.")
    if args.max_games < 0 or not 0 <= args.seed < 2 ** 32:
        raise ValueError("Invalid game limit or seed.")
    if not math.isfinite(args.tactical_ms) or not 0 <= args.tactical_ms <= 50:
        raise ValueError("Self-play tactical budget must be between 0 and 50 milliseconds.")
