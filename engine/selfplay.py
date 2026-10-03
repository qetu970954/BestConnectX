"""Batched placement decisions and policy/value updates. Games always play to a terminal."""
import math
import time
import numpy as np
import torch
from .game import DEFAULT_RULES, Game, heuristic
from .network import augment
from .search import search, choose
from .tactics import forcing_win, verifies
from .tss import search_tss, verify_tss


def observation(game, policy):
    return {"board": torch.from_numpy(game.board.copy()), "player": game.player,
            "left": game.left, "policy": torch.from_numpy(policy.astype(np.float16))}


def selfplay_batch(boards, net, simulations, rng, *, bootstrap=False, tactical_ms=2,
                   deadline=None, stopped=lambda: False, tss_enabled=True, strategies=None, plans=None):
    """Verified proofs guide placements; only unresolved roots reach batched inference.

    None means cancellation. Callers must not partially commit an interrupted batch.
    Proofs never adjudicate a game or assign its value target.
    """
    decisions, unresolved = [], []
    for i, game in enumerate(boards):
        if stopped() or (deadline is not None and time.monotonic() >= deadline):
            return None
        if game.done:
            raise ValueError("Self-play expects nonterminal positions.")
        own = game.threats(game.player, game.left)
        proof = tuple(sorted(min(own, key=lambda e: (len(e), tuple(sorted(e)))))) if own else None
        if proof is None and plans is not None and plans[i]:
            if verifies(game, plans[i]):
                proof = tuple(plans[i])
            else:
                plans[i] = []
        if proof is None and tactical_ms > 0:
            if tss_enabled and strategies is not None and strategies[i] is not None:
                plan, checked_plan, cached = [], [None], [strategies[i]]
                expires = time.perf_counter() + tactical_ms / 1000
                if _resume_tss_strategy(game, cached, plan, checked_plan, expires, stopped):
                    strategies[i] = cached[0]
                    policy = np.zeros(game.board.size, dtype=np.float32)
                    policy[plan[0]] = 1
                    decisions.append({"source": "tss_move", "policy": policy, "proof": plan})
                    continue
                strategies[i] = cached[0]
            # Reserve the final 10% of this root's budget to independently check a TSS proof.
            expires = time.perf_counter() + tactical_ms / 1000
            search_expires = time.perf_counter() + tactical_ms * .9 / 1000
            is_stopped = lambda until: lambda: stopped() or time.perf_counter() >= until
            proof_tree = None
            if tss_enabled:
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
            policy = np.zeros(game.board.size, dtype=np.float32)
            policy[proof[0]] = 1
            if plans is not None:
                plans[i] = list(proof)
            decisions.append({"source": "proof_move", "policy": policy, "proof": list(proof)})
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


def selfplay_samples(game, decision):
    sample = observation(game, decision["policy"])
    sample["source"] = decision["source"]
    return [sample]


def train_step(net, optimizer, replay, batch_size, rng, device, *, rules=DEFAULT_RULES, metrics=None):
    batch = [replay[int(i)] for i in rng.integers(len(replay), size=batch_size)]
    boards = [Game(row["board"].numpy(), row["player"], row["left"], rules=rules) for row in batch]
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
