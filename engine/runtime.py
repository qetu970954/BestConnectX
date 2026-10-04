"""Production C++ playing/inference adapter. Python only transports state and manages work."""
import ctypes as C
from functools import lru_cache
import time
import weakref
import numpy as np
from .game import Game as ReferenceGame, Rules

DEFAULT_RULES = Rules(15, 15, 5)
STOP = C.CFUNCTYPE(C.c_int)
SOURCES = ('mcts', 'heuristic', 'forced', 'proof_move', 'tss_move')


@lru_cache(maxsize=1)
def backend():
    from . import native
    lib = native.library()
    if lib is None:
        raise RuntimeError('The C++ engine is required. Run: uv run python -m engine.native')
    ints = np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
    floats = np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS')
    boards = np.ctypeslib.ndpointer(dtype=np.int8, flags='C_CONTIGUOUS')
    doubles = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
    lib.runtime_error.restype = C.c_char_p
    lib.game_play.argtypes = [boards, ints, ints, C.c_int, C.c_int]
    lib.game_query.argtypes = [boards, ints, ints, C.c_int, C.c_int, C.c_int, ints, floats]
    lib.game_start.argtypes = [boards, ints, ints, C.POINTER(C.c_int), C.c_uint64, C.c_int]
    lib.tactical.argtypes = [boards, ints, ints, C.c_int, C.c_int, ints, *([C.c_int]*5), C.c_double, STOP, ints, C.c_int]
    lib.runtime_search.argtypes = [boards, ints, ints, ints, C.c_int, C.c_void_p, C.c_int, C.c_int,
        C.c_uint64, C.c_int, C.c_double, STOP, floats, doubles]
    lib.runtime_choose.argtypes = [floats, C.c_int, C.c_uint64, C.c_double]
    lib.runtime_decide.argtypes = [boards, ints, ints, ints, C.c_int, C.c_void_p, C.c_int, C.c_int, C.c_uint64,
        C.c_int, C.c_int, C.c_int, C.c_double, C.c_double, STOP, ints, ints, ints,
        floats, ints, ints, ints, ints, ints, C.c_int, doubles]
    lib.runtime_turn.argtypes = [boards, ints, ints, C.c_int, C.c_void_p, C.c_double,
        ints, C.c_int, ints, ints, C.POINTER(C.c_int), C.c_int, C.POINTER(C.c_double)]
    lib.model_create.argtypes = [C.c_int, C.c_int, C.c_int, C.c_char_p]
    lib.model_create.restype = C.c_void_p
    lib.model_destroy.argtypes = [C.c_void_p]
    lib.model_destroy.restype = None
    lib.model_tensor.argtypes = [C.c_void_p, C.c_char_p, floats,
        np.ctypeslib.ndpointer(dtype=np.int64, flags='C_CONTIGUOUS'), C.c_int]
    lib.model_predict.argtypes = [C.c_void_p, floats, C.c_int, C.c_int, floats, floats]
    return lib


def check(code):
    if code < 0:
        message = backend().runtime_error().decode('utf-8', errors='replace')
        raise RuntimeError(message or 'Native operation failed.')
    return code


def seed(rng):
    return int(rng.integers(0, 2**63)) if rng is not None else 0


def packet(game):
    if game.board.dtype != np.int8 or game.board.shape != (game.size**2,) or np.any((game.board < -1) | (game.board > 1)):
        raise ValueError('Board cells or dimensions do not match the rules.')
    if not isinstance(game.moves, list) or len(game.moves) > game.board.size or any(type(v) is not int or not 0 <= v < game.board.size for v in game.moves):
        raise ValueError('Invalid game history.')
    meta = np.array([game.size, game.win_length, game.turn_stones, game.rules.starter_stones,
                     game.player, game.left, game.winner, game.done], dtype=np.int32)
    return np.ascontiguousarray(game.board, dtype=np.int8), meta, np.asarray(game.moves, dtype=np.int32)


class Game(ReferenceGame):
    """Serializable game state. All rule calculations and placements run in C++."""
    def __init__(self, board=None, player=1, left=None, winner=0, done=False, moves=None, rules=DEFAULT_RULES):
        super().__init__(board, player, left, winner, done, [] if moves is None else list(moves), rules)
        if self.board.dtype != np.int8 or not self.board.flags.c_contiguous or np.any((self.board < -1) | (self.board > 1)):
            raise ValueError('Board cells must be a contiguous int8 array of -1, 0, or 1.')
        if type(player) is not int or player not in (-1, 1) or type(self.left) is not int or self.left not in (1, 2):
            raise ValueError('Invalid player or remaining stones.')

    def empty(self):
        return Game(rules=self.rules)

    def copy(self):
        return Game(self.board.copy(), self.player, self.left, self.winner, self.done, self.moves, self.rules)

    def play(self, action):
        if isinstance(action, bool) or not isinstance(action, (int, np.integer)) or not 0 <= int(action) < self.board.size:
            raise ValueError('Placement must be an integer cell index within the board.')
        board, meta, moves = packet(self)
        try:
            check(backend().game_play(board, meta, moves, len(moves), int(action)))
        except RuntimeError as exc:
            raise ValueError(str(exc)) from exc
        self.board = board
        self.player, self.left, self.winner, self.done = int(meta[4]), int(meta[5]), int(meta[6]), bool(meta[7])
        self.moves.append(int(action))

    def query(self, operation, argument=0):
        board, meta, moves = packet(self)
        cells = np.empty(8000, dtype=np.int32)
        planes = np.empty((8, self.size, self.size), dtype=np.float32)
        count = check(backend().game_query(board, meta, moves, len(moves), operation, argument, cells, planes))
        return count, cells, planes

    def actions(self):
        count, cells, _ = self.query(1)
        return cells[:count].copy()

    def features(self):
        return self.query(0)[2]

    def threats(self, player, budget=None):
        if player not in (-1, 1) or type(player) is bool:
            raise ValueError('Threat player must be -1 or 1.')
        budget = self.turn_stones if budget is None else budget
        if type(budget) is not int or not 0 <= budget <= 2:
            raise ValueError('Threat budget must be 0..2.')
        count, cells, _ = self.query(3 if player == self.player else 4, budget)
        edges, offset = set(), 0
        while offset < count:
            length = int(cells[offset]); offset += 1
            edges.add(frozenset(map(int, cells[offset:offset+length]))); offset += length
        return frozenset(edges)

    @classmethod
    def from_moves(cls, moves, *, rules=DEFAULT_RULES):
        if not isinstance(moves, list) or len(moves) > rules.height**2 or any(type(move) is not int or not 0 <= move < rules.height**2 for move in moves):
            raise ValueError('Expected a legal list of integer placements.')
        game = cls(rules=rules)
        board, meta, _ = packet(game)
        history = np.zeros(game.board.size, dtype=np.int32)
        history[:len(moves)] = moves
        count = C.c_int(len(moves))
        try:
            check(backend().game_start(board, meta, history, C.byref(count), 0, 2))
        except RuntimeError as exc:
            raise ValueError(str(exc)) from exc
        return cls(board, int(meta[4]), int(meta[5]), int(meta[6]), bool(meta[7]), history[:count.value].tolist(), rules)


def opening(index, initial_seed, rules=DEFAULT_RULES, *, random_start=False):
    game = Game(rules=rules)
    board, meta, _ = packet(game)
    history = np.empty(game.board.size, dtype=np.int32); count = C.c_int(0)
    check(backend().game_start(board, meta, history, C.byref(count), (initial_seed + index) % 2**64, 0 if random_start else 1))
    return Game(board, int(meta[4]), int(meta[5]), int(meta[6]), bool(meta[7]), history[:count.value].tolist(), rules)


class NativeModel:
    """Independent inference weights. Never update them during a native search call."""
    def __init__(self, net):
        self.lib = backend()
        self.config = net.config.copy()
        self.device = str(next(net.parameters()).device)
        self.handle = self.lib.model_create(self.config['size'], self.config['channels'], self.config['blocks'], self.device.encode())
        if not self.handle:
            raise RuntimeError(self.lib.runtime_error().decode())
        self.cleanup = weakref.finalize(self, self.lib.model_destroy, self.handle)
        self.sync(net)
        self.timings = {'cpu_search_seconds': 0., 'inference_seconds': 0.}

    def sync(self, net):
        for name, tensor in net.state_dict().items():
            if name.endswith('num_batches_tracked'):
                continue
            values = tensor.detach().float().cpu().numpy().copy()
            shape = np.asarray(values.shape, dtype=np.int64)
            check(self.lib.model_tensor(self.handle, name.encode(), values, shape, len(shape)))
        net._native_dirty = False

    def evaluate_features(self, features):
        x = np.ascontiguousarray(features, dtype=np.float32)
        if x.ndim != 4 or x.shape[1:] != (8, self.config['size'], self.config['size']) or not 1 <= len(x) <= 128:
            raise ValueError('Inference requires 1..128 matching eight-plane boards.')
        policy = np.empty((len(x), x.shape[-1]**2), dtype=np.float32); value = np.empty(len(x), dtype=np.float32)
        check(self.lib.model_predict(self.handle, x, len(x), x.shape[-1], policy, value))
        return policy, value

    def evaluate(self, games):
        return self.evaluate_features(np.stack([Game(g.board, g.player, g.left, g.winner, g.done, g.moves, g.rules).features() for g in games]))


def model_for(net):
    if net is None:
        return None
    if isinstance(net, NativeModel):
        return net
    if getattr(net, '_native_model', None) is None:
        net._native_model = NativeModel(net)
    elif getattr(net, '_native_dirty', False):
        net._native_model.sync(net)
    return net._native_model


def encode_proof(proof):
    """Validate wire syntax only. C++ independently checks the claim against the game."""
    output = []
    def integers(values):
        if not isinstance(values, list) or not 1 <= len(values) <= 2 or any(type(v) is not int or not 0 <= v < 625 for v in values):
            raise ValueError('Invalid proof moves.')
        output.extend([len(values), *values])
    def node(tree, depth):
        if depth > 20 or not isinstance(tree, dict) or set(tree) != {'moves', 'responses'} or not isinstance(tree['responses'], list):
            raise ValueError('Invalid proof tree.')
        integers(tree['moves']); output.append(len(tree['responses']))
        for reply in tree['responses']:
            if not isinstance(reply, dict) or set(reply) != {'moves', 'proof'}:
                raise ValueError('Invalid proof reply.')
            integers(reply['moves']); node(reply['proof'], depth+1)
        if len(output) > 1_000_000:
            raise ValueError('Proof is too large.')
    if not isinstance(proof, dict) or set(proof) != {'version', 'attacker', 'max_turns', 'tree'}:
        raise ValueError('Invalid certificate fields.')
    if type(proof['version']) is not int or proof['version'] != 1 or type(proof['attacker']) is not int or proof['attacker'] not in (-1, 1) or type(proof['max_turns']) is not int or not 1 <= proof['max_turns'] <= 20:
        raise ValueError('Invalid certificate header.')
    output.extend([1, proof['attacker'], proof['max_turns']]); node(proof['tree'], 1)
    return output


def decode_proof(values):
    values = list(map(int, values)); offset = 3
    def moves():
        nonlocal offset
        count = values[offset]; offset += 1
        result = values[offset:offset+count]; offset += count
        return result
    def node():
        nonlocal offset
        actions = moves(); count = values[offset]; offset += 1
        return {'moves': actions, 'responses': [{'moves': moves(), 'proof': node()} for _ in range(count)]}
    proof = {'version': values[0], 'attacker': values[1], 'max_turns': values[2], 'tree': node()}
    if offset != len(values):
        raise RuntimeError('Native proof length mismatch.')
    return proof


def encode_strategy(strategy):
    if strategy is None:
        return []
    try:
        history = strategy['history']
        if not isinstance(history, list) or len(history) > 625 or any(type(v) is not int or not 0 <= v < 625 for v in history):
            return []
        return [len(history), *history, *encode_proof(strategy['proof'])]
    except (ValueError, KeyError, TypeError):
        return []


def full_turn(game, net, duration, strategy=None):
    board, meta, history = packet(game); model = model_for(net)
    stored = np.asarray(encode_strategy(strategy), dtype=np.int32)
    moves = np.empty(2, dtype=np.int32); out_strategy = np.empty(1_000_000, dtype=np.int32)
    size = C.c_int(); elapsed = C.c_double()
    count = check(backend().runtime_turn(board, meta, history, len(history), model.handle if model else None,
        duration, stored, len(stored), moves, out_strategy, C.byref(size), len(out_strategy), C.byref(elapsed)))
    values = out_strategy[:size.value]
    cached = {'history': values[1:1+values[0]].tolist(), 'proof': decode_proof(values[1+values[0]:])} if len(values) else None
    return moves[:count].tolist(), cached, elapsed.value


def tactical(game, operation, *, proof=None, moves=None, deadline, max_nodes=100_000, max_turns=3, width=24, max_candidates=256, stopped=lambda: False):
    if type(max_nodes) is not int or max_nodes < 1 or type(max_turns) is not int or not 1 <= max_turns <= 20:
        raise ValueError('Tactical search needs positive node limits and 1..20 turns.')
    if type(width) is not int or not 2 <= width <= 64 or type(max_candidates) is not int or max_candidates < 1:
        raise ValueError('Tactical width must be 2..64; candidate limit must be positive.')
    if stopped() or time.perf_counter() >= deadline:
        return None
    board, meta, history = packet(game)
    try:
        data = encode_proof(proof) if operation == 1 else (moves or [])
    except ValueError:
        return False
    data = np.asarray(data, dtype=np.int32)
    output = np.empty(1_000_000 if operation == 0 else 2, dtype=np.int32)
    code = backend().tactical(board, meta, history, len(history), operation, data, len(data), max_turns, width, max_candidates,
        max_nodes, max(0, deadline-time.perf_counter()), STOP(stopped), output, len(output))
    if code == -2:
        return None
    check(code)
    if operation in (1, 3):
        return bool(code)
    return decode_proof(output[:code]) if operation == 0 and code else tuple(map(int, output[:code])) if code else None


def search_tss(game, *, deadline, **kwargs):
    return tactical(game, 0, deadline=deadline, **kwargs)


def verify_tss(game, proof, *, deadline, **kwargs):
    return tactical(game, 1, proof=proof, deadline=deadline, **kwargs)


def batch_packet(games):
    if not games or any(g.rules != games[0].rules or g.done for g in games):
        raise ValueError('Native batches need live games with matching rules.')
    parts = [packet(game) for game in games]
    offsets = np.array([0, *np.cumsum([len(p[2]) for p in parts])], dtype=np.int32)
    return np.stack([p[0] for p in parts]), np.stack([p[1] for p in parts]), np.concatenate([p[2] for p in parts]), offsets


def search(games, net, simulations=64, rng=None, deadline=None, stopped=lambda: False, workers=6):
    boards, meta, histories, offsets = batch_packet(games); model = model_for(net)
    policies = np.empty(boards.shape, dtype=np.float32); timings = np.zeros(2, dtype=np.float64)
    completed = check(backend().runtime_search(boards, meta, histories, offsets, len(games), model.handle if model else None,
        simulations, workers, seed(rng), rng is not None, max(0, deadline-time.monotonic()) if deadline is not None else 604800,
        STOP(stopped), policies, timings))
    if model:
        model.timings = dict(zip(('cpu_search_seconds', 'inference_seconds'), timings.tolist()))
    return list(policies), completed


def selfplay_batch(games, net, simulations, rng, *, bootstrap=False, tactical_ms=2, deadline=None, stopped=lambda: False,
                   tss_enabled=True, strategies=None, plans=None, workers=6, metrics=None, _mode=0, _duration=None):
    if stopped() or (deadline is not None and time.monotonic() >= deadline):
        return None
    boards, meta, histories, offsets = batch_packet(games)
    count = len(games); model = model_for(net)
    encoded, stored_offsets = [], [0]
    for strategy in strategies or [None]*count:
        encoded.extend(encode_strategy(strategy))
        stored_offsets.append(len(encoded))
    plan_data = np.full((count, 2), -1, dtype=np.int32)
    for i, plan in enumerate(plans or [[] for _ in games]):
        if len(plan) > 2 or any(type(v) is not int or not 0 <= v < games[i].board.size for v in plan):
            raise ValueError('Invalid stored turn plan.')
        plan_data[i, :len(plan)] = plan
    policies = np.empty(boards.shape, dtype=np.float32)
    actions = np.empty(count, dtype=np.int32); sources = np.empty(count, dtype=np.int32)
    out_plans = np.empty((count, 2), dtype=np.int32); out_strategies = np.empty(1_000_000, dtype=np.int32)
    out_offsets = np.empty(count+1, dtype=np.int32); timings = np.zeros(2, dtype=np.float64)
    duration = _duration if _duration is not None else max(0, deadline-time.monotonic()) if deadline is not None else 604800
    code = backend().runtime_decide(boards, meta, histories, offsets, count, model.handle if model else None, simulations, workers,
        seed(rng), _mode, bootstrap, tss_enabled, tactical_ms, duration, STOP(stopped), np.asarray(encoded, dtype=np.int32),
        np.asarray(stored_offsets, dtype=np.int32), plan_data, policies, actions, sources, out_plans, out_strategies, out_offsets,
        len(out_strategies), timings)
    if code == -2 or stopped():
        return None
    check(code)
    decisions = []
    for i in range(count):
        plan = out_plans[i][out_plans[i] >= 0].tolist()
        if plans is not None:
            plans[i] = plan
        if strategies is not None:
            values = out_strategies[out_offsets[i]:out_offsets[i+1]]
            if len(values):
                length = int(values[0]); strategies[i] = {'history': values[1:1+length].tolist(), 'proof': decode_proof(values[1+length:])}
            else:
                strategies[i] = None
        decisions.append({'source': SOURCES[int(sources[i])], 'policy': policies[i], 'action': int(actions[i]), 'proof': plan})
    measured = dict(zip(('cpu_search_seconds', 'inference_seconds'), timings.tolist()))
    if model:
        model.timings = measured
    if metrics is not None:
        metrics.update(measured)
    return decisions


def turn_action(game, net, seconds, simulations, stopped, *, tactics=False, tss=False, plan=None, proof_plan=None, strategy=None):
    strategies = [strategy[0] if strategy else None]; plans = [list(plan or [])]
    result = selfplay_batch([game], net, simulations, None, stopped=stopped, strategies=strategies, plans=plans,
        tss_enabled=tss, tactical_ms=50 if tactics or tss else 0, workers=1, _mode=1, _duration=max(0, seconds))
    if result is None:
        return game.query(2)[0]  # Caller checks stop before committing; no artificial result.
    if plan is not None:
        plan[:] = plans[0]
    if proof_plan is not None:
        proof_plan[0] = strategies[0]['proof'] if strategies[0] else None
    if strategy is not None:
        strategy[0] = strategies[0]
    return result[0]['action']
