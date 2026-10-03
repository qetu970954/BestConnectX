"""Checks through rules, training/resume, saved datasets, and frozen-gate interfaces."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from engine.game import Game, Rules, parse_board, heuristic
from engine import training
from engine.web import history
from engine.network import Network, augment, device_for
from engine.search import search
from engine.storage import GIB, load_checkpoint, save_checkpoint, save_json
from engine.selfplay import observation, selfplay_batch, train_step
from engine.tss import search_tss, verify_tss

PROJECT = Path(__file__).resolve().parent.parent


def cli(*args, stdin=None):
    return subprocess.run([sys.executable, *args], cwd=PROJECT, input=stdin,
                          capture_output=True, text=True, timeout=90)


class EngineTests(unittest.TestCase):
    def test_single_stone_rules_and_freestyle_wins(self):
        game = Game()
        for move, player in ((0, -1), (9, 1), (1, -1), (10, 1)):
            game.play(move)
            self.assertEqual((game.player, game.left), (player, 1))
        game = Game.from_moves([0, 9, 1, 10, 2, 11, 3, 12, 4])
        self.assertEqual((game.done, game.winner), (True, 1))
        for invalid in (True, -1, 81, 1.2, None):
            with self.assertRaises(ValueError):
                Game().play(invalid)
        with self.assertRaises(ValueError):
            game.play(80)
        with self.assertRaises(ValueError):
            Game.from_moves([0, 0])
        for stride, start in ((1, 0), (5, 0), (6, 0), (4, 4)):
            # Fill a gap to make FIVE stones, not just the four-stone winning window.
            rules = Rules(5, 5, 4)
            line = [start + i * stride for i in range(5)]
            game = Game(rules=rules)
            game.board[line] = 1
            game.board[line[2]] = 0
            game.play(line[2])
            self.assertEqual((game.done, game.winner), (True, 1))

    def test_square_sizes_and_connect6_turns_preserve_configuration(self):
        rules = Rules(13, 13, 6, 2, 1)
        game = Game(rules=rules)
        for move, expected in ((84, (-1, 2)), (83, (-1, 1)), (85, (1, 2)), (71, (1, 1))):
            game.play(move)
            self.assertEqual((game.player, game.left), expected)
        self.assertEqual(game.features().shape, (8, 13, 13))
        self.assertEqual(game.copy().rules, rules)
        self.assertEqual(game.empty().rules, rules)
        self.assertEqual(Game.from_moves(game.moves, rules=rules).left, game.left)
        custom = Game(rules=Rules(13, 13, 6, 1, 2))
        custom.play(0)
        self.assertEqual((custom.player, custom.left), (1, 1))
        custom.play(1)
        self.assertEqual((custom.player, custom.left), (-1, 1))
        # A win ends the turn immediately, even with a second stone remaining.
        game = Game(rules=rules)
        game.board[:5] = 1
        game.left = 2
        game.play(5)
        self.assertTrue(game.done)
        with self.assertRaises(ValueError):
            game.play(6)
        self.assertEqual(parse_board('9*9'), (9, 9))
        for invalid in ('9', '9*9*9', '9;rm', '-9*9', '9*13', '13x9'):
            with self.assertRaises(ValueError):
                parse_board(invalid)
        self.assertEqual(Game(rules=Rules(2, 2, 2)).board.size, 4)
        for values in ((9, 9, 5, 3, 1), (1, 9, 5, 1, 1), (9, 9, 10, 1, 1), (9, 13, 5, 1, 1)):
            with self.assertRaises(ValueError):
                Rules(*values)

    def test_square_search_all_eight_symmetries_and_real_loss_updates(self):
        device_for('cpu')
        rules = Rules(5, 5, 4, 2, 1)
        net = Network(channels=8, blocks=1, size=5)
        game = Game.from_moves([12], rules=rules)
        policies, _ = search([game], net, 2)
        self.assertEqual(policies[0].shape, (25,))
        self.assertEqual(policies[0][12], 0)
        sample = observation(game, policies[0])
        sample['result'] = 1.0
        before = net.policy.weight.detach().clone()
        metric = {}
        train_step(net, torch.optim.AdamW(net.parameters()), [sample], 2,
                   np.random.default_rng(0), torch.device('cpu'), metrics=metric,
                   rules=rules)
        self.assertAlmostEqual(metric['loss'], metric['policy_loss'] + metric['value_loss'], places=5)
        self.assertFalse(torch.equal(before, net.policy.weight))
        x, p = np.zeros((1, 8, 5, 5), dtype=np.float32), np.zeros((1, 25), dtype=np.float32)
        x[0, 0, 0, 1], p[0, 1] = 1, 1
        variants = set()
        for rotation in range(4):
            for reflection in range(2):
                draws = iter((rotation, reflection))
                rng = SimpleNamespace(integers=lambda *_: next(draws))
                xx, pp = augment(x, p, rng)
                self.assertEqual(xx.shape, x.shape)
                self.assertEqual(xx[0, 0].argmax(), pp.argmax())
                variants.add(xx.tobytes())
        self.assertEqual(len(variants), 8)
        with self.assertRaises(ValueError):
            augment(np.zeros((1, 8, 5, 7)), np.zeros((1, 35)), np.random.default_rng(0))

    def test_verified_gomoku_tss_selects_moves_without_adjudicating(self):
        game = Game.from_moves([39, 0, 40, 2, 41, 4])
        proof = search_tss(game, max_turns=2, deadline=time.perf_counter() + 2,
                           max_nodes=10000, max_candidates=1000)
        self.assertIsNotNone(proof)
        self.assertIs(verify_tss(game, proof, deadline=time.perf_counter() + 2), True)
        decisions = selfplay_batch([game], None, 1, np.random.default_rng(0), bootstrap=True,
                                  tactical_ms=50, strategies=[None])
        self.assertEqual(decisions[0]['source'], 'tss_move')
        self.assertNotIn('winner', decisions[0])
        game.play(int(decisions[0]['policy'].argmax()))
        self.assertFalse(game.done)
        game.play(heuristic(game))
        game.play(heuristic(game))
        self.assertEqual((game.done, game.winner), (True, 1))

    def test_two_stone_opening_proof_continues_through_partial_turn(self):
        for tss_enabled in (True, False):
            game = Game(rules=Rules(4, 4, 3, 1, 2))
            strategies, plans = [None], [[]]
            kwargs = dict(bootstrap=True, tactical_ms=50, strategies=strategies, plans=plans,
                          tss_enabled=tss_enabled)
            first = selfplay_batch([game], None, 1, np.random.default_rng(0), **kwargs)[0]
            self.assertEqual(first['source'], 'tss_move' if tss_enabled else 'proof_move')
            self.assertEqual(len(first['proof']), 2)
            game.play(int(first['policy'].argmax()))
            if plans[0]:
                plans[0].pop(0)
            second = selfplay_batch([game], None, 1, np.random.default_rng(0), **kwargs)[0]
            self.assertEqual(int(second['policy'].argmax()), first['proof'][1])
            game.play(int(second['policy'].argmax()))
            self.assertEqual((game.player, game.left), (-1, 1))
            game.play(heuristic(game))
            game.play(heuristic(game))
            self.assertEqual((game.done, game.winner), (True, 1))

    def test_direct_cli_archives_milestones_datasets_and_resumes(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            save_json(root / 'run.json', {'rules': Rules().id, 'rule_config': Rules().to_dict()})
            command = ['train.py', '--connect', '5', '--board_size', '9*9',
                '--stones_per_turn', '1', '--starter-stones', '1', '--data', name,
                '--device', 'cpu', '--hours', '.02', '--parallel', '2', '--batch', '2',
                '--simulations', '2', '--snapshot-every', '8']
            result = cli(*command, '--max-games', '8')
            self.assertEqual(result.returncode, 0, result.stderr)
            first = training.load_state(root / 'latest.pt', Rules())
            self.assertEqual((first['games'], first['step'], first['last_milestone']), (8, 32, 8))
            self.assertTrue(all(row['board'].numel() == 81 for row in first['replay']))
            torch.manual_seed(first['settings']['seed'])
            initial = Network(**first['config'])
            self.assertFalse(torch.equal(first['weights']['policy.weight'], initial.policy.weight))
            self.assertFalse((root / 'initial.pt').exists())
            self.assertFalse((root / 'rules.json').exists())
            self.assertIn('settings', json.loads((root / 'run.json').read_text()))
            milestone = root / 'models' / 'model-00000008.pt'
            frozen = milestone.read_bytes()
            best = json.loads((root / 'incumbent.json').read_text())
            self.assertTrue(best['initial_baseline'])
            self.assertEqual(best['file'], 'models/model-00000008.pt')
            for path in sorted((root / 'selfplay').glob('*.json')):
                record = json.loads(path.read_text())
                self.assertTrue(Game.from_moves(record['moves']).done)
                self.assertTrue(record['complete'])
                self.assertNotIn('policy', json.dumps(record))
                data = load_checkpoint(root / 'replay' / (path.stem + '.pt'))
                self.assertTrue(all('policy' in row and 'result' in row for row in data['samples']))
            first['selfplay_counts'] = {'heuristic': 1234}  # Older checkpoint metadata must survive cleanup.
            save_checkpoint(root / 'latest.pt', first, 20 * GIB, root=root)
            result = cli(*command, '--simulations', '7', '--seed', '99', '--max-games', '12')
            self.assertEqual(result.returncode, 0, result.stderr)
            second = training.load_state(root / 'latest.pt')
            self.assertEqual(second['games'], 12)
            self.assertEqual(second['settings'], first['settings'])
            self.assertEqual(second['selfplay_counts'], first['selfplay_counts'])
            self.assertEqual(milestone.read_bytes(), frozen)
            self.assertEqual(len(list((root / 'selfplay').glob('*.json'))), 12)
            self.assertEqual(len(list((root / 'replay').glob('*.pt'))), 12)
            metric = json.loads(next((root / 'metrics').glob('*.json')).read_text())
            self.assertEqual(len(metric), 32)
            self.assertTrue(all(np.isfinite(row['loss']) for row in metric))
            self.assertTrue(list(root.glob('source-*.zip')))
            original = (root / 'latest.pt').read_bytes()
            result = cli(*command, '--connect', '6', '--max-games', '12')
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('rules', result.stderr.lower())
            self.assertEqual((root / 'latest.pt').read_bytes(), original)
            self.assertEqual(json.loads((root / 'status.json').read_text())['next_milestone'], 16)

    def test_connect6_style_direct_cli_and_full_bot_turn(self):
        with tempfile.TemporaryDirectory() as name:
            rules = Rules(13, 13, 6, 2, 1)
            legacy = Path(name) / 'rules.json'
            save_json(legacy, {'rules': rules.id, 'rule_config': rules.to_dict()})
            original_identity = legacy.read_bytes()
            result = cli('train.py', '--connect', '6', '--board_size', '13*13', '--stones_per_turn', '2',
                         '--starter-stones', '1', '--data', name, '--device', 'cpu', '--hours', '.02',
                         '--max-games', '2', '--parallel', '2', '--simulations', '2')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(legacy.read_bytes(), original_identity)
            saved = training.load_state(Path(name) / 'latest.pt', rules)
            self.assertEqual(saved['games'], 2)
            for path in (Path(name) / 'selfplay').glob('*.json'):
                self.assertTrue(Game.from_moves(json.loads(path.read_text())['moves'], rules=rules).done)
            result = cli('-m', 'engine', 'play', '--data', name, '--model', 'latest', '--device', 'cpu',
                         '--seconds', '.02', stdin=json.dumps({'moves': [84]}))
            self.assertEqual(result.returncode, 0, result.stderr)
            reply = json.loads(result.stdout)
            self.assertEqual(len(reply['moves']), 2)
            game = Game.from_moves([84, *reply['moves']], rules=rules)
            self.assertEqual((game.player, game.left), (1, 2))
            result = cli('-m', 'engine', 'play', '--data', name, '--model', '../latest.pt',
                         '--device', 'cpu', stdin=json.dumps({'moves': [84]}))
            self.assertNotEqual(result.returncode, 0)
            original = (Path(name) / 'latest.pt').read_bytes()
            result = cli('train.py', '--board_size', '9*13', '--data', name, '--device', 'cpu')
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('square', result.stderr)
            self.assertEqual((Path(name) / 'latest.pt').read_bytes(), original)

    def test_export_failure_recovers_from_durable_pending_dataset(self):
        with tempfile.TemporaryDirectory() as name:
            args = SimpleNamespace(data=name, rules=Rules(), disk_gib=20, device='cpu', simulations=2,
                parallel=2, batch=2, seed=5070, tactical_ms=2, snapshot_every=1000, seconds=5, hours=.02, max_games=8)
            original = training._immutable_json
            def disk_full(root, record, value, cap):
                if record.startswith('selfplay/'):
                    raise OSError('simulated full disk during export')
                original(root, record, value, cap)
            with patch.object(training, '_immutable_json', disk_full):
                with self.assertRaises(OSError):
                    training.train(args)
            durable = training.load_state(Path(name) / 'latest.pt')
            self.assertEqual(durable['games'], 8)
            self.assertEqual(len(durable['pending_games']), 8)
            training.train(args)
            self.assertEqual(len(list((Path(name) / 'selfplay').glob('*.json'))), 8)
            self.assertEqual(len(list((Path(name) / 'replay').glob('*.pt'))), 8)

    def test_frozen_gate_resumes_100_legal_color_pairs_and_checks_promotion(self):
        device = device_for('cpu')
        rules = Rules(4, 4, 4)
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            net = Network(channels=4, blocks=1, size=4)
            entries = []
            for games in (8, 16):
                filename = f'models/model-{games:08d}.pt'
                save_checkpoint(root / filename, training.snapshot(net, 0, rules, games), 20 * GIB, root=root)
                entries.append({'id': Path(filename).stem, 'file': filename, 'games': games,
                                'sha256': training.digest(root / filename)})
            opponent, candidate = entries
            save_json(root / 'incumbent.json', {**opponent, 'kind': 'network', 'rules': rules.id})
            report = {'rules': rules.id, 'candidate': candidate, 'opponent': opponent, 'incumbent': opponent['id'],
                'opening_seed': 90000016, 'seconds_per_turn': .02, 'matches': [], 'current': None,
                'max_turn_overrun_seconds': 0.0, 'promoted': False, 'code_sha256': training._source()[1]}
            filename = 'gate-model-00000016.json'
            save_json(root / filename, report)
            self.assertEqual(history(root)['gates'][0]['games'], 0)
            self.assertIsNone(history(root)['gates'][0]['score'])
            calls = [0]
            def interrupt():
                calls[0] += 1
                return calls[0] > 11
            cache = {}
            partial = training.run_gate(root, filename, rules, device, time.monotonic() + 30, interrupt,
                                        20 * GIB, model_cache=cache)
            self.assertFalse(partial['complete'])
            self.assertTrue(partial['matches'] or partial['current'])
            frozen = [(root / entry['file']).read_bytes() for entry in entries]
            cached_models = list(cache.values())
            self.assertEqual(len(cached_models), 2)
            for entry, original in zip(entries, frozen):
                path = root / entry['file']
                path.write_bytes(b'changed after caching')
                with self.assertRaisesRegex(ValueError, 'checksum'):
                    training.run_gate(root, filename, rules, device, time.monotonic() + 30,
                                      lambda: False, 20 * GIB, model_cache=cache)
                path.write_bytes(original)
            result = training.run_gate(root, filename, rules, device, time.monotonic() + 30,
                                       lambda: False, 20 * GIB, model_cache=cache)
            self.assertEqual(list(cache.values()), cached_models)
            self.assertTrue(result['decision_recorded'])
            self.assertEqual((result['games'], result['required_games'], result['score']), (100, 100, .5))
            self.assertFalse(result['promoted'])
            summary = history(root)['gates'][0]
            self.assertEqual((summary['games'], summary['score'], summary['incumbent']), (100, .5, opponent['id']))
            self.assertEqual([(root / entry['file']).read_bytes() for entry in entries], frozen)
            for i, row in enumerate(result['matches']):
                self.assertTrue(Game.from_moves(row['moves'], rules=rules).done)
                self.assertEqual(row['candidate_color'], 1 if i % 2 == 0 else -1)
                if i % 2:
                    self.assertEqual(row['opening'], result['matches'][i - 1]['opening'])
            # Replace one draw/loss with an actually played win: strictly more than 50/100 qualifies.
            better = copy.deepcopy(result)
            index = next(i for i, row in enumerate(better['matches']) if row['score'] < 1)
            row = better['matches'][index]
            rng = np.random.default_rng(42)
            for _ in range(1000):
                game = Game.from_moves(row['opening'], rules=rules)
                while not game.done:
                    game.play(int(rng.choice(np.flatnonzero(game.board == 0))))
                if game.winner == row['candidate_color']:
                    row.update(moves=game.moves, score=1.0)
                    break
            self.assertEqual(row['score'], 1.0)
            self.assertTrue(training.verify_gate(better, rules))
            late = copy.deepcopy(better)
            late['max_turn_overrun_seconds'] = .11
            self.assertFalse(training.verify_gate(late, rules))
            invalid = copy.deepcopy(better)
            invalid['matches'][1]['candidate_color'] = 1
            with self.assertRaises(ValueError):
                training.verify_gate(invalid, rules)
            training._finish_gate(root, root / filename, better, rules, 20 * GIB)
            best = json.loads((root / 'incumbent.json').read_text())
            self.assertEqual(best['file'], candidate['file'])
            self.assertEqual(best['previous']['file'], opponent['file'])
            self.assertEqual([(root / entry['file']).read_bytes() for entry in entries], frozen)
            training._finish_gate(root, root / filename, better, rules, 20 * GIB)  # Crash-safe publication retry.

    def test_two_stone_gate_preserves_overtime_across_resume(self):
        with tempfile.TemporaryDirectory() as name:
            root, rules = Path(name), Rules(5, 5, 5, 2, 1)
            report = training._gate_report(rules, {'games': 16}, {},
                {'gate_seconds': .25, 'simulations': 2}, training._source()[1])
            save_json(root / 'gate.json', report)
            clock, budgets = [0.], []
            durations = iter((.31, .06, .125))
            def action(game, net, seconds, *a, **kw):
                self.assertGreaterEqual(seconds, 0.)
                budgets.append(seconds)
                clock[0] += next(durations)
                return int(game.actions()[0])
            with patch.object(training, 'network', return_value=None), \
                    patch.object(training, 'turn_action', side_effect=action), \
                    patch.object(training.time, 'monotonic', lambda: clock[0]):
                first = training.run_gate(root, 'gate.json', rules, 'cpu', .31, lambda: False, 20 * GIB)
                self.assertAlmostEqual(first['max_turn_overrun_seconds'], .06)
                # Reload the persisted report between stones, just like a slice or process restart.
                second = training.run_gate(root, 'gate.json', rules, 'cpu', .37, lambda: False, 20 * GIB)
                self.assertAlmostEqual(second['max_turn_overrun_seconds'], .12)
                self.assertEqual(second['current']['turn_remaining'], .25)
                third = training.run_gate(root, 'gate.json', rules, 'cpu', .495, lambda: False, 20 * GIB)
                self.assertAlmostEqual(third['max_turn_overrun_seconds'], .12)
            self.assertEqual(budgets, [.125, 0., .125])

    def test_slow_gate_loading_does_not_starve_scheduled_evaluation(self):
        device_for('cpu')
        with tempfile.TemporaryDirectory() as name:
            root, rules = Path(name), Rules(4, 4, 4)
            net = Network(channels=4, blocks=1, size=4)
            settings = dict(seed=5070, parallel=2, batch=2, simulations=2, bootstrap_games=16,
                updates_per_cycle=32, replay_limit=20_000, tactical_ms=0, snapshot_every=8, gate_seconds=.25)
            for games in (8, 16):
                save_checkpoint(root / f'models/model-{games:08d}.pt', training.snapshot(net, 0, rules, games),
                                20 * GIB, root=root)
            opponent = {'file': 'models/model-00000008.pt', 'games': 8, 'kind': 'network', 'rules': rules.id,
                        'sha256': training.digest(root / 'models/model-00000008.pt')}
            save_json(root / 'incumbent.json', opponent)
            gate = training.next_gate(root, rules, settings, training._source()[1], 20 * GIB)
            state = training.snapshot(net, 0, rules, 16)
            state.update(settings=settings, optimizer=torch.optim.AdamW(net.parameters()).state_dict(),
                since_update=0, replay=[], active=[], last_milestone=16, gate=gate,
                pending_games=[], pending_metrics=[], loss_metrics=None,
                phase_seconds={'training': 80., 'evaluation': 0.},
                rng=np.random.default_rng(5070).bit_generator.state, torch_rng=torch.get_rng_state())
            save_checkpoint(root / 'latest.pt', state, 20 * GIB, root=root)
            args = SimpleNamespace(data=name, rules=rules, disk_gib=20, device='cpu', parallel=None,
                                   seconds=None, hours=.02, max_games=16, restart_gate=False)
            clock, loads = [0.], []
            original = training.load_state
            def slow_load(path, rules=None):
                if path.parent.name == 'models':
                    loads.append(path.name)
                    clock[0] += 1.1  # Two loads exceed the entire two-second evaluation slice.
                return original(path, rules)
            def action(game, *a, **kw):
                clock[0] += .05
                return int(game.actions()[0])
            with patch.object(training, 'load_state', side_effect=slow_load), \
                    patch.object(training, 'turn_action', side_effect=action) as moves, \
                    patch.object(training.time, 'monotonic', lambda: clock[0]), \
                    patch.object(training, 'train_step', side_effect=AssertionError('No training expected')), \
                    patch.object(training, 'selfplay_batch', side_effect=AssertionError('No self-play expected')):
                training.train(args)
            self.assertGreater(moves.call_count, 0)
            self.assertCountEqual(loads, ['model-00000008.pt', 'model-00000016.pt'])
            saved = training.load_state(root / 'latest.pt')
            self.assertEqual((saved['games'], saved['step']), (16, 0))
            # Setup still consumes evaluation credit; caching must not hide it from the 80/20 budget.
            self.assertAlmostEqual(saved['phase_seconds']['evaluation'], 2.2 + .05 * moves.call_count)
            self.assertLessEqual(saved['phase_seconds']['evaluation'], 20.)

    def test_evaluation_quota_keeps_training_and_survives_resume(self):
        self.assertEqual(training.eval_allowance({'training': 80., 'evaluation': 20.}), 0.)
        self.assertEqual(training.eval_allowance({'training': 80., 'evaluation': 10.}), 10.)
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            args = SimpleNamespace(data=name, rules=Rules(4, 4, 4), disk_gib=20, device='cpu', simulations=2,
                parallel=8, batch=2, seed=5070, tactical_ms=0, snapshot_every=8, seconds=.02,
                hours=.02, max_games=24, restart_gate=False)
            clock = [0.]
            original_batch, original_step, original_action = training.selfplay_batch, training.train_step, training.turn_action
            def timed_batch(*a, **kw):
                self.assertLessEqual(len(a[0]), args.parallel)
                result = original_batch(*a, **kw)
                clock[0] += 1.
                return result
            def timed_step(*a, **kw):
                result = original_step(*a, **kw)
                clock[0] += .1
                return result
            def timed_action(*a, **kw):
                self.assertEqual(a[3], args.simulations)
                result = original_action(*a, **kw)
                clock[0] += .05
                return result
            with patch.object(training.time, 'monotonic', lambda: clock[0]), \
                    patch.object(training, 'selfplay_batch', timed_batch), \
                    patch.object(training, 'train_step', timed_step), \
                    patch.object(training, 'turn_action', timed_action):
                training.train(args)
                state = training.load_state(root / 'latest.pt')
                self.assertEqual(state['games'], 24)
                self.assertTrue((root / 'models/model-00000024.pt').exists())
                report = json.loads((root / state['gate']).read_text())
                self.assertFalse(report.get('decision_recorded', False))
                self.assertTrue(report['matches'] or report['current'])
                self.assertLessEqual(state['phase_seconds']['evaluation'], state['phase_seconds']['training'] / 4 + .05)
                before = dict(state['phase_seconds'])
                training.train(args)
                resumed = training.load_state(root / 'latest.pt')
                self.assertEqual(resumed['phase_seconds']['training'], before['training'])
                self.assertLessEqual(resumed['phase_seconds']['evaluation'], before['training'] / 4 + .05)
                args.parallel, args.max_games, args.hours = 2, 32, .1
                training.train(args)
                smaller = training.load_state(root / 'latest.pt')
                self.assertEqual(smaller['games'], 32)
                self.assertEqual(smaller['settings']['parallel'], 2)
            # Saved snapshots form the queue. After a tie, the next one still challenges the old best.
            report.update(decision_recorded=True, promoted=False)
            save_json(root / state['gate'], report)
            next_name = training.next_gate(root, args.rules, state['settings'], training._source()[1], 20 * GIB)
            next_report = json.loads((root / next_name).read_text())
            self.assertEqual(next_report['candidate']['games'], 24)
            self.assertEqual(next_report['opponent']['games'], 8)
            self.assertEqual(next_report['simulations'], 2)

    def test_gate_restart_preserves_old_report_and_immutable_models(self):
        with tempfile.TemporaryDirectory() as name:
            root, rules = Path(name), Rules(4, 4, 4)
            net = Network(channels=4, blocks=1, size=4)
            for games in (8, 16):
                save_checkpoint(root / f'models/model-{games:08d}.pt', training.snapshot(net, 0, rules, games), 20 * GIB, root=root)
            opponent_path = root / 'models/model-00000008.pt'
            opponent = {'id': opponent_path.stem, 'file': 'models/' + opponent_path.name, 'games': 8,
                        'sha256': training.digest(opponent_path), 'kind': 'network', 'rules': rules.id}
            save_json(root / 'incumbent.json', opponent)
            settings = {'gate_seconds': 5., 'simulations': 64}
            gate = training.next_gate(root, rules, settings, training._source()[1], 20 * GIB)
            report = json.loads((root / gate).read_text())
            report['current'] = {'moves': [8, 0, 1, 2]}
            report['elapsed_seconds'] = 120.
            report['code_sha256'] = 'older-search-code'
            save_json(root / gate, report)
            original = (root / gate).read_bytes()
            frozen = [path.read_bytes() for path in sorted((root / 'models').glob('*.pt'))]
            settings['gate_seconds'] = .25
            restarted = training.restart_gate(root, gate, rules, settings, training._source()[1], 20 * GIB)
            self.assertEqual(restarted['matches'], [])
            self.assertIsNone(restarted['current'])
            self.assertEqual(restarted['seconds_per_turn'], .25)
            self.assertEqual(restarted['simulations'], 64)
            self.assertEqual((root / restarted['restart_from']).read_bytes(), original)
            self.assertEqual([path.read_bytes() for path in sorted((root / 'models').glob('*.pt'))], frozen)
            # A changed frozen weight must fail before the report or archive changes.
            opponent_path.write_bytes(b'changed weights')
            unchanged = (root / gate).read_bytes()
            with self.assertRaises(ValueError):
                training.restart_gate(root, gate, rules, settings, training._source()[1], 20 * GIB)
            self.assertEqual((root / gate).read_bytes(), unchanged)

    def test_pre_consolidation_checkpoint_resumes_without_old_packages(self):
        with tempfile.TemporaryDirectory() as name:
            root, rules = Path(name), Rules(4, 4, 4)
            net = Network(channels=4, blocks=1, size=4)
            settings = dict(seed=5070, parallel=2, batch=2, simulations=2, bootstrap_games=16,
                updates_per_cycle=32, replay_limit=20_000, tactical_ms=0, snapshot_every=8, gate_seconds=5.)
            for games in (8, 16):
                save_checkpoint(root / f'models/model-{games:08d}.pt', training.snapshot(net, 32, rules, games),
                                20 * GIB, root=root)
            opponent = {'file': 'models/model-00000008.pt', 'games': 8, 'kind': 'network', 'rules': rules.id,
                        'sha256': training.digest(root / 'models/model-00000008.pt')}
            save_json(root / 'incumbent.json', opponent)
            save_json(root / 'run.json', {'rules': rules.id, 'rule_config': rules.to_dict(), 'settings': settings})
            gate = training.next_gate(root, rules, settings, 'before-engine-package', 20 * GIB)
            state = training.snapshot(net, 32, rules, 16)
            state['config'] = {**state['config'], 'width': 4}  # Older square checkpoints used this field.
            game = Game.from_moves([5], rules=rules)
            policy = np.zeros(16, dtype=np.float32); policy[0] = 1
            sample = {**observation(game, policy), 'result': 1., 'source': 'mcts'}
            state.update(settings=settings, optimizer=torch.optim.AdamW(net.parameters()).state_dict(),
                since_update=0, replay=[sample], active=[{'moves': [5], 'opening': [5], 'samples': [],
                'turns': [], 'strategies': [None, None], 'plans': [[], []]}], last_milestone=16, gate=gate,
                pending_games=[], pending_metrics=[], loss_metrics=None, legacy_note={'preserve': True},
                rng=np.random.default_rng(5070).bit_generator.state, torch_rng=torch.get_rng_state())
            save_checkpoint(root / 'latest.pt', state, 20 * GIB, root=root)
            original = (root / 'latest.pt').read_bytes()
            old_report = (root / gate).read_bytes()
            frozen = [path.read_bytes() for path in sorted((root / 'models').glob('*.pt'))]
            command = ['train.py', '--data', name, '--connect', '4', '--board_size', '4*4',
                       '--device', 'cpu', '--hours', '.01', '--max-games', '16']
            result = cli(*command)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('--restart-gate', result.stderr)
            self.assertEqual((root / 'latest.pt').read_bytes(), original)
            result = cli(*command, '--restart-gate')
            self.assertEqual(result.returncode, 0, result.stderr)
            resumed = training.load_state(root / 'latest.pt', rules)
            self.assertEqual((resumed['games'], resumed['step']), (16, 32))
            self.assertEqual(resumed['active'], state['active'])
            self.assertEqual(resumed['legacy_note'], state['legacy_note'])
            self.assertEqual(resumed['rng'], state['rng'])
            torch.testing.assert_close(resumed['torch_rng'], state['torch_rng'], rtol=0, atol=0)
            torch.testing.assert_close(resumed['weights'], state['weights'], rtol=0, atol=0)
            self.assertEqual(len(resumed['replay']), 1)
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    torch.testing.assert_close(resumed['replay'][0][key], value, rtol=0, atol=0)
                else:
                    self.assertEqual(resumed['replay'][0][key], value)
            self.assertEqual(resumed['optimizer'], state['optimizer'])
            self.assertEqual([path.read_bytes() for path in sorted((root / 'models').glob('*.pt'))], frozen)
            report = json.loads((root / gate).read_text())
            self.assertEqual((root / report['restart_from']).read_bytes(), old_report)
            self.assertEqual(report['seconds_per_turn'], .25)
            self.assertFalse((PROJECT / 'connect6').exists())
            self.assertFalse((PROJECT / 'gomoku').exists())

    def test_retired_fixed_format_is_rejected_without_overwriting_data(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            save_checkpoint(root / 'latest.pt', {'format': 1, 'rules': 'connect6',
                'config': {'size': 19}, 'weights': {}}, 20 * GIB, root=root)
            original = (root / 'latest.pt').read_bytes()
            result = cli('train.py', '--data', name, '--device', 'cpu', '--hours', '.01')
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Checkpoint rules', result.stderr)
            self.assertEqual((root / 'latest.pt').read_bytes(), original)
            self.assertFalse((root / 'run.json').exists())

    def test_large_selfplay_batch_reaches_network_together(self):
        device_for('cpu')
        net = Network(channels=4, blocks=1, size=9)
        sizes, evaluate = [], net.evaluate
        def counted(games):
            sizes.append(len(games))
            return evaluate(games)
        net.evaluate = counted
        games = [Game.from_moves([40, 0, 41]) for _ in range(64)]
        decisions = selfplay_batch(games, net, 2, np.random.default_rng(0), tactical_ms=0)
        self.assertEqual(sizes, [64, 64, 64])
        self.assertTrue(all(row['source'] == 'mcts' for row in decisions))


if __name__ == '__main__':
    unittest.main()
