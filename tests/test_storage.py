"""Run-scoped accounting must remove scans, not cap or recovery guarantees."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import DEFAULT, patch
from engine import storage


class AccountingTests(unittest.TestCase):
    def test_locked_writes_scan_once_and_keep_exact_usage(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / 'old.bin').write_bytes(b'old')
            with patch('engine.storage.os.scandir', wraps=storage.os.scandir) as scans:
                with storage.run_lock(root):
                    initial_scans = scans.call_count
                    for i in range(12):
                        storage.atomic_bytes(root / 'nested' / 'state.bin', b'x' * (i + 1),
                                             cap=2 * storage.GIB, root=root)
                        # Uncapped status/recovery writes must also update the ledger.
                        storage.save_json(root / 'status.json', {'step': i})
                    self.assertEqual(scans.call_count, initial_scans)
                    expected = sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
                    self.assertEqual(storage.usage(root), expected)
                self.assertGreater(storage.usage(root), 0)
                self.assertGreater(scans.call_count, initial_scans)

    @unittest.skipUnless(storage.os.name == 'nt', 'Windows readers block atomic replacement')
    def test_json_write_survives_windows_reader(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            path = root / 'gate-model-00002000.json'
            storage.save_json(path, {'games': 1})
            replace = storage.os.replace
            blocked = []
            with storage.run_lock(root), path.open('rb') as reader:
                original = storage.usage(root)

                def replace_then_close_reader(source, destination):
                    try:
                        return replace(source, destination)
                    except PermissionError as exc:
                        blocked.append(exc.winerror)
                        raise
                    finally:
                        reader.close()  # The dashboard finishes its read before the retry.

                with patch('engine.storage.os.replace', side_effect=replace_then_close_reader) as commit:
                    storage.save_json(path, {'games': 2}, root=root, cap=2 * storage.GIB)
                self.assertEqual(blocked, [5])
                self.assertEqual(commit.call_count, 2)
                self.assertEqual(storage.load_json(path), {'games': 2})
                self.assertEqual(storage.usage(root), original)
                self.assertFalse(list(root.glob('*.tmp')))

    def test_replace_retries_windows_access_errors(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            path = root / 'state.bin'
            storage.atomic_bytes(path, b'previous')
            with storage.run_lock(root):
                for code in (5, 32, 33):
                    with self.subTest(winerror=code):
                        error = PermissionError('injected Windows file lock')
                        error.winerror = code
                        with patch('engine.storage.os.replace', wraps=storage.os.replace,
                                   side_effect=[error, DEFAULT]) as commit, \
                                patch('engine.storage.time.sleep') as sleep:
                            storage.atomic_bytes(path, b'new', cap=2 * storage.GIB, root=root)
                        self.assertEqual(commit.call_count, 2)
                        self.assertEqual(commit.call_args_list[0], commit.call_args_list[1])
                        sleep.assert_called_once_with(.05)
                        self.assertEqual(path.read_bytes(), b'new')
                        self.assertEqual(storage.usage(root), path.stat().st_size + 1)
                        self.assertFalse(list(root.glob('*.tmp')))

    def test_replace_failure_stops_retrying_and_preserves_previous(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            path = root / 'state.bin'
            storage.atomic_bytes(path, b'previous')
            with storage.run_lock(root):
                original = storage.usage(root)
                for code, attempts in ((5, 5), (32, 5), (33, 5), (None, 1), (13, 1)):
                    with self.subTest(winerror=code):
                        error = PermissionError('injected persistent access denial')
                        if code is not None:
                            error.winerror = code
                        with patch('engine.storage.os.replace', side_effect=error) as commit, \
                                patch('engine.storage.time.sleep') as sleep:
                            with self.assertRaises(PermissionError) as caught:
                                storage.atomic_bytes(path, b'new', cap=2 * storage.GIB, root=root)
                        self.assertIs(caught.exception, error)
                        self.assertEqual(commit.call_count, attempts)
                        self.assertEqual([call.args[0] for call in sleep.call_args_list],
                                         [.05 * 2 ** i for i in range(attempts - 1)])
                        self.assertEqual(path.read_bytes(), b'previous')
                        self.assertEqual(storage.usage(root), original)
                        self.assertFalse(list(root.glob('*.tmp')))

    def test_cap_failed_replace_and_restart_reconcile(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            path = root / 'state.bin'
            storage.atomic_bytes(path, b'previous')
            with storage.run_lock(root):
                original = storage.usage(root)
                with self.assertRaises(OSError):
                    storage.atomic_bytes(path, b'new', cap=original + 3 + 1024 * 1024 - 1, root=root)
                self.assertEqual(path.read_bytes(), b'previous')
                self.assertEqual(storage.usage(root), original)
                with patch('engine.storage.os.replace', side_effect=OSError('injected failure')):
                    with self.assertRaises(OSError):
                        storage.atomic_bytes(path, b'new', cap=2 * storage.GIB, root=root)
                self.assertEqual(path.read_bytes(), b'previous')
                self.assertEqual(storage.usage(root), original)
                self.assertFalse(list(root.glob('*.tmp')))
                storage.atomic_bytes(path, b'n', cap=2 * storage.GIB, root=root)
                self.assertEqual(storage.usage(root), original - len(b'previous') + 1)
            # Recovery counts interrupted temporary files and changes made between sessions.
            (root / 'leftover.tmp').write_bytes(b'interrupted')
            with storage.run_lock(root):
                expected = sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
                self.assertEqual(storage.usage(root), expected)
                with self.assertRaises(OSError):
                    storage.atomic_bytes(path, b'new', cap=expected + 3, root=root)
            # No run lock means no cached accounting; external changes cannot stay hidden.
            (root / 'outside.bin').write_bytes(b'new file')
            self.assertEqual(storage.usage(root), expected + len(b'new file'))


if __name__ == '__main__':
    unittest.main()
