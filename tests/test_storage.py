"""Run-scoped accounting must remove scans, not cap or recovery guarantees."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
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
