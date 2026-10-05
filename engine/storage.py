"""Atomic local artifacts, bounded disk usage, and a cross-process run lock."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
from threading import RLock
import time

GIB = 1024 ** 3
_ACCOUNTING = {}  # Only active, exclusively locked runs; never persisted across sessions.
_IO_LOCK = RLock()


def _disk_usage(root):
    total = 0
    try:
        with os.scandir(root) as it:
            for entry in it:
                try:
                    if entry.is_file(follow_symlinks=False):
                        # Windows directory metadata can lag the still-open lock file.
                        total += (os.stat(entry.path).st_size if entry.name == 'run.lock'
                                  else entry.stat().st_size)
                    elif entry.is_dir(follow_symlinks=False):
                        total += _disk_usage(entry.path)
                except FileNotFoundError:
                    continue  # A reader can race an atomic replacement.
    except FileNotFoundError:
        pass
    return total


def usage(root):
    root = Path(root).resolve()
    with _IO_LOCK:
        return _ACCOUNTING[root] if root in _ACCOUNTING else _disk_usage(root)


def _adjust_usage(path, delta):
    for root in _ACCOUNTING:
        if path.is_relative_to(root):
            _ACCOUNTING[root] += delta


def unlink(path, *, missing_ok=False):
    path = Path(path).resolve()
    with _IO_LOCK:
        size = path.stat().st_size if path.exists() else 0
        path.unlink(missing_ok=missing_ok)
        _adjust_usage(path, -size)


def atomic_bytes(path, payload, cap=None, *, root=None):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    root = path.parent if root is None else Path(root).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Artifact must stay inside its accounting root.")
    # ponytail: serialize in-process commits; per-run mutexes if concurrent writers matter.
    with _IO_LOCK:
        if cap is not None and usage(root) + len(payload) + 1024 * 1024 > cap:
            raise OSError("Artifact cap reached; previous checkpoints are intact. Increase --disk-gib to resume.")
        old_size = path.stat().st_size if path.exists() else 0
        fd, temp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            # Windows readers/scanners can briefly prevent an otherwise atomic replacement.
            for attempt in range(5):
                try:
                    os.replace(temp, path)
                    break
                except PermissionError as exc:
                    if getattr(exc, 'winerror', None) not in (5, 32, 33) or attempt == 4:
                        raise
                    time.sleep(.05 * 2 ** attempt)
            _adjust_usage(path, len(payload) - old_size)
        finally:
            try:
                if os.path.exists(temp):
                    os.unlink(temp)
            except OSError:
                # A leftover temp consumes space: discard ledgers rather than undercount it.
                _ACCOUNTING.clear()
                raise


def save_json(path, value, *, root=None, cap=None):
    atomic_bytes(path, (json.dumps(value, indent=2, allow_nan=False) + "\n").encode(), cap, root=root)


def load_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


def save_checkpoint(path, state, cap, *, root=None):
    import torch
    # ponytail: in-memory serialization is bounded by replay size; stream only if RAM profiling requires it.
    buffer = io.BytesIO()
    torch.save(state, buffer)
    atomic_bytes(path, buffer.getvalue(), cap, root=root)


def load_checkpoint(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=True)


@contextlib.contextmanager
def run_lock(root, *, accounting=True):
    """Cooperating writers hold this lock and use atomic_bytes/unlink for artifacts.

    Reconcile once after acquiring the lock, including interrupted temporary files.
    Unlocked readers/probes use live scans, never a stale persisted counter.
    """
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    stream = open(root / "run.lock", "a+b")
    with _IO_LOCK:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
            _adjust_usage(root / "run.lock", 1)
    stream.seek(0)
    acquired = False
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        acquired = True
        if accounting:
            with _IO_LOCK:
                _ACCOUNTING[root] = _disk_usage(root)
        yield
    finally:
        if acquired:
            if accounting:
                with _IO_LOCK:
                    _ACCOUNTING.pop(root, None)
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)
        stream.close()


def busy(root):
    try:
        with run_lock(root, accounting=False):
            return False
    except OSError:
        return True
