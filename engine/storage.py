"""Atomic local artifacts, bounded disk usage, and a cross-process run lock."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile

GIB = 1024 ** 3


def usage(root):
    total = 0
    for path in Path(root).rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except FileNotFoundError:
            pass  # Atomic writes can replace temporary files while usage is measured.
    return total


def atomic_bytes(path, payload, cap=None, *, root=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if cap is not None and usage(path.parent if root is None else root) + len(payload) + 1024 * 1024 > cap:
        raise OSError("Artifact cap reached; previous checkpoints are intact. Increase --disk-gib to resume.")
    fd, temp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


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
def run_lock(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    stream = open(root / "run.lock", "a+b")
    if stream.tell() == 0:
        stream.write(b"0")
        stream.flush()
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
        yield
    finally:
        if acquired:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)
        stream.close()


def busy(root):
    try:
        with run_lock(root):
            return False
    except OSError:
        return True
