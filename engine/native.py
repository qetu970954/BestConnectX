"""Optional original C++ kernels through stdlib ctypes. Build: python -m engine.native."""
import ctypes
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import numpy as np

SOURCE = Path(__file__).with_suffix('.cpp')
CACHE = SOURCE.parent.parent / '.native-cache'


def library_path():
    extension = '.dll' if os.name == 'nt' else '.so'
    return CACHE / ('kernels-' + hashlib.sha256(SOURCE.read_bytes()).hexdigest()[:16] + extension)


@lru_cache(maxsize=1)
def library():
    if os.environ.get('BESTCONNECT6_NATIVE') == '0':
        return None
    path = library_path()
    if not path.exists():
        return None
    backend = ctypes.CDLL(str(path))
    array = np.ctypeslib.ndpointer
    backend.connection_features.argtypes = [
        array(dtype=np.int8, ndim=2, flags='C_CONTIGUOUS'),
        array(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),
        array(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),
        array(dtype=np.int32, ndim=2, flags='C_CONTIGUOUS'),
        *([ctypes.c_int] * 5), array(dtype=np.float32, ndim=3, flags='C_CONTIGUOUS'),
        array(dtype=np.uint8, ndim=3, flags='C_CONTIGUOUS'),
        array(dtype=np.uint8, ndim=1, flags='C_CONTIGUOUS')]
    backend.connection_features.restype = ctypes.c_int
    backend.puct_select.argtypes = [array(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        array(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),
        array(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'), ctypes.c_int]
    backend.puct_select.restype = ctypes.c_int
    return backend


def features(boards, players, left, lines, stones):
    backend = library()
    if backend is None or boards.dtype != np.int8:
        return None
    batch, area = boards.shape
    if players.shape != (batch,) or left.shape != (batch,):
        raise ValueError('Feature metadata must match the board batch.')
    planes = np.empty((batch, 8, area), dtype=np.float32)
    counts = np.empty((batch, 3, len(lines)), dtype=np.uint8)
    threats = np.empty(batch, dtype=np.uint8)
    code = backend.connection_features(boards, players, left, lines, batch, area,
                                       len(lines), lines.shape[1], stones, planes, counts, threats)
    if code:
        raise ValueError('Invalid native feature dimensions, player or line indices.')
    return planes, counts, threats


def select(prior, visits, total):
    backend = library()
    if backend is None:
        return None
    if not (len(prior) == len(visits) == len(total)):
        raise ValueError('PUCT arrays must have matching lengths.')
    index = backend.puct_select(prior, visits, total, len(prior))
    if index < 0:
        raise ValueError('PUCT needs at least one action.')
    return index


def build():
    CACHE.mkdir(exist_ok=True)
    output = library_path()
    if os.name == 'nt':
        installer = Path(os.environ.get('ProgramFiles(x86)', 'C:/Program Files (x86)'))
        vswhere = installer / 'Microsoft Visual Studio/Installer/vswhere.exe'
        install = subprocess.check_output([str(vswhere), '-latest', '-products', '*',
            '-requires', 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64',
            '-property', 'installationPath'], text=True).strip()
        if not install:
            raise RuntimeError('Install MSVC C++ build tools, or use the Python backend.')
        setup = Path(install) / 'Common7/Tools/VsDevCmd.bat'
        command = (f'call "{setup}" -no_logo -arch=x64 -host_arch=x64 && '
                   f'cl /nologo /O2 /std:c++17 /EHsc /MT /LD /fp:strict "{SOURCE}" '
                   f'/Fe:"{output}" /Fo:"{CACHE / "native.obj"}" /link /INCREMENTAL:NO')
        subprocess.run('cmd.exe /d /s /c "' + command + '"', cwd=CACHE, check=True)
    else:
        compiler = shutil.which('c++')
        if compiler is None:
            raise RuntimeError('Install a C++ compiler, or use the Python backend.')
        subprocess.run([compiler, '-O3', '-std=c++17', '-shared', '-fPIC',
                        '-ffp-contract=off', str(SOURCE), '-o', str(output)], check=True)
    library.cache_clear()
    print(f'Built {output}')


if __name__ == '__main__':
    build()
