"""Shared constants, paths and small I/O helpers."""
import json
import os
import subprocess
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

SR = 44100
TUNING = [64, 59, 55, 50, 45, 40]  # string 1 (high e) .. string 6 (low E), standard tuning
MAXFRET = 19
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def note_name(p):
    return f"{NOTE_NAMES[p % 12]}{p // 12 - 1}"


class Ctx:
    """Run context: where inputs, intermediate files and outputs live."""

    def __init__(self, inp, work, out, force=()):
        self.inp = Path(inp).expanduser().resolve()
        self.work = Path(work).expanduser().resolve()
        self.out = Path(out).expanduser().resolve()
        self.force = set(force)
        self.work.mkdir(parents=True, exist_ok=True)
        self.out.mkdir(parents=True, exist_ok=True)

    def w(self, *p):
        q = self.work.joinpath(*p)
        q.parent.mkdir(parents=True, exist_ok=True)
        return q

    def o(self, *p):
        q = self.out.joinpath(*p)
        q.parent.mkdir(parents=True, exist_ok=True)
        return q

    def need(self, stage, *paths):
        """True if a stage must (re)run: forced, or any output missing."""
        return stage in self.force or "all" in self.force or not all(Path(p).exists() for p in paths)


def load(path, mono=False):
    x, sr = sf.read(str(path), always_2d=True)
    assert sr == SR, f"{path}: expected {SR} Hz, got {sr}"
    x = x.T.astype(np.float32)  # (ch, n)
    if x.shape[0] == 1:
        x = np.vstack([x, x])
    return x.mean(0) if mono else x


def save(path, x):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2 and x.shape[0] <= 2:
        x = x.T
    sf.write(str(path), x, SR)


def jdump(path, obj):
    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    Path(path).write_text(json.dumps(obj, indent=1, default=conv))


def jload(path):
    return json.loads(Path(path).read_text())


def sh(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode:
        raise RuntimeError(f"{cmd[0]} failed: {r.stderr[-2000:]}")
    return r.stdout


def log(*a):
    print("[mp3gp]", *a, flush=True)
