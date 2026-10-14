"""Ghost-note classifier: is a Basic Pitch note a real note or an overtone / artefact?

Basic Pitch fires on the 2nd/3rd/4th harmonics of guitar notes (octave, octave+5th, two octaves up).  On a
synthetic render of a known tab, ~40-70 % of its extra notes are such ghosts, and hand-tuned salience rules only
remove a fraction.  This module learns the decision from labelled synthetic data instead:

  * `make_material` generates random but *playable* guitar parts (chord shapes from the fingering module,
    strums / arpeggios with let-ring / single-note lines / double-stops, random tempo, key, register);
  * they are rendered with several GM guitar patches from two soundfonts, with and without reverb;
  * Basic Pitch runs on the audio, every detected note is labelled (real if a true note of that pitch starts within
    50 ms, otherwise ghost), and relative features are computed (activation of the note vs. the notes an octave /
    a 12th / two octaves below, onset strength, duration, polyphony, register);
  * a gradient-boosted classifier is trained (scripts/train_ghost.py) and stored in mp3gp/models/ghost.joblib.
"""
import random
from pathlib import Path

import numpy as np

from . import bp
from .common import TUNING, MAXFRET

MODEL_PATH = Path(__file__).resolve().parent / "models" / "ghost.joblib"
FEATURES = ["conf", "onset_peak", "note_act", "r_oct", "r_12th", "r_2oct", "r_oct_onset", "r_12th_onset",
            "up_oct", "dur", "poly", "pitch", "rank_in_onset", "r_strongest"]


def _win(a, t0, t1):
    i0 = int(t0 * bp.FPS)
    i1 = max(i0 + 1, int(t1 * bp.FPS))
    return a[i0:i1]


def features(notes, pg):
    """notes: decoded Basic Pitch notes (start, end, p, conf). Returns (N, F) array."""
    N, O = pg["note"], pg["onset"]
    X = []
    starts = np.array([n["start"] for n in notes]) if notes else np.zeros(0)
    for n in notes:
        k = n["p"] - 21
        t0, t1 = n["start"], min(n["end"], n["start"] + 0.12)
        def act(kk, arr=N):
            if not 0 <= kk < arr.shape[1]:
                return 1e-3
            w = _win(arr[:, kk], t0 - 0.02, t1)
            return float(w.max()) + 1e-3 if len(w) else 1e-3
        a = act(k)
        o = act(k, O)
        co = [m for i, m in enumerate(notes) if abs(starts[i] - n["start"]) < 0.05]
        confs = sorted([m["conf"] for m in co], reverse=True)
        rank = confs.index(n["conf"]) if n["conf"] in confs else 0
        X.append([n["conf"], o, a,
                  np.log(a / act(k - 12)), np.log(a / act(k - 19)), np.log(a / act(k - 24)),
                  np.log(o / act(k - 12, O)), np.log(o / act(k - 19, O)),
                  np.log(a / act(k + 12)),
                  np.log(max(n["end"] - n["start"], 0.01)), len(co), (n["p"] - 60) / 12.0, rank,
                  np.log(n["conf"] / (confs[0] + 1e-6))])
    return np.array(X, dtype=np.float32).reshape(len(notes), len(FEATURES))


_MODEL = None


def load_model():
    global _MODEL
    if _MODEL is None and MODEL_PATH.exists():
        import joblib
        _MODEL = joblib.load(MODEL_PATH)
    return _MODEL


def p_real(notes, pg):
    m = load_model()
    if m is None or not notes:
        return np.ones(len(notes))
    return m.predict_proba(features(notes, pg))[:, 1]


# ---------------------------------------------------------------- synthetic training material
SHAPES = [  # (string->fret) for strings 6..1, None = muted; movable shapes are shifted along the neck
    [None, 0, 2, 2, 2, 0], [0, 2, 2, 1, 0, 0], [3, 2, 0, 0, 0, 3], [None, None, 0, 2, 3, 2], [None, 3, 2, 0, 1, 0],
    [0, 2, 2, 0, 0, 0], [None, 0, 2, 2, 1, 0], [None, None, 0, 2, 3, 1], [3, 2, 0, 0, 3, 3], [None, 0, 2, 0, 2, 0],
    [1, 3, 3, 2, 1, 1], [1, 3, 3, 1, 1, 1], [None, 1, 3, 3, 3, 1], [None, 1, 3, 3, 2, 1], [None, 1, 3, 1, 3, 1],
    [1, 3, 1, 2, 1, 1], [None, None, 1, 3, 4, 3], [None, 1, 3, 3, 1, None], [1, 3, 3, None, None, None],
    [None, 1, 3, 3, None, None], [None, None, 1, 3, 3, None], [None, 2, 1, 3, 4, None],
]


def _shape_notes(shape, shift):
    out = []
    for i, f in enumerate(shape):
        if f is None:
            continue
        s = 5 - i  # string index 0 = high e
        fr = f + (shift if f > 0 or shift == 0 else shift)
        if shift and f == 0:
            fr = shift  # barre over the open strings
        if 0 <= fr <= MAXFRET:
            out.append((s, fr, TUNING[s] + fr))
    return out


def make_material(seed, seconds=60.0):
    """Random playable guitar part -> list of (start, end, pitch, velocity)."""
    rng = random.Random(seed)
    bpm = rng.uniform(70, 170)
    beat = 60 / bpm
    t = 0.3
    notes = []
    while t < seconds:
        mode = rng.choice(["arp", "arp", "strum", "line", "dyad", "arp_high"])
        shape = rng.choice(SHAPES)
        shift = rng.choice([0, 0, 1, 2, 3, 5, 7, 9, 12]) if any(f and f > 0 for f in shape) else 0
        ch = _shape_notes(shape, shift)
        if len(ch) < 2:
            continue
        bars = rng.choice([1, 2])
        sub = rng.choice([2, 4, 4])  # 8ths or 16ths
        steps = int(bars * 4 * sub)
        dt = beat / sub
        if mode == "strum":
            for k in range(0, steps, rng.choice([sub, sub * 2])):
                vel = rng.randint(60, 120)
                for j, (_, _, p) in enumerate(sorted(ch, key=lambda c: -c[0])):
                    s = t + k * dt + j * rng.uniform(0.004, 0.015)
                    notes.append((s, s + rng.uniform(0.3, 1.2) * beat * 2, p, vel))
        elif mode in ("arp", "arp_high"):
            order = sorted(ch, key=lambda c: -c[0])
            if mode == "arp_high":
                order = order[len(order) // 2:] + [rng.choice(order)]
            pat = [rng.choice(order) for _ in range(steps)] if rng.random() < 0.3 else \
                [order[k % len(order)] for k in range(steps)]
            ring = rng.random() < 0.7
            last = {}
            for k, (s_, f_, p) in enumerate(pat):
                if rng.random() < 0.15:
                    continue
                st = t + k * dt + rng.gauss(0, 0.008)
                if s_ in last:  # same string re-picked stops the previous note
                    i = last[s_]
                    notes[i] = (notes[i][0], min(notes[i][1], st), notes[i][2], notes[i][3])
                notes.append((st, st + (rng.uniform(1.0, 3.0) * beat if ring else dt * 0.9), p, rng.randint(50, 115)))
                last[s_] = len(notes) - 1
        elif mode == "line":
            pos = rng.randint(0, 12)
            for k in range(steps):
                if rng.random() < 0.2:
                    continue
                s_ = rng.randint(0, 5)
                p = TUNING[s_] + pos + rng.randint(0, 3)
                st = t + k * dt + rng.gauss(0, 0.008)
                notes.append((st, st + dt * rng.uniform(0.6, 2.0), p, rng.randint(55, 120)))
        else:  # dyads / octaves / power-chord pairs
            for k in range(0, steps, rng.choice([1, 2, sub])):
                base = rng.choice(ch)[2]
                iv = rng.choice([12, 7, 5, 3, 4, 9, 19])
                st = t + k * dt
                v = rng.randint(55, 115)
                for p in (base, base + iv):
                    if p <= TUNING[0] + MAXFRET:
                        notes.append((st, st + dt * rng.uniform(0.8, 2.5), p, v))
        t += steps * dt + rng.choice([0, 0, beat])
    return [n for n in notes if n[0] < seconds]
