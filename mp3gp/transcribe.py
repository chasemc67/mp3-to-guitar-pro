"""Per-part note transcription onto the beat grid.

candidates (Basic Pitch on the part stem, thresholds from params)
  -> salience filter (harmonic NMF activation: absolute vs. bar level, relative to simultaneous notes;
     removes overtone ghosts / weak octave doubles / mush)
  -> ownership filter (energy share of this part's stem vs. the other part stems at the note's harmonics;
     removes bleed from other guitars)
  -> quantise onset/offset to the 16th grid of the detected (tempo-varying) beat grid.
"""
import librosa
import numpy as np

from .common import MAXFRET, SR, TUNING, load
from . import bp, salience

DEFAULT = dict(onset=0.45, frame=0.28, min_len_ms=70, rel_min=0.18, abs_min=0.10, share_min=0.40,
               octave_rel=0.35, max_poly=6, drum_max=3.0)
SUB = 4  # grid slots per beat (16ths in x/4)


class Grid:
    def __init__(self, bars, meter):
        self.meter = meter
        self.beats = np.array([t for b in bars for t in b["beats"]] + [bars[-1]["end"]])
        self.bars = bars
        self.spb = meter * SUB  # slots per bar

    def pos(self, t):
        """time (s) -> position in slots (float), linear between beats, extrapolated at the ends"""
        idx = np.arange(len(self.beats))
        if t <= self.beats[0]:
            return (t - self.beats[0]) / (self.beats[1] - self.beats[0]) * SUB
        if t >= self.beats[-1]:
            return (len(self.beats) - 1 + (t - self.beats[-1]) / (self.beats[-1] - self.beats[-2])) * SUB
        return float(np.interp(t, self.beats, idx)) * SUB

    def time(self, slot):
        b = slot / SUB
        i = int(np.clip(np.floor(b), 0, len(self.beats) - 2))
        return float(self.beats[i] + (b - i) * (self.beats[i + 1] - self.beats[i]))


class StemEnergy:
    """STFT power of a stem, for 'who owns this note' energy comparisons."""

    def __init__(self, path):
        self.x = load(path, mono=True)
        self.S = np.abs(librosa.stft(self.x, n_fft=4096, hop_length=512)) ** 2

    def harm_energy(self, p, t0, t1):
        f0 = librosa.midi_to_hz(p)
        a = int(t0 * SR / 512)
        b = max(a + 2, int(min(t1, t0 + 0.12) * SR / 512))
        ks = [int(round(f0 * h * 4096 / SR)) + d for h in (1, 2, 3) for d in (-1, 0, 1)]
        ks = [k for k in ks if k < self.S.shape[0]]
        return float(self.S[ks, a:b].sum())


_DRUMS = {}


def drums_for(ctx):
    p = ctx.w("stems/htdemucs_ft/song/drums.wav")
    if p not in _DRUMS:
        _DRUMS[p] = StemEnergy(p)
    return _DRUMS[p]


class PartAudio(StemEnergy):
    """Cached analysis of one part stem (posteriorgram, NMF activations, STFT for ownership)."""

    def __init__(self, ctx, part):
        self.drums = drums_for(ctx)
        self.part = part
        self.x = load(part["stem"], mono=True)
        self.pg = bp.posteriorgram(part["stem"], ctx.w("bp"))
        cache = ctx.w("nmf", part["id"] + ".npz")
        if cache.exists() and "transcribe" not in ctx.force:
            d = np.load(cache)
            self.P, self.W, self.H = d["P"], d["W"], d["H"]
        else:
            self.P, self.W, self.H = salience.salience_for(self.x, self.pg["note"], bp.FPS)
            np.savez_compressed(cache, P=self.P, W=self.W, H=self.H)
        self.S = np.abs(librosa.stft(self.x, n_fft=4096, hop_length=512)) ** 2


def candidates(pa, prm, t0, t1):
    notes = bp.decode(pa.pg, onset=prm["onset"], frame=prm["frame"], min_len_ms=prm["min_len_ms"])
    return [n for n in notes if t0 <= n["start"] < t1]


def filter_notes(notes, pa, others, prm, grid, bar_level):
    H = pa.H
    out = []
    for n in notes:
        s = salience.note_strength(H, n["p"], n["start"], n["end"])
        n["sal"] = s
    for n in notes:
        co = [m for m in notes if abs(m["start"] - n["start"]) < 0.06]
        mx = max(m["sal"] for m in co)
        bar = int(grid.pos(n["start"]) // grid.spb)
        lvl = bar_level.get(bar, mx)
        n["rel"] = n["sal"] / (mx + 1e-9)
        n["abs"] = n["sal"] / (lvl + 1e-9)
        if n["rel"] < prm["rel_min"] or n["abs"] < prm["abs_min"]:
            continue
        # octave-above ghost: a note 12 (or 19, 24) above a stronger simultaneous note needs to be strong itself
        lower = [m for m in co if n["p"] - m["p"] in (12, 19, 24) and m["sal"] > n["sal"]]
        if lower and n["sal"] < prm["octave_rel"] * max(m["sal"] for m in lower):
            continue
        e_self = pa.harm_energy(n["p"], n["start"], n["end"])
        n["drum_ratio"] = pa.drums.harm_energy(n["p"], n["start"], n["end"]) / (e_self + 1e-12)
        if n["drum_ratio"] > prm["drum_max"]:
            continue
        if others:
            e_oth = sum(o.harm_energy(n["p"], n["start"], n["end"]) for o in others)
            n["share"] = e_self / (e_self + e_oth + 1e-12)
            if n["share"] < prm["share_min"]:
                continue
        out.append(n)
    return out


def bar_levels(pa, grid, nbars):
    """95th percentile of NMF activation per bar (the reference for 'absolute' note strength)."""
    lv = {}
    for b in range(nbars):
        a = int(grid.bars[b]["start"] * SR / salience.HOP)
        e = int(grid.bars[b]["end"] * SR / salience.HOP)
        seg = pa.H[:, a:e]
        lv[b] = float(np.percentile(seg.max(0), 95)) if seg.size else 1.0
    return lv


def quantise(notes, grid, prm, bar_range):
    out = {}
    lo, hi = bar_range[0] * grid.spb, bar_range[1] * grid.spb
    for n in notes:
        s = int(round(grid.pos(n["start"])))
        e = int(round(grid.pos(n["end"])))
        if not lo <= s < hi:
            continue
        p = n["p"]
        if p < 40:
            p += 12 * int(np.ceil((40 - p) / 12))
        if p > TUNING[0] + MAXFRET:  # above the 19th fret of the high e: almost always an overtone
            continue
        d = int(np.clip(e - s, 1, 2 * grid.spb))
        key = (s, p)
        v = int(np.clip(40 + 87 * min(1.0, n.get("abs", 0.7)), 40, 127))
        if key in out:
            out[key]["dur"] = max(out[key]["dur"], d)
            out[key]["vel"] = max(out[key]["vel"], v)
        else:
            out[key] = dict(slot=s, p=p, dur=d, vel=v, sal=n.get("sal", 0))
    # polyphony cap per onset
    by = {}
    for q in out.values():
        by.setdefault(q["slot"], []).append(q)
    res = []
    for s, qs in by.items():
        qs.sort(key=lambda q: -q["sal"])
        res += qs[:prm["max_poly"]]
    res.sort(key=lambda q: (q["slot"], q["p"]))
    return res


def transcribe(pa, others, grid, bar_range, prm=None):
    prm = dict(DEFAULT, **(prm or {}))
    t0 = grid.bars[bar_range[0]]["start"] - 0.05
    t1 = grid.bars[bar_range[1] - 1]["end"]
    notes = candidates(pa, prm, t0, t1)
    lv = bar_levels(pa, grid, len(grid.bars))
    kept = filter_notes(notes, pa, others, prm, grid, lv)
    return quantise(kept, grid, prm, bar_range)
