"""Bar-by-bar comparison of a rendered tab against reference audio (part stems / guitars stem / original mix).

Each bar of both signals is resampled to FPB frames (bar-relative time), so the tab's GP tempo map and the
recording's detected beat grid are aligned bar to bar even if they drift.
  chroma  : mean cosine similarity of CQT chroma frames (frames where the reference is silent are skipped)
  pitch_f : frame-level multi-pitch F-measure between binarised Basic Pitch note posteriorgrams (midi 40-88)
            -- the same transcriber on both sides, so timbre differences of the synth matter less
  onset_f : onset F-measure (+-50 ms) of Basic Pitch onset-posteriorgram peaks, render vs reference
  score   : mean of the three
"""
import librosa
import mir_eval
import numpy as np
from scipy.signal import find_peaks

from .common import SR
from . import bp

FPB = 24
P_LO, P_HI = 40, 88
ONSET_THR = 0.5


def _bar_frames(M, fps, bounds):
    out = []
    for a, b in bounds:
        ia, ib = a * fps, b * fps
        idx = np.clip(np.round(np.linspace(ia, ib, FPB, endpoint=False)).astype(int), 0, M.shape[1] - 1)
        out.append(M[:, idx])
    return out


def features(x_mono, bounds, pg=None, wav_for_bp=None, cache_dir=None):
    hop = 512
    ch = librosa.feature.chroma_cqt(y=x_mono, sr=SR, hop_length=hop, bins_per_octave=36)
    rms = librosa.feature.rms(y=x_mono, hop_length=hop)[0]
    if pg is None:
        pg = bp.posteriorgram(wav_for_bp, cache_dir)
    note = pg["note"][:, P_LO - 21:P_HI - 21 + 1].T  # (49, frames)
    # onsets = peaks of Basic Pitch's onset posteriorgram (max over guitar pitches); same detector on both sides.
    # (Broadband / CQT flux detectors missed ~40% of the re-picked notes inside ringing arpeggios on the synth.)
    o = pg["onset"][:, P_LO - 21:P_HI - 21 + 1].max(1)
    pk, _ = find_peaks(o, height=ONSET_THR, distance=max(1, int(0.05 * bp.FPS)))
    on = pk / bp.FPS
    fps = SR / hop
    chb = _bar_frames(ch, fps, bounds)
    rmsb = _bar_frames(rms[None], fps, bounds)
    nb = _bar_frames(note, bp.FPS, bounds)
    ons = [((on[(on >= a) & (on < b)] - a) / (b - a)) for a, b in bounds]
    durs = [b - a for a, b in bounds]
    return dict(chroma=chb, rms=rmsb, note=nb, onsets=ons, durs=durs)


def compare(fr, fo, bars=None, thr_ref=0.3, thr_ren=0.3):
    """fr: reference features, fo: rendered features. Returns list of per-bar dicts."""
    n = len(fr["chroma"])
    bars = range(n) if bars is None else bars
    out = []
    for b in bars:
        A, B = fr["chroma"][b], fo["chroma"][b]
        live = fr["rms"][b][0] > 1e-4
        if live.sum() >= 3:
            cs = (A * B).sum(0) / (np.linalg.norm(A, axis=0) * np.linalg.norm(B, axis=0) + 1e-9)
            chroma = float(cs[live].mean())
        else:
            chroma = np.nan
        R = fr["note"][b] > thr_ref
        O = fo["note"][b] > thr_ren
        tp = (R & O).sum()
        p = tp / max(O.sum(), 1)
        r = tp / max(R.sum(), 1)
        pf = 2 * p * r / (p + r) if (p + r) else (1.0 if R.sum() == 0 and O.sum() == 0 else 0.0)
        ro, oo = fr["onsets"][b], fo["onsets"][b]
        win = 0.05 / fr["durs"][b]
        if len(ro) == 0 and len(oo) == 0:
            of = 1.0
        elif len(ro) == 0 or len(oo) == 0:
            of = 0.0
        else:
            of = mir_eval.onset.f_measure(np.sort(ro), np.sort(oo), window=win)[0]
        vals = [v for v in (chroma, pf, of) if not np.isnan(v)]
        out.append(dict(bar=b, chroma=chroma, pitch_f=float(pf), pitch_p=float(p), pitch_r=float(r),
                        onset_f=float(of), score=float(np.mean(vals)) if vals else np.nan))
    return out


def summary(rows):
    keys = ["chroma", "pitch_f", "onset_f", "score"]
    return {k: float(np.nanmean([r[k] for r in rows])) for k in keys} | {"n_bars": len(rows)}
