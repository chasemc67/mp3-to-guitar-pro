"""Harmonic NMF pitch salience.

Basic Pitch's posteriorgram fires on overtones (octave/12th ghost notes) and on bleed. Here a CQT magnitude
spectrogram V (bins x frames) is factorised V ~ W H with one template per guitar pitch whose non-zero entries are
restricted to that pitch's harmonic bins (multiplicative updates keep zeros at zero, so templates stay harmonic but
learn this recording's timbre).  H is initialised from Basic Pitch.  Because the true fundamental's template
already explains its own overtones, ghost notes end up with little activation.
"""
import librosa
import numpy as np

from .common import SR

LO, HI = 38, 90  # D2 .. F#6 (covers standard-tuned guitar incl. drop-ish low notes and 21st+ fret)
BPO = 36
FMIN = librosa.midi_to_hz(LO - 1)
NB = BPO * 7
HOP = 512
NH = 12  # harmonics per template


def cqt(x):
    C = np.abs(librosa.cqt(x, sr=SR, hop_length=HOP, fmin=FMIN, n_bins=NB, bins_per_octave=BPO))
    return C.astype(np.float32)


def init_templates():
    P = np.arange(LO, HI + 1)
    W = np.zeros((NB, len(P)), np.float32)
    for j, p in enumerate(P):
        f0 = librosa.midi_to_hz(p)
        for h in range(1, NH + 1):
            f = f0 * h * (1 + 0.0002 * h * h)  # slight string inharmonicity
            b = BPO * np.log2(f / FMIN)
            if b >= NB - 1:
                break
            for d in (-1, 0, 1):
                k = int(round(b)) + d
                if 0 <= k < NB:
                    W[k, j] += np.exp(-0.5 * ((k - b) / 0.6) ** 2) / h ** 0.9
    return P, W


def fit(V, H0, iters=80, lock_W=False, W0=None):
    P, W = init_templates()
    if W0 is not None:
        W = W0.copy()
    mask = W > 0
    H = np.maximum(H0, 1e-3).astype(np.float32)
    eps = 1e-9
    scale = V.max() + eps
    V = V / scale
    for _ in range(iters):
        WH = W @ H + eps
        H *= (W.T @ (V / WH)) / (W.T.sum(1, keepdims=True) + eps)  # KL-divergence updates
        if not lock_W:
            WH = W @ H + eps
            W *= ((V / WH) @ H.T) / (H.sum(1)[None, :] + eps)
            W *= mask
            # keep the fundamental present (stop templates from turning into "octave-up" templates)
            for j in range(W.shape[1]):
                col = W[:, j]
                nz = np.nonzero(col)[0]
                if len(nz) == 0:
                    continue
                f0bins = nz[:3]
                m = col.max()
                if col[f0bins].max() < 0.25 * m:
                    col[f0bins] = np.maximum(col[f0bins], 0.25 * m * (col[f0bins] > 0))
            W /= W.sum(0, keepdims=True) + eps
    return P, W, H * scale


def bp_to_H(note_post, n_frames, fps_bp):
    """Resample Basic Pitch note posterior (frames x 88, midi 21..) to NMF frames for pitches LO..HI."""
    t_nmf = np.arange(n_frames) * HOP / SR
    idx = np.clip(np.round(t_nmf * fps_bp).astype(int), 0, len(note_post) - 1)
    return note_post[idx][:, LO - 21:HI - 21 + 1].T.copy()


def salience_for(x_mono, note_post, fps_bp, W0=None):
    V = cqt(x_mono)
    H0 = bp_to_H(note_post, V.shape[1], fps_bp) + 0.02
    P, W, H = fit(V, H0, W0=W0, lock_W=W0 is not None, iters=60 if W0 is not None else 80)
    return P, W, H


def note_strength(H, p, t0, t1):
    """Peak activation of pitch p in [t0, t0+min(dur,0.15)]."""
    j = p - LO
    if not 0 <= j < H.shape[0]:
        return 0.0
    a = int(t0 * SR / HOP)
    b = max(a + 2, int(min(t1, t0 + 0.15) * SR / HOP))
    return float(H[j, a:b].max()) if b <= H.shape[1] else 0.0
