"""Beat / downbeat grid (tempo may vary), meter and key detection."""
import numpy as np
import librosa

from .common import SR, NOTE_NAMES, jdump, jload, load, log


def track_beats(wav):
    from beat_this.inference import File2Beats

    f2b = File2Beats(checkpoint_path="final0", device="mps", dbn=False)
    beats, downbeats = f2b(str(wav))
    return np.asarray(beats, float), np.asarray(downbeats, float)


def clean_beats(beats, dur):
    """Drop spurious beats (too close), fill gaps (missed beats), extend to the end of the audio."""
    b = list(beats)
    changed = True
    while changed:
        changed = False
        ibi = np.diff(b)
        med = np.median(ibi)
        for i, d in enumerate(ibi):
            if d < 0.6 * med:
                # drop whichever of the pair makes the neighbourhood more regular
                cand = [i, i + 1]
                best = min(cand, key=lambda k: _irreg(np.delete(np.array(b), k), max(k - 2, 0), k + 2, med))
                del b[best]
                changed = True
                break
    out = [b[0]]
    med = np.median(np.diff(b))
    for t in b[1:]:
        gap = t - out[-1]
        n = int(round(gap / med))
        if n >= 2:
            out += list(np.linspace(out[-1], t, n + 1)[1:-1])
        out.append(t)
    while out[-1] + med < dur:
        out.append(out[-1] + med)
    return np.array(out)


def _irreg(b, lo, hi, med):
    seg = np.diff(b[lo:hi + 1])
    return float(np.abs(seg - med).sum()) if len(seg) else 0.0


def smooth_beats(beats, half=6):
    """Local linear regression over +-half beats: removes tracker frame jitter, keeps real tempo drift."""
    n = len(beats)
    out = np.empty(n)
    idx = np.arange(n)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        a, b = np.polyfit(idx[lo:hi], beats[lo:hi], 1)
        out[i] = a * i + b
    return out


def meter_and_phase(beats, downbeats):
    idx = np.array([int(np.argmin(np.abs(beats - d))) for d in downbeats])
    best = None
    for m in (3, 4, 2, 5, 6, 7):
        counts = np.bincount(idx % m, minlength=m)
        frac = counts.max() / len(idx)
        # prefer small meters that explain nearly all downbeats
        score = frac - 0.02 * m
        if best is None or score > best[0]:
            best = (score, m, int(counts.argmax()), frac)
    _, m, phase, frac = best
    if m in (6,) and np.bincount(idx % 3, minlength=3).max() / len(idx) > 0.9:
        m, phase = 3, int(np.bincount(idx % 3).argmax())
    return m, phase, frac


def build_grid(beats, meter, phase):
    """Bars as lists of beat times; beats before the first full downbeat are extrapolated backwards."""
    beats = list(beats)
    med = float(np.median(np.diff(beats)))
    while phase != 0:
        # prepend extrapolated beats so the first bar is complete (pickup handled as rests)
        if beats[0] - med < 0:
            beats = beats[(meter - phase):]  # drop the incomplete pickup instead
            phase = 0
            break
        beats.insert(0, beats[0] - med)
        phase = (phase + 1) % meter
    nb = len(beats) // meter
    bars = [beats[i * meter:(i + 1) * meter] for i in range(nb)]
    ends = [bars[i + 1][0] if i + 1 < nb else bars[i][-1] + med for i in range(nb)]
    return [dict(beats=bars[i], start=bars[i][0], end=ends[i]) for i in range(nb)]


def smooth_tempo(bars, tol=2.0):
    """Piecewise-constant integer tempo per bar: only change when the local tempo drifts > tol BPM."""
    raw = np.array([60.0 * len(b["beats"]) / (b["end"] - b["start"]) for b in bars])
    sm = np.convolve(np.pad(raw, 2, mode="edge"), np.ones(5) / 5, mode="valid")
    # segment where the smoothed tempo leaves the running region mean by > tol, then use region means
    segs, start = [], 0
    for i in range(1, len(sm) + 1):
        if i == len(sm) or abs(sm[i] - sm[start:i].mean()) > tol:
            segs.append((start, i))
            start = i
    out = [0] * len(raw)
    for a, b in segs:
        v = int(round(raw[a:b].mean()))
        for i in range(a, b):
            out[i] = v
    return raw, out


KS_MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KS_MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def detect_key(y):
    ch = librosa.feature.chroma_cqt(y=y, sr=SR, hop_length=2048, bins_per_octave=36)
    prof = ch.sum(1)
    prof = prof / prof.sum()
    res = []
    for k in range(12):
        for mode, P in (("major", KS_MAJ), ("minor", KS_MIN)):
            r = np.corrcoef(prof, np.roll(P, k))[0, 1]
            res.append((float(r), f"{NOTE_NAMES[k]} {mode}"))
    res.sort(reverse=True)
    return res[:4], prof


def run(ctx, guitar_stem):
    path = ctx.w("rhythm.json")
    if not ctx.need("rhythm", path):
        return jload(path)
    mix = load(ctx.w("song.wav"))
    dur = mix.shape[1] / SR
    beats_raw, downs = track_beats(ctx.w("song.wav"))
    beats = smooth_beats(clean_beats(beats_raw, dur))
    meter, phase, frac = meter_and_phase(beats, downs)
    bars = build_grid(beats, meter, phase)
    raw_bpm, gp_bpm = smooth_tempo(bars)
    key, prof = detect_key(load(guitar_stem, mono=True))
    for b, r, g in zip(bars, raw_bpm, gp_bpm):
        b["bpm_raw"], b["bpm"] = float(r), g
    info = dict(meter=meter, downbeat_consistency=frac, n_beats_raw=len(beats_raw), n_beats=len(beats),
                bpm_median=float(60 / np.median(np.diff(beats))), bpm_min=float(raw_bpm.min()),
                bpm_max=float(raw_bpm.max()), key=key[0][1], key_candidates=key,
                pitch_class_profile=prof.tolist(), bars=bars, duration=dur)
    jdump(path, info)
    log(f"rhythm: {meter}/4, {len(bars)} bars, median {info['bpm_median']:.1f} BPM "
        f"(bar range {info['bpm_min']:.1f}-{info['bpm_max']:.1f}), key {info['key']}")
    return info
