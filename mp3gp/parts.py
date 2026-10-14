"""Find the guitar parts and build one stem per part.

Model separators give one "all guitars" stem, so parts are found from stereo placement over time:
  centre  = coherent, centre-panned primary component (pan mask on the PCA primary)
  wide    = guitars - centre (panned and decorrelated material)
  L-only / R-only = tiles clearly louder on one side (ADRess-style), used to classify the wide material.
Per bar, the wide material is labelled 'right' / 'left' (one-sided) or 'double' (both sides at similar level,
low inter-channel coherence = double-tracked takes hard-panned L/R).  Contiguous runs of a label become parts.
"""
import librosa
import numpy as np

from .common import SR, jdump, jload, load, log, save
from . import separate


def _db(x):
    return 10 * np.log10(np.mean(x ** 2) + 1e-12)


def bar_energies(g, comps, bars):
    rows = []
    for b in bars:
        s, e = int(b["start"] * SR), int(b["end"] * SR)
        ref = _db(g[:, s:e])
        rows.append({k: _db(v[:, s:e]) - ref for k, v in comps.items()} | {"abs": ref})
    return rows


def label_bars(E, wide_on=-14.0, side_db=6.0):
    labels = []
    for r in E:
        if r["abs"] < -55 or r["wide"] < wide_on:
            labels.append(None)
        elif r["R_only"] - r["L_only"] > side_db:
            labels.append("right")
        elif r["L_only"] - r["R_only"] > side_db:
            labels.append("left")
        else:
            labels.append("double")
    # smooth: majority over a 5-bar window, then drop runs shorter than 4 bars
    sm = []
    for i in range(len(labels)):
        w = [x for x in labels[max(0, i - 2):i + 3]]
        sm.append(max(set(w), key=w.count))
    runs, i = [], 0
    while i < len(sm):
        j = i
        while j < len(sm) and sm[j] == sm[i]:
            j += 1
        runs.append([sm[i], i, j])
        i = j
    for r in runs:
        if r[0] is not None and r[2] - r[1] < 4:
            r[0] = None
    return [(lab, a, b) for lab, a, b in runs if lab is not None]


def run(ctx, bars):
    path = ctx.w("parts.json")
    if not ctx.need("parts", path):
        return jload(path)
    g = load(ctx.w("guitars.wav"))
    n = g.shape[1]
    centre = load(ctx.w("comp", "centre.wav"))[:, :n]
    wide = g - centre
    L = librosa.stft(g[0], n_fft=separate.NFFT, hop_length=separate.HOP)
    R = librosa.stft(g[1], n_fft=separate.NFFT, hop_length=separate.HOP)
    pan = np.degrees(np.arctan2(np.abs(R), np.abs(L))) - 45
    lonly = librosa.istft(L * (pan < -10), hop_length=separate.HOP, length=n)
    ronly = librosa.istft(R * (pan > 10), hop_length=separate.HOP, length=n)
    comps = dict(centre=centre, wide=wide, L_only=np.vstack([lonly, lonly]), R_only=np.vstack([ronly, ronly]))
    E = bar_energies(g, comps, bars)
    segs = label_bars(E)

    parts = [dict(id="gtr1", name="Gtr 1 (centre)", kind="centre", bars=[0, len(bars)],
                  stem=str(ctx.w("parts", "gtr1.wav")))]
    save(parts[0]["stem"], centre)
    # merge wide segments of the same label into one part (a part may drop out and come back)
    by_label = {}
    for lab, a, b in segs:
        by_label.setdefault(lab, []).append([a, b])
    order = sorted(by_label, key=lambda k: by_label[k][0][0])
    for i, lab in enumerate(order):
        pid = f"gtr{i + 2}"
        name = {"right": "panned right", "left": "panned left", "double": "wide, double-tracked L+R"}[lab]
        mask = np.zeros(n, np.float32)
        fade = int(0.05 * SR)
        for a, b in by_label[lab]:
            s, e = int(bars[a]["start"] * SR), int(bars[b - 1]["end"] * SR)
            mask[s:e] = 1
            mask[max(0, s - fade):s] = np.linspace(0, 1, s - max(0, s - fade))
            mask[e:e + fade] = np.linspace(1, 0, len(mask[e:e + fade]))
        if lab == "right":
            sig = np.vstack([wide[1], wide[1]])  # the right channel carries it; the left mostly carries leakage
        elif lab == "left":
            sig = np.vstack([wide[0], wide[0]])
        else:
            sig = wide
        stem = ctx.w("parts", f"{pid}.wav")
        save(stem, sig * mask)
        parts.append(dict(id=pid, name=f"Gtr {i + 2} ({name})", kind=lab, segments=by_label[lab],
                          bars=[by_label[lab][0][0], by_label[lab][-1][1]], stem=str(stem)))
    info = dict(parts=parts, bar_energy=E, segments=segs)
    jdump(path, info)
    for p in parts:
        log(f"part {p['id']}: {p['name']} bars {p['bars'][0] + 1}-{p['bars'][1]}")
    return info
