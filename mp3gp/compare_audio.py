"""Listening material: original vs. tab playback for the same section, sequential and side by side."""
import numpy as np

from .common import SR, load, log, save, sh


def pick_section(rhythm, parts, secs):
    bars = rhythm["bars"]
    active = np.zeros(len(bars))
    for p in parts:
        for a, e in p.get("segments", [p["bars"]]):
            active[a:e] += 1
    best = None
    for s in range(len(bars)):
        e = s
        while e < len(bars) and bars[e]["end"] - bars[s]["start"] < secs:
            e += 1
        if e >= len(bars):
            break
        aud = all(b.get("audible", True) for b in bars[s:e + 1])
        sc = active[s:e + 1].sum() + (0 if aud else -1e3)
        if best is None or sc > best[0]:
            best = (sc, s, e)
    _, s, e = best
    return s, e, bars[s]["start"], bars[s]["start"] + secs


def _rms(x):
    return np.sqrt(np.mean(x ** 2) + 1e-12)


def make(ctx, rhythm, ev, parts, secs=30.0):
    orig = load(ctx.w("song.wav"))
    tab = load(ctx.w("render", "final_full.wav"))
    n = min(orig.shape[1], tab.shape[1])
    orig, tab = orig[:, :n], tab[:, :n]
    b0, b1, t0, t1 = pick_section(rhythm, parts, secs)
    a, e = int(t0 * SR), int(t1 * SR)
    o, t = orig[:, a:e], tab[:, a:e]
    t = t * (_rms(o) / _rms(t))
    pk = max(np.abs(o).max(), np.abs(t).max())
    o, t = o / pk * 0.95, t / pk * 0.95
    fade = np.ones(e - a)
    k = int(0.3 * SR)
    fade[:k] = np.linspace(0, 1, k)
    fade[-k:] = np.linspace(1, 0, k)
    gap = np.zeros((2, int(1.0 * SR)))
    save(ctx.o("comparison.wav"), np.hstack([o * fade, gap, t * fade]))
    save(ctx.o("comparison_side_by_side.wav"), np.vstack([o.mean(0) * fade, t.mean(0) * fade]))
    # whole song side by side as well (left = original, right = tab)
    tt = tab * (_rms(orig) / _rms(tab))
    pk = max(np.abs(orig).max(), np.abs(tt).max())
    save(ctx.w("compare_full_lr.wav"), np.vstack([orig.mean(0), tt.mean(0)]) / pk * 0.95)
    sh(["ffmpeg", "-y", "-loglevel", "error", "-i", str(ctx.w("compare_full_lr.wav")), "-b:a", "192k",
        str(ctx.o("comparison_side_by_side_full_song.mp3"))])
    # per-part: separated stem (L) vs that track's tab playback (R)
    for p in parts:
        st = load(p["stem"]).mean(0)[:n]
        rn = load(ctx.w("render", f"final_{p['id']}.wav")).mean(0)[:n]
        rn = rn * (_rms(st) / (_rms(rn) + 1e-9))
        pk = max(np.abs(st).max(), np.abs(rn).max()) + 1e-9
        save(ctx.w("cmp_part.wav"), np.vstack([st, rn]) / pk * 0.95)
        sh(["ffmpeg", "-y", "-loglevel", "error", "-i", str(ctx.w("cmp_part.wav")), "-b:a", "160k",
            str(ctx.o("parts", f"{p['id']}_stem_L_vs_tab_R.mp3"))])
        sh(["ffmpeg", "-y", "-loglevel", "error", "-i", p["stem"], "-b:a", "192k",
            str(ctx.o("parts", f"{p['id']}_separated_stem.mp3"))])
    info = dict(bars=[b0 + 1, b1 + 1], t0=t0, t1=t1)
    (ctx.w("comparison_section.txt")).write_text(f"{info}\n")
    log(f"comparison section: bars {b0 + 1}-{b1 + 1} ({t0:.1f}-{t1:.1f}s)")
    return info
