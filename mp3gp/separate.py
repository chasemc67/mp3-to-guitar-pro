"""Source separation.

Stage 1 (model separators): demucs htdemucs / htdemucs_6s / htdemucs_ft and BS-Roformer (no-drum-bass) give
candidate "all guitars" stems.  They are scored per bar (drum leakage vs. harmonic content lost to the residual)
and the best stem is chosen per section (crossfaded) -> guitars.wav.

Stage 2 (spatial): model separators never split two guitars from each other, so guitars.wav is decomposed with
primary/ambient extraction (per time-frequency tile PCA of the stereo covariance):
  * primary = the coherent, amplitude-panned part; its pan angle per tile splits it into centre / left / right;
  * ambient = the decorrelated residual (double-tracked guitars hard-panned L/R, stereo effects, reverb);
    its L and R channels are kept separately so a doubled part can be checked for L/R agreement.
Mid/side signals are written too (centre-panned parts cancel perfectly in S).
"""
import glob
import shutil
from pathlib import Path

import librosa
import numpy as np
from scipy.ndimage import uniform_filter

from .common import SR, jdump, jload, load, log, save, sh

DEMUCS = ["htdemucs", "htdemucs_6s", "htdemucs_ft"]
NFFT, HOP = 4096, 1024


def run_models(ctx, py):
    song = ctx.w("song.wav")
    for m in DEMUCS:
        if ctx.need("demucs", ctx.w("stems", m, "song", "drums.wav")):
            log(f"demucs {m} ...")
            sh([py, "-m", "demucs", "-n", m, "-d", "mps", "-o", str(ctx.w("stems")), str(song)])
    bdir = ctx.w("stems", "bsroformer")
    target = bdir / "nodrumbass.wav"
    if ctx.need("demucs", target):
        try:
            log("bs-roformer (no drum-bass) ...")
            exe = str(Path(py).parent / "audio-separator")
            sh([exe, str(song), "-m", "model_bs_roformer_ep_937_sdr_10.5309.ckpt", "--output_dir", str(bdir),
                "--model_file_dir", str(ctx.w("models"))])
            f = [x for x in glob.glob(str(bdir / "*No Drum-Bass*"))][0]
            sh(["ffmpeg", "-y", "-loglevel", "error", "-i", f, "-ar", str(SR), str(target)])
        except Exception as e:  # optional separator
            log(f"bs-roformer unavailable: {e}")


def candidates(ctx):
    c = {
        "htdemucs_6s:guitar": ctx.w("stems/htdemucs_6s/song/guitar.wav"),
        "htdemucs:other": ctx.w("stems/htdemucs/song/other.wav"),
        "htdemucs_ft:other": ctx.w("stems/htdemucs_ft/song/other.wav"),
        "bs_roformer:no_drum_bass": ctx.w("stems/bsroformer/nodrumbass.wav"),
    }
    out = {k: load(v) for k, v in c.items() if Path(v).exists()}
    # the 6-stem model sometimes puts guitar into "other"/"piano": add them back (they are guitar here if anything)
    six = ctx.w("stems/htdemucs_6s/song")
    if "htdemucs_6s:guitar" in out:
        out["htdemucs_6s:guitar+other+piano"] = out["htdemucs_6s:guitar"] + load(six / "other.wav") + load(six / "piano.wav")
    n = min(x.shape[1] for x in out.values())
    out = {k: v[:, :n] for k, v in out.items()}
    out["ensemble_mean"] = np.mean([out[k] for k in c if k in out], axis=0)
    return out


def _mag(x):
    return np.abs(librosa.stft(x.mean(0), n_fft=NFFT, hop_length=HOP))


def score_candidates(ctx, cands, bars):
    """Per-bar quality of each candidate guitar stem.

    leak = TF overlap with the drum stem (min(|stem|,|drums|)^2 / |stem|^2): drum bleed into the stem.
    loss = harmonic (HPSS) energy left in mix - stem relative to the stem: guitar the separator threw away.
    score = -(leak + loss) in dB-ish units (higher is better).
    """
    mix = load(ctx.w("song.wav"))
    drums = load(ctx.w("stems/htdemucs_ft/song/drums.wav"))
    n = min(x.shape[1] for x in cands.values())
    D = _mag(drums[:, :n])
    fr = lambda t: min(int(t * SR / HOP), D.shape[1])
    table = {}
    for k, x in cands.items():
        S = _mag(x)
        Rh, _ = librosa.decompose.hpss(_mag(mix[:, :n] - x), margin=2.0)
        leak, loss = [], []
        for b in bars:
            s = slice(fr(b["start"]), fr(b["end"]))
            e = (S[:, s] ** 2).sum() + 1e-9
            leak.append(10 * np.log10((np.minimum(S[:, s], D[:, s]) ** 2).sum() / e + 1e-6))
            loss.append(10 * np.log10((Rh[:, s] ** 2).sum() / e + 1e-6))
        table[k] = dict(leak=leak, loss=loss)
    return table


def choose_per_section(table, bars, seg=4):
    """Pick the best candidate per `seg`-bar section (sum of per-bar scores; leakage weighted a bit higher,
    because drum transients become fake onsets in transcription)."""
    keys = list(table)
    choice = []
    for i in range(0, len(bars), seg):
        sc = {k: -np.mean(1.3 * np.array(table[k]["leak"][i:i + seg]) + np.array(table[k]["loss"][i:i + seg]))
              for k in keys}
        best = max(sc, key=sc.get)
        choice += [best] * len(bars[i:i + seg])
    return choice


def assemble(cands, choice, bars, fade=0.05):
    n = min(x.shape[1] for x in cands.values())
    out = np.zeros((2, n), np.float32)
    w_total = np.zeros(n, np.float32)
    nf = int(fade * SR)
    for i, b in enumerate(bars):
        a = int(b["start"] * SR) if i else 0
        e = int(b["end"] * SR) if i + 1 < len(bars) else n
        a0, e1 = max(0, a - nf), min(n, e + nf)
        w = np.ones(e1 - a0, np.float32)
        if a0 < a:
            w[:a - a0] = np.linspace(0, 1, a - a0)
        if e1 > e:
            w[-(e1 - e):] = np.linspace(1, 0, e1 - e)
        out[:, a0:e1] += w * cands[choice[i]][:, a0:e1]
        w_total[a0:e1] += w
    return out / np.maximum(w_total, 1e-6)


def primary_ambient(x, smooth=(3, 7)):
    """Per-tile PCA primary/ambient extraction. Returns (primary LR STFTs, ambient LR STFTs, pan deg, coherence)."""
    L = librosa.stft(x[0], n_fft=NFFT, hop_length=HOP)
    R = librosa.stft(x[1], n_fft=NFFT, hop_length=HOP)
    f = lambda z: uniform_filter(z, size=smooth, mode="nearest")
    rll, rrr = f(np.abs(L) ** 2), f(np.abs(R) ** 2)
    rlr = f((L * np.conj(R)).real) + 1j * f((L * np.conj(R)).imag)
    # principal eigenvector of [[rll, rlr],[conj(rlr), rrr]]
    tr, det = rll + rrr, rll * rrr - np.abs(rlr) ** 2
    lam = tr / 2 + np.sqrt(np.maximum(tr ** 2 / 4 - det, 0))
    vl, vr = rlr, lam - rll  # eigenvector (unnormalised)
    small = np.abs(vl) ** 2 + np.abs(vr) ** 2 < 1e-12
    vl = np.where(small, 1 / np.sqrt(2), vl)
    vr = np.where(small, 1 / np.sqrt(2), vr)
    nrm = np.sqrt(np.abs(vl) ** 2 + np.abs(vr) ** 2)
    vl, vr = vl / nrm, vr / nrm
    proj = np.conj(vl) * L + np.conj(vr) * R
    PL, PR = vl * proj, vr * proj
    AL, AR = L - PL, R - PR
    pan = np.degrees(np.arctan2(np.abs(vr), np.abs(vl))) - 45.0  # -45 hard left .. +45 hard right
    coh = np.abs(rlr) / np.sqrt(rll * rrr + 1e-20)
    return (PL, PR), (AL, AR), pan, coh


def istft(Z, n):
    return librosa.istft(Z, hop_length=HOP, n_fft=NFFT, length=n)


def spatial_components(x, centre_width=7.0):
    n = x.shape[1]
    (PL, PR), (AL, AR), pan, coh = primary_ambient(x)
    wc = np.exp(-0.5 * (pan / centre_width) ** 2)  # soft pan masks on the primary
    wr = (1 - wc) * (pan > 0)
    wl = (1 - wc) * (pan <= 0)
    comp = {
        "centre": np.vstack([istft(PL * wc, n), istft(PR * wc, n)]),
        "right": np.vstack([istft(PL * wr, n), istft(PR * wr, n)]),
        "left": np.vstack([istft(PL * wl, n), istft(PR * wl, n)]),
        "ambient": np.vstack([istft(AL, n), istft(AR, n)]),
        "mid": np.vstack([(x[0] + x[1]) / 2] * 2),
        "side": np.vstack([(x[0] - x[1]) / 2] * 2),
    }
    comp["ambient_L"] = np.vstack([comp["ambient"][0]] * 2)
    comp["ambient_R"] = np.vstack([comp["ambient"][1]] * 2)
    return comp


def run(ctx, py, bars):
    meta_p = ctx.w("separation.json")
    gpath = ctx.w("guitars.wav")
    if not ctx.need("separate", meta_p, gpath, ctx.w("comp", "ambient.wav")):
        return jload(meta_p)
    run_models(ctx, py)
    cands = candidates(ctx)
    table = score_candidates(ctx, cands, bars)
    choice = choose_per_section(table, bars)
    g = assemble(cands, choice, bars)
    save(gpath, g)
    comp = spatial_components(g)
    for k, v in comp.items():
        save(ctx.w("comp", f"{k}.wav"), v)
    summary = {k: dict(mean_leak_db=float(np.mean(v["leak"])), mean_loss_db=float(np.mean(v["loss"])))
               for k, v in table.items()}
    meta = dict(candidates=summary, per_bar=table, choice=choice, components=sorted(comp))
    jdump(meta_p, meta)
    from collections import Counter
    log("separation choice per section:", dict(Counter(choice)))
    return meta
