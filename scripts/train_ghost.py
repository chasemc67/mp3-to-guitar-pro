"""Train the ghost-note classifier (mp3gp/models/ghost.joblib) on synthetic guitar renders.

    .venv/bin/python scripts/train_ghost.py [--n 36] [--work /tmp/ghost]

Material: mp3gp.ghost.make_material (random playable parts) rendered with GM guitar patches 24-28 from
GeneralUser GS and FluidR3 (if present), reverb on/off.  Validation is leave-one-soundfont/patch-out.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pretty_midi

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mp3gp import bp, ghost  # noqa: E402
from mp3gp.common import SR, sh  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=36)
ap.add_argument("--work", default="/tmp/mp3gp_ghost")
a = ap.parse_args()
work = Path(a.work)
work.mkdir(parents=True, exist_ok=True)
root = Path(__file__).resolve().parent.parent
sfs = [str(p) for p in [root / "soundfonts/GeneralUser-GS.sf2", root / "soundfonts/FluidR3Mono_GM.sf3"] if p.exists()]
programs = [24, 25, 26, 27, 28]
X, Y, G = [], [], []
for i in range(a.n):
    mat = ghost.make_material(i)
    prog = programs[i % len(programs)]
    sf = sfs[(i // len(programs)) % len(sfs)]
    rev = i % 2
    pm = pretty_midi.PrettyMIDI()
    ins = pretty_midi.Instrument(prog)
    for s, e, p, v in mat:
        ins.notes.append(pretty_midi.Note(v, p, s, max(e, s + 0.05)))
    pm.instruments.append(ins)
    mid, wav = work / f"m{i}.mid", work / f"m{i}.wav"
    pm.write(str(mid))
    sh(["fluidsynth", "-ni", "-q", "-g", "0.6", "-R", str(rev), "-C", "0", "-r", str(SR),
        "-o", "synth.reverb.room-size=0.7", "-o", "synth.reverb.level=0.6", "-F", str(wav), sf, str(mid)])
    pg = bp.posteriorgram(wav, work)
    det = bp.decode(pg, onset=0.35, frame=0.22, min_len_ms=50)
    truth = {}
    for s, e, p, v in mat:
        truth.setdefault(p, []).append(s)
    y = [int(any(abs(t - n["start"]) < 0.05 for t in truth.get(n["p"], []))) for n in det]
    X.append(ghost.features(det, pg))
    Y += y
    G += [f"{Path(sf).stem}:{prog}"] * len(det)
    print(f"{i:3d} sf={Path(sf).stem[:12]} prog={prog} rev={rev}: {len(mat)} true, {len(det)} detected, "
          f"{sum(y)} real, {len(det) - sum(y)} ghost/false", flush=True)
X = np.vstack(X)
Y = np.array(Y)
G = np.array(G)
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
import joblib  # noqa: E402

print("\nleave-one-timbre-out validation (threshold 0.5):")
for g in sorted(set(G)):
    tr, te = G != g, G == g
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=31).fit(X[tr], Y[tr])
    pr = m.predict_proba(X[te])[:, 1] > 0.5
    tp = (pr & (Y[te] == 1)).sum()
    keep_real = tp / max(1, (Y[te] == 1).sum())
    drop_ghost = ((~pr) & (Y[te] == 0)).sum() / max(1, (Y[te] == 0).sum())
    print(f"  held out {g:24s}: real notes kept {keep_real:.3f}, ghosts removed {drop_ghost:.3f}, "
          f"precision after {tp / max(1, pr.sum()):.3f} (before {Y[te].mean():.3f})")
m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=31).fit(X, Y)
ghost.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(m, ghost.MODEL_PATH)
print("saved", ghost.MODEL_PATH, X.shape)
