"""End-to-end driver:  python -m mp3gp.pipeline INPUT.mp3 --work WORK --out OUT"""
import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from .common import SR, Ctx, jdump, jload, load, log, save, sh


def prepare(ctx):
    wav = ctx.w("song.wav")
    if ctx.need("prepare", wav):
        sh(["ffmpeg", "-y", "-loglevel", "error", "-i", str(ctx.inp), "-ar", str(SR), "-ac", "2", str(wav)])
    return wav


def tag_metadata(ctx):
    try:
        import json
        out = sh(["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(ctx.inp)])
        t = json.loads(out)["format"].get("tags", {})
        return dict(title=t.get("title", ctx.inp.stem), artist=t.get("artist", ""), album=t.get("album", ""))
    except Exception:
        return dict(title=ctx.inp.stem, artist="", album="")


def mark_audible(ctx, rhythm, floor_db=-30.0):
    g = load(ctx.w("guitars.wav"))
    lv = []
    for b in rhythm["bars"]:
        s, e = int(b["start"] * SR), int(b["end"] * SR)
        lv.append(10 * np.log10(np.mean(g[:, s:e] ** 2) + 1e-12))
    top = np.percentile(lv, 90)
    for b, v in zip(rhythm["bars"], lv):
        b["level_db"] = float(v - top)
        b["audible"] = bool(v - top > floor_db)


def choose_programs(ref, notes0, candidates=(24, 25, 26, 27, 28, 29)):
    """Pick the GM guitar patch per part whose render best matches the part stem on a sample of bars
    (chroma + pitch-F only: onset-F rewards hard-attack patches like overdrive regardless of fit)."""
    progs = {}
    table = {}
    for p in ref.parts:
        pid = p["id"]
        bars = [b for b in ref.active[pid] if notes0[pid].get(b)]
        bars = bars[:: max(1, len(bars) // 10)][:10]
        res = {}
        for prog in candidates:
            ref.programs[pid] = prog
            sc = ref.score_candidates(pid, [(b, [("current", notes0[pid][b])]) for b in bars])
            res[prog] = float(np.mean([0.5 * (sc[b][0][1]["chroma"] + sc[b][0][1]["pitch_f"]) for b in bars]))
        progs[pid] = max(res, key=res.get)
        ref.programs[pid] = progs[pid]
        table[pid] = res
        log(f"program for {pid}: {progs[pid]}  " + " ".join(f"{k}:{v:.3f}" for k, v in res.items()))
    return progs, table


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--work", default="work")
    ap.add_argument("--out", default="out")
    ap.add_argument("--force", default="", help="comma list of stages to recompute (or 'all')")
    ap.add_argument("--max-passes", type=int, default=10)
    ap.add_argument("--mode", choices=["ensemble", "stems"], default="ensemble",
                    help="ensemble: estimated playable parts optimised for the combined sound (default); "
                         "stems: each part fitted to its own separated stem")
    ap.add_argument("--section", type=float, default=30.0, help="seconds for comparison.wav")
    a = ap.parse_args(argv)
    ctx = Ctx(a.input, a.work, a.out, force=[f for f in a.force.split(",") if f])
    t0 = time.time()
    from . import rhythm as RH, separate, parts as PT, refine, lily, compare_audio, report

    prepare(ctx)
    meta = tag_metadata(ctx)
    py = sys.executable
    # separators first need a rough guitar stem for key detection; rhythm uses the mix
    separate.run_models(ctx, py)
    R = RH.run(ctx, ctx.w("stems/htdemucs_6s/song/guitar.wav"))
    sep = separate.run(ctx, py, R["bars"])
    mark_audible(ctx, R)
    P = PT.run(ctx, R["bars"])
    parts = P["parts"]
    pans = {"centre": 64, "right": 100, "left": 28, "double": 64}
    roles = {"centre": "arpeggio/rhythm, centre", "right": "lead, panned right", "left": "lead, panned left",
             "double": "upper part, wide double-tracked"}
    for p in parts:
        p["gp_pan"] = pans.get(p["kind"], 64)
        p["name"] = f"Gtr {p['id'][3:]} ({roles.get(p['kind'], p['kind'])})"

    from .ensemble import Ensemble
    cls = Ensemble if a.mode == "ensemble" else refine.Refiner
    ref = cls(ctx, R, parts, {p["id"]: 27 for p in parts}, meta)
    notes0 = ref.initial()
    progs, prog_table = ref.choose_programs(notes0) if a.mode == "ensemble" else choose_programs(ref, notes0)
    passes = ref.run(max_passes=a.max_passes)
    final = ctx.o(f"{_slug(meta['title'])}.gp5")
    # rebuild the best pass with section markers and copy out
    ref.meta["markers"] = {p["bars"][0]: f"{p['name']} in" for p in parts if p["bars"][0] > 0}
    fstats = ref.build_gp5(ref.notes, final)
    ev = ref.evaluate_gp5(final, "final")
    jdump(ctx.o("passes.json"), dict(
        mode=a.mode,
        objective=("mean of full_vs_guitars and full_vs_mix bar scores (combined playback of all tracks)"
                   if a.mode == "ensemble" else "mean part-vs-own-stem score"),
        criteria="per-bar mean of chroma cosine, frame pitch F (Basic Pitch on render vs reference), onset F (50 ms); "
                 "parts are scored against their own separated stem, 'full_vs_guitars' / 'full_vs_mix' score the "
                 "summed render against the all-guitar stem / the original mix (faded-out bars excluded)",
        passes=passes, final=dict(gp5=str(final), summary=ev["summary"], best_pass=ref.best_pass),
        programs=progs, program_scores=prog_table))
    jdump(ctx.w("final_eval.json"), {k: v for k, v in ev["per_bar"].items()})
    pngs = lily.render(final, ctx.o("tab"), parts, R, meta)
    compare_audio.make(ctx, R, ev, parts, a.section)
    report.write(ctx, final, a.mode, meta, R, sep, P, passes, ev, fstats, progs, pngs, time.time() - t0)
    log(f"done in {time.time() - t0:.0f}s -> {ctx.out}")


def _slug(s):
    return "".join(c if c.isalnum() else "_" for c in s.lower()).strip("_") or "song"


if __name__ == "__main__":
    main()
