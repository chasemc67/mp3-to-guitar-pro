"""Analysis-by-synthesis refinement loop.

pass k:
  1. finger every part (global DP), write pass_k.gp5, read it back, render each track + the full arrangement with
     FluidSynth, score every bar (evaluate.py) against the part stems, the guitars stem and the original mix.
  2. pick the worst bars of each part; for each build candidate re-transcriptions:
       - Basic Pitch decodes at other thresholds / salience / ownership settings
       - salience-guided edits: drop the weakest note, move notes an octave to where the NMF activation is stronger,
         add strong pitches missing from the bar
       - copy of the notes of a musically matching bar (reference chroma correlation) that scores well
     render all candidates of all selected bars in one FluidSynth batch, score them the same way (the current notes
     are scored in the same isolated way as candidate 0) and keep a candidate only if it beats the current notes.
  3. stop when `patience` passes in a row gain less than `tol` over the best so far, or nothing is accepted;
     the best-scoring pass is kept.
"""
import copy
import itertools
import tempfile
from pathlib import Path

import numpy as np
import pretty_midi

from .common import SR, jdump, load, log, save
from . import bp, evaluate as E, gpio, render, salience, score as S, transcribe as T

VARIANTS = [
    dict(),
    dict(onset=0.35, frame=0.22),
    dict(onset=0.55, frame=0.32),
    dict(onset=0.6, frame=0.4, min_len_ms=100),
    dict(rel_min=0.3, abs_min=0.18),
    dict(rel_min=0.1, abs_min=0.06),
    dict(octave_rel=0.6),
    dict(share_min=0.55),
    dict(share_min=0.25, onset=0.4),
    dict(onset=0.3, frame=0.2, rel_min=0.25, abs_min=0.15),
    dict(drum_max=1.0),
    dict(drum_max=20.0, share_min=0.3),
]


class Refiner:
    def __init__(self, ctx, rhythm, parts, programs, title_meta):
        self.ctx = ctx
        self.R = rhythm
        self.parts = parts
        self.grid = T.Grid(rhythm["bars"], rhythm["meter"])
        self.spb = self.grid.spb
        self.nbars = len(rhythm["bars"])
        self.pas = {p["id"]: T.PartAudio(ctx, p) for p in parts}
        self.programs = programs
        self.meta = dict(title_meta, meter=rhythm["meter"], spb=self.spb, nbars=self.nbars,
                         tempos=[b["bpm"] for b in rhythm["bars"]], key=rhythm["key"])
        self.bounds_ref = [(b["start"], b["end"]) for b in rhythm["bars"]]
        self.ref_feats = {}
        for p in parts:
            x = load(p["stem"], mono=True)
            self.ref_feats[p["id"]] = E.features(x, self.bounds_ref, wav_for_bp=p["stem"], cache_dir=ctx.w("bp"))
        g = ctx.w("guitars.wav")
        self.ref_feats["guitars"] = E.features(load(g, mono=True), self.bounds_ref, wav_for_bp=g, cache_dir=ctx.w("bp"))
        mix = ctx.w("song.wav")
        self.ref_feats["mix"] = E.features(load(mix, mono=True), self.bounds_ref, wav_for_bp=mix, cache_dir=ctx.w("bp"))
        self.n_samples = load(mix).shape[1]
        self._decode_cache = {}
        self.active = {p["id"]: self._active_bars(p) for p in parts}
        self.similar = self._similar_bars()

    # ---------- helpers
    def _active_bars(self, p):
        segs = p.get("segments", [p["bars"]])
        return sorted(set(b for a, e in segs for b in range(a, e)))

    def _others(self, pid):
        return [self.pas[q["id"]] for q in self.parts if q["id"] != pid]

    def _similar_bars(self):
        """bars whose reference (guitars) chroma is nearly identical -> repeated material"""
        F = np.array([c.mean(1) for c in self.ref_feats["guitars"]["chroma"]])
        F = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-9)
        Fl = np.array([c.ravel() for c in self.ref_feats["guitars"]["chroma"]])
        Fl = (Fl - Fl.mean(1, keepdims=True))
        Fl = Fl / (np.linalg.norm(Fl, axis=1, keepdims=True) + 1e-9)
        sim = Fl @ Fl.T
        out = {}
        for b in range(len(F)):
            order = [j for j in np.argsort(-sim[b]) if j != b and sim[b, j] > 0.75]
            out[b] = order[:6]
        return out

    def transcribe_range(self, pid, bar_range, prm):
        pa = self.pas[pid]
        return T.transcribe(pa, self._others(pid), self.grid, bar_range, prm)

    def initial(self, prm=None):
        notes = {}
        for p in self.parts:
            bars = {b: [] for b in range(self.nbars)}
            segs = p.get("segments", [p["bars"]])
            for a, e in segs:
                for n in self.transcribe_range(p["id"], (a, e), prm):
                    bars.setdefault(n["slot"] // self.spb, []).append(n)
            notes[p["id"]] = bars
        return notes

    # ---------- full render + evaluation of a pass
    def build_gp5(self, notes, path):
        parts_out, fstats = [], {}
        for p in self.parts:
            flat = [n for b in sorted(notes[p["id"]]) for n in notes[p["id"]][b]]
            fn, st = S.finger_part(flat)
            fstats[p["id"]] = st
            parts_out.append(dict(name=p["name"], program=self.programs[p["id"]], pan=p.get("gp_pan", 64), notes=fn))
        S.build(self.meta, parts_out, path)
        return fstats

    def evaluate_gp5(self, path, tag):
        song = gpio.read(path)
        tl = render.Timeline(song["tempos"], song["meter"], self.R["bars"][0]["start"])
        bren = [(tl.bar_start[i], tl.bar_start[i + 1]) for i in range(self.nbars)]
        res = {"per_bar": {}, "summary": {}}
        tracks_audio = []
        for i, p in enumerate(self.parts):
            wav = self.ctx.w("render", f"{tag}_{p['id']}.wav")
            x = render.render_tracks(song["tracks"], tl, wav, which=[i], length=self.n_samples)
            save(wav, x)
            tracks_audio.append(x)
            fo = E.features(x.mean(0), bren, wav_for_bp=wav, cache_dir=self.ctx.w("bp"))
            rows = E.compare(self.ref_feats[p["id"]], fo, bars=self.active[p["id"]])
            res["per_bar"][p["id"]] = rows
            res["summary"][p["id"]] = E.summary(rows)
        full = np.sum(tracks_audio, axis=0)
        full = full / (np.abs(full).max() + 1e-9) * 0.9
        wav = self.ctx.w("render", f"{tag}_full.wav")
        save(wav, full)
        fo = E.features(full.mean(0), bren, wav_for_bp=wav, cache_dir=self.ctx.w("bp"))
        audible = [b for b in range(self.nbars) if self.R["bars"][b].get("audible", True)]
        for ref in ("guitars", "mix"):
            rows = E.compare(self.ref_feats[ref], fo, bars=audible)
            res["per_bar"]["full_vs_" + ref] = rows
            res["summary"]["full_vs_" + ref] = E.summary(rows)
        res["song"] = song
        res["timeline"] = tl
        return res

    # ---------- candidate generation for one bar
    def _decode_pool(self, pid, b, onset=0.3, frame=0.2):
        """low-threshold note pool of a bar (for 'add missing note' edits)"""
        key = (pid, onset, frame)
        if key not in self._decode_cache:
            self._decode_cache[key] = bp.decode(self.pas[pid].pg, onset=onset, frame=frame, min_len_ms=50)
        t0, t1 = self.R["bars"][b]["start"] - 0.03, self.R["bars"][b]["end"] - 0.03
        return [n for n in self._decode_cache[key] if t0 <= n["start"] < t1]

    def _sal(self, pid, p, slot):
        t = self.grid.time(slot)
        return salience.note_strength(self.pas[pid].H, p, t, t + 0.15)

    def candidates(self, pid, b, cur, scores_now):
        cands = [("current", cur)]
        seen = {self._key(cur)}

        def add(name, notes):
            k = self._key(notes)
            if k not in seen and len(cands) < 24:
                seen.add(k)
                cands.append((name, notes))

        for i, v in enumerate(VARIANTS[1:], 1):
            add(f"params{i}", self.transcribe_range(pid, (b, b + 1), v))
        if cur:
            sal = [self._sal(pid, n["p"], n["slot"]) for n in cur]
            weakest = int(np.argmin(sal))
            add("drop_weakest", [n for i, n in enumerate(cur) if i != weakest])
            order = np.argsort(sal)
            add("drop_2_weakest", [n for i, n in enumerate(cur) if i not in set(order[:2])])
            # octave moves where the other octave has more activation
            moved = []
            for n in cur:
                best = n
                for d in (-12, 12):
                    q = n["p"] + d
                    if 40 <= q <= 83 and self._sal(pid, q, n["slot"]) > 1.3 * self._sal(pid, n["p"], n["slot"]):
                        best = dict(n, p=q)
                moved.append(best)
            add("octave_fix", moved)
        # add strong pool notes that are missing
        pool = self._decode_pool(pid, b)
        have = {(n["slot"], n["p"]) for n in cur}
        extra = []
        for n in pool:
            s = int(round(self.grid.pos(n["start"])))
            if (s, n["p"]) in have or not 40 <= n["p"] <= 83:
                continue
            sal = salience.note_strength(self.pas[pid].H, n["p"], n["start"], n["end"])
            extra.append((sal, dict(slot=s, p=n["p"], dur=max(1, int(round(self.grid.pos(n["end"]) - s))), vel=80, sal=sal)))
        extra.sort(key=lambda x: -x[0])
        if extra:
            add("add_strongest", cur + [extra[0][1]])
            add("add_3_strongest", cur + [e[1] for e in extra[:3]])
        # repeated material: borrow the notes of the best-scoring similar bar
        for j in self.similar.get(b, [])[:4]:
            if j in self.active[pid] and scores_now.get(j, 0) > scores_now.get(b, 0) + 0.05:
                src = self.notes[pid].get(j, [])
                shift = (b - j) * self.spb
                add(f"copy_bar{j + 1}", [dict(n, slot=n["slot"] + shift) for n in src])
        return cands

    @staticmethod
    def _key(notes):
        return tuple(sorted((n["slot"], n["p"]) for n in notes))

    # ---------- batch-render candidates (isolated bars) and score them
    def score_candidates(self, pid, jobs):
        """jobs: list of (bar, [(name, notes)]). Returns {bar: [(name, score_row)]}."""
        tempo = float(np.median(self.meta["tempos"]))
        spb_sec = 60.0 / tempo / T.SUB
        bar_sec = self.spb * spb_sec
        gap = 1.0
        pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        ins = pretty_midi.Instrument(program=int(self.programs[pid]))
        layout = []
        t = 0.5
        for b, cl in jobs:
            for name, notes in cl:
                base = b * self.spb
                for n in notes:
                    s = t + (n["slot"] - base) * spb_sec
                    e = s + max(1, n["dur"]) * spb_sec
                    if 0 <= n["slot"] - base < self.spb:
                        ins.notes.append(pretty_midi.Note(velocity=int(n["vel"]), pitch=int(n["p"]), start=s, end=e))
                layout.append((b, name, t, t + bar_sec))
                t += bar_sec + gap
        pm.instruments.append(ins)
        with tempfile.TemporaryDirectory() as td:
            mid, wav = Path(td) / "c.mid", Path(td) / "c.wav"
            pm.write(str(mid))
            render.synth(mid, wav)
            x = load(wav, mono=True)
            n_need = int((t + 0.5) * SR)
            if len(x) < n_need:
                x = np.pad(x, (0, n_need - len(x)))
            from basic_pitch.inference import run_inference
            import contextlib, io
            with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
                save(wav, x)
                pg = run_inference(str(wav), bp._model())
            fo = E.features(x, [(a, e) for _, _, a, e in layout], pg=pg)
        out = {}
        ref = self.ref_feats[pid]
        for i, (b, name, a, e) in enumerate(layout):
            sub_ref = {k: [v[b]] for k, v in ref.items() if isinstance(v, list)}
            sub_ren = {k: [v[i]] for k, v in fo.items() if isinstance(v, list)}
            row = E.compare(sub_ref, sub_ren)[0]
            out.setdefault(b, []).append((name, row))
        return out

    # ---------- main loop
    def run(self, max_passes=10, tol=0.002, worst_frac=0.2, min_worst=4, patience=2):
        self.notes = self.initial()
        passes = []
        prev_mean = None
        best = (-1, None, None, None)
        low = 0
        tried = {p["id"]: {} for p in self.parts}
        for k in range(1, max_passes + 1):
            gp5 = self.ctx.w("passes", f"pass{k}.gp5")
            fstats = self.build_gp5(self.notes, gp5)
            ev = self.evaluate_gp5(gp5, f"pass{k}")
            mean_parts = float(np.mean([ev["summary"][p["id"]]["score"] for p in self.parts]))
            rec = dict(pass_=k, gp5=str(gp5), summary=ev["summary"], mean_part_score=mean_parts,
                       fingering=fstats, n_notes={p["id"]: sum(len(v) for v in self.notes[p["id"]].values())
                                                  for p in self.parts})
            log(f"pass {k}: mean part score {mean_parts:.4f} | " + " | ".join(
                f"{pid}: {s['score']:.3f} (chroma {s['chroma']:.3f}, pitchF {s['pitch_f']:.3f}, onsetF {s['onset_f']:.3f})"
                for pid, s in ev["summary"].items()))
            if mean_parts > best[0]:
                best = (mean_parts, k, copy.deepcopy(self.notes), (ev, gp5))
            gain = None if prev_mean is None else mean_parts - prev_mean
            low = low + 1 if gain is not None and gain < tol else 0
            rec["gain"] = gain
            if low >= patience:
                rec["stopped"] = f"plateau ({patience} passes in a row gaining < {tol})"
                passes.append(rec)
                log(rec["stopped"])
                break
            if k == max_passes:
                rec["stopped"] = "max passes"
                passes.append(rec)
                break
            prev_mean = mean_parts if prev_mean is None else max(prev_mean, mean_parts)
            # ---- redo worst bars
            changes = []
            for p in self.parts:
                pid = p["id"]
                rows = [r for r in ev["per_bar"][pid] if not np.isnan(r["score"])]
                now = {r["bar"]: r["score"] for r in rows}
                loud = [r for r in rows if self.R["bars"][r["bar"]].get("audible", True)]
                loud.sort(key=lambda r: r["score"])
                n_w = max(min_worst, int(worst_frac * len(loud)))
                # skip bars already retried with no improvement twice
                worst = [r["bar"] for r in loud if tried[pid].get(r["bar"], 0) < 2][:n_w]
                jobs = [(b, self.candidates(pid, b, self.notes[pid].get(b, []), now)) for b in worst]
                scored = self.score_candidates(pid, jobs)
                for b in worst:
                    lst = scored[b]
                    cur_row = dict(lst)["current"]
                    # better overall, and not paying for it with pitch accuracy
                    ok = [x for x in lst if x[1]["pitch_f"] >= cur_row["pitch_f"] - 0.03]
                    name, row = max(ok, key=lambda x: x[1]["score"])
                    if name != "current" and row["score"] > cur_row["score"] + 0.01:
                        newn = dict(jobs[[j[0] for j in jobs].index(b)][1])[name]
                        self.notes[pid][b] = [n for n in newn if b * self.spb <= n["slot"] < (b + 1) * self.spb]
                        changes.append(dict(part=pid, bar=b + 1, edit=name, isolated_before=cur_row["score"],
                                            isolated_after=row["score"], full_before=now.get(b)))
                        tried[pid][b] = 0
                    else:
                        tried[pid][b] = tried[pid].get(b, 0) + 1
            rec["redone_bars"] = {p["id"]: len([c for c in changes if c["part"] == p["id"]]) for p in self.parts}
            rec["changes"] = changes
            passes.append(rec)
            log(f"  redo: {len(changes)} bars changed " + str(rec["redone_bars"]))
            if not changes:
                rec["stopped"] = "no accepted changes"
                # evaluate once more is unnecessary: notes unchanged
                break
        _, self.best_pass, self.notes, (self.last_eval, self.last_gp5) = best
        for r in passes:
            r["best"] = r["pass_"] == self.best_pass
        return passes
