"""Estimated parts optimised for the combined sound ("ensemble" mode).

Where the audio cannot cleanly split the guitars, the parts are *estimated*: one note pool is transcribed from
the all-guitars stem (Basic Pitch + learned ghost-note classifier), then distributed over the detected roles
(centre arpeggio/rhythm, panned lead, wide doubled upper part) so that every part is playable on its own:
  * spatial energy share of each note in the part stems is only a hint for which role gets it;
  * per onset a lead part takes at most 2 notes, the rhythm/arpeggio part at most 4 new notes;
  * each part's notes at an onset must have a valid fingering (one note per string, <=5-fret span, <=4 fingers);
    notes that do not fit are moved to another active part where they fit, else dropped;
  * lead parts prefer the upper register when the spatial evidence is ambiguous.
The loop then optimises the COMBINED playback: every pass renders all tracks from the .gp5, scores each bar
against the all-guitars stem and the original mix (objective = mean of the two bar scores), and re-estimates the
worst bars (pool thresholds, salience-guided edits, repeated-bar copies, alternative role assignments), keeping a
candidate only if its combined render scores better.  Per-part-vs-stem scores are still logged as diagnostics.
"""
import copy
import tempfile
from pathlib import Path

import numpy as np
import pretty_midi

from .common import SR, load, log
from . import bp, evaluate as E, ghost, salience, transcribe as T
from .fingering import event_candidates
from .refine import Refiner

CAP = {"centre": 4, "right": 2, "left": 2, "double": 2}
LEAD_MIN_PITCH = 62  # D4: below this, ambiguous notes stay in the rhythm/arpeggio part


class Ensemble(Refiner):
    def __init__(self, ctx, rhythm, parts, programs, title_meta):
        super().__init__(ctx, rhythm, parts, programs, title_meta)
        gpart = dict(id="guitars", stem=str(ctx.w("guitars.wav")), bars=[0, self.nbars])
        self.g = T.PartAudio(ctx, gpart)
        self.det = bp.decode(self.g.pg, onset=0.35, frame=0.22, min_len_ms=50)
        self.preal = ghost.p_real(self.det, self.g.pg)
        for n, q in zip(self.det, self.preal):
            n["preal"] = float(q)
        self.kind = {p["id"]: p["kind"] for p in parts}
        self.bar_parts = {b: [p["id"] for p in parts if b in set(self.active[p["id"]])] for b in range(self.nbars)}
        self._share_cache = {}

    # ---------- pool + role assignment
    def pool(self, b, thr=0.35, onset_min=0.0):
        t0, t1 = self.R["bars"][b]["start"] - 0.03, self.R["bars"][b]["end"] - 0.03
        ns = [dict(n) for n in self.det if t0 <= n["start"] < t1 and n["preal"] > thr and n["conf"] >= onset_min]
        for n in ns:
            n["sal"] = n["preal"]
        prm = dict(T.DEFAULT, max_poly=8)
        q = T.quantise(ns, self.grid, prm, (b, b + 1))
        for x in q:
            x["vel"] = int(np.clip(50 + 70 * x["sal"], 40, 120))
        return q

    def shares(self, n):
        key = (n["slot"], n["p"])
        if key not in self._share_cache:
            t = self.grid.time(n["slot"])
            e = {pid: self.pas[pid].harm_energy(n["p"], t, t + 0.12) for pid in self.pas}
            tot = sum(e.values()) + 1e-12
            self._share_cache[key] = {k: v / tot for k, v in e.items()}
        return self._share_cache[key]

    def assign(self, b, notes, mode="spatial"):
        act = self.bar_parts[b]
        out = {pid: [] for pid in act}
        if not act:
            return out
        centre = next((p for p in act if self.kind[p] == "centre"), act[0])
        leads = [p for p in act if p != centre]
        by = {}
        for n in notes:
            by.setdefault(n["slot"], []).append(n)
        for s in sorted(by):
            ns = sorted(by[s], key=lambda n: -n["p"])
            want = {}
            for n in ns:
                if mode == "register" or not leads:
                    pref = [p for p in leads if n["p"] >= LEAD_MIN_PITCH][:1] + [centre] + leads
                else:
                    sh = self.shares(n)
                    ranked = sorted(act, key=lambda p: -sh.get(p, 0))
                    if sh.get(ranked[0], 0) < 0.5:  # ambiguous: register decides
                        ranked = ([p for p in leads if n["p"] >= LEAD_MIN_PITCH][:1] + [centre] + leads)
                    pref = list(dict.fromkeys(ranked + [centre] + leads))
                want[id(n)] = pref
            slot_parts = {pid: [] for pid in act}
            for n in sorted(ns, key=lambda n: -n.get("sal", 0)):  # strongest notes choose first
                for pid in want[id(n)]:
                    cur = slot_parts[pid]
                    if len(cur) >= CAP.get(self.kind[pid], 2):
                        continue
                    if not event_candidates([m["p"] for m in cur] + [n["p"]], limit=1):
                        continue
                    cur.append(n)
                    break
            for pid in act:
                out[pid] += slot_parts[pid]
        return out

    def choose_programs(self, notes0, candidates=(24, 25, 26, 27, 28, 29), n_bars=14):
        """Coordinate ascent over each part's GM patch, scored on the COMBINED render (mean guitars/mix score)."""
        bars = [b for b in range(self.nbars) if self.R["bars"][b].get("audible", True) and self.bar_parts[b]]
        bars = bars[:: max(1, len(bars) // n_bars)][:n_bars]
        self.notes = notes0

        def objective():
            jobs = [(b, [("current", {pid: notes0[pid].get(b, []) for pid in self.bar_parts[b]})]) for b in bars]
            sc = self.score_combined(jobs)
            return float(np.mean([sc[b][0][1] for b in bars]))

        table = {}
        best = objective()
        for _ in range(2):
            for p in self.parts:
                pid = p["id"]
                res = {}
                for prog in candidates:
                    self.programs[pid] = prog
                    res[prog] = objective()
                self.programs[pid] = max(res, key=res.get)
                table[pid] = res
                best = res[self.programs[pid]]
                log(f"program for {pid}: {self.programs[pid]}  " + " ".join(f"{k}:{v:.3f}" for k, v in res.items()))
        return dict(self.programs), table

    # ---------- passes
    def initial(self, prm=None):
        notes = {p["id"]: {b: [] for b in range(self.nbars)} for p in self.parts}
        for b in range(self.nbars):
            for pid, ns in self.assign(b, self.pool(b)).items():
                notes[pid][b] = ns
        return notes

    def bar_objective(self, ev):
        g = {r["bar"]: r for r in ev["per_bar"]["full_vs_guitars"]}
        m = {r["bar"]: r for r in ev["per_bar"]["full_vs_mix"]}
        return {b: 0.5 * (g[b]["score"] + m[b]["score"]) for b in g if b in m}

    def combined_candidates(self, b, scores_now):
        cur = {pid: self.notes[pid].get(b, []) for pid in self.bar_parts[b]}
        flat = [n for v in cur.values() for n in v]
        cands = [("current", cur)]
        seen = {self._ckey(cur)}

        def add(name, assignment):
            k = self._ckey(assignment)
            if k not in seen and len(cands) < 20:
                seen.add(k)
                cands.append((name, assignment))

        for thr in (0.2, 0.5, 0.7):
            add(f"pool_thr{thr}", self.assign(b, self.pool(b, thr)))
        add("pool_strong_onsets", self.assign(b, self.pool(b, 0.35, onset_min=0.45)))
        add("reassign_register", self.assign(b, flat, mode="register"))
        if flat:
            sal = [n.get("sal", 0.5) for n in flat]
            order = np.argsort(sal)
            add("drop_weakest", self.assign(b, [n for i, n in enumerate(flat) if i != order[0]]))
            add("drop_2_weakest", self.assign(b, [n for i, n in enumerate(flat) if i not in set(order[:2])]))
            moved = []
            for n in flat:
                best = n
                for d in (-12, 12):
                    q = n["p"] + d
                    t = self.grid.time(n["slot"])
                    if 40 <= q <= 83 and salience.note_strength(self.g.H, q, t, t + .15) > \
                            1.3 * salience.note_strength(self.g.H, n["p"], t, t + .15):
                        best = dict(n, p=q)
                moved.append(best)
            add("octave_fix", self.assign(b, moved))
        have = {(n["slot"], n["p"]) for n in flat}
        extra = [n for n in self.pool(b, 0.1) if (n["slot"], n["p"]) not in have]
        extra.sort(key=lambda n: -n["sal"])
        if extra:
            add("add_strongest", self.assign(b, flat + extra[:1]))
            add("add_3_strongest", self.assign(b, flat + extra[:3]))
        for j in self.similar.get(b, [])[:4]:
            if scores_now.get(j, 0) > scores_now.get(b, 0) + 0.04 and self.bar_parts[j] == self.bar_parts[b]:
                shift = (b - j) * self.spb
                add(f"copy_bar{j + 1}", {pid: [dict(n, slot=n["slot"] + shift) for n in self.notes[pid].get(j, [])]
                                         for pid in self.bar_parts[b]})
        return cands

    @staticmethod
    def _ckey(assignment):
        return tuple(sorted((pid, n["slot"], n["p"]) for pid, v in assignment.items() for n in v))

    def score_combined(self, jobs):
        """Render each candidate bar with all its parts together (one batch), score vs guitars stem and mix."""
        tempo = float(np.median(self.meta["tempos"]))
        sl = 60.0 / tempo / T.SUB
        bar_sec = self.spb * sl
        gap = 1.0
        pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        ins = {p["id"]: pretty_midi.Instrument(program=int(self.programs[p["id"]])) for p in self.parts}
        layout, t = [], 0.5
        for b, cl in jobs:
            for name, asg in cl:
                base = b * self.spb
                for pid, ns in asg.items():
                    for n in ns:
                        if 0 <= n["slot"] - base < self.spb:
                            s = t + (n["slot"] - base) * sl
                            ins[pid].notes.append(pretty_midi.Note(velocity=int(n["vel"]), pitch=int(n["p"]), start=s,
                                                                   end=s + max(1, n["dur"]) * sl))
                layout.append((b, name, t, t + bar_sec))
                t += bar_sec + gap
        pm.instruments += list(ins.values())
        from . import render
        import contextlib, io
        from basic_pitch.inference import run_inference
        with tempfile.TemporaryDirectory() as td:
            mid, wav = Path(td) / "c.mid", Path(td) / "c.wav"
            pm.write(str(mid))
            render.synth(mid, wav)
            x = load(wav, mono=True)
            need = int((t + 0.5) * SR)
            if len(x) < need:
                x = np.pad(x, (0, need - len(x)))
            from .common import save
            save(wav, x)
            with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
                pg = run_inference(str(wav), bp._model())
            fo = E.features(x, [(a, e) for _, _, a, e in layout], pg=pg)
        out = {}
        for i, (b, name, a, e) in enumerate(layout):
            sub_ren = {k: [v[i]] for k, v in fo.items() if isinstance(v, list)}
            rows = {}
            for ref in ("guitars", "mix"):
                sub_ref = {k: [v[b]] for k, v in self.ref_feats[ref].items() if isinstance(v, list)}
                rows[ref] = E.compare(sub_ref, sub_ren)[0]
            obj = 0.5 * (rows["guitars"]["score"] + rows["mix"]["score"])
            out.setdefault(b, []).append((name, obj, rows))
        return out

    def run(self, max_passes=10, tol=0.002, worst_frac=0.3, min_worst=6, patience=2):
        self.notes = self.initial()
        passes, best, prev, low = [], (-1, None, None, None), None, 0
        tried = {}
        for k in range(1, max_passes + 1):
            gp5 = self.ctx.w("passes", f"pass{k}.gp5")
            fstats = self.build_gp5(self.notes, gp5)
            ev = self.evaluate_gp5(gp5, f"pass{k}")
            fg, fm = ev["summary"]["full_vs_guitars"], ev["summary"]["full_vs_mix"]
            obj = 0.5 * (fg["score"] + fm["score"])
            rec = dict(pass_=k, gp5=str(gp5), objective=obj, mean_part_score=obj, summary=ev["summary"],
                       fingering=fstats, n_notes={pid: sum(len(v) for v in self.notes[pid].values()) for pid in self.notes})
            log(f"pass {k}: objective {obj:.4f} | full_vs_guitars {fg['score']:.3f} (c {fg['chroma']:.3f} p {fg['pitch_f']:.3f} "
                f"o {fg['onset_f']:.3f}) | full_vs_mix {fm['score']:.3f} (c {fm['chroma']:.3f} p {fm['pitch_f']:.3f} "
                f"o {fm['onset_f']:.3f}) | notes {rec['n_notes']}")
            if obj > best[0]:
                best = (obj, k, copy.deepcopy(self.notes), (ev, gp5))
            gain = None if prev is None else obj - prev
            rec["gain"] = gain
            low = low + 1 if gain is not None and gain < tol else 0
            if low >= patience or k == max_passes:
                rec["stopped"] = f"plateau ({patience} passes in a row gaining < {tol})" if low >= patience else "max passes"
                passes.append(rec)
                log(rec["stopped"])
                break
            prev = obj if prev is None else max(prev, obj)
            now = self.bar_objective(ev)
            cand_bars = [b for b in now if self.R["bars"][b].get("audible", True) and self.bar_parts[b]
                         and tried.get(b, 0) < 2]
            cand_bars.sort(key=lambda b: now[b])
            worst = cand_bars[:max(min_worst, int(worst_frac * len(cand_bars)))]
            jobs = [(b, self.combined_candidates(b, now)) for b in worst]
            scored = self.score_combined(jobs)
            changes = []
            for b, cl in jobs:
                lst = scored[b]
                cur_obj, cur_rows = next((o, r) for nme, o, r in lst if nme == "current")
                ok = [x for x in lst if x[2]["guitars"]["pitch_f"] >= cur_rows["guitars"]["pitch_f"] - 0.03]
                name, o, rows = max(ok, key=lambda x: x[1])
                if name != "current" and o > cur_obj + 0.01:
                    asg = dict(cl)[name]
                    for pid in self.bar_parts[b]:
                        self.notes[pid][b] = [n for n in asg.get(pid, []) if b * self.spb <= n["slot"] < (b + 1) * self.spb]
                    changes.append(dict(bar=b + 1, edit=name, isolated_before=cur_obj, isolated_after=o,
                                        full_before=now[b]))
                    tried[b] = 0
                else:
                    tried[b] = tried.get(b, 0) + 1
            rec["redone_bars"] = {"bars_tried": len(worst), "bars_changed": len(changes)}
            rec["changes"] = changes
            passes.append(rec)
            log(f"  redo: {len(changes)}/{len(worst)} bars changed")
            if not changes:
                rec["stopped"] = "no accepted changes"
                break
        _, self.best_pass, self.notes, (self.last_eval, self.last_gp5) = best
        for r in passes:
            r["best"] = r["pass_"] == self.best_pass
        return passes
