"""Fret/string assignment by dynamic programming over (fingering, hand position).

State per event (notes starting together) = one candidate fingering + the index-finger fret h ("position").
Hard constraints per event: one note per string, frets within reach of the hand (fretted span <= 5, i.e. at
most a 4-5 fret stretch), at most 4 fretting fingers (notes on the lowest fret may be barred by the index).
Costs:
  * stretch: span 4 -> +1.5, span 5 -> +4 (span <= 3 free)
  * position: small preference for low positions and open strings
  * hand movement between events: shift cost grows with distance, discounted by the time available
  * ringing: a note that is still sounding on a string would be cut by a new note on that string (+2.5);
    re-picking the same string/fret is free.  Uses the path's actual ringing notes (path-aware Viterbi).
  * chord-shape consistency: pass 2 penalises fingerings of a pitch set that differ from the shape most often
    chosen for that pitch set in pass 1 (and arpeggiated bars that repeat a pitch content reuse the same shape).
"""
import itertools
from collections import Counter, defaultdict

import numpy as np

from .common import MAXFRET, TUNING

COMFORT = 3


def note_options(p):
    return [(s, p - TUNING[s]) for s in range(6) if 0 <= p - TUNING[s] <= MAXFRET]


def event_candidates(pitches, limit=80):
    per = [note_options(p) for p in pitches]
    if any(not x for x in per):
        return []
    out = []
    for combo in itertools.product(*per):
        strings = [c[0] for c in combo]
        if len(set(strings)) != len(strings):
            continue
        fr = [c[1] for c in combo if c[1] > 0]
        if fr:
            lo, hi = min(fr), max(fr)
            span = hi - lo
            if span > 5:
                continue
            fingers = (1 if lo in fr else 0) + sum(1 for f in fr if f > lo)
            if fingers > 4:
                continue
            base = {0: 0, 1: 0, 2: 0, 3: 0, 4: 1.5, 5: 4.0}[span]
        else:
            lo = hi = None
            base = 0
        n_open = sum(1 for c in combo if c[1] == 0)
        cost = base - 0.25 * n_open + (0.06 * lo if lo else 0) + 0.5 * sum(1 for f in fr if f > 12)
        # hand positions compatible: index finger at h covers h..h+3 (h+4/h+5 with a stretch)
        hs = [None] if lo is None else list(range(max(1, hi - COMFORT), lo + 1)) or [lo]
        out.append(dict(assign=tuple(combo), lo=lo, hi=hi, cost=cost, hs=hs))
    out.sort(key=lambda o: o["cost"])
    return out[:limit]


def _shape_key(assign):
    """Translation-invariant-ish key: strings + frets (exact)."""
    return tuple(sorted(assign))


def assign(events, shape_pref=None, w_shape=1.5):
    """events: list of dict(t (beats), notes=[dict(p, end_t (beats))]).
    Returns list of tuples [(string, fret), ...] aligned with each event's notes, and total cost."""
    # build state lists
    states = []  # per event: list of (cand_idx, h)
    cands = []
    for ev in events:
        pitches = [n["p"] for n in ev["notes"]]
        c = event_candidates(pitches)
        while not c and len(pitches) > 1:  # unplayable cluster: drop the weakest note
            k = int(np.argmin([n.get("vel", 64) for n in ev["notes"]]))
            ev["notes"].pop(k)
            pitches.pop(k)
            c = event_candidates(pitches)
        if not c:
            ev["notes"] = []
        cands.append(c)
        st = []
        for i, o in enumerate(c):
            for h in o["hs"]:
                st.append((i, h))
        states.append(st)
    keep = [k for k in range(len(events)) if states[k]]
    if len(keep) < len(events):
        sub_assign, total = assign([events[k] for k in keep], shape_pref, w_shape)
        full = [()] * len(events)
        for k, a in zip(keep, sub_assign):
            full[k] = a
        return full, total
    N = len(events)
    if N == 0:
        return [], 0.0
    pkey = [tuple(sorted(n["p"] for n in ev["notes"])) for ev in events]

    def local_cost(k, i):
        o = cands[k][i]
        c = o["cost"]
        if shape_pref is not None and len(o["assign"]) >= 2:
            pref = shape_pref.get(pkey[k])
            if pref is not None and _shape_key(o["assign"]) != pref:
                c += w_shape
        return c

    # DP (vectorised over state pairs)
    def fret_vec(k):
        """per candidate: fret on each string (-1 = unused) and whether that note is still ringing at next onset"""
        c = cands[k]
        fv = -np.ones((len(c), 6), int)
        ring_end = np.zeros((len(c), 6))
        for i, o in enumerate(c):
            for (s_, f_), n in zip(o["assign"], events[k]["notes"]):
                fv[i, s_] = f_
                ring_end[i, s_] = n["end_t"]
        return fv, ring_end

    st_idx = [np.array([i for i, _ in st]) for st in states]
    st_h = [np.array([np.nan if h is None else h for _, h in st], float) for st in states]
    lcost = [np.array([local_cost(k, i) for i, _ in states[k]]) for k in range(N)]
    best = [lcost[0].copy()]
    back = [np.full(len(states[0]), -1)]
    hand = [st_h[0].copy()]
    fv_prev, re_prev = fret_vec(0)
    for k in range(1, N):
        fv_cur, re_cur = fret_vec(k)
        t = events[k]["t"]
        dt = max(t - events[k - 1]["t"], 0.01)
        # ringing conflicts between candidate pairs (prev cand j, cur cand i)
        used_prev = (fv_prev >= 0) & (re_prev > t + 0.05)
        used_cur = fv_cur >= 0
        conflict = (used_prev[:, None, :] & used_cur[None, :, :] & (fv_prev[:, None, :] != fv_cur[None, :, :])).sum(2)
        ring = 2.5 * conflict  # (Cprev, Ccur)
        R = ring[st_idx[k - 1]][:, st_idx[k]]  # (Sprev, Scur)
        hp = hand[k - 1][:, None]
        hc = st_h[k][None, :]
        d = np.abs(hc - hp)
        move = np.where(np.isnan(d) | (d == 0), 0.0, (0.6 + 0.35 * d + 0.8 * (d > 4)) / (1.0 + 0.5 * dt))
        tot = best[k - 1][:, None] + R + move
        pj = np.argmin(tot, axis=0)
        best.append(tot[pj, np.arange(len(pj))] + lcost[k])
        back.append(pj)
        hand.append(np.where(np.isnan(st_h[k]), hand[k - 1][pj], st_h[k]))
        fv_prev, re_prev = fv_cur, re_cur
    j = int(np.argmin(best[-1]))
    total = float(best[-1][j])
    out = [None] * N
    for k in range(N - 1, -1, -1):
        i, _ = states[k][j]
        out[k] = cands[k][i]["assign"]
        j = back[k][j]
    return out, total


def fingering(events):
    """Two-pass DP: second pass prefers the modal shape for each repeated pitch set."""
    if not events:
        return [], {}
    a1, c1 = assign(events)
    pkey = [tuple(sorted(n["p"] for n in ev["notes"])) for ev in events]
    shapes = defaultdict(Counter)
    for k, a in enumerate(a1):
        if len(a) >= 2:
            shapes[pkey[k]][_shape_key(a)] += 1
    pref = {k: v.most_common(1)[0][0] for k, v in shapes.items() if sum(v.values()) >= 2}
    a2, c2 = assign(events, shape_pref=pref)
    stats = dict(cost_pass1=c1, cost_pass2=c2, chord_pitchsets=len(shapes), repeated_pitchsets=len(pref),
                 shape_consistency=_consistency(a2, pkey))
    return a2, stats


def _consistency(assigns, pkey):
    d = defaultdict(Counter)
    for a, k in zip(assigns, pkey):
        if len(a) >= 2:
            d[k][_shape_key(a)] += 1
    tot = sum(sum(v.values()) for v in d.values())
    mod = sum(v.most_common(1)[0][1] for v in d.values())
    return mod / tot if tot else 1.0


def playability(events, assigns):
    """Diagnostics: max span, mean hand shift, fraction of events in open/first position."""
    spans, shifts, pos = [], [], []
    last = None
    for a in assigns:
        fr = [f for _, f in a if f > 0]
        if fr:
            spans.append(max(fr) - min(fr))
            if last is not None:
                shifts.append(abs(min(fr) - last))
            last = min(fr)
            pos.append(min(fr))
    return dict(max_span=max(spans) if spans else 0, mean_span=float(np.mean(spans)) if spans else 0,
                mean_shift=float(np.mean(shifts)) if shifts else 0, median_position=float(np.median(pos)) if pos else 0,
                frac_span_gt3=float(np.mean(np.array(spans) > 3)) if spans else 0)
