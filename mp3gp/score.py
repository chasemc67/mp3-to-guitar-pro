"""Glue: grid notes per part -> fingered score -> .gp5."""
from .fingering import fingering, playability
from .transcribe import SUB
from . import gpio


def finger_part(notes):
    by = {}
    for n in notes:
        by.setdefault(n["slot"], []).append(n)
    slots = sorted(by)
    events = [dict(t=s / SUB, notes=[dict(p=n["p"], end_t=(s + n["dur"]) / SUB, vel=n["vel"], ref=n)
                                     for n in sorted(by[s], key=lambda n: -n["p"])]) for s in slots]
    assigns, stats = fingering(events)
    out = []
    for ev, a in zip(events, assigns):
        for (s, f), en in zip(a, ev["notes"]):
            q = dict(en["ref"])
            q["string"], q["fret"] = s, f
            out.append(q)
    stats |= playability(events, assigns)
    stats["dropped_unplayable"] = len(notes) - len(out)
    return out, stats


def build(meta, parts_notes, path):
    """meta: title/artist/key/meter/spb/nbars/tempos/markers; parts_notes: list of dict(name, program, pan, notes)"""
    score = dict(meta, parts=parts_notes)
    gpio.write(score, path)
    return path
