"""Guitar Pro 5 writing, and reading a .gp5 back into timed notes (so playback checks use the actual tab)."""
import guitarpro as gp
from guitarpro import models as m

from .common import TUNING

QUARTER = 960
RING_CAP = 1.5  # beats a let-ring note sounds at most in our playback (unless re-struck earlier)
DUR = {12: (2, True), 8: (2, False), 6: (4, True), 4: (4, False), 3: (8, True), 2: (8, False), 1: (16, False)}
KEYS = {"C major": "CMajor", "G major": "GMajor", "D major": "DMajor", "A major": "AMajor", "E major": "EMajor",
        "B major": "BMajor", "F# major": "FMajorSharp", "C# major": "CMajorSharp", "F major": "FMajor",
        "A# major": "BMajorFlat", "D# major": "EMajorFlat", "G# major": "AMajorFlat",
        "A minor": "AMinor", "E minor": "EMinor", "B minor": "BMinor", "F# minor": "FMinorSharp",
        "C# minor": "CMinorSharp", "G# minor": "GMinorSharp", "D# minor": "DMinorSharp", "D minor": "DMinor",
        "G minor": "GMinor", "C minor": "CMinor", "F minor": "FMinor", "A# minor": "BMinorFlat"}


def _split(n, pos, spb_beat=4):
    """Split n slots starting at in-bar slot `pos` into notatable pieces; off-beat pieces stop at the next beat."""
    out = []
    while n > 0:
        off = pos % spb_beat
        allowed = (12, 8, 6, 4, 3, 2, 1) if off == 0 else tuple(v for v in (3, 2, 1) if v <= spb_beat - off)
        v = next(v for v in allowed if v <= n)
        out.append(v)
        pos += v
        n -= v
    return out


def layout_track(notes, nbars, spb):
    """notes: dict(slot, p, dur, vel, string, fret) -> per bar list of (pos, length, [notes], tie, let_ring)."""
    by = {}
    for n in notes:
        by.setdefault(n["slot"], []).append(n)
    slots = sorted(by)
    end_total = nbars * spb
    items = []
    if not slots:
        items.append((0, end_total, None))
    else:
        if slots[0] > 0:
            items.append((0, slots[0], None))
        for k, s in enumerate(slots):
            nxt = slots[k + 1] if k + 1 < len(slots) else min(end_total, s + max(n["dur"] for n in by[s]))
            nxt = max(nxt, s + 1)
            gap = nxt - s
            ring = max(n["dur"] for n in by[s])
            L = gap
            if gap - ring >= 2:  # note stops well before the next onset -> rest
                L = max(1, ring)
            items.append((s, L, by[s]))
            if L < gap:
                items.append((s + L, gap - L, None))
        last = items[-1][0] + items[-1][1]
        if last < end_total:
            items.append((last, end_total - last, None))
    bars = [[] for _ in range(nbars)]
    for s, L, nts in items:
        pos, rem, first = s, L, True
        while rem > 0 and pos < end_total:
            b = pos // spb
            room = (b + 1) * spb - pos
            seg = min(rem, room)
            for piece in _split(seg, pos % spb):
                bars[b].append((pos, piece, nts, not first))
                first = False
                pos += piece
                rem -= piece
    return bars


def write(score, path):
    """score: dict(title, artist, key, meter, spb, nbars, tempos[per bar], parts=[dict(name, program, notes)])"""
    song = m.Song()
    song.title = score.get("title", "")
    song.artist = score.get("artist", "")
    song.album = score.get("album", "")
    song.tab = "mp3gp auto-transcription"
    song.comments = score.get("comments", "")
    tempos = score["tempos"]
    song.tempo = int(tempos[0])
    meter, spb, nbars = score["meter"], score["spb"], score["nbars"]
    song.measureHeaders = []
    song.tracks[0].measures = []
    ks = getattr(m.KeySignature, KEYS.get(score.get("key", ""), "CMajor"))
    for b in range(nbars):
        song.newMeasure()
        h = song.measureHeaders[-1]
        h.number = b + 1
        h.start = QUARTER + b * QUARTER * meter
        h.timeSignature.numerator = meter
        h.timeSignature.denominator.value = 4
        h.keySignature = ks
        if b in score.get("markers", {}):
            h.marker = m.Marker(title=score["markers"][b])
    for ti, part in enumerate(score["parts"]):
        if ti == 0:
            tr = song.tracks[0]
        else:
            tr = m.Track(song, number=ti + 1)
            tr.measures = [m.Measure(tr, h) for h in song.measureHeaders]
            song.tracks.append(tr)
        ch = 2 * ti if 2 * ti < 9 else 2 * ti + 1
        tr.channel = m.MidiChannel(channel=ch, effectChannel=ch + 1, instrument=part.get("program", 27))
        tr.channel.volume = 104
        tr.channel.balance = part.get("pan", 64)
        tr.name = part["name"]
        tr.strings = [m.GuitarString(i + 1, v) for i, v in enumerate(TUNING)]
        bars = layout_track(part["notes"], nbars, spb)
        cur_tempo = None
        for b, meas in enumerate(tr.measures):
            voice = meas.voices[0]
            voice.beats = []
            for k, (pos, piece, nts, tie) in enumerate(bars[b]):
                val, dot = DUR[piece]
                beat = m.Beat(voice, duration=m.Duration(value=val, isDotted=dot))
                if ti == 0 and k == 0 and tempos[b] != cur_tempo:
                    mt = m.MixTableChange(tempo=m.MixTableItem(value=int(tempos[b]), duration=0))
                    beat.effect.mixTableChange = mt
                    cur_tempo = tempos[b]
                if nts is None:
                    beat.status = m.BeatStatus.rest
                else:
                    beat.status = m.BeatStatus.normal
                    for n in sorted(nts, key=lambda n: n["string"]):
                        note = m.Note(beat, value=n["fret"], string=n["string"] + 1,
                                      velocity=int(min(127, max(30, n["vel"]))),
                                      type=m.NoteType.tie if tie else m.NoteType.normal)
                        if n["dur"] > piece and not tie:
                            note.effect.letRing = True
                        beat.notes.append(note)
                voice.beats.append(beat)
    gp.write(song, str(path), version=(5, 1, 0))
    return path


def read(path):
    """Parse a .gp5 into dict(tempos per bar, meter, tracks=[dict(name, program, notes=[dict(t_beats, dur_beats, p,
    vel, string, fret, bar)])]) using Guitar Pro playback semantics: tied notes extend, let-ring notes sound until
    the next note on the same string (capped), others for their beat duration."""
    song = gp.parse(str(path))
    hdrs = song.measureHeaders
    meter = hdrs[0].timeSignature.numerator
    tempos = []
    cur = song.tempo
    tr0 = song.tracks[0]
    for meas in tr0.measures:
        for beat in meas.voices[0].beats[:1]:
            mt = beat.effect.mixTableChange
            if mt is not None and mt.tempo is not None and mt.tempo.value > 0:
                cur = mt.tempo.value
        tempos.append(cur)
    tracks = []
    for tr in song.tracks:
        tun = {s.number: s.value for s in tr.strings}
        notes = []
        last_on_string = {}
        for bi, meas in enumerate(tr.measures):
            t = bi * meter
            for beat in meas.voices[0].beats:
                d = 4.0 / beat.duration.value * (1.5 if beat.duration.isDotted else 1.0)
                if beat.duration.tuplet and beat.duration.tuplet.enters > 1:
                    d *= beat.duration.tuplet.times / beat.duration.tuplet.enters
                for n in beat.notes:
                    if n.type == gp.NoteType.tie and n.string in last_on_string:
                        prev = last_on_string[n.string]
                        prev["dur"] = t + d - prev["t"]
                        continue
                    if n.type == gp.NoteType.dead:
                        continue
                    if n.string in last_on_string:  # a new note on a string stops the previous one
                        prev = last_on_string[n.string]
                        if prev["ring"]:
                            prev["dur"] = t - prev["t"]
                    note = dict(t=t, dur=d, p=tun[n.string] + n.value, vel=n.velocity, string=n.string - 1,
                                fret=n.value, bar=bi, ring=bool(n.effect.letRing))
                    if note["ring"]:
                        note["dur"] = max(d, RING_CAP)  # until next note on the string (cut above), capped
                    notes.append(note)
                    last_on_string[n.string] = note
                t += d
        tracks.append(dict(name=tr.name, program=tr.channel.instrument, notes=notes))
    return dict(tempos=tempos, meter=meter, nbars=len(hdrs), tracks=tracks)
