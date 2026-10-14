"""Render tab notes to audio with FluidSynth (GeneralUser GS guitar patches)."""
import tempfile
from pathlib import Path

import numpy as np
import pretty_midi

from .common import SR, load, sh

SOUNDFONT = None


def find_soundfont():
    global SOUNDFONT
    if SOUNDFONT:
        return SOUNDFONT
    here = Path(__file__).resolve().parent.parent
    for c in [here / "soundfonts" / "GeneralUser-GS.sf2", Path.home() / "tab-sim/work/sf/GeneralUser-GS.sf2",
              here / "soundfonts" / "FluidR3Mono_GM.sf3"]:
        if c.exists():
            SOUNDFONT = str(c)
            return SOUNDFONT
    raise FileNotFoundError("no soundfont: run scripts/setup.sh")


class Timeline:
    """beat position (in beats from bar 1) -> seconds, from per-bar GP tempos, offset to the recording's bar 1."""

    def __init__(self, tempos, meter, offset):
        self.meter = meter
        self.offset = offset
        self.bar_start = [offset]
        for t in tempos:
            self.bar_start.append(self.bar_start[-1] + meter * 60.0 / t)
        self.tempos = tempos

    def sec(self, beat):
        b = int(min(max(beat // self.meter, 0), len(self.tempos) - 1))
        return self.bar_start[b] + (beat - b * self.meter) * 60.0 / self.tempos[b]


def to_midi(tracks, tl, path, which=None):
    pm = pretty_midi.PrettyMIDI(initial_tempo=float(tl.tempos[0]))
    for i, tr in enumerate(tracks):
        if which is not None and i not in which:
            continue
        ins = pretty_midi.Instrument(program=int(tr.get("program", 27)), name=tr.get("name", f"t{i}"))
        for n in tr["notes"]:
            s = tl.sec(n["t"])
            e = tl.sec(n["t"] + max(n["dur"], 0.25))
            ins.notes.append(pretty_midi.Note(velocity=int(n["vel"]), pitch=int(n["p"]), start=s, end=e))
        pm.instruments.append(ins)
    pm.write(str(path))


def synth(midi_path, wav_path, gain=0.6):
    sh(["fluidsynth", "-ni", "-q", "-g", str(gain), "-R", "0", "-C", "0", "-r", str(SR), "-F", str(wav_path),
        find_soundfont(), str(midi_path)])


def render_tracks(tracks, tl, wav_path, which=None, length=None):
    with tempfile.TemporaryDirectory() as td:
        mid = Path(td) / "x.mid"
        to_midi(tracks, tl, mid, which)
        synth(mid, wav_path)
    x = load(wav_path)
    if length is not None:
        x = x[:, :length] if x.shape[1] >= length else np.pad(x, ((0, 0), (0, length - x.shape[1])))
    return x
