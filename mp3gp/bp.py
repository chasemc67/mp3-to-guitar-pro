"""Basic Pitch wrapper: cache the raw posteriorgrams once, decode notes at any threshold cheaply."""
import contextlib
import io
from pathlib import Path

import numpy as np

from .common import SR, load, save

FPS = 22050 / 256  # basic-pitch frame rate (annotation frames)
_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            from basic_pitch import ICASSP_2022_MODEL_PATH
            from basic_pitch.inference import Model
            _MODEL = Model(ICASSP_2022_MODEL_PATH)
    return _MODEL


def posteriorgram(wav, cache_dir, mono_fn=None):
    """Return dict(note, onset, contour) arrays for an audio file (cached as .npz next to cache_dir)."""
    wav = Path(wav)
    cache = Path(cache_dir) / (wav.stem + ".bp.npz")
    if cache.exists() and cache.stat().st_mtime >= wav.stat().st_mtime:
        d = np.load(cache)
        return {k: d[k] for k in d.files}
    from basic_pitch.inference import run_inference

    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        out = run_inference(str(wav), _model())
    out = {k: np.asarray(v, np.float32) for k, v in out.items()}
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **out)
    return out


def posteriorgram_array(x, cache_dir, name):
    """Same, for an in-memory signal (written to a temp wav first)."""
    p = Path(cache_dir) / f"{name}.wav"
    p.parent.mkdir(parents=True, exist_ok=True)
    save(p, x)
    return posteriorgram(p, cache_dir)


def decode(pg, onset=0.5, frame=0.3, min_len_ms=60, fmin=70.0, fmax=1400.0):
    """Notes as dicts: start, end (s), p (midi), v (0-127), conf."""
    from basic_pitch.note_creation import model_output_to_notes

    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        _, ev = model_output_to_notes(pg, onset_thresh=onset, frame_thresh=frame,
                                      min_note_len=max(1, int(round(min_len_ms / 1000 * FPS))),
                                      min_freq=fmin, max_freq=fmax, include_pitch_bends=False,
                                      melodia_trick=True)
    notes = [dict(start=float(s), end=float(e), p=int(p), v=int(round(127 * a)), conf=float(a))
             for s, e, p, a, *_ in ev]
    notes.sort(key=lambda n: (n["start"], n["p"]))
    return notes


def frame_activation(pg):
    """(frames, 88) note posterior, midi 21..108."""
    return pg["note"]


def midi_of_bin(k):
    return 21 + k
