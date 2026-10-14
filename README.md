# mp3-to-guitar-pro

Turns a mixed recording (mp3) into a multi-track, playable Guitar Pro 5 tab. The tab is then checked by playing it
back with FluidSynth and comparing it bar by bar with the recording, and the worst bars are redone over several
passes until the accuracy stops improving.

```
scripts/setup.sh                                   # brew ffmpeg/fluidsynth/lilypond, Python 3.11 venv, soundfont
.venv/bin/python -m mp3gp input/song.mp3 --work work --out out          # default: --mode ensemble
.venv/bin/python -m mp3gp input/song.mp3 --mode stems                    # alternative: fit tracks to stems
.venv/bin/python -m mp3gp ... --force all                                # recompute cached stages
```

Outputs (`--out`): `<title>.gp5`, `tab/tab-page*.png` + `tab.pdf` (LilyPond), `comparison.wav` (about 30 s of the
original, then the same section played from the tab), `comparison_side_by_side.wav` (original left / tab right;
there is also a whole-song mp3), `parts/` (each part's stem and a stem-vs-tab check), `passes.json` and `REPORT.md`.
Runtime is about 1.5 min with cached stems (about 5 min from scratch on an M4).

## Pipeline

| stage | module | what it does |
|---|---|---|
| prepare | `pipeline.py` | ffmpeg → 44.1 kHz stereo wav, tags → title/artist |
| rhythm | `rhythm.py` | beat_this beats + downbeats → spurious/missed beats fixed → local-linear smoothing (keeps tempo drift, removes 20 ms tracker jitter) → meter + downbeat phase → bars; piecewise-constant GP tempo map; key by Krumhansl–Schmuckler on CQT chroma |
| separation | `separate.py` | demucs htdemucs / htdemucs_6s / htdemucs_ft + BS-Roformer (no drum-bass) give candidate all-guitar stems, scored per bar (drum-leak TF overlap and harmonic loss); best candidate per 4-bar section, crossfaded. Then PCA primary/ambient decomposition per TF tile, centre/left/right pan masks, ambient L/R, mid/side |
| parts | `parts.py` | per-bar energy of centre / wide / L-only / R-only → each bar labelled `right`, `left` or `double` (both sides, low coherence = double-tracked) → contiguous runs become parts; one stem per part |
| transcription | `bp.py`, `ghost.py`, `salience.py`, `transcribe.py` | Basic Pitch posteriorgrams (cached) → **ghost-note classifier** (gradient boosting on relative activation features, trained on synthetic guitar renders, `scripts/train_ghost.py`) → harmonic NMF salience (templates restricted to harmonic bins) for octave checks → quantised to the 16th grid of the detected (tempo-varying) beat grid |
| roles | `ensemble.py` | one note pool from the all-guitar stem, distributed over the parts as **playable roles**: spatial energy share is a hint; per onset a lead part takes ≤2 notes and the arpeggio/rhythm part ≤4 new notes; each part's chord must be fingerable, otherwise the note moves to another active part or is dropped |
| fingering | `fingering.py` | Viterbi DP over (fingering, hand position) states: ≤5-fret span (spans of 4–5 cost extra), ≤4 fingers with index barre, hand-shift cost scaled by the available time, penalty for cutting ringing strings, second pass enforcing the most common chord shape for repeated pitch sets |
| GP5 | `gpio.py` | pyguitarpro writer (ties across barlines, let-ring, tempo changes, markers, pan); reader that turns any .gp5 back into timed notes |
| check | `render.py`, `evaluate.py` | .gp5 → MIDI on the recording's timeline → FluidSynth (GeneralUser GS). Per bar (time-normalised): CQT-chroma cosine, frame pitch-F (Basic Pitch on both sides), onset-F (±50 ms) |
| loop | `ensemble.py` (`refine.py` for `--mode stems`) | each pass renders the whole tab, scores it against the **all-guitar stem and the mix**, re-estimates the worst ~30% of bars (pool thresholds, octave fix, drop/add, register re-assignment, copy of a matching repeated bar), renders every candidate with all parts, and keeps it only if the combined score improves; stops after 2 passes without gain and keeps the best pass |
| outputs | `lily.py`, `compare_audio.py`, `report.py` | LilyPond tab + rhythm staves → PNG/PDF; comparison audio; REPORT.md (optionally appends `<work>/analyst_notes.md`) |

### Modes
* `ensemble` (default): where the audio can't cleanly split the guitars, the parts are *estimated*. Each part
  is playable on its own, and together they are optimised to sound like the guitars and the mix.
* `stems`: each track is fitted to its own separated stem (useful when the parts are hard-panned and separate cleanly).

### Lessons built in (from the earlier attempt)
Demucs merges guitars, so parts come from spatial analysis plus role estimation. Pan-only splitting fails for
centred or doubled parts, so primary/ambient decomposition and a coherence-based `double` label are used, and
the final assignment is optimised for the sound rather than trusting the masks. Basic Pitch ghosts and octave
errors are handled by the learned classifier, NMF octave checks and loop edits. Fingering uses a real DP with hand
position, not the lowest fret. Every tab is verified by playback, and only the GP5 that is read back gets scored.

## Caveats
Standard tuning is assumed. There are no articulations (bends, slides, harmonics, palm mutes). Metrics are
relative to separated stems that contain bleed, so read them as pass-to-pass trends. The ghost classifier is
trained on GM soundfont guitars only. See `REPORT.md` → Open problems.
