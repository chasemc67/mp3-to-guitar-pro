"""GP5 -> LilyPond (stacked TabStaff + rhythm per track) -> PNG pages."""
import glob
from pathlib import Path

import guitarpro as gp

from .common import log, sh

NAMES = ["c", "cis", "d", "dis", "e", "f", "fis", "g", "gis", "a", "ais", "b"]
BARS_PER_LINE = 4


def _pitch(midi):
    o = midi // 12 - 4
    return NAMES[midi % 12] + ("'" * o if o > 0 else "," * (-o))


def _dur(d):
    return str(d.value) + ("." if d.isDotted else "")


def _track_body(tr, nbars, marks):
    tun = {s.number: s.value for s in tr.strings}
    beats = [(bi, b) for bi, ms in enumerate(tr.measures) for b in ms.voices[0].beats]
    tie_from = set()
    for i, (bi, b) in enumerate(beats):
        for n in b.notes:
            if n.type == gp.NoteType.tie:
                # tie from the most recent earlier beat that has a note on this string
                for j in range(i - 1, -1, -1):
                    if any(m.string == n.string for m in beats[j][1].notes):
                        tie_from.add((j, n.string))
                        break
    body, rbody, line, rline, cur = [], [], [], [], -1
    for i, (bi, b) in enumerate(beats):
        if bi != cur:
            if cur >= 0:
                body.append("  " + " ".join(line) + " |")
                rbody.append("  " + " ".join(rline) + " |")
                if (cur + 1) % BARS_PER_LINE == 0 and nbars - (cur + 1) > 1:
                    body.append("  \\break")
                line, rline = [], []
            cur = bi
            line.append("%% bar %d\n " % (bi + 1))
            if bi + 1 in marks:
                line.append(marks[bi + 1])
        d = _dur(b.duration)
        if b.status == gp.BeatStatus.rest or not b.notes:
            line.append("r" + d)
            rline.append("r" + d)
            continue
        held = any((i, n.string) in tie_from for n in b.notes)
        rline.append("c" + d + ("~" if held else ""))
        parts = []
        for n in sorted(b.notes, key=lambda n: -n.string):
            p = _pitch(tun[n.string] + n.value) + "\\%d" % n.string
            if (i, n.string) in tie_from:
                p += "~"
            parts.append(p)
        line.append("<" + " ".join(parts) + ">" + d)
    body.append("  " + " ".join(line) + ' \\bar "|."')
    rbody.append("  " + " ".join(rline) + ' \\bar "|."')
    return "\n".join(body), "\n".join(rbody)


def to_ly(gp5, ly_path, parts, rhythm, meta):
    song = gp.parse(str(gp5))
    nb = len(song.measureHeaders)
    meter = song.measureHeaders[0].timeSignature.numerator
    marks = {}
    for p in parts:
        if p["bars"][0] > 0:
            marks[p["bars"][0] + 1] = r'\mark \markup \small \bold "%s enters"' % p["name"].split(" (")[0]
    defs, staves = [], []
    for ti, tr in enumerate(song.tracks):
        b, r = _track_body(tr, nb, marks if ti == 0 else {})
        tag = chr(ord("A") + ti)
        tempo = ("\\tempo 4 = %d" % song.tempo) if ti == 0 else ""
        defs.append("tab%s = {\n  \\timing\n  %s\n  \\bar \"\"\n%s\n}\nrhythm%s = {\n  \\timing\n  \\bar \"\"\n%s\n}"
                    % (tag, tempo, b, tag, r))
        short = tr.name.split(" (")[0]
        staves.append(r"""    \new TabStaff \with {
      \RemoveAllEmptyStaves
      stringTunings = #guitar-tuning
      instrumentName = \markup \bold "%(n)s"
      shortInstrumentName = \markup \bold "%(n)s"
      \override TabNoteHead.font-size = #-0.5
      \override TabNoteHead.font-series = #'bold
      \override TimeSignature.stencil = #ly:time-signature::print
      \override VerticalAxisGroup.staff-staff-spacing = #'((basic-distance . 3) (minimum-distance . 2) (padding . 0.3))
    } { \tab%(t)s }
    \new RhythmicStaff \with {
      \RemoveAllEmptyStaves
      \remove Staff_symbol_engraver
      \remove Time_signature_engraver
      \remove Clef_engraver
      \override BarLine.transparent = ##t
      \override NoteHead.transparent = ##t
      \override NoteHead.no-ledgers = ##t
      \override Stem.direction = #DOWN
      \override Beam.direction = #DOWN
      \override Tie.direction = #UP
      \override Stem.length = #6
      fontSize = #-1
      \override StaffSymbol.staff-space = #(magstep -1)
      \override VerticalAxisGroup.staff-staff-spacing = #'((basic-distance . 11) (minimum-distance . 9) (padding . 2))
    } { \rhythm%(t)s }""" % dict(t=tag, n=short))
    legend = " · ".join(p["name"] for p in parts)
    title = meta.get("title", "") + (f" — {meta['artist']}" if meta.get("artist") else "")
    ly = r"""\version "2.24.0"
\header {
  title = "%(title)s"
  subtitle = "Standard tuning (E A D G B E) · ♩ = %(tempo)d · %(meter)d/4 · key %(key)s · %(nb)d bars"
  subsubtitle = \markup \center-column { "%(legend)s" "Auto-transcribed by mp3gp (separation + Basic Pitch + harmonic NMF + DP fingering, refined by re-synthesis). Verify by ear." }
  tagline = ##f
}
\paper {
  #(set-paper-size "letter")
  top-margin = 12\mm  bottom-margin = 12\mm  left-margin = 12\mm  right-margin = 12\mm
  system-system-spacing.basic-distance = #16
  system-system-spacing.minimum-distance = #13
  system-separator-markup = \slashSeparator
  ragged-last = ##f  ragged-last-bottom = ##t  print-page-number = ##t
}
timing = {
  \time %(meter)d/4
  \set Timing.beamExceptions = #'()
  \set Timing.baseMoment = #(ly:make-moment 1/4)
  \set Timing.beatStructure = %(bs)s
}
%(defs)s
\score {
  \new StaffGroup <<
%(staves)s
  >>
  \layout {
    indent = 11\mm
    short-indent = 11\mm
    \context { \Score
      \override BarNumber.break-visibility = ##(#f #t #t)
      barNumberVisibility = #all-bar-numbers-visible
      \override BarNumber.font-size = #-2
      \override RehearsalMark.self-alignment-X = #LEFT
      \override RehearsalMark.font-size = #-1
      \override SpacingSpanner.base-shortest-duration = #(ly:make-moment 1/32)
    }
  }
}
""" % dict(title=title, tempo=song.tempo, meter=meter, key=rhythm.get("key", "?"), nb=nb, legend=legend,
           defs="\n".join(defs), staves="\n".join(staves), bs=",".join(["1"] * meter))
    Path(ly_path).write_text(ly)
    return ly_path


def render(gp5, out_dir, parts, rhythm, meta):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in glob.glob(str(out_dir / "tab*.png")) + glob.glob(str(out_dir / "tab*.pdf")):
        Path(f).unlink()
    ly = to_ly(gp5, out_dir / "tab.ly", parts, rhythm, meta)
    try:
        sh(["lilypond", "--png", "-dresolution=110", "-o", str(out_dir / "tab"), str(ly)])
        sh(["lilypond", "--pdf", "-o", str(out_dir / "tab"), str(ly)])
    except RuntimeError as e:
        log(f"lilypond failed: {str(e)[-500:]}")
    pngs = sorted(glob.glob(str(out_dir / "tab*.png")))
    log(f"lilypond: {len(pngs)} page(s)")
    return pngs
