#!/usr/bin/env bash
# One-time setup on macOS (Apple Silicon tested): system tools, Python 3.11 venv, models, soundfont.
set -euo pipefail
cd "$(dirname "$0")/.."
command -v brew >/dev/null || { echo "Homebrew required"; exit 1; }
brew list ffmpeg >/dev/null 2>&1 || brew install ffmpeg
brew list fluid-synth >/dev/null 2>&1 || brew install fluidsynth
brew list lilypond >/dev/null 2>&1 || brew install lilypond
command -v uv >/dev/null || brew install uv
[ -d .venv ] || uv venv -p 3.11 .venv
uv pip install -p .venv/bin/python -r requirements.txt
# guitar-capable GM soundfont for the FluidSynth playback check (GeneralUser GS, free license)
[ -f soundfonts/GeneralUser-GS.sf2 ] || curl -fL -o soundfonts/GeneralUser-GS.sf2 \
  https://github.com/mrbumpy409/GeneralUser-GS/raw/main/GeneralUser-GS.sf2
echo "setup ok"
