#!/usr/bin/env bash
# Setup for Jetson Nano P3450 (NVIDIA JetPack / Ubuntu) and ordinary Linux boxes.
# Everything runs on the CPU: the GPU stays free for vision or other CUDA work.
#
# Usage: ./scripts/install_jetson.sh [lang]     lang = de | en | all (default: all)
set -euo pipefail

cd "$(dirname "$0")/.."
LANG_ARG="${1:-all}"

echo "== voice-interaction setup =="

# --- Python ---------------------------------------------------------------
# piper-tts needs >= 3.9. JetPack 4.6 (Ubuntu 18.04) ships Python 3.6, so on a
# stock Nano image a newer interpreter has to be installed first.
PY="${PYTHON:-python3}"
if ! "$PY" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)'; then
  echo "ERROR: $("$PY" -V) is too old — piper-tts requires Python >= 3.9." >&2
  echo >&2
  echo "On JetPack 4.6 (Ubuntu 18.04) install a newer interpreter:" >&2
  echo "  sudo add-apt-repository ppa:deadsnakes/ppa" >&2
  echo "  sudo apt update && sudo apt install -y python3.9 python3.9-venv python3.9-dev" >&2
  echo "  PYTHON=python3.9 ./scripts/install_jetson.sh $LANG_ARG" >&2
  exit 1
fi
echo "Python: $("$PY" -V)"

# --- System audio ---------------------------------------------------------
# sounddevice ships no Linux wheel with a bundled PortAudio, so the system
# library must be there. (On Windows and macOS the wheel brings its own.)
if ! ldconfig -p | grep -q libportaudio.so.2; then
  echo "Installing PortAudio (needs sudo) ..."
  sudo apt-get update
  sudo apt-get install -y libportaudio2
else
  echo "PortAudio: present"
fi

# --- Python environment ---------------------------------------------------
"$PY" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# --- Models ---------------------------------------------------------------
python -m voice_interaction download --lang "$LANG_ARG"

cat <<'EOF'

Setup complete.

  source .venv/bin/activate
  python -m voice_interaction selftest          # verify without a microphone
  python -m voice_interaction devices           # find your mic/speaker
  python -m voice_interaction echo              # the example application

On a Nano, run it at full clocks for the lowest latency:
  sudo nvpmodel -m 0 && sudo jetson_clocks
EOF
