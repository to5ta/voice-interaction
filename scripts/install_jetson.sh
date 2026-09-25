#!/usr/bin/env bash
# Setup for Jetson Nano P3450 (NVIDIA JetPack / Ubuntu) and ordinary Linux boxes.
# Everything runs on the CPU: the GPU stays free for vision or other CUDA work.
#
# Usage: ./scripts/install_jetson.sh [lang] [--llm]
#   lang  = de | en | all (default: all)
#   --llm = also install llama-cpp-python and the chat model (~470 MB)
set -euo pipefail

cd "$(dirname "$0")/.."
LANG_ARG="all"
WITH_LLM=0
for arg in "$@"; do
  case "$arg" in
    --llm) WITH_LLM=1 ;;
    de|en|all) LANG_ARG="$arg" ;;
    *) echo "Usage: $0 [de|en|all] [--llm]" >&2; exit 2 ;;
  esac
done

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

# --- Local LLM (optional) -------------------------------------------------
# --only-binary in requirements-llm.txt is deliberate: building llama.cpp from
# source on a Nano takes the better part of an hour, and a prebuilt aarch64
# wheel exists. If pip cannot find one, that is the message you want to see.
if [ "$WITH_LLM" = "1" ]; then
  pip install -r requirements-llm.txt
fi

# --- Models ---------------------------------------------------------------
python -m voice_interaction download --lang "$LANG_ARG"
if [ "$WITH_LLM" = "1" ]; then
  python -m voice_interaction download --llm
fi

cat <<'EOF'

Setup complete.

  source .venv/bin/activate
  python -m voice_interaction selftest          # verify without a microphone
  python -m voice_interaction devices           # find your mic/speaker
  python -m voice_interaction echo              # the example application
  python -m voice_interaction chat              # ... with a local LLM (--llm installs)

On a Nano, run it at full clocks for the lowest latency:
  sudo nvpmodel -m 0 && sudo jetson_clocks
EOF
