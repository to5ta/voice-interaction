"""Central configuration. Every value is overridable by an environment
variable so a robot deployment can be tuned via its systemd unit without
touching code."""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Language:
    """A language preset pairs an STT model with a matching TTS voice."""

    code: str
    label: str
    vosk_model: str
    piper_voice: str


LANGUAGES = {
    "de": Language(
        code="de",
        label="Deutsch",
        vosk_model="vosk-model-small-de-0.15",
        piper_voice="de_DE-thorsten-medium",
    ),
    "en": Language(
        code="en",
        label="English",
        vosk_model="vosk-model-small-en-us-0.15",
        piper_voice="en_US-lessac-medium",
    ),
}

DEFAULT_LANG = os.environ.get("VI_LANG", "de")

MODELS_DIR = Path(os.environ.get("VI_MODELS_DIR", "models")).resolve()
PIPER_DIR = MODELS_DIR / "piper"
VOSK_DIR = MODELS_DIR / "vosk"

# Vosk's small models are 16 kHz; capturing at their native rate avoids a
# resampling step in the hot path.
SAMPLE_RATE = 16000

# 1024 samples = 64 ms at 16 kHz: fine enough for responsive pause detection,
# coarse enough to keep per-chunk overhead negligible.
BLOCK_SIZE = 1024

# Audio devices: sounddevice index (int) or a substring of the device name.
# None lets PortAudio pick the system default.
INPUT_DEVICE = os.environ.get("VI_INPUT_DEVICE") or None
OUTPUT_DEVICE = os.environ.get("VI_OUTPUT_DEVICE") or None

# Thread caps. The Nano has 4 cores and must also run the robot's own logic,
# so neither engine gets to claim all of them.
TTS_THREADS = int(os.environ.get("VI_TTS_THREADS", "2"))

# --- Endpointing ("Threshold-Pause") -------------------------------------
# Silence after speech that ends an utterance.
SILENCE_SEC = float(os.environ.get("VI_SILENCE_SEC", "0.8"))
# RMS level (int16 scale, 0..32767) above which a chunk counts as speech.
# Unset -> measured from the room at startup, which is what you want on a
# robot whose own fans and motors set the noise floor.
_threshold = os.environ.get("VI_SILENCE_THRESHOLD")
SILENCE_THRESHOLD = float(_threshold) if _threshold else None
# Utterances shorter than this are treated as noise (door slam, cough).
MIN_SPEECH_SEC = float(os.environ.get("VI_MIN_SPEECH_SEC", "0.3"))
# Hard cap so a stuck-open mic can't record forever.
MAX_UTTERANCE_SEC = float(os.environ.get("VI_MAX_UTTERANCE_SEC", "15.0"))


def get_language(code: str = None) -> Language:
    code = (code or DEFAULT_LANG).lower()
    if code not in LANGUAGES:
        raise ValueError(
            f"Unknown language '{code}'. Available: {', '.join(sorted(LANGUAGES))}"
        )
    return LANGUAGES[code]


def piper_paths(voice: str):
    return PIPER_DIR / f"{voice}.onnx", PIPER_DIR / f"{voice}.onnx.json"


def vosk_path(model: str) -> Path:
    return VOSK_DIR / model
