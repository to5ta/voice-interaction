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


@dataclass(frozen=True)
class LlmModel:
    """A GGUF file on HuggingFace, plus what it costs to run it."""

    name: str
    repo: str
    filename: str
    size_mb: int
    note: str


# Deliberately tiny models: the whole point of `chat` is that it still fits
# next to Vosk and Piper on a 4 GB Nano. Everything here is Apache-2.0 and
# runs on the CPU in a few hundred MB.
LLM_MODELS = {
    "qwen2.5-0.5b": LlmModel(
        name="qwen2.5-0.5b",
        repo="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        filename="qwen2.5-0.5b-instruct-q4_k_m.gguf",
        size_mb=469,
        note="default - the smallest model that answers usable German",
    ),
    "qwen2.5-1.5b": LlmModel(
        name="qwen2.5-1.5b",
        repo="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        filename="qwen2.5-1.5b-instruct-q4_k_m.gguf",
        size_mb=1066,
        note="noticeably better answers, ~3x slower - desktops, not a Nano",
    ),
    "smollm2-360m": LlmModel(
        name="smollm2-360m",
        repo="HuggingFaceTB/SmolLM2-360M-Instruct-GGUF",
        filename="smollm2-360m-instruct-q8_0.gguf",
        size_mb=368,
        note="fastest, English only",
    ),
}

DEFAULT_LLM = os.environ.get("VI_LLM_MODEL", "qwen2.5-0.5b")

MODELS_DIR = Path(os.environ.get("VI_MODELS_DIR", "models")).resolve()
PIPER_DIR = MODELS_DIR / "piper"
VOSK_DIR = MODELS_DIR / "vosk"
LLM_DIR = MODELS_DIR / "llm"

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
# so no engine gets to claim all of them — and in `chat` two of them do run at
# once, because the next sentence is generated while the current one is spoken.
TTS_THREADS = int(os.environ.get("VI_TTS_THREADS", "2"))
LLM_THREADS = int(os.environ.get("VI_LLM_THREADS", "2"))

# --- Local LLM (`chat`) ---------------------------------------------------
# Short answers are not a style choice: every token is both CPU time and
# time the user spends waiting for the robot to start talking.
LLM_MAX_TOKENS = int(os.environ.get("VI_LLM_MAX_TOKENS", "80"))
# Context window. Small keeps prompt processing cheap on a Cortex-A57; it
# bounds system prompt + history + question + answer.
LLM_CONTEXT = int(os.environ.get("VI_LLM_CONTEXT", "1024"))
# How many previous exchanges the model still sees. Zero makes every turn
# independent, which is the cheapest and most predictable mode.
LLM_HISTORY_TURNS = int(os.environ.get("VI_LLM_HISTORY_TURNS", "3"))
LLM_TEMPERATURE = float(os.environ.get("VI_LLM_TEMPERATURE", "0.7"))
# Overrides the built-in per-language persona, e.g. to give the robot a job.
LLM_SYSTEM_PROMPT = os.environ.get("VI_LLM_SYSTEM_PROMPT") or None

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


def get_llm_model(name: str = None) -> LlmModel:
    name = (name or DEFAULT_LLM).lower()
    if name not in LLM_MODELS:
        raise ValueError(
            f"Unknown LLM '{name}'. Available: {', '.join(sorted(LLM_MODELS))}"
        )
    return LLM_MODELS[name]


def llm_path(model: LlmModel) -> Path:
    return LLM_DIR / model.filename
