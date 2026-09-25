"""Text-to-speech via Piper (ONNX Runtime, CPU only).

Piper's own PiperVoice.load() does not expose onnxruntime's thread counts,
so the InferenceSession is built here instead — capping threads is the main
lever for keeping TTS from crowding out the recognizer on a 4-core Nano."""

import io
import json
import logging
import threading
import wave
from typing import Optional

import onnxruntime
from piper import PiperVoice
from piper.config import PiperConfig

from . import audio
from . import config as cfg

_LOGGER = logging.getLogger(__name__)


class Synthesizer:
    """One resident voice, with synthesis serialized so peak CPU and RAM stay
    predictable when several callers share the process."""

    def __init__(self, lang: str = None, voice: str = None, num_threads: int = None):
        if voice is not None:
            self.voice_name = voice
            self.language = None
        else:
            self.language = cfg.get_language(lang)
            self.voice_name = self.language.piper_voice
        self.num_threads = num_threads or cfg.TTS_THREADS
        self._lock = threading.Lock()
        self._voice: Optional[PiperVoice] = None

    def load(self) -> None:
        onnx_path, config_path = cfg.piper_paths(self.voice_name)
        if not onnx_path.exists() or not config_path.exists():
            raise FileNotFoundError(
                f"Piper voice not found: {onnx_path}\n"
                f"Run: python -m voice_interaction download --voice {self.voice_name}"
            )

        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)

        sess_options = onnxruntime.SessionOptions()
        sess_options.intra_op_num_threads = self.num_threads
        sess_options.inter_op_num_threads = 1
        sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL

        session = onnxruntime.InferenceSession(
            str(onnx_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        self._voice = PiperVoice(config=PiperConfig.from_dict(config_dict), session=session)
        _LOGGER.info(
            "Loaded Piper voice '%s' (%d Hz, %d threads)",
            self.voice_name,
            self._voice.config.sample_rate,
            self.num_threads,
        )

    @property
    def sample_rate(self) -> int:
        assert self._voice is not None, "call load() first"
        return self._voice.config.sample_rate

    def synthesize(self, text: str) -> bytes:
        """Return a complete WAV file as bytes."""
        assert self._voice is not None, "call load() first"
        with self._lock:
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wav_file:
                self._voice.synthesize_wav(text, wav_file)
            return buf.getvalue()

    def speak(self, text: str, device=None) -> None:
        """Synthesize and play through the speaker, blocking until done."""
        audio.play_wav(self.synthesize(text), device=device)
