"""Speech-to-text via Vosk (Apache-2.0), streaming and fully offline.

Audio is fed in as it arrives, so by the time the endpointer reports a pause
the transcript is essentially already decoded — that is what keeps the
round trip near real time on a Nano."""

import json
import logging
from typing import Optional

import vosk

from . import config as cfg

_LOGGER = logging.getLogger(__name__)

# Vosk logs Kaldi internals to stderr at default level; -1 silences it.
vosk.SetLogLevel(-1)


class SpeechRecognizer:
    def __init__(self, lang: str = None, sample_rate: int = None):
        self.language = cfg.get_language(lang)
        self.sample_rate = sample_rate or cfg.SAMPLE_RATE
        self._model: Optional[vosk.Model] = None
        self._recognizer: Optional[vosk.KaldiRecognizer] = None

    def load(self) -> None:
        model_path = cfg.vosk_path(self.language.vosk_model)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Vosk model not found: {model_path}\n"
                f"Run: python -m voice_interaction download --lang {self.language.code}"
            )
        self._model = vosk.Model(str(model_path))
        self._recognizer = vosk.KaldiRecognizer(self._model, self.sample_rate)
        # Word-level timing costs CPU and nothing here consumes it.
        self._recognizer.SetWords(False)
        _LOGGER.info("Loaded Vosk model '%s'", self.language.vosk_model)

    def accept(self, chunk: bytes) -> bool:
        """Feed one chunk. True means Vosk itself detected an utterance end;
        we ignore that signal and rely on our own endpointer so the pause
        length stays a single configurable knob."""
        assert self._recognizer is not None, "call load() first"
        return self._recognizer.AcceptWaveform(chunk)

    def partial_text(self) -> str:
        assert self._recognizer is not None, "call load() first"
        return json.loads(self._recognizer.PartialResult()).get("partial", "").strip()

    def final_text(self) -> str:
        """Flush the decoder and return the utterance, resetting for the next."""
        assert self._recognizer is not None, "call load() first"
        return json.loads(self._recognizer.FinalResult()).get("text", "").strip()

    def reset(self) -> None:
        """Drop anything buffered — used after playback so the robot's own
        voice cannot bleed into the next utterance."""
        assert self._recognizer is not None, "call load() first"
        self._recognizer.Reset()
