"""Utterance endpointing: decides when the speaker has finished.

Energy-based rather than a learned VAD on purpose — it costs almost nothing
on a Cortex-A57, has no model to load, and its one knob (the threshold) can
be calibrated against the actual room, which is what a robot with audible
fans and motors needs."""

import enum
from typing import Iterable, Optional

from . import config as cfg
from .audio import rms

# Calibration: a chunk counts as speech at this multiple of the measured
# noise floor, never below the absolute minimum (a silent room must not
# produce a hair-trigger threshold).
NOISE_FACTOR = 3.0
MIN_THRESHOLD = 150.0


class Event(enum.Enum):
    IDLE = "idle"
    """Silence, no utterance in progress."""

    SPEECH_START = "speech_start"
    SPEECH = "speech"
    """Utterance in progress (SPEECH_START only on the first chunk)."""

    ENDPOINT = "endpoint"
    """Pause threshold reached — the utterance is complete."""

    TOO_SHORT = "too_short"
    """Pause reached, but the sound was too brief to be speech; discarded."""


class Endpointer:
    def __init__(
        self,
        threshold: Optional[float] = None,
        sample_rate: int = None,
        block_size: int = None,
        silence_sec: float = None,
        min_speech_sec: float = None,
        max_utterance_sec: float = None,
    ):
        self.sample_rate = sample_rate or cfg.SAMPLE_RATE
        self.block_size = block_size or cfg.BLOCK_SIZE
        self.threshold = threshold if threshold is not None else cfg.SILENCE_THRESHOLD
        self.silence_sec = silence_sec if silence_sec is not None else cfg.SILENCE_SEC
        self.min_speech_sec = min_speech_sec if min_speech_sec is not None else cfg.MIN_SPEECH_SEC
        self.max_utterance_sec = (
            max_utterance_sec if max_utterance_sec is not None else cfg.MAX_UTTERANCE_SEC
        )
        self.block_sec = self.block_size / self.sample_rate
        self.last_level = 0.0
        self.reset()

    def reset(self) -> None:
        self._started = False
        self._speech_blocks = 0
        self._silence_blocks = 0
        self._total_blocks = 0

    @property
    def speech_seconds(self) -> float:
        return self._speech_blocks * self.block_sec

    def calibrate(self, chunks: Iterable[bytes]) -> float:
        """Measure the room's noise floor and derive a threshold from it."""
        levels = [rms(chunk) for chunk in chunks]
        noise_floor = sorted(levels)[len(levels) // 2] if levels else 0.0
        self.threshold = max(noise_floor * NOISE_FACTOR, MIN_THRESHOLD)
        return self.threshold

    def update(self, chunk: bytes) -> Event:
        if self.threshold is None:
            raise RuntimeError("threshold not set — call calibrate() or pass one")

        level = rms(chunk)
        self.last_level = level
        is_speech = level >= self.threshold

        if not self._started:
            if not is_speech:
                return Event.IDLE
            self._started = True
            self._speech_blocks = 1
            self._silence_blocks = 0
            self._total_blocks = 1
            return Event.SPEECH_START

        self._total_blocks += 1
        if is_speech:
            self._speech_blocks += 1
            self._silence_blocks = 0
        else:
            self._silence_blocks += 1

        if self._silence_blocks * self.block_sec >= self.silence_sec:
            long_enough = self.speech_seconds >= self.min_speech_sec
            self.reset()
            return Event.ENDPOINT if long_enough else Event.TOO_SHORT

        if self._total_blocks * self.block_sec >= self.max_utterance_sec:
            self.reset()
            return Event.ENDPOINT

        return Event.SPEECH
