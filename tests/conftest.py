"""Shared fixtures: a microphone that isn't one, and something to say into it.

Both loop tests (`echo` and `chat`) need the same thing — a recording of real
synthesized speech with silence around it — and synthesizing it once per
session keeps the suite quick.
"""

import numpy as np
import pytest

from voice_interaction import audio
from voice_interaction import config as cfg
from voice_interaction import models
from voice_interaction.tts import Synthesizer


class FakeMicrophone:
    """Replays a fixed signal as if it came from a mic, then ends the stream
    so the app's loop terminates instead of blocking forever."""

    def __init__(self, samples: np.ndarray):
        self.samples = samples
        self.muted_for = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def drain(self):
        pass

    def mute(self):
        self.muted_for += 1

    def unmute(self):
        pass

    def read(self) -> bytes:
        chunk, self.samples = self.samples[:cfg.BLOCK_SIZE], self.samples[cfg.BLOCK_SIZE:]
        return chunk.tobytes()

    def chunks(self):
        while len(self.samples) >= cfg.BLOCK_SIZE:
            yield self.read()


def to_mic_rate(samples: np.ndarray, rate: int) -> np.ndarray:
    """Piper renders at 22.05 kHz; the mic path runs at 16 kHz."""
    if rate == cfg.SAMPLE_RATE:
        return samples
    length = int(len(samples) * cfg.SAMPLE_RATE / rate)
    return np.interp(
        np.linspace(0, len(samples) - 1, length),
        np.arange(len(samples)),
        samples.astype(np.float32),
    ).astype(np.int16)


def quiet(seconds: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0, 8, int(seconds * cfg.SAMPLE_RATE)).astype(np.int16)


PHRASE = "hallo roboter wie geht es dir"

needs_models = pytest.mark.skipif(
    bool(models.missing_for("de")),
    reason="models not downloaded: python -m voice_interaction download --lang de",
)


@pytest.fixture(scope="session")
def spoken_phrase() -> np.ndarray:
    """PHRASE as Piper says it, resampled to what the microphone delivers."""
    synth = Synthesizer(lang="de")
    synth.load()
    samples, rate = audio.wav_to_array(synth.synthesize(PHRASE))
    return to_mic_rate(samples, rate)
