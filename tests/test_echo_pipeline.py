"""End-to-end test of the echo loop with a simulated microphone.

Covers the seam the other tests do not: endpointer, recognizer and
synthesizer working together inside EchoApp. Needs the models, so it skips
when they have not been downloaded.

Run with: pytest
"""

import numpy as np
import pytest

from voice_interaction import audio
from voice_interaction import config as cfg
from voice_interaction import models
from voice_interaction.echo import EchoApp
from voice_interaction.tts import Synthesizer

pytestmark = pytest.mark.skipif(
    bool(models.missing_for("de")),
    reason="models not downloaded: python -m voice_interaction download --lang de",
)

PHRASE = "hallo roboter wie geht es dir"


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


@pytest.fixture(scope="module")
def spoken_phrase() -> np.ndarray:
    synth = Synthesizer(lang="de")
    synth.load()
    samples, rate = audio.wav_to_array(synth.synthesize(PHRASE))
    return to_mic_rate(samples, rate)


def test_echo_recognizes_and_speaks_back(monkeypatch, spoken_phrase, capsys):
    mic = FakeMicrophone(np.concatenate([quiet(1.2, 1), spoken_phrase, quiet(2.0, 2)]))
    monkeypatch.setattr(audio, "Microphone", lambda **kwargs: mic)

    spoken = []
    monkeypatch.setattr(audio, "play_wav", lambda wav_bytes, device=None: spoken.append(wav_bytes))

    assert EchoApp(lang="de", show_partial=False).run() == 0

    assert len(spoken) == 1, "expected exactly one spoken response"
    # What it says back must be audible and roughly as long as the input.
    played, rate = audio.wav_to_array(spoken[0])
    assert len(played) / rate > 0.5
    assert np.abs(played).max() > 1000

    transcript = capsys.readouterr().out
    assert "roboter" in transcript.lower()


def test_microphone_is_muted_while_speaking(monkeypatch, spoken_phrase):
    """Without this the robot transcribes its own voice and answers itself."""
    mic = FakeMicrophone(np.concatenate([quiet(1.2, 3), spoken_phrase, quiet(2.0, 4)]))
    monkeypatch.setattr(audio, "Microphone", lambda **kwargs: mic)

    muted_during_playback = []
    monkeypatch.setattr(
        audio, "play_wav",
        lambda wav_bytes, device=None: muted_during_playback.append(mic.muted_for),
    )

    EchoApp(lang="de", show_partial=False).run()

    assert muted_during_playback == [1], "mic was not muted before playback"


def test_silence_alone_produces_no_speech(monkeypatch):
    mic = FakeMicrophone(quiet(3.0, 5))
    monkeypatch.setattr(audio, "Microphone", lambda **kwargs: mic)

    spoken = []
    monkeypatch.setattr(audio, "play_wav", lambda wav_bytes, device=None: spoken.append(wav_bytes))

    EchoApp(lang="de", show_partial=False).run()

    assert spoken == []
