"""Endpointer tests. Deliberately free of models and audio hardware so they
run anywhere — CI, a fresh Windows box, a headless Jetson.

Run with: pytest
"""

import numpy as np
import pytest

from voice_interaction import config as cfg
from voice_interaction.vad import Endpointer, Event

BLOCK = cfg.BLOCK_SIZE
RATE = cfg.SAMPLE_RATE


def signal(seconds: float, level: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0, level, int(seconds * RATE)).astype(np.int16)


def run(endpointer: Endpointer, samples: np.ndarray):
    """Feed a signal block by block, returning (event, timestamp) for the
    events that mark utterance boundaries."""
    marks = []
    for offset in range(0, len(samples) - BLOCK, BLOCK):
        event = endpointer.update(samples[offset:offset + BLOCK].tobytes())
        if event in (Event.SPEECH_START, Event.ENDPOINT, Event.TOO_SHORT):
            marks.append((event, offset / RATE))
    return marks


@pytest.mark.parametrize("noise_level,label", [(5, "quiet room"), (300, "robot fan")])
def test_endpoint_fires_after_configured_pause(noise_level, label):
    speech_sec, pause_sec, lead_sec = 1.5, 0.8, 1.0
    stream = np.concatenate([
        signal(lead_sec, noise_level, seed=1),
        signal(speech_sec, noise_level * 20 + 4000, seed=2),
        signal(2.0, noise_level, seed=3),
    ])

    endpointer = Endpointer(silence_sec=pause_sec)
    endpointer.calibrate([signal(lead_sec, noise_level, seed=1)[i:i + BLOCK].tobytes()
                          for i in range(0, BLOCK * 16, BLOCK)])

    marks = run(endpointer, stream)
    assert [event for event, _ in marks] == [Event.SPEECH_START, Event.ENDPOINT]

    start_at, end_at = marks[0][1], marks[1][1]
    assert lead_sec - 0.2 <= start_at <= lead_sec + 0.2
    assert abs(end_at - (lead_sec + speech_sec + pause_sec)) < 0.25


def test_threshold_adapts_to_noise_floor():
    quiet = Endpointer()
    quiet.calibrate([signal(0.064, 5, seed=i).tobytes() for i in range(16)])
    noisy = Endpointer()
    noisy.calibrate([signal(0.064, 300, seed=i).tobytes() for i in range(16)])

    assert noisy.threshold > quiet.threshold
    # A silent room must not produce a hair-trigger threshold.
    assert quiet.threshold >= 150


def test_short_blip_is_rejected():
    endpointer = Endpointer(silence_sec=0.5, min_speech_sec=0.3)
    endpointer.calibrate([signal(0.064, 5, seed=i).tobytes() for i in range(16)])

    stream = np.concatenate([signal(0.1, 6000, seed=9), signal(1.0, 5, seed=10)])
    events = [event for event, _ in run(endpointer, stream)]

    assert Event.TOO_SHORT in events
    assert Event.ENDPOINT not in events


def test_max_utterance_forces_endpoint():
    """A stuck-open or permanently noisy mic must not record forever: a
    monologue past the cap is cut into successive utterances instead."""
    endpointer = Endpointer(silence_sec=5.0, max_utterance_sec=1.0, threshold=100)
    marks = run(endpointer, signal(3.0, 5000, seed=11))

    assert marks[0][0] is Event.SPEECH_START
    assert marks[1][0] is Event.ENDPOINT
    assert marks[1][1] <= 1.2
    # And it keeps cutting rather than stopping after the first cap.
    assert sum(1 for event, _ in marks if event is Event.ENDPOINT) >= 2


def test_silence_alone_never_starts_an_utterance():
    endpointer = Endpointer(threshold=1000)
    assert run(endpointer, signal(2.0, 5, seed=12)) == []
