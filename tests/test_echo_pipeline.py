"""End-to-end test of the echo loop with a simulated microphone.

Covers the seam the other tests do not: endpointer, recognizer and
synthesizer working together inside EchoApp. Needs the models, so it skips
when they have not been downloaded.

Run with: pytest
"""

import numpy as np

from voice_interaction import audio
from voice_interaction.echo import EchoApp

from conftest import FakeMicrophone, needs_models, quiet

pytestmark = needs_models


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
