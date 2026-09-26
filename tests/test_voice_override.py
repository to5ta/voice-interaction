"""`--voice` has to reach the synthesizer in every mode, and the model check
has to look for the voice that will actually be loaded — otherwise you are
told to download the preset's voice you never asked for.

No models and no audio hardware needed.
"""

import pytest

from voice_interaction import config as cfg
from voice_interaction import models
from voice_interaction.echo import EchoApp
from voice_interaction.tts import Synthesizer

OVERRIDE = "de_DE-kerstin-low"


def test_synthesizer_prefers_the_override_over_the_preset():
    assert Synthesizer(lang="de").voice_name == cfg.get_language("de").piper_voice
    assert Synthesizer(lang="de", voice=OVERRIDE).voice_name == OVERRIDE


def test_echo_app_passes_the_voice_through():
    assert EchoApp(lang="de", voice=OVERRIDE).synthesizer.voice_name == OVERRIDE
    assert EchoApp(lang="de").synthesizer.voice_name == cfg.get_language("de").piper_voice


def test_missing_report_names_the_requested_voice(tmp_path, monkeypatch):
    """The whole point of the override reaching models.py: the error must not
    demand a voice the caller did not ask for."""
    monkeypatch.setattr(cfg, "PIPER_DIR", tmp_path / "piper")
    monkeypatch.setattr(cfg, "VOSK_DIR", tmp_path / "vosk")

    missing = models.missing_for("de", OVERRIDE)
    assert any(OVERRIDE in entry for entry in missing)
    assert not any(cfg.get_language("de").piper_voice in entry for entry in missing)


def test_missing_report_falls_back_to_the_preset(tmp_path, monkeypatch):
    """Without an override the preset is what gets checked — asserted against
    an empty models dir so both sides actually report something."""
    monkeypatch.setattr(cfg, "PIPER_DIR", tmp_path / "piper")
    monkeypatch.setattr(cfg, "VOSK_DIR", tmp_path / "vosk")
    preset = cfg.get_language("de").piper_voice

    missing = models.missing_for("de", None)
    assert any(preset in entry for entry in missing)
    assert missing == models.missing_for("de", preset)


def test_require_language_suggests_downloading_the_override(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "PIPER_DIR", tmp_path / "piper")
    monkeypatch.setattr(cfg, "VOSK_DIR", tmp_path / "vosk")

    with pytest.raises(SystemExit) as excinfo:
        models.require_language("de", OVERRIDE)

    message = str(excinfo.value)
    assert f"download --voice {OVERRIDE}" in message
