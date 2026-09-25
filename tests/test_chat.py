"""The chat mode: sentence streaming, conversation memory, and the loop.

The model itself is never loaded here — a 470 MB GGUF would make the suite
useless as a quick check, and what can actually break is the plumbing around
it: where sentences are cut, what the model is told about earlier turns, and
whether the microphone stays muted while the robot talks.

Run with: pytest
"""

import types

import numpy as np
import pytest

from voice_interaction import audio
from voice_interaction import chat as chat_module
from voice_interaction.chat import ChatApp
from voice_interaction.llm import Responder, clean_for_speech, sentences

from conftest import FakeMicrophone, needs_models, quiet


# --- Sentence streaming ---------------------------------------------------

def test_sentences_are_emitted_as_they_complete():
    stream = iter(["Mir geht", " es gut.", " Und selbst", "?"])
    assert list(sentences(stream)) == ["Mir geht es gut.", "Und selbst?"]


def test_numbers_and_abbreviations_do_not_end_a_sentence():
    text = "Das sind 3.5 Grad, z.B. heute. Und morgen mehr."
    assert list(sentences([text])) == ["Das sind 3.5 Grad, z.B. heute.", "Und morgen mehr."]


def test_short_fragments_are_not_spoken_separately():
    """'Ja.' on its own would just add a gap before the rest of the answer."""
    assert list(sentences(["Ja. Gerne doch."])) == ["Ja. Gerne doch."]


def test_an_answer_without_punctuation_still_arrives():
    assert list(sentences(["kein satzende in sicht"])) == ["kein satzende in sicht"]


def test_markdown_and_emoji_are_stripped():
    """Small models produce them however firmly the system prompt forbids it,
    and Piper would read the marks out loud."""
    assert clean_for_speech("**Hallo** \U0001F600 \n- erstens") == "Hallo - erstens"


# --- Conversation memory --------------------------------------------------

class ScriptedResponder(Responder):
    """A Responder whose llama.cpp call is replaced by a canned token stream."""

    def __init__(self, tokens, **kwargs):
        super().__init__(**kwargs)
        self.tokens = tokens
        self.prompts = []

    def _complete(self, text):
        self.prompts.append(self._messages(text))
        return iter([{"choices": [{"delta": {"content": token}}]} for token in self.tokens])


def test_a_finished_answer_is_remembered():
    responder = ScriptedResponder(["Mir ", "geht ", "es ", "gut."])
    assert responder.reply("wie geht es dir") == "Mir geht es gut."

    responder.reply("und sonst")
    roles = [m["role"] for m in responder.prompts[1]]
    assert roles == ["system", "user", "assistant", "user"]
    assert responder.prompts[1][1]["content"] == "wie geht es dir"
    assert responder.prompts[1][2]["content"] == "Mir geht es gut."


def test_an_abandoned_answer_is_not_remembered():
    """Ctrl+C mid-sentence must not leave half a turn in the history."""
    responder = ScriptedResponder(["Erstens ist alles gut. ", "Zweitens auch."])
    stream = responder.stream("wie geht es dir")
    next(stream)
    stream.close()

    responder.reply("und sonst")
    assert [m["role"] for m in responder.prompts[1]] == ["system", "user"]


def test_history_can_be_switched_off():
    responder = ScriptedResponder(["Alles gut."], history_turns=0)
    responder.reply("wie geht es dir")
    responder.reply("und sonst")
    assert [m["role"] for m in responder.prompts[1]] == ["system", "user"]


def test_history_is_bounded():
    """A robot left running for a day must not grow its prompt all day."""
    responder = ScriptedResponder(["Alles gut."], history_turns=2)
    for i in range(5):
        responder.reply(f"frage {i}")
    # system + two remembered exchanges + the new question
    assert len(responder.prompts[-1]) == 1 + 2 * 2 + 1


# --- Model and device selection -------------------------------------------

def test_the_model_stays_on_the_cpu_by_default():
    """A Nano's GPU belongs to the vision pipeline, so offload is opt-in."""
    assert Responder().gpu_layers == 0


def test_model_and_gpu_choice_reach_the_responder():
    app = ChatApp(lang="de", llm_model="qwen2.5-7b", gpu_layers=-1, n_threads=8)
    assert app.responder.model.name == "qwen2.5-7b"
    assert app.responder.gpu_layers == -1
    assert app.responder.n_threads == 8


# --- The loop -------------------------------------------------------------

class FakeResponder:
    """Stands in for the LLM inside ChatApp."""

    answer = ["Mir geht es gut.", "Und dir?"]

    def __init__(self, **kwargs):
        self.model = types.SimpleNamespace(name="fake")
        self.n_threads = 1
        self.gpu_layers = 0
        self.max_tokens = 32
        self.asked = []

    def load(self):
        pass

    def stream(self, text):
        self.asked.append(text)
        yield from self.answer


@pytest.fixture
def no_llm(monkeypatch):
    """ChatApp with the model replaced — no GGUF, no llama-cpp-python."""
    responder = FakeResponder()
    monkeypatch.setattr(chat_module, "Responder", lambda **kwargs: responder)
    monkeypatch.setattr(chat_module.models, "require_llm", lambda name=None: None)
    return responder


@needs_models
def test_chat_speaks_every_sentence_of_the_answer(monkeypatch, no_llm, spoken_phrase):
    mic = FakeMicrophone(np.concatenate([quiet(1.2, 1), spoken_phrase, quiet(2.0, 2)]))
    monkeypatch.setattr(audio, "Microphone", lambda **kwargs: mic)
    spoken = []
    monkeypatch.setattr(audio, "play_wav", lambda wav_bytes, device=None: spoken.append(wav_bytes))

    assert ChatApp(lang="de", show_partial=False).run() == 0

    assert "roboter" in no_llm.asked[0], "the transcript never reached the model"
    assert len(spoken) == len(FakeResponder.answer), "not every sentence was spoken"
    # The first sentence must be audible, not an empty buffer.
    played, rate = audio.wav_to_array(spoken[0])
    assert np.abs(played).max() > 1000


@needs_models
def test_microphone_stays_muted_for_the_whole_answer(monkeypatch, no_llm, spoken_phrase):
    """Muting once per sentence would reopen the mic between them, and the
    robot would hear the end of its own answer."""
    mic = FakeMicrophone(np.concatenate([quiet(1.2, 3), spoken_phrase, quiet(2.0, 4)]))
    monkeypatch.setattr(audio, "Microphone", lambda **kwargs: mic)
    muted_during_playback = []
    monkeypatch.setattr(
        audio, "play_wav",
        lambda wav_bytes, device=None: muted_during_playback.append(mic.muted_for),
    )

    ChatApp(lang="de", show_partial=False).run()

    assert muted_during_playback == [1, 1], "mic was not muted for the whole turn"


@needs_models
def test_a_failing_model_does_not_stop_the_robot(monkeypatch, no_llm, spoken_phrase):
    def explode(text):
        no_llm.asked.append(text)
        raise RuntimeError("out of context")
        yield  # pragma: no cover - makes this a generator

    no_llm.stream = explode
    mic = FakeMicrophone(np.concatenate([quiet(1.2, 5), spoken_phrase, quiet(2.0, 6)]))
    monkeypatch.setattr(audio, "Microphone", lambda **kwargs: mic)
    spoken = []
    monkeypatch.setattr(audio, "play_wav", lambda wav_bytes, device=None: spoken.append(wav_bytes))

    assert ChatApp(lang="de", show_partial=False).run() == 0
    assert spoken == []
