"""Answer generation with a small local LLM (llama.cpp, CPU only).

The point of `chat` is that the entire loop — hearing, thinking, speaking —
stays on the device, which caps how big the model may be: it has to fit next
to Vosk and Piper in a Nano's memory and answer at a speed a person will wait
for. That lands at half a billion parameters quantized to 4 bits (~470 MB),
with answers capped at a couple of sentences.

Generation is streamed and handed to the synthesizer **sentence by sentence**,
so the robot starts talking after the first sentence instead of after the last
one. On slow hardware that is the difference between a pause you notice and
one you don't.

llama-cpp-python is imported lazily: the speech stack must keep working — and
its tests keep running — on machines that never installed it."""

import collections
import logging
import unicodedata
from typing import Iterable, Iterator, List, Optional

from . import config as cfg

_LOGGER = logging.getLogger(__name__)

# Spoken, not written: no lists, no markdown, no emoji, and short — every
# token is CPU time on the robot and waiting time for the person in front of
# it. Small models follow short, concrete rules far better than long ones.
SYSTEM_PROMPTS = {
    "de": (
        "Du bist die Stimme eines kleinen Roboters. Antworte in höchstens zwei "
        "kurzen Sätzen auf Deutsch, in einfacher gesprochener Sprache. Keine "
        "Listen, keine Emojis, keine Formatierung, keine Bemerkungen über dich "
        "selbst. Wenn du etwas nicht weißt, sage das in einem Satz."
    ),
    "en": (
        "You are the voice of a small robot. Answer in at most two short "
        "sentences of plain spoken English. No lists, no emoji, no formatting, "
        "no remarks about yourself. If you do not know something, say so in one "
        "sentence."
    ),
}

# Sentence ends the synthesizer can be handed separately without the result
# sounding chopped.
_SENTENCE_ENDS = ".!?…"

# Below this a fragment is not worth its own synthesis call — it would only
# add a gap in the middle of something like "Ja. Gerne."
MIN_SENTENCE_CHARS = 12

# Characters Piper would read out or stumble over instead of pronouncing.
_DROP_CHARS = str.maketrans({c: None for c in "*_`#<>|~^[]{}"})


def clean_for_speech(text: str) -> str:
    """Strip what a TTS voice cannot say: markdown marks, emoji, line breaks.

    Small instruct models slip into bullet points and emoji whatever the system
    prompt says, so this is a guarantee rather than a fallback."""
    text = text.translate(_DROP_CHARS)
    # Category 'So' is "symbol, other" — where emoji and pictographs live.
    text = "".join(" " if unicodedata.category(ch) == "So" else ch for ch in text)
    return " ".join(text.split())


def _split_point(buffer: str, min_chars: int) -> Optional[int]:
    """Index just past the first sentence end in `buffer`, or None.

    A sentence end only counts when whitespace and then something that looks
    like a new sentence follow it. That keeps "3.5" and "z.B. heute" in one
    piece; the length floor keeps "Ja." from becoming its own synthesis call.
    A missed split costs a little streaming latency, a wrong one costs an
    audible gap mid-sentence — so the rule errs towards waiting."""
    for i, ch in enumerate(buffer):
        if ch not in _SENTENCE_ENDS:
            continue
        end = i + 1
        while end < len(buffer) and buffer[end] in _SENTENCE_ENDS:
            end += 1  # "?!" and "..." end the sentence once, not three times
        if end >= len(buffer):
            return None  # may still be growing — wait for the next token
        if not buffer[end].isspace():
            continue
        if end < min_chars:
            continue
        start = end
        while start < len(buffer) and buffer[start].isspace():
            start += 1
        if start >= len(buffer):
            return None
        if buffer[start].isupper() or buffer[start].isdigit():
            return end
    return None


def sentences(chunks: Iterable[str], min_chars: int = MIN_SENTENCE_CHARS) -> Iterator[str]:
    """Regroup a stream of model tokens into speakable sentences."""
    buffer = ""
    for chunk in chunks:
        buffer += chunk
        while True:
            cut = _split_point(buffer, min_chars)
            if cut is None:
                break
            head, buffer = buffer[:cut], buffer[cut:]
            spoken = clean_for_speech(head)
            if spoken:
                yield spoken
    rest = clean_for_speech(buffer)
    if rest:
        yield rest


class Responder:
    """One resident GGUF model that turns an utterance into a short answer."""

    def __init__(
        self,
        lang: str = None,
        model: str = None,
        n_threads: int = None,
        max_tokens: int = None,
        n_ctx: int = None,
        temperature: float = None,
        system_prompt: str = None,
        history_turns: int = None,
    ):
        self.language = cfg.get_language(lang)
        self.model = cfg.get_llm_model(model)
        self.n_threads = n_threads or cfg.LLM_THREADS
        self.max_tokens = max_tokens or cfg.LLM_MAX_TOKENS
        self.n_ctx = n_ctx or cfg.LLM_CONTEXT
        self.temperature = cfg.LLM_TEMPERATURE if temperature is None else temperature
        self.system_prompt = (
            system_prompt
            or cfg.LLM_SYSTEM_PROMPT
            or SYSTEM_PROMPTS.get(self.language.code, SYSTEM_PROMPTS["en"])
        )
        turns = cfg.LLM_HISTORY_TURNS if history_turns is None else history_turns
        # Two messages per exchange: the question and the answer.
        self._history = collections.deque(maxlen=max(0, turns) * 2)
        self._llm = None

    def load(self) -> None:
        path = cfg.llm_path(self.model)
        if not path.exists():
            raise FileNotFoundError(
                f"LLM not found: {path}\n"
                f"Run: python -m voice_interaction download --llm {self.model.name}"
            )
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise SystemExit(
                "The chat mode needs llama-cpp-python, which is not installed.\n"
                "Run: pip install -r requirements-llm.txt"
            ) from exc

        self._llm = Llama(
            model_path=str(path),
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            # The GPU stays out of this: on a Nano it belongs to the vision
            # pipeline, and a 0.5B model would not repay the contention.
            n_gpu_layers=0,
            verbose=False,
        )
        _LOGGER.info(
            "Loaded LLM '%s' (%d threads, ctx %d)",
            self.model.name,
            self.n_threads,
            self.n_ctx,
        )

    def reset(self) -> None:
        """Forget the conversation so far."""
        self._history.clear()

    def _messages(self, text: str) -> List[dict]:
        return [
            {"role": "system", "content": self.system_prompt},
            *self._history,
            {"role": "user", "content": text},
        ]

    def _complete(self, text: str):
        assert self._llm is not None, "call load() first"
        return self._llm.create_chat_completion(
            messages=self._messages(text),
            stream=True,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=0.9,
            repeat_penalty=1.1,
        )

    def _tokens(self, text: str, collected: List[str]) -> Iterator[str]:
        try:
            stream = self._complete(text)
        except ValueError:
            # Context exhausted by a long conversation: drop the history and
            # answer the question on its own rather than die mid-dialogue.
            _LOGGER.warning("context window exceeded - dropping conversation history")
            self.reset()
            stream = self._complete(text)

        for piece in stream:
            delta = piece["choices"][0].get("delta", {}).get("content")
            if delta:
                collected.append(delta)
                yield delta

    def stream(self, text: str) -> Iterator[str]:
        """Answer `text`, yielding whole sentences as they become available.

        The exchange is remembered only once the stream has been consumed to
        the end, so an interrupted answer leaves no half turn in the history."""
        collected: List[str] = []
        yield from sentences(self._tokens(text, collected))
        answer = clean_for_speech("".join(collected))
        if answer and self._history.maxlen:
            self._history.append({"role": "user", "content": text})
            self._history.append({"role": "assistant", "content": answer})

    def reply(self, text: str) -> str:
        """The whole answer as one string, for the non-streaming callers."""
        return " ".join(self.stream(text))
