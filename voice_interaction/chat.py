"""Example application: listen, think, answer out loud.

Same loop as `echo` — microphone, endpointer, Vosk, Piper, speaker — with a
small local LLM spliced into the one place echo hands the text straight back
to the synthesizer. Nothing leaves the device here either: the model is a
GGUF file on disk, run by llama.cpp on the CPU.

The answer is spoken sentence by sentence as it is generated. On a Nano that
matters more than anything else in this file: the robot starts talking after
the first sentence (about a second) instead of after the last one."""

import queue
import sys
import threading
import time

from . import config as cfg
from . import models
from .echo import EchoApp
from .llm import Responder


def _where(responder) -> str:
    """Where the model actually ended up — the one thing you want in the log
    when the same command is fast on one machine and slow on another."""
    if responder.gpu_layers:
        layers = "all layers" if responder.gpu_layers < 0 else f"{responder.gpu_layers} layers"
        return f"{layers} on the GPU"
    return f"{responder.n_threads} threads, CPU"


class ChatApp(EchoApp):
    def __init__(
        self,
        llm_model: str = None,
        max_tokens: int = None,
        system_prompt: str = None,
        history_turns: int = None,
        n_threads: int = None,
        gpu_layers: int = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.responder = Responder(
            lang=self.language.code,
            model=llm_model,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            history_turns=history_turns,
            n_threads=n_threads,
            gpu_layers=gpu_layers,
        )

    def _load(self) -> None:
        super()._load()
        models.require_llm(self.responder.model.name)
        started = time.monotonic()
        self.responder.load()
        print(f"LLM loaded in {time.monotonic() - started:.1f}s "
              f"({self.responder.model.name}, {_where(self.responder)}, "
              f"max {self.responder.max_tokens} tokens)")

    def _reply_parts(self, text: str):
        """Yield the answer sentence by sentence as the model produces it.

        Generation runs in its own thread, so the next sentence is being
        decoded while the current one is still being spoken. That overlap is
        most of the reason a 3 tokens/s machine does not feel like one — only
        the first sentence is ever actually waited for."""
        sentences = queue.Queue()

        def generate():
            try:
                for sentence in self.responder.stream(text):
                    sentences.put(sentence)
            except Exception as exc:  # one bad answer must not end the loop
                sentences.put(exc)
            finally:
                sentences.put(None)

        threading.Thread(target=generate, daemon=True).start()

        started = time.monotonic()
        answered = False
        while True:
            item = sentences.get()
            if item is None:
                break
            if isinstance(item, Exception):
                print(f"  answer  : (generation failed: {item})")
                return
            if not answered:
                print(f"  thinking: {time.monotonic() - started:.1f}s to the first sentence")
                answered = True
            print(f"  answer  : {item}")
            yield item
        if not answered:
            print("  answer  : (the model said nothing)")


def main(**kwargs) -> int:
    try:
        return ChatApp(**kwargs).run()
    except KeyboardInterrupt:
        print("\nStopped.")
        return 0


def ask(text: str, lang: str = None, llm_model: str = None, max_tokens: int = None,
        system_prompt: str = None, n_threads: int = None, gpu_layers: int = None,
        speak: bool = False, output_device=None) -> int:
    """One question, one answer — no microphone involved.

    The way to check the chat mode over SSH on a headless robot, the fastest
    way to see what a given model does with a prompt, and — because it prints
    tokens per second — the way to compare models and machines."""
    responder = Responder(lang=lang, model=llm_model, max_tokens=max_tokens,
                          system_prompt=system_prompt, n_threads=n_threads,
                          gpu_layers=gpu_layers)
    models.require_llm(responder.model.name)
    started = time.monotonic()
    responder.load()
    print(f"LLM loaded in {time.monotonic() - started:.1f}s ({responder.model.name}, "
          f"{_where(responder)})")

    synthesizer = None
    if speak:
        from .tts import Synthesizer

        models.require_language(cfg.get_language(lang).code)
        synthesizer = Synthesizer(lang=cfg.get_language(lang).code)
        synthesizer.load()

    print(f"\n> {text}")
    started = time.monotonic()
    first_sec = None
    for sentence in responder.stream(text):
        if first_sec is None:
            first_sec = time.monotonic() - started
        print(sentence)
        if synthesizer is not None:
            synthesizer.speak(sentence, device=output_device)

    total = time.monotonic() - started
    if first_sec is None:
        print("(the model said nothing)")
        return 1
    tokens = responder.last_token_count
    print(f"\n{first_sec:.1f}s to the first sentence, {total:.1f}s in total "
          f"({tokens} tokens, {tokens / total:.1f} tok/s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
