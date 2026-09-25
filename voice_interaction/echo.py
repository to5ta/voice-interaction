"""Example application: listen, wait for the pause, say it back.

This is the smallest complete round trip through the stack — microphone →
endpointer → Vosk → Piper → speaker — and doubles as the hardware smoke
test for a new robot: if echo works, every piece is wired up correctly."""

import collections
import sys
import time

from . import audio
from . import config as cfg
from . import models
from .stt import SpeechRecognizer
from .tts import Synthesizer
from .vad import Endpointer, Event

# Audio kept before the threshold is crossed, so the first syllable — which
# is what pushes the level over the threshold — still reaches the decoder.
PRE_ROLL_BLOCKS = 5

CALIBRATION_BLOCKS = 16  # ~1 s at 64 ms per block


class EchoApp:
    def __init__(
        self,
        lang: str = None,
        input_device=None,
        output_device=None,
        threshold: float = None,
        silence_sec: float = None,
        show_partial: bool = True,
    ):
        self.language = cfg.get_language(lang)
        self.input_device = input_device
        self.output_device = output_device
        self.show_partial = show_partial
        self.recognizer = SpeechRecognizer(lang=self.language.code)
        self.synthesizer = Synthesizer(lang=self.language.code)
        self.endpointer = Endpointer(threshold=threshold, silence_sec=silence_sec)

    def _load(self) -> None:
        models.require_language(self.language.code)
        started = time.monotonic()
        self.recognizer.load()
        self.synthesizer.load()
        print(f"Models loaded in {time.monotonic() - started:.1f}s "
              f"(Vosk: {self.language.vosk_model}, Piper: {self.language.piper_voice})")

    def _calibrate(self, mic) -> None:
        if self.endpointer.threshold is not None:
            print(f"Threshold: {self.endpointer.threshold:.0f} (fixed)")
            return
        print("Measuring background noise - please stay quiet ...", end="", flush=True)
        mic.drain()  # ignore the transient right after the stream opens
        chunks = [mic.read() for _ in range(CALIBRATION_BLOCKS)]
        threshold = self.endpointer.calibrate(chunks)
        print(f" done. Threshold: {threshold:.0f}")

    def _reply_parts(self, text: str):
        """What to say back, in pieces that are synthesized and played one
        after the other. This is the substitution point: echo says the text
        itself, ChatApp streams an LLM's answer sentence by sentence so
        playback can start before the answer is finished."""
        yield text

    def _handle_utterance(self, mic, text: str) -> None:
        print(f"  heard   : {text}")

        # The robot must not hear itself: muting around the whole reply is
        # what stops the echo app from echoing its own echo forever.
        mic.mute()
        try:
            for part in self._reply_parts(text):
                started = time.monotonic()
                wav_bytes = self.synthesizer.synthesize(part)
                synth_sec = time.monotonic() - started
                print(f"  speaking ... ({synth_sec:.2f}s synthesis)")
                audio.play_wav(wav_bytes, device=self.output_device)
        finally:
            mic.unmute()
            self.recognizer.reset()

    def run(self) -> int:
        self._load()

        with audio.Microphone(device=self.input_device) as mic:
            self._calibrate(mic)
            print(f"\nReady - speak {self.language.label}. "
                  f"A pause of {self.endpointer.silence_sec:.1f}s ends your turn. "
                  f"Stop with Ctrl+C.\n")

            pre_roll = collections.deque(maxlen=PRE_ROLL_BLOCKS)
            last_partial = ""

            for chunk in mic.chunks():
                event = self.endpointer.update(chunk)

                if event is Event.IDLE:
                    pre_roll.append(chunk)
                    continue

                if event is Event.SPEECH_START:
                    print("  listening ...")
                    for buffered in pre_roll:
                        self.recognizer.accept(buffered)
                    pre_roll.clear()

                self.recognizer.accept(chunk)

                if event is Event.SPEECH and self.show_partial:
                    partial = self.recognizer.partial_text()
                    if partial and partial != last_partial:
                        print(f"  ... {partial}", end="\r", flush=True)
                        last_partial = partial

                if event is Event.TOO_SHORT:
                    self.recognizer.reset()
                    last_partial = ""
                    continue

                if event is Event.ENDPOINT:
                    text = self.recognizer.final_text()
                    last_partial = ""
                    if text:
                        self._handle_utterance(mic, text)
                    else:
                        print("  (nothing understood)")
                    print()

        return 0


def main(**kwargs) -> int:
    try:
        return EchoApp(**kwargs).run()
    except KeyboardInterrupt:
        print("\nStopped.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
