"""Command line entry point: python -m voice_interaction <command>"""

import argparse
import logging
import sys
import time
from pathlib import Path

from . import audio
from . import config as cfg
from . import models


def cmd_devices(args) -> int:
    print(audio.list_devices())
    print("\nPick one with --input-device / --output-device (index or name substring),")
    print("or via the VI_INPUT_DEVICE / VI_OUTPUT_DEVICE environment variables.")
    return 0


def cmd_download(args) -> int:
    codes = sorted(cfg.LANGUAGES) if args.lang == "all" else [args.lang]
    if args.voice:
        models.download_piper_voice(args.voice, force=args.force)
        return 0
    for code in codes:
        models.ensure_language(code, force=args.force)
    print(f"\nModels in {cfg.MODELS_DIR}")
    return 0


def cmd_speak(args) -> int:
    from .tts import Synthesizer

    synth = Synthesizer(lang=args.lang, voice=args.voice)
    synth.load()
    started = time.monotonic()
    wav_bytes = synth.synthesize(args.text)
    elapsed = time.monotonic() - started

    if args.out:
        Path(args.out).write_bytes(wav_bytes)
        print(f"wrote {args.out} ({len(wav_bytes)} bytes, {synth.sample_rate} Hz, {elapsed:.2f}s)")
    else:
        print(f"synthesized in {elapsed:.2f}s - playing ...")
        audio.play_wav(wav_bytes, device=args.output_device)
    return 0


def cmd_listen(args) -> int:
    """Speech to text only — prints transcripts, speaks nothing."""
    from .stt import SpeechRecognizer
    from .vad import Endpointer, Event
    from .echo import CALIBRATION_BLOCKS, PRE_ROLL_BLOCKS
    import collections

    lang = cfg.get_language(args.lang)
    models.require_language(lang.code)
    recognizer = SpeechRecognizer(lang=lang.code)
    recognizer.load()
    endpointer = Endpointer(threshold=args.threshold, silence_sec=args.silence_sec)

    with audio.Microphone(device=args.input_device) as mic:
        if endpointer.threshold is None:
            print("Calibrating ...", end="", flush=True)
            mic.drain()
            print(f" threshold: {endpointer.calibrate([mic.read() for _ in range(CALIBRATION_BLOCKS)]):.0f}")
        print(f"Ready ({lang.label}). Stop with Ctrl+C.\n")

        pre_roll = collections.deque(maxlen=PRE_ROLL_BLOCKS)
        try:
            for chunk in mic.chunks():
                event = endpointer.update(chunk)
                if event is Event.IDLE:
                    pre_roll.append(chunk)
                    continue
                if event is Event.SPEECH_START:
                    for buffered in pre_roll:
                        recognizer.accept(buffered)
                    pre_roll.clear()
                recognizer.accept(chunk)
                if event is Event.TOO_SHORT:
                    recognizer.reset()
                elif event is Event.ENDPOINT:
                    text = recognizer.final_text()
                    print(f"> {text}" if text else "> (nothing understood)")
        except KeyboardInterrupt:
            print("\nStopped.")
    return 0


def cmd_meter(args) -> int:
    """Live level meter with the endpointer's decision, for setting the
    threshold on a machine whose noise floor you cannot guess (a robot with
    fans, an open-plan room) and for answering 'why did it not react?'."""
    from .vad import Endpointer, Event
    from .echo import CALIBRATION_BLOCKS

    endpointer = Endpointer(threshold=args.threshold, silence_sec=args.silence_sec)

    with audio.Microphone(device=args.input_device) as mic:
        if endpointer.threshold is None:
            print("Calibrating ...", end="", flush=True)
            mic.drain()
            endpointer.calibrate([mic.read() for _ in range(CALIBRATION_BLOCKS)])
            print(f" threshold: {endpointer.threshold:.0f}")
        else:
            print(f"Threshold: {endpointer.threshold:.0f} (fixed)")
        print("Speak — the bar shows the level, '|' marks the threshold. Ctrl+C to stop.\n")

        width = 50
        scale = endpointer.threshold * 2.5
        try:
            for chunk in mic.chunks():
                event = endpointer.update(chunk)
                level = endpointer.last_level
                filled = min(width, int(width * level / scale))
                mark = min(width - 1, int(width * endpointer.threshold / scale))
                bar = "".join("|" if i == mark else ("#" if i < filled else " ")
                              for i in range(width))
                note = {
                    Event.SPEECH_START: "speech starts",
                    Event.ENDPOINT: "ENDPOINT -> would transcribe now",
                    Event.TOO_SHORT: "discarded (too short)",
                }.get(event, "")
                end = "\n" if note else "\r"
                print(f"  [{bar}] {level:6.0f}  {note}", end=end, flush=True)
        except KeyboardInterrupt:
            print("\nStopped.")
    return 0


def cmd_echo(args) -> int:
    from .echo import main as echo_main

    return echo_main(
        lang=args.lang,
        input_device=args.input_device,
        output_device=args.output_device,
        threshold=args.threshold,
        silence_sec=args.silence_sec,
        show_partial=not args.no_partial,
    )


def cmd_selftest(args) -> int:
    """Round trip without any audio hardware: synthesize a sentence, feed the
    samples straight into the recognizer, compare. Verifies an install on a
    headless Jetson or a fresh Windows box before a mic is plugged in."""
    import numpy as np
    from .stt import SpeechRecognizer
    from .tts import Synthesizer

    lang = cfg.get_language(args.lang)
    models.require_language(lang.code)
    phrase = args.text or ("Das ist ein Test der Sprachausgabe." if lang.code == "de"
                           else "This is a test of the speech pipeline.")

    print(f"language  : {lang.label}")
    print(f"phrase    : {phrase}")

    synth = Synthesizer(lang=lang.code)
    synth.load()
    started = time.monotonic()
    wav_bytes = synth.synthesize(phrase)
    synth_sec = time.monotonic() - started

    samples, sample_rate = audio.wav_to_array(wav_bytes)
    duration = len(samples) / sample_rate
    print(f"synthesis : {synth_sec:.2f}s for {duration:.2f}s of audio "
          f"(RTF {synth_sec / duration:.2f}) @ {sample_rate} Hz")

    # Piper renders at 22.05 kHz, Vosk's small models want 16 kHz: resample by
    # linear interpolation, which is plenty for a plausibility check.
    target_rate = cfg.SAMPLE_RATE
    if sample_rate != target_rate:
        new_length = int(len(samples) * target_rate / sample_rate)
        resampled = np.interp(
            np.linspace(0, len(samples) - 1, new_length),
            np.arange(len(samples)),
            samples.astype(np.float32),
        ).astype(np.int16)
    else:
        resampled = samples

    recognizer = SpeechRecognizer(lang=lang.code, sample_rate=target_rate)
    recognizer.load()
    started = time.monotonic()
    chunk_bytes = cfg.BLOCK_SIZE * 2
    raw = resampled.tobytes()
    for offset in range(0, len(raw), chunk_bytes):
        recognizer.accept(raw[offset:offset + chunk_bytes])
    text = recognizer.final_text()
    stt_sec = time.monotonic() - started
    print(f"recognition: {stt_sec:.2f}s (RTF {stt_sec / duration:.2f})")
    print(f"result    : {text!r}")

    if not text:
        print("\nFAILED: nothing recognized.")
        return 1
    print("\nOK - synthesis and recognition work together.")
    print("Note: this is synthetic speech; accuracy with a real microphone will differ.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voice_interaction",
        description="Offline speech interaction (Vosk STT + Piper TTS) for Jetson Nano and desktops.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_lang(p):
        p.add_argument("--lang", default=cfg.DEFAULT_LANG, choices=sorted(cfg.LANGUAGES),
                       help="language preset (default: %(default)s)")

    def add_listen_opts(p):
        p.add_argument("--input-device", default=None, help="mic index or name substring")
        p.add_argument("--threshold", type=float, default=None,
                       help="RMS speech threshold; default is measured at startup")
        p.add_argument("--silence-sec", type=float, default=None,
                       help=f"pause that ends an utterance (default: {cfg.SILENCE_SEC})")

    p = sub.add_parser("devices", help="list audio input/output devices")
    p.set_defaults(func=cmd_devices)

    p = sub.add_parser("download", help="download STT model and TTS voice")
    p.add_argument("--lang", default=cfg.DEFAULT_LANG,
                   choices=sorted(cfg.LANGUAGES) + ["all"], help="(default: %(default)s)")
    p.add_argument("--voice", default=None, help="download only this Piper voice, e.g. de_DE-thorsten-low")
    p.add_argument("--force", action="store_true", help="re-download even if present")
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("echo", help="example app: speak, hear it back after a pause")
    add_lang(p)
    add_listen_opts(p)
    p.add_argument("--output-device", default=None, help="speaker index or name substring")
    p.add_argument("--no-partial", action="store_true", help="do not print live partial transcripts")
    p.set_defaults(func=cmd_echo)

    p = sub.add_parser("listen", help="speech to text only")
    add_lang(p)
    add_listen_opts(p)
    p.set_defaults(func=cmd_listen)

    p = sub.add_parser("meter", help="live input level meter, for setting --threshold")
    add_listen_opts(p)
    p.set_defaults(func=cmd_meter)

    p = sub.add_parser("speak", help="text to speech")
    p.add_argument("text")
    add_lang(p)
    p.add_argument("--voice", default=None, help="override the preset's Piper voice")
    p.add_argument("--out", default=None, help="write a WAV file instead of playing it")
    p.add_argument("--output-device", default=None, help="speaker index or name substring")
    p.set_defaults(func=cmd_speak)

    p = sub.add_parser("selftest", help="TTS -> STT round trip, no audio hardware needed")
    add_lang(p)
    p.add_argument("--text", default=None, help="phrase to test with")
    p.set_defaults(func=cmd_selftest)

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
