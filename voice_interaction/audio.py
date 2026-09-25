"""Cross-platform microphone capture and playback via sounddevice/PortAudio.

The same code path works on Linux (ALSA), the Jetson's Ubuntu image and
Windows (WASAPI/MME) — only the device selection differs, which is why
devices can be named by substring rather than by index."""

import io
import queue
import wave
from typing import Iterator, Optional, Union

import numpy as np
import sounddevice as sd

from . import config as cfg


def list_devices() -> str:
    return str(sd.query_devices())


def resolve_device(device: Union[int, str, None], want_input: bool) -> Optional[int]:
    """Accept a device index, a case-insensitive name substring, or None.

    Name matching matters in practice: USB audio device indices shuffle
    across reboots and between Linux and Windows, names mostly don't."""
    if device is None or device == "":
        return None
    if isinstance(device, int) or str(device).isdigit():
        return int(device)

    needle = str(device).lower()
    for index, info in enumerate(sd.query_devices()):
        channels = info["max_input_channels"] if want_input else info["max_output_channels"]
        if channels > 0 and needle in info["name"].lower():
            return index
    raise ValueError(f"No {'input' if want_input else 'output'} device matching '{device}'")


class Microphone:
    """Streams mono 16-bit PCM chunks from the default or selected mic.

    Supports muting: while the robot is speaking we drop captured audio so
    it does not transcribe its own voice and answer itself."""

    def __init__(self, device=None, sample_rate=None, block_size=None):
        self.device = resolve_device(device if device is not None else cfg.INPUT_DEVICE, want_input=True)
        self.sample_rate = sample_rate or cfg.SAMPLE_RATE
        self.block_size = block_size or cfg.BLOCK_SIZE
        self._queue: queue.Queue = queue.Queue()
        self._stream: Optional[sd.RawInputStream] = None
        self._muted = False

    def _callback(self, indata, frames, time_info, status):
        if not self._muted:
            self._queue.put(bytes(indata))

    def __enter__(self):
        self._stream = sd.RawInputStream(
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            device=self.device,
            dtype="int16",
            channels=1,
            callback=self._callback,
        )
        self._stream.start()
        return self

    def __exit__(self, *exc):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def mute(self) -> None:
        self._muted = True

    def unmute(self) -> None:
        """Unmuting drops whatever queued up meanwhile, so audio recorded
        during playback never reaches the recognizer."""
        self.drain()
        self._muted = False

    def drain(self) -> None:
        with self._queue.mutex:
            self._queue.queue.clear()

    def read(self) -> bytes:
        """Block until the next chunk of captured audio is available."""
        return self._queue.get()

    def chunks(self) -> Iterator[bytes]:
        while True:
            yield self.read()


def wav_to_array(wav_bytes: bytes):
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())
    return np.frombuffer(frames, dtype=np.int16), sample_rate


def play_wav(wav_bytes: bytes, device=None) -> None:
    """Blocking playback — callers that must not transcribe their own output
    mute the microphone around this call."""
    samples, sample_rate = wav_to_array(wav_bytes)
    out_device = resolve_device(device if device is not None else cfg.OUTPUT_DEVICE, want_input=False)
    sd.play(samples, samplerate=sample_rate, device=out_device)
    sd.wait()


def rms(chunk: bytes) -> float:
    """Root-mean-square level of an int16 chunk, on the 0..32767 scale."""
    samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples * samples)))
