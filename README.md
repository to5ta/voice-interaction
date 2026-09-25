# voice-interaction

Offline speech interaction for a robot: **speech → text → speech**, in German and English,
running entirely on the CPU. Built for a **Jetson Nano P3450** that also has to run other
workloads, and developed/tested on ordinary Linux and Windows machines.

Nothing leaves the device — no cloud, no API keys, no network at runtime.

```
  microphone ──► endpointer ──► Vosk (STT) ──► your logic ──► Piper (TTS) ──► speaker
   16 kHz        "pause of        German/         (here:         German/
   mono          0.8 s ends      English          echo it        English
                 the turn"       offline         back)          offline
```

| Stage | Engine | License | Why |
|---|---|---|---|
| Speech → text | [Vosk](https://alphacephei.com/vosk/) small models | Apache-2.0 | Streaming, offline, 45 MB models, runs on a Pi-class CPU |
| Text → speech | [Piper](https://github.com/OHF-voice/piper1-gpl) | GPL-3.0-or-later | Neural quality, ONNX-on-CPU, no GPU and no PyTorch |
| Audio I/O | [sounddevice](https://python-sounddevice.readthedocs.io/)/PortAudio | MIT | One API for ALSA, Windows and the Jetson image |

Both engines run on the **CPU only**, deliberately: the Nano's Maxwell GPU stays free for
vision work, and there is no CUDA/TensorRT version matching to maintain.

---

## Hardware

**The Jetson Nano P3450 has no analog audio input.** Its only onboard audio path is HDMI out.
You need one of:

- a **USB headset** (simplest — mic and speaker in one device, and the headset physically
  prevents the robot from hearing its own voice),
- a **USB microphone** plus HDMI/USB speakers,
- a USB audio interface, or an I2S MEMS mic on the 40-pin header (extra driver work).

Also worth knowing:

- **4 GB Nano recommended.** Both models resident need ~460 MB (measured, see below). On a 2 GB
  Nano run headless (no desktop) and add swap.
- A **USB 2.0 port** is fine; audio needs almost no bandwidth.
- Mic placement matters far more than model size for recognition accuracy. A cheap mic 30 cm from
  the speaker beats a good mic 3 m away.

---

## Quick start

### Linux and Jetson Nano

```bash
./scripts/install_jetson.sh all      # de + en; use "de" or "en" for just one
```

The script checks the Python version, installs PortAudio, creates the venv, and downloads the
models (~90 MB per language). Then:

```bash
source .venv/bin/activate
python -m voice_interaction selftest     # works without any microphone
python -m voice_interaction devices      # find your mic/speaker
python -m voice_interaction echo         # the example application
```

### Windows

Windows needs no extra system libraries — the `sounddevice` wheel bundles PortAudio.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m voice_interaction download --lang all
python -m voice_interaction selftest
python -m voice_interaction echo
```

Use Python 3.9–3.12. If `echo` picks the wrong device, list them with
`python -m voice_interaction devices` and pass `--input-device "Headset"` — name matching is
more stable than indices, which shuffle between reboots.

---

## The example application: `echo`

The user speaks; once a pause of `--silence-sec` (default 0.8 s) has elapsed, whatever was
understood is **spoken back**. That is the whole app — it exists to prove every link in the chain
works, and it is the smoke test to run first on new hardware.

```
$ python -m voice_interaction echo --lang de
Models loaded in 1.7s (Vosk: vosk-model-small-de-0.15, Piper: de_DE-thorsten-medium)
Measuring background noise - please stay quiet ... done. Threshold: 150

Ready - speak Deutsch. A pause of 0.8s ends your turn. Stop with Ctrl+C.

  listening ...
  heard   : hallo roboter wie geht es dir
  speaking ... (0.21s synthesis)
```

What happens under the hood, and why:

- **Noise calibration at startup.** The speech threshold is measured from the actual room rather
  than hard-coded, because a robot's own fans and motors set the noise floor. Measured: a quiet
  room gives a threshold of ~150, an audible fan ~900, and both endpoint correctly.
- **Pre-roll buffer.** The 320 ms *before* the threshold was crossed are fed to the recognizer
  too — otherwise the first syllable, which is what crossed the threshold, gets clipped.
- **Streaming recognition.** Audio goes into Vosk as it arrives, so when the pause is detected
  the transcript is essentially already decoded. That is what keeps the response near real time.
- **The microphone is muted during playback.** Without this the robot transcribes its own voice
  and answers itself forever. (A headset makes this robust; open speakers rely on the muting.)
- **Short blips are discarded.** Sounds under `--min-speech-sec` (0.3 s) — a door, a cough — never
  reach the recognizer.

---

## Commands

```bash
python -m voice_interaction <command> [options]
```

| Command | What it does |
|---|---|
| `echo` | The example app: listen, wait for the pause, speak it back |
| `listen` | Speech to text only — prints transcripts, says nothing |
| `speak "text"` | Text to speech; `--out file.wav` writes instead of playing |
| `devices` | List audio inputs/outputs with their indices and names |
| `meter` | Live input level meter with the endpointer's verdict — for setting `--threshold` |
| `download --lang de\|en\|all` | Fetch the Vosk model and Piper voice for a language |
| `selftest` | TTS → STT round trip **without audio hardware** — verifies an install |

`selftest` is the one to run first on a fresh machine: it synthesizes a sentence, feeds the
samples straight back into the recognizer, and reports timings. No mic, no speaker, no desktop
session needed, so it also works over SSH on a headless Nano.

Useful options: `--lang de|en`, `--input-device`/`--output-device` (index or name substring),
`--silence-sec`, `--threshold`.

---

## Configuration

Every setting is an environment variable, so a deployment can be tuned without touching code.
CLI flags override them.

| Variable | Default | Meaning |
|---|---|---|
| `VI_LANG` | `de` | Language preset (`de`, `en`) |
| `VI_MODELS_DIR` | `models` | Where models live |
| `VI_INPUT_DEVICE` | system default | Mic index or name substring |
| `VI_OUTPUT_DEVICE` | system default | Speaker index or name substring |
| `VI_SILENCE_SEC` | `0.8` | Pause that ends an utterance |
| `VI_SILENCE_THRESHOLD` | measured at startup | RMS speech threshold (0–32767) |
| `VI_MIN_SPEECH_SEC` | `0.3` | Shorter sounds are discarded as noise |
| `VI_MAX_UTTERANCE_SEC` | `15.0` | Hard cap so a stuck mic can't record forever |
| `VI_TTS_THREADS` | `2` | ONNX Runtime threads for Piper |

### Tuning the pause

`VI_SILENCE_SEC` is the single knob that decides how the system feels:

- **0.5–0.6 s** — snappy, but cuts people off mid-sentence when they pause to think.
- **0.8 s** (default) — a good compromise for command-style interaction.
- **1.2–1.5 s** — for longer, more thoughtful utterances; noticeably laggier.

If it triggers on background noise, raise the threshold (`--threshold 600`); if quiet speech is
missed, lower it. `meter` shows the live level against the threshold and prints when it would
endpoint — the fastest way to set this on a robot whose fans you cannot silence:

```
$ python -m voice_interaction meter
Calibrating ... threshold: 150
  [########            |                             ]    412
  [####################|####                         ]   1180  speech starts
  [                    |                             ]      3  ENDPOINT -> would transcribe now
```

---

## Resource budget

Measured on an x86 development machine, both languages' engines in one process:

| | RSS |
|---|---|
| Python + numpy + onnxruntime + vosk imports | 103 MB |
| `+` Vosk small German model | 282 MB |
| `+` Piper `de_DE-thorsten-medium` | 392 MB |
| Steady state after synthesizing and recognizing | **462 MB** |

Speed on the same machine: model load 1.7 s once at startup; synthesis RTF 0.06; recognition
RTF 0.14 (de) / 0.29 (en) — i.e. far faster than real time, with headroom to spare.

Per-block timing matters more than the averages, because recognition has to keep up with the
microphone in real time. Each 64 ms block of audio must be decoded in under 64 ms; measured on
the same desktop, `accept()` takes **6.8 ms on average but 36 ms at p95** (the spikes are Kaldi's
lattice work). Calling `PartialResult()` every block — which is what draws the live transcript —
adds only ~0.5 ms, so live partials are effectively free.

**On the Nano expect roughly 4–6× those CPU times** (Cortex-A57 @ 1.43 GHz vs a desktop core).
That puts the average around 55 % of the real-time budget, with p95 spikes exceeding one block —
absorbed by the input queue during pauses, but it means recognition on a Nano runs with far less
margin than these desktop averages suggest, and English (RTF 0.29 here vs 0.14 for German) is the
tighter of the two. These Nano figures are extrapolated, **not yet measured on real P3450
hardware** — run `selftest` on the device for the real numbers.

To claw back headroom:

- `VI_TTS_THREADS=2` (default) keeps Piper from taking all 4 cores. Raise to 3 only if nothing
  else runs on the device.
- Use a `-low` Piper voice instead of `-medium` (`--voice de_DE-thorsten-low`): ~30 MB less RAM
  and noticeably faster, at some quality cost.
- Run one language at a time — loading both doubles the model memory.
- `sudo nvpmodel -m 0 && sudo jetson_clocks` puts the Nano in its 10 W all-core mode. Worth
  ~30–40 % latency on this workload; make sure the cooling can take it.

---

## Languages and models

| | German | English |
|---|---|---|
| Vosk model | `vosk-model-small-de-0.15` (45 MB, WER 13.75) | `vosk-model-small-en-us-0.15` (40 MB, WER 9.85) |
| Piper voice | `de_DE-thorsten-medium` | `en_US-lessac-medium` |

Presets live in [`voice_interaction/config.py`](voice_interaction/config.py) — adding a language
means adding one entry there with a Vosk model name and a Piper voice name.

Alternative German voices (`--voice`, or change the preset): `de_DE-thorsten-low` (faster),
`de_DE-eva_k-x_low` (21 MB, robotic), `de_DE-thorsten-high` (114 MB, slower),
`de_DE-kerstin-low`, `de_DE-ramona-low`, `de_DE-karlsson-low`, `de_DE-pavoque-low`.

Bigger Vosk models (`vosk-model-de-0.21`, 1.9 GB) are considerably more accurate but do not fit
a Nano's memory budget alongside everything else.

**Both languages at once is not supported in one process.** Vosk needs a model per language, and
a running recognizer is bound to one. For a bilingual robot, either switch language on a command
(reload the recognizer, ~0.5 s) or run one process per language if RAM allows.

---

## Testing

```bash
pip install pytest
pytest
```

Nine tests, none of which need a microphone or speaker:

- `tests/test_vad.py` — pause detection: endpoint timing, threshold adaptation from the noise
  floor, rejection of short blips, the maximum-utterance cap. No models needed, runs anywhere.
- `tests/test_echo_pipeline.py` — the whole echo loop against a simulated microphone: synthesized
  speech goes in, the transcript and a spoken response come out, and the mic is verified to be
  muted during playback. Skips automatically if the models are not downloaded.

Plus `python -m voice_interaction selftest` as the on-device install check.

---

## Licenses

This matters if the robot ever ships as a product:

| Component | License | Implication |
|---|---|---|
| Vosk + its small models | Apache-2.0 | Permissive, no copyleft |
| Piper (`piper-tts` ≥ 1.3) | **GPL-3.0-or-later** | Copyleft — affects distribution of a combined work |
| Piper voices | MIT / CC0 / CC-BY (per voice) | Check the voice card; `thorsten` and `lessac` are permissive |
| sounddevice | MIT | Permissive |

`piper-tts` was MIT up to version 1.2.0 (the original Rhasspy project) and became GPL-3.0 with
the 1.3 rewrite under new maintainership. Version 1.2.0 is still installable and permissive, but
unmaintained, and its `piper-phonemize` dependency only has wheels for Python 3.9–3.11. If
copyleft is a problem for you, that pin is the escape hatch — otherwise stay on the current
release. For personal, internal or research robots, GPL-3.0 imposes no practical restriction.

---

## Troubleshooting

**`selftest` passes but `echo` hears nothing.** Wrong input device. Run `devices`, then
`echo --input-device "<name fragment>"`. On Linux check the mic isn't muted in `alsamixer`.

**It answers itself / loops forever.** Speaker audio is reaching the mic. Use a headset, lower
the volume, or increase the distance. The mic is already muted during playback; what leaks is
room reverb after playback ends.

**It triggers on fan noise.** Raise `--threshold`; use `meter` to pick the value. Startup
calibration assumes the room is quiet during that first second — if the robot was moving then,
the measurement is off; restart or set the threshold explicitly.

**It printed a partial transcript and then seemed to stop.** That is the app waiting out the
pause — the partial line is rewritten in place, so it looks frozen. Give it
`--silence-sec` plus the synthesis time (well under a second for a short sentence) before
concluding anything is wrong. `meter` shows exactly when the endpoint fires.

**It cuts me off mid-sentence.** Raise `--silence-sec` to 1.2.

**Recognition is poor.** Get the mic closer before reaching for a bigger model. Vosk's small
models are trained on clean wideband speech and degrade quickly with distance and reverb. Check
you are on the right language — German speech through the English model produces confident
nonsense.

**`OSError: PortAudio library not found` (Linux).** `sudo apt install libportaudio2`.

**`piper-tts` won't install on the Nano.** JetPack 4.6 ships Python 3.6; piper needs ≥ 3.9. See
the deadsnakes instructions the install script prints.

---

## Where to take it next

`echo.py` is deliberately the thinnest possible app so the substitution point is obvious: in
`EchoApp._handle_utterance`, the text goes straight to the synthesizer. Replace that one call
with intent parsing, a state machine, a local LLM, or ROS message publishing, and the rest of
the stack — capture, endpointing, recognition, playback, muting — stays as it is.

Likely next steps for a robot:

- **Wake word** so it doesn't react to every noise ([openWakeWord](https://github.com/dscripka/openWakeWord)
  is Apache-2.0 and runs on the same CPU budget).
- **Barge-in**: let the user interrupt playback instead of muting the mic for its duration.
- **Language switching** on a spoken command, reloading the recognizer.
- **A long-running service** with the recognizer kept warm, talking to the rest of the robot over
  ROS topics, MQTT or a socket, instead of a foreground CLI.
