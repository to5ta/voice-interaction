# voice-interaction

Offline speech interaction for a robot: **speech → text → speech**, in German and English,
running entirely on the CPU — optionally with a **small local LLM in the middle**, so the robot
answers instead of repeating. Built for a **Jetson Nano P3450** that also has to run other
workloads, and developed/tested on ordinary Linux and Windows machines.

Nothing leaves the device — no cloud, no API keys, no network at runtime.

```
  microphone ──► endpointer ──► Vosk (STT) ──► your logic ──► Piper (TTS) ──► speaker
   16 kHz        "pause of        German/       echo: say it     German/
   mono          0.8 s ends      English        straight back    English
                 the turn"       offline        chat: a 0.5B     offline
                                                LLM answers
```

| Stage | Engine | License | Why |
|---|---|---|---|
| Speech → text | [Vosk](https://alphacephei.com/vosk/) small models | Apache-2.0 | Streaming, offline, 45 MB models, runs on a Pi-class CPU |
| Text → speech | [Piper](https://github.com/OHF-voice/piper1-gpl) | GPL-3.0-or-later | Neural quality, ONNX-on-CPU, no GPU and no PyTorch |
| Audio I/O | [sounddevice](https://python-sounddevice.readthedocs.io/)/PortAudio | MIT | One API for ALSA, Windows and the Jetson image |
| Answering (optional) | [llama.cpp](https://github.com/ggml-org/llama.cpp) + Qwen2.5-0.5B | MIT / Apache-2.0 | 470 MB at 4 bits, prebuilt CPU wheels incl. aarch64 |

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

- **4 GB Nano recommended.** One language needs ~400 MB resident, and the `chat` model another
  ~500 MB on top (measured, see below). On a 2 GB Nano run headless (no desktop) and add swap —
  and expect `chat` to be tight.
- A **USB 2.0 port** is fine; audio needs almost no bandwidth.
- Mic placement matters far more than model size for recognition accuracy. A cheap mic 30 cm from
  the speaker beats a good mic 3 m away.

---

## Quick start

### Linux and Jetson Nano

```bash
./scripts/install_jetson.sh all         # de + en; use "de" or "en" for just one
./scripts/install_jetson.sh de --llm    # ... and the local LLM for `chat` (~470 MB more)
```

The script checks the Python version, installs PortAudio, creates the venv, and downloads the
models (~90 MB per language). Then:

```bash
source .venv/bin/activate
python -m voice_interaction selftest     # works without any microphone
python -m voice_interaction devices      # find your mic/speaker
python -m voice_interaction echo         # say something, hear it back
python -m voice_interaction chat         # say something, get an answer (needs --llm)
```

### Windows

Windows needs no extra system libraries — the `sounddevice` wheel bundles PortAudio. What it
does need is a **real Python installation**: a fresh Windows 11 has none, and the `python` /
`python3` you find on `PATH` are Microsoft Store stubs that only print *"Python was not found"*
(and `py` may not exist at all). Install one first, either way:

```powershell
winget install Python.Python.3.11        # gives you py, python and pip
```

```powershell
winget install astral-sh.uv              # alternative: uv brings its own Python
uv python install 3.11
```

Then, from the project directory:

```powershell
py -3.11 -m venv .venv                   # with uv: uv venv --seed --python 3.11 .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m voice_interaction download --lang all
python -m voice_interaction selftest
python -m voice_interaction echo
```

For the `chat` mode, add the LLM — a prebuilt CPU wheel, no compiler involved:

```powershell
pip install -r requirements-llm.txt
python -m voice_interaction download --llm
python -m voice_interaction ask "hallo roboter wer bist du"
python -m voice_interaction chat
```

Tested on Windows 11 with Python 3.11; use Python 3.9–3.12. Everything is pure wheels, no
compiler and no admin rights beyond installing Python itself.

- **Activation is the usual stumbling block.** In PowerShell the script is `Activate.ps1` and the
  leading `.\` is required; if it is refused, see Troubleshooting. In `cmd.exe` use
  `.venv\Scripts\activate.bat`. You can also skip activation entirely and call
  `.venv\Scripts\python.exe -m voice_interaction ...` directly — that is what the commands above
  do under the hood.
- If `echo` picks the wrong device, list them with `python -m voice_interaction devices` and pass
  `--input-device "Headset"` — name matching is more stable than indices, which shuffle between
  reboots. Windows lists each device once per host API (MME, DirectSound, WASAPI, WDM-KS); MME
  truncates names to 31 characters, so match on the first few words.

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

## The conversation mode: `chat`

Same loop as `echo`, with a small LLM where echo hands the text straight back to the
synthesizer. Still offline: the model is a 470 MB file on disk, run by llama.cpp on the CPU.

```
$ python -m voice_interaction chat --lang de
Models loaded in 1.7s (Vosk: vosk-model-small-de-0.15, Piper: de_DE-thorsten-medium)
LLM loaded in 0.7s (qwen2.5-0.5b, 2 threads, max 80 tokens)
Measuring background noise - please stay quiet ... done. Threshold: 150

Ready - speak Deutsch. A pause of 0.8s ends your turn. Stop with Ctrl+C.

  listening ...
  heard   : hallo roboter wer bist du
  thinking: 0.9s to the first sentence
  answer  : Ich bin ein kleines Programm, das von Menschen inspiriert wird.
  speaking ... (0.23s synthesis)
```

What is different from `echo`, and why:

- **The answer is spoken sentence by sentence.** As soon as the model has finished a sentence it
  goes to Piper, while the next one is still being generated in a background thread. Only the
  *first* sentence is ever really waited for — which is what makes 4 tokens/s on a Nano feel
  acceptable instead of glacial.
- **Answers are capped at two sentences** (`VI_LLM_MAX_TOKENS`, 80 tokens) by the system prompt
  *and* by a hard limit. Every token is CPU time on the robot and waiting time for the person in
  front of it; a chatty model is a worse robot.
- **Markdown and emoji are stripped** before synthesis. Small instruct models produce bullet
  points and smileys however firmly the prompt forbids it, and Piper would read the asterisks out.
- **It remembers the last three exchanges** (`VI_LLM_HISTORY_TURNS`), which is enough for "and
  how about tomorrow?" without the prompt — and the CPU cost of processing it — growing all day.
  `VI_LLM_HISTORY_TURNS=0` makes every turn independent.
- **The microphone stays muted for the whole answer**, not per sentence, so the robot cannot hear
  the end of its own reply.

Give the robot a job instead of a personality with `--system`:

```bash
python -m voice_interaction chat --system "Du bist ein Lagerroboter. Antworte in einem Satz."
```

And test the model without a microphone at all — the way to check it over SSH:

```bash
python -m voice_interaction ask "wie viele beine hat eine spinne"
python -m voice_interaction ask --speak "erzaehl mir etwas ueber roboter"
python -m voice_interaction selftest --llm      # speech in, model, speech out, no hardware
```

### What a 0.5B model can and cannot do

Be clear-eyed about this: `qwen2.5-0.5b` is roughly a thousandth the size of a hosted assistant.
It produces fluent, grammatical German and English and follows simple instructions. It also gets
arithmetic wrong, invents facts confidently, and has no idea what day it is:

```
$ python -m voice_interaction ask "wie viele beine hat eine spinne"
Ein Spinne hat zwei Beine.
```

It is a **language** model in the literal sense — good for a robot that should answer
conversationally, useless as a knowledge source.

That is a fine fit for a robot that mostly needs to acknowledge, confirm and chat; it is the wrong
tool for anything that must be correct. For those, parse intents from the transcript in
`ChatApp._reply_parts` before (or instead of) asking the model — the transcript is right there.

---

## Commands

```bash
python -m voice_interaction <command> [options]
```

| Command | What it does |
|---|---|
| `echo` | The example app: listen, wait for the pause, speak it back |
| `chat` | Same, but a local LLM answers instead of echoing |
| `ask "text"` | One question to the LLM without a microphone; `--speak` says it too |
| `listen` | Speech to text only — prints transcripts, says nothing |
| `speak "text"` | Text to speech; `--out file.wav` writes instead of playing |
| `devices` | List audio inputs/outputs with their indices and names |
| `meter` | Live input level meter with the endpointer's verdict — for setting `--threshold` |
| `download --lang de\|en\|all` | Fetch the Vosk model and Piper voice for a language |
| `download --llm` | Fetch the GGUF model for `chat` (~470 MB) |
| `selftest` | TTS → STT round trip **without audio hardware** — verifies an install |

`selftest` is the one to run first on a fresh machine: it synthesizes a sentence, feeds the
samples straight back into the recognizer, and reports timings. No mic, no speaker, no desktop
session needed, so it also works over SSH on a headless Nano.

Useful options: `--lang de|en`, `--input-device`/`--output-device` (index or name substring),
`--silence-sec`, `--threshold`; for `chat` and `ask` also `--llm-model`, `--max-tokens` and
`--system`.

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
| `VI_LLM_MODEL` | `qwen2.5-0.5b` | Which chat model (see below) |
| `VI_LLM_THREADS` | `2` | llama.cpp threads |
| `VI_LLM_MAX_TOKENS` | `80` | Hard cap on the answer length |
| `VI_LLM_CONTEXT` | `1024` | Context window (system prompt + history + turn) |
| `VI_LLM_HISTORY_TURNS` | `3` | Exchanges the model still sees; `0` = no memory |
| `VI_LLM_TEMPERATURE` | `0.7` | Lower is more repetitive and more predictable |
| `VI_LLM_SYSTEM_PROMPT` | per language | Replaces the built-in persona |

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

### Memory

Measured RSS on an x86 development machine:

| | RSS |
|---|---|
| Python + numpy + onnxruntime + vosk imports | 103 MB |
| `+` German Vosk model and Piper voice, steady state | **404 MB** |
| `+` English models loaded alongside | **618 MB** |
| German models `+` the chat model (instead of English), after a few answers | **~900 MB** |

Memory, not CPU, is the binding constraint on a Nano. One language fits comfortably on a 4 GB
board; both at once is possible but leaves little for anything else. On a 2 GB board, run
headless, use one language, and add swap.

The chat row is a +500 MB delta measured against the German-only figure. The 469 MB GGUF is
memory-mapped, so RSS only climbs to its full size once the whole model has been touched — after
the first answer or two, not at load. One language plus the chat model still leaves ~3 GB on a
4 GB Nano; two languages plus chat does not leave enough to be worth it.

### CPU

Steady-state CPU time per second of audio (warmed up — the first utterance after startup costs
several times more while the decoder allocates):

| Stage | i7-4790K @ 4.4 GHz | Nano estimate (÷6–8) |
|---|---|---|
| Idle, nobody speaking (VAD only) | 0.02 % of a core | ~0.15 % — free |
| Recognizing German | 2.8 % of a core | ~20 % of a core |
| Recognizing English | 5.3 % of a core | ~37 % of a core |
| Synthesizing (2 threads) | 12.2 % of a core | ~85 %, in bursts |

Live partial transcripts are free: `PartialResult()` on every block does not measurably change
the totals.

Answer generation, measured over five short questions with the default 0.5B model:

| | i7-4790K @ 4.4 GHz | Nano estimate (÷6–8) |
|---|---|---|
| Generation, 2 threads | 26 tokens/s | ~3–4 tokens/s |
| To the **first spoken sentence** | 0.8 s mean, 1.3 s max | ~5–9 s |
| Whole answer (18–58 tokens) | 1.1 s mean, 1.7 s max | ~7–13 s |

Only the middle row is really felt, because the rest of the answer is generated
while the first sentence is already being spoken. Throwing cores at it helps less than expected —
1 → 2 threads nearly doubles throughput (14.9 → 28.4 tokens/s), 2 → 4 adds a quarter (35.3):
generation is memory-bandwidth bound, not compute bound. On a Nano, whose bandwidth is far
scarcer than an i7's, taking cores from the rest of the robot is unlikely to repay itself.

What matters for real-time behaviour is the per-block deadline — each 64 ms of audio must be
decoded in under 64 ms:

| | mean | p95 | max |
|---|---|---|---|
| German, per 64 ms block | 1.7 ms | 7.1 ms | 9.5 ms |
| English, per 64 ms block | 2.9 ms | 15.3 ms | 29.6 ms |

Scaled 6–8× for the Nano, German stays well inside the deadline while **English p95 blocks land
at or past it**. Occasional overruns are absorbed by the input queue and caught up during pauses,
so the effect is a little added latency rather than lost audio — but English has materially less
margin than German, because the small English model is roughly twice the decoding work.

These Nano figures are extrapolated from an i7-4790K and are **not measured on real P3450
hardware**. Run `selftest` on the device for the real numbers.

To claw back headroom:

- `VI_TTS_THREADS=2` (default) keeps Piper from taking all 4 cores. Raise to 3 only if nothing
  else runs on the device.
- Use a `-low` Piper voice instead of `-medium` (`--voice de_DE-thorsten-low`): ~30 MB less RAM
  and noticeably faster, at some quality cost.
- Run one language at a time — loading both doubles the model memory.
- Lower `VI_LLM_MAX_TOKENS` (48 is still two sentences) before touching anything else: it is a
  hard bound on the worst case, and the worst case is what makes a robot feel slow.
- For English, `--llm-model smollm2-360m` is smaller and faster than the default — but it does
  not speak German at all.
- **Use the 10 W power mode.** `sudo nvpmodel -m 0 && sudo jetson_clocks` gives all 4 cores at
  1.43 GHz. The 5 W mode (`-m 1`) runs only 2 cores at 918 MHz — roughly half the throughput per
  core and half the cores, which is where English recognition stops keeping up. Check the cooling
  can sustain 10 W.
- The **GPU is entirely unused** by this stack, so a vision pipeline can have it in full.

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

### The chat model

| | `qwen2.5-0.5b` (default) | `qwen2.5-1.5b` | `smollm2-360m` |
|---|---|---|---|
| File (4-bit GGUF) | 469 MB | 1066 MB | 368 MB (8-bit) |
| License | Apache-2.0 | Apache-2.0 | Apache-2.0 |
| German | usable | noticeably better | not really |
| Fits a 4 GB Nano | yes | in theory, but ~3x slower | yes |

Pick one with `--llm-model` or `VI_LLM_MODEL`, fetch it with `download --llm <name>`. Adding
another is one entry in `LLM_MODELS` in [`voice_interaction/config.py`](voice_interaction/config.py):
a HuggingFace repo and a GGUF filename. Anything larger than ~1B parameters stops making sense on
a Nano long before it stops fitting in RAM.

**Both languages at once is not supported in one process.** Vosk needs a model per language, and
a running recognizer is bound to one. For a bilingual robot, either switch language on a command
(reload the recognizer, ~0.5 s) or run one process per language if RAM allows.

---

## Testing

```bash
pip install pytest
pytest
```

Twenty-one tests, none of which need a microphone, a speaker, or llama-cpp-python:

- `tests/test_vad.py` — pause detection: endpoint timing, threshold adaptation from the noise
  floor, rejection of short blips, the maximum-utterance cap. No models needed, runs anywhere.
- `tests/test_echo_pipeline.py` — the whole echo loop against a simulated microphone: synthesized
  speech goes in, the transcript and a spoken response come out, and the mic is verified to be
  muted during playback. Skips automatically if the models are not downloaded.
- `tests/test_chat.py` — where sentences are cut out of a token stream (decimals and "z.B." must
  not end one), what the model is told about earlier turns, that an interrupted answer leaves no
  half-turn in the history, and that a model which throws does not take the robot down with it.
  The LLM is scripted rather than loaded, so this runs in milliseconds.

Plus `python -m voice_interaction selftest` (add `--llm`) as the on-device install check.

---

## Licenses

This matters if the robot ever ships as a product:

| Component | License | Implication |
|---|---|---|
| Vosk + its small models | Apache-2.0 | Permissive, no copyleft |
| Piper (`piper-tts` ≥ 1.3) | **GPL-3.0-or-later** | Copyleft — affects distribution of a combined work |
| Piper voices | MIT / CC0 / CC-BY (per voice) | Check the voice card; `thorsten` and `lessac` are permissive |
| sounddevice | MIT | Permissive |
| llama.cpp / llama-cpp-python | MIT | Permissive |
| Qwen2.5 and SmolLM2 GGUF weights | Apache-2.0 | Permissive; the model card still governs use |

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

**`py` is not recognized, or `python` prints "Python was not found; run without arguments to
install from the Microsoft Store" (Windows).** No Python is installed — `python.exe` and
`python3.exe` on the `PATH` of a fresh Windows are Store stubs, not interpreters. Install one
(`winget install Python.Python.3.11`) and open a new terminal; optionally turn the stubs off
under *Settings > Apps > Advanced app settings > App execution aliases*.

**`Activate.ps1 cannot be loaded because running scripts is disabled on this system`
(Windows).** PowerShell's execution policy. Either
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, or use `cmd.exe` with
`.venv\Scripts\activate.bat`, or skip activation and run `.venv\Scripts\python.exe -m
voice_interaction ...`.

**`No module named pip` inside a venv made with `uv venv` (Windows or Linux).** uv does not put
pip in the venv unless asked. Use `uv venv --seed`, or install with `uv pip install -r
requirements.txt` instead.

**`chat` says llama-cpp-python is not installed.** It is deliberately not in
`requirements.txt`; `pip install -r requirements-llm.txt` adds it. That file also points pip at
the prebuilt CPU wheels — without it, pip falls back to the source tarball, which needs cmake and
a compiler and does not even unpack on Windows (llama.cpp's vendored tree exceeds the 260-character
path limit).

**The answers are confidently wrong.** That is a 0.5B model, not a mistake in the wiring. It is
fluent, not knowledgeable. Use `qwen2.5-1.5b` on a desktop, or handle the questions that must be
answered correctly by parsing the transcript yourself before it reaches the model.

**The robot takes forever to answer on the Nano.** Lower `VI_LLM_MAX_TOKENS`, keep
`VI_LLM_HISTORY_TURNS` small (every remembered turn is re-processed on the next question), and
check you are in the 10 W power mode. Raising `VI_LLM_THREADS` past 2 buys little — see the CPU
section.

**`piper-tts` won't install on the Nano.** JetPack 4.6 ships Python 3.6; piper needs ≥ 3.9. See
the deadsnakes instructions the install script prints.

---

## Where to take it next

`echo.py` is deliberately the thinnest possible app, and `chat.py` shows what that buys:
`EchoApp._reply_parts` is the one method that decides what to say, `ChatApp` overrides exactly
that method and nothing else, and capture, endpointing, recognition, playback and muting are
inherited unchanged. Intent parsing, a state machine or ROS message publishing belong in the same
place.

Likely next steps for a robot:

- **Wake word** so it doesn't react to every noise ([openWakeWord](https://github.com/dscripka/openWakeWord)
  is Apache-2.0 and runs on the same CPU budget).
- **Barge-in**: let the user interrupt playback instead of muting the mic for its duration.
- **Language switching** on a spoken command, reloading the recognizer and the LLM's persona.
- **Tools for the LLM**: let it call into the robot (drive, turn, report the battery) instead of
  only talking. A 0.5B model will not do reliable function calling — match intents yourself and
  use the model for the wording.
- **A long-running service** with the recognizer kept warm, talking to the rest of the robot over
  ROS topics, MQTT or a socket, instead of a foreground CLI.
