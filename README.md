# voice-synth

Lightweight text-to-speech service built on [Piper](https://github.com/rhasspy/piper) (MIT-licensed,
ONNX Runtime CPU inference). Designed to run alongside Vosk (STT) and other workloads on a
Jetson Nano P3450 (2/4GB, 2022 model) without starving them of CPU, RAM, or GPU.

## Why Piper

| Engine | License | RAM | Speed on Cortex-A57 | GPU needed | Quality |
|---|---|---|---|---|---|
| **Piper** | MIT | ~150-300MB per voice | RTF ~0.1-0.3 (faster than real time) | No | Good (neural, VITS-derived) |
| Coqui TTS (VITS/XTTS) | MPL-2.0 / non-commercial for some models | 1GB+ (PyTorch) | Often slower than real time on A57 | Recommended | Very good/excellent |
| eSpeak-NG | GPL-3.0 | <20MB | Instant | No | Robotic |
| Mimic3 | AGPL/MIT mix, deprecated by Rhasspy in favor of Piper | similar to Piper but heavier runtime | slower | No | Good |

Piper wins here because:
- **No GPU dependency** — leaves the Nano's Maxwell GPU free for camera/vision pipelines or CUDA-accelerated
  Vosk/other models, and avoids TensorRT/CUDA version-matching headaches.
- **Small footprint** — ONNX Runtime + a single small model, no PyTorch/TensorFlow runtime to load.
- **Fast enough on CPU alone** — small/medium voice models synthesize faster than real-time on 4x Cortex-A57.
- **Fully free/open** — MIT license for the engine, voices are MIT or CC-BY/CC0 (check each voice's card).

## Resource budget on a Jetson Nano (4GB recommended, 2GB workable)

The Nano has 4 CPU cores total. Assume Vosk (streaming STT) and other services are also running.
This project defaults to a conservative budget:

- **1 persistent Piper process**, model loaded once (avoids repeated ~100-300ms model-load cost per request).
- **`PIPER_NUM_THREADS=2`** by default — caps ONNX Runtime intra-op threads so Piper doesn't compete for
  all 4 cores against Vosk's own recognizer thread(s). Tune via env var (see below).
- **One voice model resident in RAM at a time** (~50-150MB for `low`/`medium` quality `en_US` voices).
  Swapping voices reloads the model — avoid doing this per-request in production.
- **Request queue, not thread-per-request** — synthesis requests are serialized through a single worker
  so peak RAM/CPU stays bounded and predictable instead of spiking with concurrent synthesis.
- Prefer `x_low` or `medium` quality voice variants over `high` — `high` models are ~4x larger and slower
  for a quality gain that matters less on a robot/embedded speaker than latency and headroom for Vosk.

Typical numbers on Jetson Nano 4GB, `en_US-lessac-medium`, 2 threads: model load ~1-2s (once, at startup),
synthesis RTF (real-time factor) commonly 0.15-0.4 — i.e. a 3s sentence synthesizes in well under 1s.

## Install (on the Jetson, aarch64 Ubuntu/JetPack)

```bash
./scripts/install_jetson.sh
```

This creates a venv, installs `piper-tts` + `onnxruntime` (both have aarch64 wheels on PyPI), and
downloads a default small English voice. See the script for details/overrides.

## Install (dev machine, any platform)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m voice_synth.cli download en_US-lessac-medium
```

## Usage

### CLI (quick test, no server)

```bash
python -m voice_synth.cli speak "Hello from the Jetson Nano." --out hello.wav
aplay hello.wav   # or paplay, depending on your audio stack
```

### Server (persistent process, HTTP API)

```bash
PIPER_NUM_THREADS=2 python -m voice_synth.server
```

Then:

```bash
curl -X POST http://localhost:5002/synthesize \
  -H 'Content-Type: application/json' \
  -d '{"text": "Hello from the Jetson Nano."}' \
  --output out.wav
```

`GET /health` reports the loaded voice and queue depth.

### Config (env vars)

| Var | Default | Purpose |
|---|---|---|
| `PIPER_VOICE` | `en_US-lessac-medium` | Voice model name (must be downloaded first) |
| `PIPER_MODELS_DIR` | `./models` | Where `.onnx`/`.onnx.json` voice files live |
| `PIPER_NUM_THREADS` | `2` | ONNX Runtime intra-op threads — keep low to leave cores for Vosk |
| `PIPER_HOST` / `PIPER_PORT` | `0.0.0.0` / `5002` | HTTP server bind address |

## Running alongside Vosk

- Keep `PIPER_NUM_THREADS` + Vosk's own thread count ≤ number of physical cores (4 on Nano). E.g.
  Piper=2, Vosk=2 is a safe split; both are usually idle waiting on I/O/audio buffers rather than
  pegging CPU continuously, so some oversubscription is fine in practice — measure with `tegrastats`.
- Run both as separate systemd services (see `systemd/`) rather than in one process, so either can be
  restarted independently and you can see per-service CPU/RSS in `tegrastats` / `top`.
- Don't load multiple Piper voices simultaneously unless you have RAM to spare (each resident voice
  model costs its own ONNX Runtime session + weights, roughly 50-150MB depending on quality tier).

## Voice models

Grab additional voices from the official Piper voice samples page and place the `.onnx` +
`.onnx.json` pair into `PIPER_MODELS_DIR`. All are free to use; check individual voice cards for
attribution requirements (most are MIT or CC0, a few are CC-BY which just requires credit).

### German

Works out of the box, same download/CLI/server flow, just point `PIPER_VOICE` at a German voice
name:

```bash
python -m voice_synth.cli download de_DE-thorsten-medium
python -m voice_synth.cli speak "Hallo, hier spricht der Jetson Nano." --voice de_DE-thorsten-medium
```

```bash
PIPER_VOICE=de_DE-thorsten-medium python -m voice_synth.server
```

Available German (`de_DE`) voices, smallest/fastest to largest/best quality — pick one:

| Voice | Quality | Size | Notes |
|---|---|---|---|
| `de_DE-eva_k-x_low` | x_low | ~21MB | Smallest/fastest, noticeably robotic |
| `de_DE-thorsten-low` | low | ~63MB | Good speed/quality tradeoff |
| `de_DE-karlsson-low`, `de_DE-kerstin-low`, `de_DE-pavoque-low`, `de_DE-ramona-low` | low | ~63MB | Alternative voices/speakers, same tier |
| `de_DE-thorsten-medium` | medium | ~63MB | Recommended default — same tier as the English default above |
| `de_DE-thorsten_emotional-medium` | medium | ~77MB | Multi-speaker, 8 emotional styles (`speaker_id`) |
| `de_DE-mls-medium` | medium | ~77MB | 236 speakers |
| `de_DE-thorsten-high` | high | ~114MB | Best quality, slower — usually not worth it on a Nano |

Tested on this repo's dev setup: `de_DE-thorsten-medium` downloads and synthesizes correctly
through both the CLI and the HTTP server, same resource profile as the English medium voice
(~1.5s including model load for a short sentence, ~210MB RSS once loaded).

Only one voice is loaded per running server process (see resource budget above) — to serve both
English and German, run two `voice-synth` processes on different ports (e.g. `PIPER_PORT=5002` /
`5003`), each with its own systemd unit and thread cap, rather than hot-swapping voices per
request.

## systemd

See [`systemd/voice-synth.service`](systemd/voice-synth.service) for a unit file that runs the
server with a capped thread count and restarts on failure.
# voice-interaction
