# Streaming AgenticASR

`system/` contains the streaming implementation used by the AgenticASR desktop App. The packaged Windows/macOS application is distributed through the [VibeXASR product page](https://vibexasr.speech.wiki/); this directory is the reproducible Python implementation.

## Runtime Path

```text
WAV/microphone -> VAD -> online sherpa-onnx ASR -> ChunkManager -> K=3 Refiner
```

The full App path requires a Refiner. The current backend is MLX-LM on macOS. `--identity-refiner` is only an ASR/chunking diagnostic mode.

## Models

Install dependencies:

```bash
python -m pip install -r system/requirements.txt
python -m pip install mlx mlx-lm
```

Place an online sherpa-onnx checkpoint under `models/asr/` with `tokens.txt` and either transducer files (`encoder*.onnx`, `decoder*.onnx`, `joiner*.onnx`) or a Wenet CTC model (`model*.onnx`).

Download the default Silero VAD:

```bash
bash system/download_vad.sh models
```

Other supported modes are `--vad energy` (no model) and `--vad firered` (install `fireredvad` and provide `--firered-dir`).

Convert a trained Refiner for the current Mac backend:

```bash
python pipeline/scripts/convert_to_mlx.py \
  --input /path/to/huggingface-refiner \
  --output models/refiner-mlx \
  --quantize q4_0
```

Run:

```bash
python -m system.live_asr \
  --wav path/to/example.wav \
  --asr-dir models/asr \
  --refiner models/refiner-mlx
```

## Selectable ASR backends for the browser frontend

The browser frontend is ASR-backend agnostic. It talks to a streaming HTTP
contract (`/stream/start`, `/stream/chunk`, and `/stream/finish`) and applies
the same local Refiner to whatever text the backend returns.

The web UI provides one microphone mode and two local-file modes. Online mode
uses the microphone and refines changed ASR hypotheses synchronously as speech
arrives. Offline mode selects a local audio file, transcribes the entire file,
then runs one final Refiner pass. Streaming mode also selects a local audio
file, sends it in PCM chunks, and refines only the latest pending hypothesis in
a background task so Refiner latency does not block transcription updates.

Qwen3-ASR is the low-latency online backend:

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n qwen3-asr \
  python -m system.qwen_asr_stream_server \
  --model /path/to/Qwen3-ASR-0.6B \
  --gpu-memory-utilization 0.55 \
  --max-model-len 32768 \
  --port 8766
```

The service bounds each Qwen streaming state to protect long recordings from
the upstream implementation's growing full-audio reprocessing cost. It starts
a fresh state at a sentence boundary after 30 seconds and always rotates by 45
seconds, while preserving the cumulative transcript. Use `--segment-seconds`
and `--max-segment-seconds` to tune these limits. Abandoned browser sessions
are cancelled so their queued chunks do not continue occupying the GPU.

Whisper is also supported through a rolling-window backend. Whisper is not an
online transducer, so it re-transcribes the accumulated utterance every few
seconds; this is more compute-heavy but uses the same browser UI:

```bash
CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n agentic-asr \
  python -m system.whisper_stream_server \
  --model /path/to/whisper-model \
  --device cuda:0 \
  --language Chinese \
  --update-interval-seconds 1.5 \
  --port 8766
```

Start the frontend against either backend:

```bash
CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n agentic-asr \
  python -m system.web_app \
  --refiner-model /path/to/AgenticASR-Refiner \
  --refiner-device cuda:0 \
  --asr-url http://127.0.0.1:8766 \
  --language Chinese \
  --port 8081
```

The Whisper backend accepts a local Hugging Face Whisper directory or a model
id such as `openai/whisper-small`; use a local directory for offline runs.

## Protected entity database and session memory

The frontend audits numbers, dates, email addresses, URLs, identifiers, and
verified domain entities alongside each Refiner request. When a verified alias
is explicitly present in an ASR hypothesis, its canonical form is supplied as
a short Refiner glossary. For an entity using `normalize`, that exact alias is
then deterministically normalized in the Refiner output and recorded in the
session log. This does not use fuzzy global replacement: an unrelated or
unmatched database entry cannot change the displayed text.

Create or update the local SQLite term database:

```bash
python -m system.manage_entities --db data/entities.db add AgenticASR \
  --alias "Agentic SR" \
  --type PROJECT \
  --policy normalize \
  --priority 10

python -m system.manage_entities --db data/entities.db add Qwen3-ASR \
  --alias "Qwen ASR" \
  --type MODEL \
  --policy normalize

python -m system.manage_entities --db data/entities.db list
```

`preserve` keeps the exact matched surface. `normalize` restores the verified
canonical spelling when an explicit alias is matched. Fuzzy matching has four
server modes: `off`, `shadow`, `hint`, and `auto`. The default is `shadow`,
which records candidates without changing output. `auto` only normalizes
high-confidence `TERM`, `PROJECT`, and `MODEL` matches in a completed
transcript; streaming intermediate updates and high-risk `PERSON`/`ORG`
matches are never fuzzy-auto-replaced.

Enable the database in the browser frontend:

```bash
CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n agentic-asr \
  python -m system.web_app \
  --refiner-model /path/to/AgenticASR-Refiner \
  --refiner-device cuda:0 \
  --asr-url http://127.0.0.1:8766 \
  --entity-db data/entities.db \
  --entity-fuzzy-mode shadow \
  --output results/web/session.jsonl \
  --port 8081
```

When the server is started with `--entity-db`, the browser page also exposes a
local **术语库管理** panel. It can search, add, edit, enable/disable, and
permanently delete verified entities and aliases. Changes are stored only in
the configured SQLite file and are loaded by the next transcription session;
they do not interrupt a running session or force a rewrite of displayed text.

Entity types and domains are fixed selectable categories in the panel. The
main page also provides a **术语领域** selector. A new session loads entries in
the chosen domain plus entries in `general`; for example, selecting `医疗`
loads `医疗` and `通用` entities. This selection only changes entity auditing,
the Refiner glossary, and verified-alias normalization; it does not change the
ASR model itself.

Each WebSocket connection also gets a bounded in-memory entity memory.
Database entities are immediately trusted. A newly observed entity is not
allowed to constrain later text unless it has multiple independent
high-confidence observations with distinct stable-segment IDs. Repeated
partial hypotheses do not count as independent evidence, and missing ASR
confidence never promotes an entity automatically. This avoids turning one
ASR or Refiner error into persistent session memory.

## Local Qwen3-ASR + Refiner microphone mode

The original `live_asr.py` requires a sherpa-onnx online ASR model and an MLX
Refiner. For a Linux machine with a local Qwen3-ASR checkpoint and a Hugging
Face Refiner checkpoint, run the two processes below in separate environments.
They communicate only through `127.0.0.1`.

Start the Qwen3-ASR service in an environment with `qwen-asr==0.0.6` and
`transformers==4.57.6`:

```bash
python system/qwen_asr_server.py \
  --model /path/to/Qwen3-ASR-0.6B \
  --device cuda:0
```

In a second terminal, install microphone support in the Refiner environment:

```bash
python -m pip install sounddevice
```

Then start the microphone client in that environment:

```bash
python -m system.live_qwen_refiner \
  --refiner-model /path/to/AgenticASR-Refiner \
  --refiner-device cuda:1 \
  --entity-db data/entities.db \
  --entity-domain general \
  --entity-fuzzy-mode shadow \
  --output results/live/session.jsonl
```

After reviewing shadow candidates in the JSONL output, use
`--entity-fuzzy-mode auto` to enable deterministic final-utterance
normalization. Candidate scores, decisions, reasons, matcher latency, and
applied normalizations are included in each output record.

The client uses local energy VAD and submits one completed utterance after a
short silence. It prints raw and refined text, and optionally records each
utterance to JSONL. Use `--list-devices` to select a microphone.

## Browser microphone mode for VMs

For a PVE VM, microphone access is usually easier through a browser on the
physical computer. This mode uses Qwen's vLLM streaming backend. Stop an
existing `qwen_asr_server.py` process first, then start the streaming service
in the Qwen environment:

```bash
python system/qwen_asr_stream_server.py \
  --model /path/to/Qwen3-ASR-0.6B \
  --gpu-memory-utilization 0.75
```

In a second terminal, start this Web service in the Refiner environment:

```bash
python -m system.web_app \
  --refiner-model /path/to/AgenticASR-Refiner \
  --refiner-device cuda:1 \
  --output results/web/session.jsonl
```

Keep the service bound to `127.0.0.1`. From the physical computer, forward the
VM port over SSH and open the localhost URL in a browser:

```bash
ssh -L 8081:127.0.0.1:8081 user@vm-address
```

Open `http://localhost:8081`. Browsers permit microphone access on localhost;
opening an HTTP URL at the VM's LAN address generally will not. The browser
records WAV audio locally and sends it through the SSH tunnel, so no VM audio
passthrough is required.

## Files

- `live_asr.py`: terminal entry point and streaming loop.
- `backends.py`: audio input, VAD, and sherpa-onnx adapters.
- `chunking.py`: stable bounded chunks from incremental hypotheses.
- `refiner.py`: Refiner protocol, MLX backend, identity diagnostic backend, and sliding-window session.
- `qwen_asr_server.py`: local Qwen3-ASR offline HTTP service for the separate Qwen environment.
- `qwen_asr_stream_server.py`: local Qwen3-ASR vLLM streaming HTTP service.
- `live_qwen_refiner.py`: microphone VAD and Refiner client for the separate AgenticASR environment.
- `web_app.py`: browser API and local Refiner service for VM use.
- `web/index.html`: browser microphone interface.
- `download_vad.sh`: download the Silero VAD model.
- `requirements.txt`: streaming dependencies.
- `__init__.py`: package exports.
