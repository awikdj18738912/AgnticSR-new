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
  --output results/live/session.jsonl
```

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
