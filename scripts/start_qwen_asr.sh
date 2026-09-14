#!/usr/bin/env bash
set -euo pipefail

# Resolve the repository from this script so the command remains valid when
# the checkout is renamed or moved.  Keep the ASR process in its own Conda
# environment because vLLM/Qwen dependencies are not the Refiner runtime.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CONDA_EXE="${CONDA_EXE:-/home/aim0/anaconda3/bin/conda}"
QWEN_ENV="${QWEN_ENV:-qwen3-asr}"
QWEN_MODEL="${QWEN_MODEL:-/home/aim0/data/models/ASR/Qwen3-ASR-1.7B}"
VAD_MODEL="${VAD_MODEL:-${PROJECT_ROOT}/models/silero_vad.onnx}"
QWEN_GPU="${QWEN_GPU:-0}"
QWEN_PORT="${QWEN_PORT:-8766}"

cd "${PROJECT_ROOT}"

if [[ ! -f "${VAD_MODEL}" ]]; then
  echo "VAD model not found: ${VAD_MODEL}" >&2
  exit 1
fi

exec env CUDA_VISIBLE_DEVICES="${QWEN_GPU}" "${CONDA_EXE}" run --no-capture-output -n "${QWEN_ENV}" \
  python -m system.qwen_asr_stream_server \
  --model "${QWEN_MODEL}" \
  --gpu-memory-utilization "${QWEN_GPU_MEMORY_UTILIZATION:-0.55}" \
  --max-model-len "${QWEN_MAX_MODEL_LEN:-32768}" \
  --confidence-logprobs "${QWEN_CONFIDENCE_LOGPROBS:-5}" \
  --vad silero \
  --vad-model "${VAD_MODEL}" \
  --vad-min-silence "${QWEN_VAD_MIN_SILENCE:-0.7}" \
  --vad-min-speech "${QWEN_VAD_MIN_SPEECH:-0.25}" \
  --vad-preroll "${QWEN_VAD_PREROLL:-0.35}" \
  --silence-rms-threshold "${QWEN_SILENCE_RMS_THRESHOLD:-0.002}" \
  --port "${QWEN_PORT}"
