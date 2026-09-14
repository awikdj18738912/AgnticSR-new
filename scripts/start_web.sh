#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CONDA_EXE="${CONDA_EXE:-/home/aim0/anaconda3/bin/conda}"
AGENTIC_ENV="${AGENTIC_ENV:-agentic-asr}"
REFINER_MODEL="${REFINER_MODEL:-/home/aim0/data/models/ASR/AgenticASR-Refiner}"
ASR_URL="${ASR_URL:-http://127.0.0.1:8766}"
ENTITY_DB="${ENTITY_DB:-${PROJECT_ROOT}/data/entities.db}"
WEB_GPU="${WEB_GPU:-1}"
WEB_PORT="${WEB_PORT:-8081}"
ENTITY_FUZZY_MODE="${ENTITY_FUZZY_MODE:-auto}"
REFINEMENT_GATE_MODE="${REFINEMENT_GATE_MODE:-conservative}"
OUTPUT_PATH="${OUTPUT_PATH:-${PROJECT_ROOT}/results/web/session.jsonl}"

cd "${PROJECT_ROOT}"

if [[ ! -d "${REFINER_MODEL}" ]]; then
  echo "Refiner model not found: ${REFINER_MODEL}" >&2
  exit 1
fi
if [[ ! -f "${ENTITY_DB}" ]]; then
  echo "Entity database not found: ${ENTITY_DB}" >&2
  exit 1
fi

exec env CUDA_VISIBLE_DEVICES="${WEB_GPU}" "${CONDA_EXE}" run --no-capture-output -n "${AGENTIC_ENV}" \
  python -m system.web_app \
  --refiner-model "${REFINER_MODEL}" \
  --refiner-device cuda:0 \
  --asr-url "${ASR_URL}" \
  --language "${LANGUAGE:-Chinese}" \
  --entity-db "${ENTITY_DB}" \
  --entity-fuzzy-mode "${ENTITY_FUZZY_MODE}" \
  --refinement-gate-mode "${REFINEMENT_GATE_MODE}" \
  --output "${OUTPUT_PATH}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-256}" \
  --port "${WEB_PORT}"
