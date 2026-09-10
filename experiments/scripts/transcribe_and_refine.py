#!/usr/bin/env python3
"""Run a selectable ASR backend followed by AgenticASR refinement.

The ASR and Refiner dependencies may live in separate Conda environments.  The
intermediate JSONL file is the stable boundary between the two stages.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
POSTPROCESS_SCRIPT = SCRIPT_DIR / "postprocess_asr.py"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "e2e" / "result.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe audio with Qwen3-ASR or Whisper, then refine it with AgenticASR."
    )
    parser.add_argument("audio", type=Path)
    parser.add_argument("output_jsonl", type=Path, nargs="?", default=DEFAULT_OUTPUT)
    parser.add_argument("--asr-backend", choices=("qwen3", "whisper"), default="qwen3")
    parser.add_argument("--asr-model", required=True, help="local model path or Whisper Hugging Face id")
    parser.add_argument("--refiner-model", type=Path, required=True)
    parser.add_argument("--asr-env", default=None, help="Conda env for ASR (default: qwen3-asr or agentic-asr)")
    parser.add_argument("--refiner-env", default="agentic-asr")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--language", default="Chinese", help="use auto for automatic detection")
    parser.add_argument("--source-record-id", default="audio-001")
    parser.add_argument("--asr-max-new-tokens", type=int, default=256)
    parser.add_argument("--refiner-max-new-tokens", type=int, default=1024)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument(
        "--entity-db",
        type=Path,
        help="optional SQLite database containing verified protected entities",
    )
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> int:
    args = parse_args()
    audio = args.audio.resolve()
    refiner_model = args.refiner_model.resolve()
    output = args.output_jsonl.resolve()
    raw_output = output.with_name(f"{output.stem}.raw.jsonl")
    json_output = args.json_output.resolve() if args.json_output else output.with_suffix(".json")

    if not audio.exists():
        raise FileNotFoundError(f"audio does not exist: {audio}")
    if not refiner_model.exists():
        raise FileNotFoundError(f"Refiner model does not exist: {refiner_model}")
    if args.asr_backend == "qwen3" and not Path(args.asr_model).exists():
        raise FileNotFoundError(f"Qwen3-ASR model does not exist: {args.asr_model}")
    if args.asr_max_new_tokens < 1 or args.refiner_max_new_tokens < 1:
        raise ValueError("token limits must be at least 1")
    output.parent.mkdir(parents=True, exist_ok=True)

    language = None if args.language.lower() == "auto" else args.language
    asr_env = args.asr_env or ("qwen3-asr" if args.asr_backend == "qwen3" else "agentic-asr")
    if args.asr_backend == "qwen3":
        asr_code = """
import json
import sys
from pathlib import Path
import torch
from qwen_asr import Qwen3ASRModel

model_path, audio_path, output_path, device, language, record_id, max_tokens = sys.argv[1:]
model = Qwen3ASRModel.from_pretrained(
    model_path,
    dtype=torch.bfloat16,
    device_map=device,
    max_inference_batch_size=1,
    max_new_tokens=int(max_tokens),
)
result = model.transcribe(audio=audio_path, language=None if language == "" else language)[0]
record = {"source_record_id": record_id, "output": {"raw_text": result.text}}
Path(output_path).write_text(json.dumps(record, ensure_ascii=False) + "\\n", encoding="utf-8")
print(f"ASR backend: qwen3")
print(f"ASR language: {result.language}")
print(f"ASR raw text: {result.text}")
"""
        asr_command = [
            "conda", "run", "--no-capture-output", "-n", asr_env, "python", "-c", asr_code,
            str(Path(args.asr_model).resolve()), str(audio), str(raw_output), args.device,
            language or "", args.source_record_id, str(args.asr_max_new_tokens),
        ]
    else:
        asr_code = """
import json
import sys
from pathlib import Path
import numpy as np
import soundfile as sf
from system.whisper_backend import SAMPLE_RATE, WhisperBackend

model_path, audio_path, output_path, device, language, record_id, max_tokens = sys.argv[1:]
audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
mono = audio.mean(axis=1)
if sample_rate != SAMPLE_RATE:
    output_size = round(len(mono) * SAMPLE_RATE / sample_rate)
    mono = np.interp(np.linspace(0, 1, output_size), np.linspace(0, 1, len(mono)), mono).astype("float32")
backend = WhisperBackend(model_path, device=device, max_new_tokens=int(max_tokens))
text = backend.transcribe(mono, language=None if language == "" else language)
record = {"source_record_id": record_id, "output": {"raw_text": text}}
Path(output_path).write_text(json.dumps(record, ensure_ascii=False) + "\\n", encoding="utf-8")
print("ASR backend: whisper")
print(f"ASR raw text: {text}")
"""
        asr_command = [
            "conda", "run", "--no-capture-output", "-n", asr_env, "python", "-c", asr_code,
            args.asr_model, str(audio), str(raw_output), args.device,
            language or "", args.source_record_id, str(args.asr_max_new_tokens),
        ]

    run(asr_command)
    refine_command = [
        "conda", "run", "--no-capture-output", "-n", args.refiner_env, "python",
        str(POSTPROCESS_SCRIPT), str(raw_output), str(output), "--model", str(refiner_model),
        "--batch-size", "1", "--max-new-tokens", str(args.refiner_max_new_tokens), "--overwrite",
    ]
    if args.entity_db is not None:
        refine_command.extend(["--entity-db", str(args.entity_db.resolve())])
    run(refine_command)
    result_lines = output.read_text(encoding="utf-8").splitlines()
    if len(result_lines) != 1:
        raise ValueError(f"expected one refined result record, found {len(result_lines)}")
    json_output.write_text(json.dumps(json.loads(result_lines[0]), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Raw ASR output: {raw_output}")
    print(f"Refined output: {output}")
    print(f"Formatted JSON output: {json_output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, json.JSONDecodeError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
