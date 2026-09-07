#!/usr/bin/env python3
"""Run local Qwen3-ASR transcription followed by AgenticASR refinement.

The ASR and Refiner dependencies may live in separate Conda environments. This
wrapper preserves the AgenticASR JSONL contract between the two processes.
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
        description="Transcribe one audio file with Qwen3-ASR and refine it with AgenticASR."
    )
    parser.add_argument("audio", type=Path, help="input audio file")
    parser.add_argument(
        "output_jsonl",
        type=Path,
        nargs="?",
        default=DEFAULT_OUTPUT,
        help="refined JSONL output path (default: results/e2e/result.jsonl)",
    )
    parser.add_argument("--asr-model", type=Path, required=True, help="local Qwen3-ASR model")
    parser.add_argument("--refiner-model", type=Path, required=True, help="local Refiner model")
    parser.add_argument("--qwen-env", default="qwen3-asr", help="Conda environment containing qwen-asr")
    parser.add_argument("--refiner-env", default="agentic-asr", help="Conda environment containing torch and transformers")
    parser.add_argument("--device", default="cuda:0", help="CUDA device used by both sequential stages")
    parser.add_argument("--language", default="Chinese", help="Qwen3-ASR language hint; use auto to disable")
    parser.add_argument("--source-record-id", default="audio-001")
    parser.add_argument("--asr-max-new-tokens", type=int, default=256)
    parser.add_argument("--refiner-max-new-tokens", type=int, default=1024)
    parser.add_argument(
        "--json-output",
        type=Path,
        help="formatted JSON result path (default: output_jsonl with a .json suffix)",
    )
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> int:
    args = parse_args()
    audio = args.audio.resolve()
    asr_model = args.asr_model.resolve()
    refiner_model = args.refiner_model.resolve()
    output = args.output_jsonl.resolve()
    raw_output = output.with_name(f"{output.stem}.raw.jsonl")
    json_output = (
        args.json_output.resolve() if args.json_output else output.with_suffix(".json")
    )

    for path, label in ((audio, "audio"), (asr_model, "ASR model"), (refiner_model, "Refiner model")):
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    output.parent.mkdir(parents=True, exist_ok=True)

    language = None if args.language.lower() == "auto" else args.language
    if args.asr_max_new_tokens < 1 or args.refiner_max_new_tokens < 1:
        raise ValueError("token limits must be at least 1")

    qwen_code = """
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
record = {
    "source_record_id": record_id,
    "output": {"raw_text": result.text},
}
Path(output_path).write_text(json.dumps(record, ensure_ascii=False) + "\\n", encoding="utf-8")
print(f"ASR language: {result.language}")
print(f"ASR raw text: {result.text}")
"""
    run(
        [
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            args.qwen_env,
            "python",
            "-c",
            qwen_code,
            str(asr_model),
            str(audio),
            str(raw_output),
            args.device,
            language or "",
            args.source_record_id,
            str(args.asr_max_new_tokens),
        ]
    )
    run(
        [
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            args.refiner_env,
            "python",
            str(POSTPROCESS_SCRIPT),
            str(raw_output),
            str(output),
            "--model",
            str(refiner_model),
            "--batch-size",
            "1",
            "--max-new-tokens",
            str(args.refiner_max_new_tokens),
            "--overwrite",
        ]
    )
    result_lines = output.read_text(encoding="utf-8").splitlines()
    if len(result_lines) != 1:
        raise ValueError(f"expected one refined result record, found {len(result_lines)}")
    json_output.write_text(
        json.dumps(json.loads(result_lines[0]), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
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
