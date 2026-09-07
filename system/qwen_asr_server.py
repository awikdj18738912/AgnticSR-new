"""Local HTTP wrapper for Qwen3-ASR in its dedicated Conda environment."""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve local Qwen3-ASR transcription")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be at least 1")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    model_path = args.model.resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Qwen3-ASR model not found: {model_path}")
    try:
        import soundfile as sf
        import torch
        from flask import Flask, jsonify, request
        from qwen_asr import Qwen3ASRModel
    except ImportError as error:
        raise RuntimeError(
            "Server requires qwen-asr, flask, soundfile, and torch"
        ) from error

    print("Loading Qwen3-ASR...", flush=True)
    model = Qwen3ASRModel.from_pretrained(
        str(model_path),
        dtype=torch.bfloat16,
        device_map=args.device,
        max_inference_batch_size=1,
        max_new_tokens=args.max_new_tokens,
    )
    app = Flask(__name__)

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.post("/transcribe")
    def transcribe():
        data = request.get_data(cache=False)
        if not data:
            return jsonify(error="request body must contain WAV audio"), 400
        try:
            audio, sample_rate = sf.read(
                io.BytesIO(data), dtype="float32", always_2d=True
            )
        except RuntimeError as error:
            return jsonify(error=f"invalid audio: {error}"), 400
        mono = audio.mean(axis=1)
        if len(mono) == 0:
            return jsonify(error="audio is empty"), 400
        language = request.args.get("language") or None
        result = model.transcribe(audio=(mono, sample_rate), language=language)[0]
        return jsonify(text=result.text, language=result.language)

    print(f"Listening on http://{args.host}:{args.port}", flush=True)
    app.run(host=args.host, port=args.port, debug=False, threaded=False, use_reloader=False)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
