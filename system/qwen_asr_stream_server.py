"""Local vLLM streaming service for Qwen3-ASR."""

from __future__ import annotations

import argparse
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request


@dataclass
class Session:
    state: object
    last_seen: float


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve local Qwen3-ASR streaming")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--chunk-size-seconds", type=float, default=1.0)
    parser.add_argument("--unfixed-chunk-num", type=int, default=4)
    parser.add_argument("--unfixed-token-num", type=int, default=5)
    args = parser.parse_args(argv)
    if not args.model.exists():
        parser.error(f"Qwen3-ASR model not found: {args.model}")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("--gpu-memory-utilization must be in (0, 1]")
    if args.chunk_size_seconds <= 0:
        parser.error("--chunk-size-seconds must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        from qwen_asr import Qwen3ASRModel
    except ImportError as error:
        raise RuntimeError("Streaming service requires `qwen-asr[vllm]`") from error

    print("Loading Qwen3-ASR vLLM streaming backend...", flush=True)
    asr = Qwen3ASRModel.LLM(
        model=str(args.model.resolve()),
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_new_tokens=32,
    )
    app = Flask(__name__)
    sessions: dict[str, Session] = {}
    model_lock = threading.Lock()
    session_lock = threading.Lock()

    def get_session(session_id: str) -> Session | None:
        with session_lock:
            session = sessions.get(session_id)
            if session is not None:
                session.last_seen = time.monotonic()
            return session

    @app.get("/health")
    def health():
        return {"ok": True, "mode": "streaming"}

    @app.post("/stream/start")
    def start():
        language = request.args.get("language") or None
        with model_lock:
            state = asr.init_streaming_state(
                language=language,
                unfixed_chunk_num=args.unfixed_chunk_num,
                unfixed_token_num=args.unfixed_token_num,
                chunk_size_sec=args.chunk_size_seconds,
            )
        session_id = uuid.uuid4().hex
        with session_lock:
            sessions[session_id] = Session(state=state, last_seen=time.monotonic())
        return jsonify(session_id=session_id)

    @app.post("/stream/chunk")
    def chunk():
        session = get_session(request.args.get("session_id", ""))
        if session is None:
            return jsonify(error="invalid session_id"), 400
        payload = request.get_data(cache=False)
        if len(payload) % 4:
            return jsonify(error="PCM payload must contain float32 samples"), 400
        pcm16k = np.frombuffer(payload, dtype=np.float32)
        with model_lock:
            asr.streaming_transcribe(pcm16k, session.state)
        return jsonify(
            language=getattr(session.state, "language", "") or "",
            text=getattr(session.state, "text", "") or "",
        )

    @app.post("/stream/finish")
    def finish():
        session_id = request.args.get("session_id", "")
        with session_lock:
            session = sessions.pop(session_id, None)
        if session is None:
            return jsonify(error="invalid session_id"), 400
        with model_lock:
            asr.finish_streaming_transcribe(session.state)
        return jsonify(
            language=getattr(session.state, "language", "") or "",
            text=getattr(session.state, "text", "") or "",
        )

    print(f"Streaming service listening on http://{args.host}:{args.port}", flush=True)
    app.run(host=args.host, port=args.port, debug=False, threaded=True, use_reloader=False)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(f"error: {error}", flush=True)
        raise SystemExit(1) from error
