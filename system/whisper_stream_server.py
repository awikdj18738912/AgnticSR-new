"""Rolling-window Whisper streaming service using the AgenticASR HTTP contract.

Whisper is not an online transducer.  This service re-transcribes the bounded
audio accumulated for each session at a configurable interval, which provides
the same ``/stream/start``, ``/stream/chunk`` and ``/stream/finish`` contract as
the Qwen3-ASR vLLM service while keeping the ASR backend replaceable.
"""

from __future__ import annotations

import argparse
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request

from .whisper_backend import SAMPLE_RATE, WhisperBackend


@dataclass
class Session:
    samples: list[np.ndarray] = field(default_factory=list)
    sample_count: int = 0
    last_transcribed_count: int = 0
    last_seen: float = 0.0
    text: str = ""
    cancelled: bool = False


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve rolling-window Whisper streaming ASR")
    parser.add_argument("--model", required=True, help="local Whisper directory or Hugging Face model id")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="auto", help="torch dtype, e.g. auto, float16, bfloat16")
    parser.add_argument("--language", default="Chinese", help="Whisper language hint; use auto for detection")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--update-interval-seconds", type=float, default=1.5)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args(argv)
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    if args.update_interval_seconds <= 0:
        parser.error("--update-interval-seconds must be positive")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print("Loading Whisper streaming backend...", flush=True)
    whisper = WhisperBackend(args.model, args.device, args.dtype, args.max_new_tokens)
    update_samples = round(args.update_interval_seconds * SAMPLE_RATE)
    language = None if args.language.lower() == "auto" else args.language

    app = Flask(__name__)
    sessions: dict[str, Session] = {}
    model_lock = threading.Lock()
    sessions_lock = threading.Lock()

    def get_session(session_id: str) -> Session | None:
        with sessions_lock:
            session = sessions.get(session_id)
            if session is not None:
                session.last_seen = time.monotonic()
            return session

    def transcribe_if_ready(session: Session, force: bool = False) -> None:
        if not force and session.sample_count - session.last_transcribed_count < update_samples:
            return
        waveform = np.concatenate(session.samples) if session.samples else np.zeros(0, dtype=np.float32)
        if waveform.size == 0:
            return
        with model_lock:
            if session.cancelled:
                return
            session.text = whisper.transcribe(waveform, language)
        session.last_transcribed_count = session.sample_count

    @app.get("/health")
    def health():
        return {"ok": True, "mode": "whisper-rolling"}

    @app.post("/stream/start")
    def start():
        session_id = uuid.uuid4().hex
        with sessions_lock:
            sessions[session_id] = Session(last_seen=time.monotonic())
        return jsonify(session_id=session_id)

    @app.post("/stream/chunk")
    def chunk():
        session = get_session(request.args.get("session_id", ""))
        if session is None:
            return jsonify(error="invalid session_id"), 400
        payload = request.get_data(cache=False)
        if len(payload) % 4:
            return jsonify(error="PCM payload must contain float32 samples"), 400
        samples = np.frombuffer(payload, dtype=np.float32).copy()
        if samples.size:
            session.samples.append(samples)
            session.sample_count += samples.size
            transcribe_if_ready(session)
        return jsonify(language=language or "", text=session.text)

    @app.post("/stream/finish")
    def finish():
        session_id = request.args.get("session_id", "")
        with sessions_lock:
            session = sessions.pop(session_id, None)
        if session is None:
            return jsonify(error="invalid session_id"), 400
        transcribe_if_ready(session, force=True)
        return jsonify(language=language or "", text=session.text)

    @app.post("/stream/cancel")
    def cancel():
        session_id = request.args.get("session_id", "")
        with sessions_lock:
            session = sessions.pop(session_id, None)
            if session is not None:
                session.cancelled = True
        if session is None:
            return jsonify(error="invalid session_id"), 400
        return jsonify(cancelled=True)

    print(f"Whisper streaming service listening on http://{args.host}:{args.port}", flush=True)
    app.run(host=args.host, port=args.port, debug=False, threaded=True, use_reloader=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
