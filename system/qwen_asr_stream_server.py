"""Local vLLM streaming service for Qwen3-ASR."""

from __future__ import annotations

import argparse
import re
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
    language: str | None = None
    detected_language: str = ""
    committed_text: str = ""
    segment_samples: int = 0
    cancelled: bool = False


SAMPLE_RATE = 16_000
_SENTENCE_END_RE = re.compile(r"[.!?。！？][\"'”’》〉』】）)]*$")


def _join_transcripts(prefix: str, suffix: str) -> str:
    prefix = prefix.strip()
    suffix = suffix.strip()
    if not prefix:
        return suffix
    if not suffix:
        return prefix
    separator = " " if prefix[-1].isascii() and suffix[0].isascii() else ""
    return f"{prefix}{separator}{suffix}"


def _should_rotate_segment(
    text: str,
    sample_count: int,
    soft_limit_samples: int,
    hard_limit_samples: int,
) -> bool:
    if sample_count >= hard_limit_samples:
        return True
    return sample_count >= soft_limit_samples and bool(
        _SENTENCE_END_RE.search(text.strip())
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve local Qwen3-ASR streaming")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=32768,
        help="vLLM context limit; bounded streaming does not require the model's 65K maximum",
    )
    parser.add_argument("--chunk-size-seconds", type=float, default=1.0)
    parser.add_argument("--unfixed-chunk-num", type=int, default=4)
    parser.add_argument("--unfixed-token-num", type=int, default=5)
    parser.add_argument(
        "--segment-seconds",
        type=float,
        default=30.0,
        help="prefer starting a fresh bounded ASR state after this duration",
    )
    parser.add_argument(
        "--max-segment-seconds",
        type=float,
        default=45.0,
        help="always start a fresh ASR state after this duration",
    )
    args = parser.parse_args(argv)
    if not args.model.exists():
        parser.error(f"Qwen3-ASR model not found: {args.model}")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("--gpu-memory-utilization must be in (0, 1]")
    if args.max_model_len < 1024:
        parser.error("--max-model-len must be at least 1024")
    if args.chunk_size_seconds <= 0:
        parser.error("--chunk-size-seconds must be positive")
    if args.segment_seconds <= 0:
        parser.error("--segment-seconds must be positive")
    if args.max_segment_seconds < args.segment_seconds:
        parser.error("--max-segment-seconds must be at least --segment-seconds")
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
        max_model_len=args.max_model_len,
        max_new_tokens=32,
    )
    app = Flask(__name__)
    sessions: dict[str, Session] = {}
    model_lock = threading.Lock()
    session_lock = threading.Lock()
    soft_limit_samples = round(args.segment_seconds * SAMPLE_RATE)
    hard_limit_samples = round(args.max_segment_seconds * SAMPLE_RATE)

    def new_state(language: str | None):
        return asr.init_streaming_state(
            language=language,
            unfixed_chunk_num=args.unfixed_chunk_num,
            unfixed_token_num=args.unfixed_token_num,
            chunk_size_sec=args.chunk_size_seconds,
        )

    def full_text(session: Session) -> str:
        return _join_transcripts(
            session.committed_text,
            getattr(session.state, "text", "") or "",
        )

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
            state = new_state(language)
        session_id = uuid.uuid4().hex
        with session_lock:
            sessions[session_id] = Session(
                state=state,
                last_seen=time.monotonic(),
                language=language,
            )
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
            # The browser may disconnect while this request waits behind a
            # long GPU call. Do not spend more GPU time on a session that was
            # cancelled in the meantime.
            if session.cancelled:
                return jsonify(error="session cancelled"), 409
            asr.streaming_transcribe(pcm16k, session.state)
            session.segment_samples += pcm16k.size
            segment_text = getattr(session.state, "text", "") or ""
            detected_language = getattr(session.state, "language", "") or ""
            if detected_language:
                session.detected_language = detected_language
            if _should_rotate_segment(
                segment_text,
                session.segment_samples,
                soft_limit_samples,
                hard_limit_samples,
            ):
                if getattr(session.state, "buffer", np.zeros(0)).size:
                    asr.finish_streaming_transcribe(session.state)
                    segment_text = getattr(session.state, "text", "") or ""
                session.committed_text = _join_transcripts(
                    session.committed_text, segment_text
                )
                session.state = new_state(session.language)
                session.segment_samples = 0
        return jsonify(
            language=session.detected_language,
            text=full_text(session),
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
            detected_language = getattr(session.state, "language", "") or ""
            if detected_language:
                session.detected_language = detected_language
        return jsonify(
            language=session.detected_language,
            text=full_text(session),
        )

    @app.post("/stream/cancel")
    def cancel():
        session_id = request.args.get("session_id", "")
        with session_lock:
            session = sessions.pop(session_id, None)
            if session is not None:
                session.cancelled = True
        if session is None:
            return jsonify(error="invalid session_id"), 400
        return jsonify(cancelled=True)

    print(f"Streaming service listening on http://{args.host}:{args.port}", flush=True)
    app.run(host=args.host, port=args.port, debug=False, threaded=True, use_reloader=False)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(f"error: {error}", flush=True)
        raise SystemExit(1) from error
