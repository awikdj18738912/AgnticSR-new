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

from .audio_activity import is_silence, pcm_peak, pcm_rms
from .backends import SAMPLE_RATE, build_vad
from .qwen_confidence import QwenVLLMConfidenceDecoder, state_confidence_payload
from .stream_vad import StreamingVADGate


@dataclass
class Session:
    state: object
    last_seen: float
    vad_gate: StreamingVADGate | None = None
    language: str | None = None
    detected_language: str = ""
    committed_text: str = ""
    segment_samples: int = 0
    last_confidence: dict[str, object] | None = None
    silence_skipped_chunks: int = 0
    vad_skipped_chunks: int = 0
    vad_last_speech: bool = False
    cancelled: bool = False


_SENTENCE_END_RE = re.compile(r"[.!?。！？][\"'”’》〉』】）)]*$")


def _scope_confidence_payload(
    payload: dict[str, object],
    *,
    has_committed_prefix: bool,
) -> dict[str, object]:
    """Describe which portion of the returned transcript owns the score."""

    scoped = dict(payload)
    token_count = scoped.get("confidence_token_count")
    available = (
        scoped.get("confidence") is not None
        and isinstance(token_count, int)
        and not isinstance(token_count, bool)
        and token_count > 0
    )
    covers_full_state = (
        available and scoped.get("confidence_covers_full_state") is True
    )
    covers_full_text = covers_full_state and not has_committed_prefix
    if not available:
        scope = "unavailable"
    elif covers_full_text:
        scope = "full_text"
    elif covers_full_state:
        scope = "current_state"
    else:
        scope = "generated_suffix"
    scoped["confidence_covers_full_text"] = covers_full_text
    scoped["confidence_scope"] = scope
    return scoped


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
        "--silence-rms-threshold",
        type=float,
        default=0.002,
        help="RMS fallback for chunks below this level; set 0 to disable",
    )
    parser.add_argument(
        "--vad",
        choices=("off", "silero", "energy", "firered"),
        default="silero",
        help="speech activity detector used before Qwen decoding",
    )
    parser.add_argument(
        "--vad-model",
        type=Path,
        default=Path("models/silero_vad.onnx"),
        help="Silero VAD ONNX model path",
    )
    parser.add_argument(
        "--firered-dir",
        type=Path,
        default=Path("models/firered_vad"),
        help="FireRedVAD model directory when --vad firered is selected",
    )
    parser.add_argument(
        "--vad-provider",
        default="cpu",
        help="sherpa-onnx provider for Silero VAD (normally cpu)",
    )
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=0.5,
        help="speech probability threshold for Silero/FireRed VAD",
    )
    parser.add_argument(
        "--vad-min-silence",
        type=float,
        default=0.7,
        help="seconds of silence required to end a speech region",
    )
    parser.add_argument(
        "--vad-min-speech",
        type=float,
        default=0.25,
        help="seconds of speech required to start a speech region",
    )
    parser.add_argument(
        "--vad-energy-threshold",
        type=float,
        default=0.02,
        help="RMS threshold when --vad energy is selected",
    )
    parser.add_argument(
        "--vad-preroll",
        type=float,
        default=0.35,
        help="seconds of audio retained before VAD speech start",
    )
    parser.add_argument(
        "--confidence-logprobs",
        type=int,
        default=0,
        help="capture this many vLLM token alternatives; 0 keeps the original fast path",
    )
    parser.add_argument(
        "--confidence-temperature",
        type=float,
        default=1.0,
        help="temperature used to convert mean token logprob into a score",
    )
    parser.add_argument(
        "--confidence-calibrated",
        action="store_true",
        help="mark the score as calibrated after fitting temperature on a dev set",
    )
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
    if args.confidence_logprobs < 0:
        parser.error("--confidence-logprobs must be non-negative")
    if args.confidence_temperature <= 0:
        parser.error("--confidence-temperature must be positive")
    if args.confidence_calibrated and args.confidence_logprobs == 0:
        parser.error("--confidence-calibrated requires --confidence-logprobs")
    if args.silence_rms_threshold < 0:
        parser.error("--silence-rms-threshold must be non-negative")
    if not 0 <= args.vad_threshold <= 1:
        parser.error("--vad-threshold must be between 0 and 1")
    if args.vad_min_silence <= 0:
        parser.error("--vad-min-silence must be positive")
    if args.vad_min_speech <= 0:
        parser.error("--vad-min-speech must be positive")
    if args.vad_energy_threshold < 0:
        parser.error("--vad-energy-threshold must be non-negative")
    if args.vad_preroll < 0:
        parser.error("--vad-preroll must be non-negative")
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
    decoder = (
        QwenVLLMConfidenceDecoder(
            asr,
            top_logprobs=args.confidence_logprobs,
            temperature=args.confidence_temperature,
            calibrated=args.confidence_calibrated,
        )
        if args.confidence_logprobs
        else None
    )
    asr_runner = decoder or asr
    app = Flask(__name__)
    sessions: dict[str, Session] = {}
    model_lock = threading.Lock()
    session_lock = threading.Lock()
    soft_limit_samples = round(args.segment_seconds * SAMPLE_RATE)
    hard_limit_samples = round(args.max_segment_seconds * SAMPLE_RATE)

    if args.vad == "silero" and not args.vad_model.exists():
        raise RuntimeError(
            f"Silero VAD model not found: {args.vad_model}. "
            "Run `bash system/download_vad.sh models` or use --vad off."
        )

    def new_vad_gate() -> StreamingVADGate | None:
        if args.vad == "off":
            return None
        detector = build_vad(
            args.vad,
            str(args.vad_model.resolve()),
            str(args.firered_dir.resolve()),
            args.vad_threshold,
            args.vad_min_silence,
            args.vad_min_speech,
            args.vad_energy_threshold,
            args.vad_provider,
        )
        return StreamingVADGate(
            detector,
            sample_rate=SAMPLE_RATE,
            preroll_seconds=args.vad_preroll,
        )

    def new_state(language: str | None):
        return asr_runner.init_streaming_state(
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

    def confidence_payload(session: Session) -> dict[str, object]:
        current = state_confidence_payload(session.state)
        if current.get("confidence") is not None:
            return _scope_confidence_payload(
                current,
                has_committed_prefix=bool(session.committed_text.strip()),
            )
        # Once a new state has produced text without usable logprobs, an older
        # segment score no longer describes the returned transcript.
        if (getattr(session.state, "text", "") or "").strip():
            return _scope_confidence_payload(
                current,
                has_committed_prefix=bool(session.committed_text.strip()),
            )
        return session.last_confidence or _scope_confidence_payload(
            current,
            has_committed_prefix=bool(session.committed_text.strip()),
        )

    def response_payload(
        session: Session,
        *,
        audio_rms: float = 0.0,
        audio_peak: float = 0.0,
        silence_skipped: bool = False,
    ) -> dict[str, object]:
        response: dict[str, object] = {
            "language": session.detected_language,
            "text": full_text(session),
            "audio_rms": audio_rms,
            "audio_peak": audio_peak,
            "silence_skipped": silence_skipped,
            "silence_skipped_chunks": session.silence_skipped_chunks,
            "vad": args.vad,
            "vad_speech_detected": session.vad_last_speech,
            "vad_skipped_chunks": session.vad_skipped_chunks,
        }
        response.update(confidence_payload(session))
        return response

    @app.get("/health")
    def health():
        return {
            "ok": True,
            "mode": "streaming",
            "confidence_logprobs": args.confidence_logprobs,
            "confidence_calibrated": args.confidence_calibrated,
            "silence_rms_threshold": args.silence_rms_threshold,
            "vad": args.vad,
            "vad_model": str(args.vad_model),
            "vad_threshold": args.vad_threshold,
            "vad_min_silence": args.vad_min_silence,
            "vad_min_speech": args.vad_min_speech,
            "vad_preroll": args.vad_preroll,
        }

    @app.post("/stream/start")
    def start():
        language = request.args.get("language") or None
        with model_lock:
            state = new_state(language)
            vad_gate = new_vad_gate()
        session_id = uuid.uuid4().hex
        with session_lock:
            sessions[session_id] = Session(
                state=state,
                last_seen=time.monotonic(),
                vad_gate=vad_gate,
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
        audio_rms = pcm_rms(pcm16k)
        audio_peak = pcm_peak(pcm16k)
        forwarded_pcm: np.ndarray | None = pcm16k
        vad_skipped = False
        with model_lock:
            # The browser may disconnect while this request waits behind a
            # long GPU call. Do not spend more GPU time on a session that was
            # cancelled in the meantime.
            if session.cancelled:
                return jsonify(error="session cancelled"), 409
            if session.vad_gate is not None:
                decision = session.vad_gate.accept(pcm16k)
                session.vad_last_speech = decision.speech_detected
                forwarded_pcm = decision.forward
                vad_skipped = forwarded_pcm is None
            elif args.silence_rms_threshold > 0 and is_silence(
                pcm16k, rms_threshold=args.silence_rms_threshold
            ):
                # RMS remains an explicit fallback when VAD is disabled.
                forwarded_pcm = None
                vad_skipped = True

            if forwarded_pcm is None or forwarded_pcm.size == 0:
                session.silence_skipped_chunks += 1
                if session.vad_gate is not None:
                    session.vad_skipped_chunks += 1
            else:
                asr_runner.streaming_transcribe(forwarded_pcm, session.state)
                session.segment_samples += forwarded_pcm.size
            segment_text = getattr(session.state, "text", "") or ""
            detected_language = getattr(session.state, "language", "") or ""
            if detected_language:
                session.detected_language = detected_language
            if forwarded_pcm is not None and _should_rotate_segment(
                segment_text,
                session.segment_samples,
                soft_limit_samples,
                hard_limit_samples,
            ):
                if getattr(session.state, "buffer", np.zeros(0)).size:
                    asr_runner.finish_streaming_transcribe(session.state)
                    segment_text = getattr(session.state, "text", "") or ""
                segment_confidence = _scope_confidence_payload(
                    state_confidence_payload(session.state),
                    has_committed_prefix=bool(session.committed_text.strip()),
                )
                session.committed_text = _join_transcripts(
                    session.committed_text, segment_text
                )
                session.last_confidence = segment_confidence
                session.state = new_state(session.language)
                session.segment_samples = 0
        return jsonify(
            response_payload(
                session,
                audio_rms=audio_rms,
                audio_peak=audio_peak,
                silence_skipped=vad_skipped,
            )
        )

    @app.post("/stream/finish")
    def finish():
        session_id = request.args.get("session_id", "")
        with session_lock:
            session = sessions.pop(session_id, None)
        if session is None:
            return jsonify(error="invalid session_id"), 400
        with model_lock:
            asr_runner.finish_streaming_transcribe(session.state)
            detected_language = getattr(session.state, "language", "") or ""
            if detected_language:
                session.detected_language = detected_language
        return jsonify(response_payload(session))

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
