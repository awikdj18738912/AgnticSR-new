"""Microphone AgenticASR client using a local Qwen3-ASR service.

Run ``system.qwen_asr_server`` in the qwen3-asr environment first. This client
runs in the agentic-asr environment and sends each VAD-delimited utterance to
that localhost service before applying the local Transformers Refiner.
"""

from __future__ import annotations

import argparse
import io
import json
import queue
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import wave
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

from .backends import EnergyVad, SAMPLE_RATE, VAD_WINDOW, normalize_cjk
from .entity_store import EntityStore
from .protection import EntityProtector
from .session_memory import SessionEntityMemory

SYSTEM_PROMPT = (
    "你是 ASR 文本纠错助手。保留原意，最小修改：去口癖/重复，修错字，补必要标点，"
    "规范数字、日期、术语和代码符号，处理自我修正。不要总结、扩写或解释。"
    "输入末尾的 <KEY>[词1、词2] 是已验证术语表；仅在原文已出现对应名称或别名时"
    "使用它来纠错或规范为标准名称，不得据此添加原文未提及的实体，也不要在输出中保留 <KEY>。"
)


class TransformersRefiner:
    """Local Transformers backend matching the offline Refiner prompt."""

    # Prevent a malformed or unusually long request from keeping the WebSocket
    # open forever. ``generate(max_time=...)`` returns the best partial output
    # available; the caller quality-checks it and falls back to source text
    # when it is incomplete.
    GENERATION_MAX_TIME_SECONDS = 12.0

    def __init__(self, model_path: Path, device_map: str, max_new_tokens: int) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "Refiner requires torch and transformers in the active environment"
            ) from error
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype="auto", device_map=device_map
        ).eval()
        self._max_new_tokens = max_new_tokens
        template = self._tokenizer.get_chat_template()
        self._template_kwargs = (
            {"enable_thinking": False} if "enable_thinking" in template else {}
        )

    def refine(
        self, raw_text: str, *, entity_hints: Iterable[str] = ()
    ) -> tuple[str, float]:
        hints = tuple(dict.fromkeys(item.strip() for item in entity_hints if item.strip()))
        user_content = raw_text
        if hints:
            user_content = f"{raw_text}\n<KEY>[{'、'.join(hints[:16])}]"
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        inputs = self._tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            **self._template_kwargs,
        )
        device = self._model.get_input_embeddings().weight.device
        inputs = inputs.to(device)
        started = time.perf_counter()
        with self._torch.inference_mode():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                max_time=self.GENERATION_MAX_TIME_SECONDS,
                do_sample=False
            )
        input_width = inputs["input_ids"].shape[1]
        text = self._tokenizer.decode(
            generated[0, input_width:], skip_special_tokens=True
        ).strip()
        return text, (time.perf_counter() - started) * 1000


def _sounddevice():
    try:
        import sounddevice as sd
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "Microphone input requires sounddevice and the system PortAudio library. "
            "On Ubuntu: `sudo apt install libportaudio2`"
        ) from error
    return sd


def _wav_payload(samples: list[np.ndarray]) -> tuple[bytes, float]:
    audio = np.concatenate(samples).astype("float32", copy=False)
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(SAMPLE_RATE)
        writer.writeframes(pcm.tobytes())
    return buffer.getvalue(), len(audio) / SAMPLE_RATE


def _transcribe(
    asr_url: str, audio: bytes, language: str | None, timeout: float
) -> tuple[str, str | None]:
    query = urllib.parse.urlencode({"language": language}) if language else ""
    separator = "&" if "?" in asr_url else "?"
    url = f"{asr_url}{separator}{query}" if query else asr_url
    request = urllib.request.Request(
        url, data=audio, headers={"Content-Type": "audio/wav"}, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as error:
        raise RuntimeError(f"Qwen3-ASR service request failed: {error}") from error
    text = payload.get("text")
    if not isinstance(text, str):
        raise RuntimeError(f"Qwen3-ASR returned no text: {payload}")
    detected_language = payload.get("language")
    return text, detected_language if isinstance(detected_language, str) else None


def _append_record(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Microphone AgenticASR via a local Qwen3-ASR service"
    )
    parser.add_argument("--refiner-model", type=Path)
    parser.add_argument("--refiner-device", default="cuda:1")
    parser.add_argument("--asr-url", default="http://127.0.0.1:8765/transcribe")
    parser.add_argument("--asr-timeout", type=float, default=120.0)
    parser.add_argument("--device-index", type=int)
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--language", default="Chinese", help="use auto for automatic detection")
    parser.add_argument("--vad-threshold", type=float, default=0.02)
    parser.add_argument("--vad-min-silence", type=float, default=0.7)
    parser.add_argument("--vad-min-speech", type=float, default=0.25)
    parser.add_argument("--preroll", type=float, default=0.5)
    parser.add_argument("--refiner-max-new-tokens", type=int, default=256)
    parser.add_argument("--output", type=Path, help="append utterance records as JSONL")
    parser.add_argument(
        "--entity-db",
        type=Path,
        help="optional SQLite database containing verified protected entities",
    )
    args = parser.parse_args(argv)
    if args.preroll < 0 or args.asr_timeout <= 0:
        parser.error("--preroll must be non-negative and --asr-timeout must be positive")
    if args.vad_min_silence <= 0 or args.vad_min_speech <= 0:
        parser.error("VAD durations must be positive")
    if args.refiner_max_new_tokens < 1:
        parser.error("--refiner-max-new-tokens must be at least 1")
    if not args.list_devices and args.refiner_model is None:
        parser.error("--refiner-model is required for transcription")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    sd = _sounddevice()
    if args.list_devices:
        print(sd.query_devices())
        return 0

    refiner_model_path = args.refiner_model.resolve()
    if not refiner_model_path.exists():
        raise FileNotFoundError(f"Refiner model not found: {refiner_model_path}")
    print("Loading AgenticASR Refiner...", flush=True)
    refiner = TransformersRefiner(
        refiner_model_path, args.refiner_device, args.refiner_max_new_tokens
    )
    memory = SessionEntityMemory()
    if args.entity_db is not None:
        entity_store = EntityStore(args.entity_db.resolve())
        entity_definitions = entity_store.list_entities()
        print(
            f"Loaded {len(entity_definitions)} protected entities from "
            f"{args.entity_db.resolve()}",
            flush=True,
        )
    else:
        entity_definitions = ()
    protector = EntityProtector(entity_definitions, session_memory=memory)
    print(f"Using Qwen3-ASR service at {args.asr_url}")
    print("Listening. Press Ctrl+C to stop.", flush=True)

    vad = EnergyVad(args.vad_threshold, args.vad_min_silence, args.vad_min_speech)
    windows: queue.Queue[np.ndarray] = queue.Queue()
    preroll = deque(maxlen=max(1, round(args.preroll * SAMPLE_RATE / VAD_WINDOW)))
    active = False
    utterance: list[np.ndarray] = []
    language = None if args.language.lower() == "auto" else args.language

    def transcribe_and_refine(samples: list[np.ndarray]) -> None:
        payload, duration = _wav_payload(samples)
        raw_text, detected_language = _transcribe(
            args.asr_url, payload, language, args.asr_timeout
        )
        raw_text = normalize_cjk(raw_text).strip()
        if not raw_text:
            return
        protection = protector.protect(raw_text)
        refined_text, latency_ms = refiner.refine(
            raw_text, entity_hints=protector.refinement_hints(protection)
        )
        refined_text, entity_normalizations = protector.normalize_verified_aliases(
            refined_text, protection
        )
        entity_audit_issues = protector.audit_unmasked(refined_text, protection)
        print(f"[raw] {raw_text}")
        print(f"[refined {latency_ms:.0f}ms] {refined_text}", flush=True)
        if entity_audit_issues:
            print(f"[entity audit] {', '.join(entity_audit_issues)}", flush=True)
        if args.output:
            _append_record(
                args.output.resolve(),
                {
                    "captured_at": datetime.now(timezone.utc).isoformat(),
                    "audio_seconds": round(duration, 3),
                    "asr_language": detected_language,
                    "output": {
                        "raw_text": raw_text,
                        "clean_text": refined_text,
                        "llm_latency_ms": latency_ms,
                        "refiner_accepted": True,
                        "refiner_reject_reasons": [],
                        "entity_audit_issues": list(entity_audit_issues),
                        "entity_normalizations": list(entity_normalizations),
                        "protected_entities": [
                            span.public_dict() for span in protection.spans
                        ],
                    },
                },
            )

    def callback(indata, frames, timing, status) -> None:  # noqa: ANN001, ARG001
        if status:
            print(status, file=sys.stderr)
        windows.put(indata[:, 0].copy())

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=VAD_WINDOW,
            device=args.device_index,
            callback=callback,
        ):
            while True:
                window = windows.get()
                vad.accept_waveform(window)
                speech = vad.is_speech_detected()
                if speech and not active:
                    active = True
                    utterance = list(preroll)
                if active:
                    utterance.append(window)
                preroll.append(window)
                if active and not speech:
                    transcribe_and_refine(utterance)
                    active = False
                    utterance = []
                while not vad.empty():
                    vad.pop()
    except KeyboardInterrupt:
        print()
        if active and utterance:
            transcribe_and_refine(utterance)
        print("Stopped.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
