"""Browser WebSocket UI for a pluggable streaming ASR backend and AgenticASR."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
import uvicorn

from .live_qwen_refiner import TransformersRefiner, _append_record

WEB_DIR = Path(__file__).resolve().parent / "web"


def _stream_request(
    asr_url: str,
    endpoint: str,
    session_id: str | None = None,
    data: bytes = b"",
    params: dict[str, str] | None = None,
) -> dict[str, object]:
    query_params = dict(params or {})
    if session_id:
        query_params["session_id"] = session_id
    query = urllib.parse.urlencode(query_params)
    url = f"{asr_url.rstrip('/')}{endpoint}"
    if query:
        url = f"{url}?{query}"
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/octet-stream"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as error:
        raise RuntimeError(f"streaming ASR request failed: {error}") from error
    if not isinstance(payload, dict) or payload.get("error"):
        raise RuntimeError(f"streaming ASR error: {payload}")
    return payload


def create_app(
    refiner_model: Path,
    refiner_device: str,
    asr_url: str,
    language: str | None,
    max_new_tokens: int,
    output: Path | None,
) -> FastAPI:
    app = FastAPI(docs_url=None, redoc_url=None)
    print("Loading AgenticASR Refiner...", flush=True)
    refiner = TransformersRefiner(refiner_model, refiner_device, max_new_tokens)
    refiner_lock = threading.Lock()

    def refine_update(raw_text: str, detected_language: str | None, final: bool) -> dict[str, object]:
        with refiner_lock:
            clean_text, latency_ms = refiner.refine(raw_text)
        return {
            "event": "final" if final else "update",
            "raw_text": raw_text,
            "clean_text": clean_text,
            "asr_language": detected_language,
            "refiner_latency_ms": round(latency_ms),
        }

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/health")
    def health() -> dict[str, bool]:
        return {"ok": True}

    @app.websocket("/ws/stream")
    async def stream(websocket: WebSocket) -> None:
        await websocket.accept()
        requested_language = websocket.query_params.get("language") or language
        session_id: str | None = None
        last_raw_text = ""
        finished = False
        try:
            start = _stream_request(
                asr_url,
                "/stream/start",
                params={"language": requested_language} if requested_language else None,
            )
            session_id = str(start["session_id"])
            await websocket.send_json({"event": "ready"})
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                text = message.get("text")
                if text:
                    command = json.loads(text)
                    if command.get("event") != "finish":
                        continue
                    payload = _stream_request(asr_url, "/stream/finish", session_id)
                    final_raw = str(payload.get("text", "")).strip()
                    detected_language = payload.get("language")
                    if final_raw:
                        result = refine_update(
                            final_raw,
                            detected_language if isinstance(detected_language, str) else None,
                            final=True,
                        )
                        if output is not None:
                            _append_record(
                                output,
                                {
                                    "captured_at": datetime.now(timezone.utc).isoformat(),
                                    "asr_language": result["asr_language"],
                                    "output": {
                                        "raw_text": result["raw_text"],
                                        "clean_text": result["clean_text"],
                                        "llm_latency_ms": result["refiner_latency_ms"],
                                    },
                                },
                            )
                        await websocket.send_json(result)
                    else:
                        await websocket.send_json({"event": "final"})
                    finished = True
                    break
                audio = message.get("bytes")
                if not audio:
                    continue
                payload = _stream_request(asr_url, "/stream/chunk", session_id, audio)
                raw_text = str(payload.get("text", "")).strip()
                if raw_text and raw_text != last_raw_text:
                    last_raw_text = raw_text
                    detected_language = payload.get("language")
                    await websocket.send_json(
                        refine_update(
                            raw_text,
                            detected_language if isinstance(detected_language, str) else None,
                            final=False,
                        )
                    )
        except (RuntimeError, ValueError, KeyError, json.JSONDecodeError) as error:
            await websocket.send_json({"event": "error", "detail": str(error)})
        except WebSocketDisconnect:
            pass
        finally:
            if session_id is not None and not finished:
                try:
                    _stream_request(asr_url, "/stream/finish", session_id)
                except RuntimeError:
                    pass

    return app


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the local AgenticASR streaming browser UI")
    parser.add_argument("--refiner-model", type=Path, required=True)
    parser.add_argument("--refiner-device", default="cuda:1")
    parser.add_argument("--asr-url", default="http://127.0.0.1:8766")
    parser.add_argument("--language", default="Chinese", help="use auto for automatic detection")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if not args.refiner_model.exists():
        parser.error(f"Refiner model not found: {args.refiner_model}")
    if args.max_new_tokens < 1 or not 1 <= args.port <= 65535:
        parser.error("--max-new-tokens must be positive and --port must be valid")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    app = create_app(
        args.refiner_model.resolve(),
        args.refiner_device,
        args.asr_url,
        None if args.language.lower() == "auto" else args.language,
        args.max_new_tokens,
        args.output.resolve() if args.output else None,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
