"""Browser WebSocket UI for a pluggable streaming ASR backend and AgenticASR."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import threading
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import uvicorn

from .live_qwen_refiner import TransformersRefiner, _append_record
from .entity_store import EntityDefinition, EntityStore
from .entity_matcher import EntityCandidateMatcher, EntityFuzzyMode
from .entity_pipeline import (
    finalize_entity_segment,
    has_placeholder_failure,
    has_retryable_integrity_failure,
    prepare_entity_segment,
)
from .protection import EntityProtector
from .refinement_guard import join_refined_segments, split_for_refinement
from .window_refinement import CumulativeWindowRefinement
from .session_memory import SessionEntityMemory

WEB_DIR = Path(__file__).resolve().parent / "web"
# Refining the complete cumulative hypothesis for every streaming chunk is
# expensive and can monopolize the single GPU-backed refiner when a long file
# is uploaded (especially if the browser has more than one open connection).
# Keep short early hypotheses responsive, then perform one complete refinement
# when the client sends ``finish``.
STREAMING_INTERMEDIATE_MAX_CHARS = 240
REFINER_LOCK_TIMEOUT_SECONDS = 30.0
FINAL_REFINEMENT_TIMEOUT_SECONDS = 30.0
# Long recordings are refined one bounded segment at a time.  Give the final
# pass a length-aware deadline while retaining a hard upper bound for broken
# model calls or disconnected clients.
FINAL_REFINEMENT_MAX_TIMEOUT_SECONDS = 300.0
WEBSOCKET_SEND_TIMEOUT_SECONDS = 10.0
ASR_CONTROL_REQUEST_TIMEOUT_SECONDS = 120.0
# Streaming recognition becomes more expensive as the active Qwen context
# grows.  A long recording can therefore have an occasional slow chunk even
# though the session is healthy.  Keep control requests bounded more tightly,
# but allow chunk processing to finish before the browser's watchdog fires.
ASR_CHUNK_REQUEST_TIMEOUT_SECONDS = 300.0
ASR_CHUNK_STATUS_INTERVAL_SECONDS = 10.0


class EntityInput(BaseModel):
    canonical_text: str
    entity_type: str = "TERM"
    domain: str = "general"
    normalization_policy: str = "preserve"
    priority: int = 0
    aliases: list[str] = Field(default_factory=list)


class EntityEnabledInput(BaseModel):
    enabled: bool


def _entity_dict(definition: EntityDefinition) -> dict[str, object]:
    return {
        "entity_id": definition.entity_id,
        "canonical_text": definition.canonical_text,
        "entity_type": definition.entity_type,
        "domain": definition.domain,
        "normalization_policy": definition.normalization_policy,
        "priority": definition.priority,
        "aliases": list(definition.aliases),
        "enabled": definition.enabled,
        "source": definition.source,
        "created_at": definition.created_at,
        "updated_at": definition.updated_at,
    }


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
        timeout = (
            ASR_CHUNK_REQUEST_TIMEOUT_SECONDS
            if endpoint == "/stream/chunk"
            else ASR_CONTROL_REQUEST_TIMEOUT_SECONDS
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError) as error:
        raise RuntimeError(f"streaming ASR request failed: {error}") from error
    if not isinstance(payload, dict) or payload.get("error"):
        raise RuntimeError(f"streaming ASR error: {payload}")
    return payload


async def _stream_request_async(
    asr_url: str,
    endpoint: str,
    session_id: str | None = None,
    data: bytes = b"",
    params: dict[str, str] | None = None,
) -> dict[str, object]:
    """Run the blocking ASR HTTP adapter without stopping the WebSocket loop."""

    return await asyncio.to_thread(
        _stream_request,
        asr_url,
        endpoint,
        session_id,
        data,
        params,
    )


def create_app(
    refiner_model: Path,
    refiner_device: str,
    asr_url: str,
    language: str | None,
    max_new_tokens: int,
    output: Path | None,
    entity_db: Path | None = None,
    entity_fuzzy_mode: str = "shadow",
) -> FastAPI:
    app = FastAPI(docs_url=None, redoc_url=None)
    print("Loading AgenticASR Refiner...", flush=True)
    refiner = TransformersRefiner(refiner_model, refiner_device, max_new_tokens)
    refiner_lock = threading.Lock()
    entity_store = EntityStore(entity_db) if entity_db is not None else None
    fuzzy_mode = EntityFuzzyMode.parse(entity_fuzzy_mode)

    def managed_entity_store() -> EntityStore:
        if entity_store is None:
            raise HTTPException(
                status_code=409,
                detail="实体数据库未启用；请使用 --entity-db 启动服务。",
            )
        return entity_store

    def refine_update(
        raw_text: str,
        detected_language: str | None,
        final: bool,
        protector: EntityProtector,
        asr_confidence: float | None,
        matcher: EntityCandidateMatcher | None = None,
        *, single_window: bool = False,
    ) -> dict[str, object]:
        clean_parts: list[str] = []
        protected_entities: list[dict[str, str]] = []
        entity_hints: list[str] = []
        entity_normalizations: list[dict[str, str]] = []
        entity_audit_issues: list[str] = []
        entity_candidates: list[dict[str, object]] = []
        refiner_reject_reasons: list[str] = []
        refiner_masked_outputs: list[str] = []
        placeholder_retry_count = 0
        refiner_retry_count = 0
        refiner_retry_reasons: list[str] = []
        total_latency_ms = 0.0
        matcher_latency_ms = 0.0
        refiner_available = True
        segments = (raw_text,) if single_window and raw_text else split_for_refinement(raw_text)
        for segment_index, segment in enumerate(segments):
            prepared = prepare_entity_segment(
                segment,
                protector,
                matcher,
                allow_auto=final,
                confidence=asr_confidence,
            )
            protection = prepared.protection
            hints = prepared.hints
            # A stale browser connection must not be able to block the final
            # result forever.  After one timeout, preserve all remaining source
            # segments and report the quality fallback in the response.
            lock_acquired = (
                refiner_available
                and refiner_lock.acquire(timeout=REFINER_LOCK_TIMEOUT_SECONDS)
            )
            if not lock_acquired:
                refiner_available = False
                clean_segment = prepared.baseline_text
                changes = list(prepared.normalizations)
                refiner_reject_reasons.append(
                    f"segment_{segment_index + 1}:refiner_busy"
                )
            else:
                try:
                    refined_candidate, latency_ms = refiner.refine(
                        protection.masked_text, entity_hints=hints
                    )
                    total_latency_ms += latency_ms
                    refiner_masked_outputs.append(refined_candidate)
                    finalized = finalize_entity_segment(
                        refined_candidate, prepared, protector
                    )
                    if has_retryable_integrity_failure(finalized.reject_reasons):
                        initial_reasons = finalized.reject_reasons
                        retry_candidate, retry_latency_ms = refiner.refine(
                            protection.masked_text,
                            entity_hints=hints,
                            strict_placeholders=True,
                        )
                        refiner_retry_count += 1
                        if has_placeholder_failure(initial_reasons):
                            placeholder_retry_count += 1
                        refiner_retry_reasons.extend(
                            f"segment_{segment_index + 1}:{reason}"
                            for reason in initial_reasons
                        )
                        total_latency_ms += retry_latency_ms
                        refiner_masked_outputs.append(retry_candidate)
                        finalized = finalize_entity_segment(
                            retry_candidate, prepared, protector
                        )
                finally:
                    refiner_lock.release()
                clean_segment = finalized.text
                changes = list(prepared.normalizations)
                if finalized.reject_reasons:
                    refiner_reject_reasons.extend(
                        f"segment_{segment_index + 1}:{reason}"
                        for reason in finalized.reject_reasons
                    )
            clean_parts.append(clean_segment)
            protected_entities.extend(span.public_dict() for span in protection.spans)
            entity_hints.extend(hints)
            entity_normalizations.extend(changes)
            entity_candidates.extend(
                match.public_dict() for match in prepared.report.matches
            )
            matcher_latency_ms += prepared.report.matcher_latency_ms
            entity_audit_issues.extend(protector.audit_unmasked(clean_segment, protection))
        clean_text = join_refined_segments(clean_parts)
        return {
            "event": "final" if final else "update",
            "raw_text": raw_text,
            "clean_text": clean_text,
            "asr_language": detected_language,
            "asr_confidence": asr_confidence,
            "refiner_latency_ms": round(total_latency_ms),
            "refiner_accepted": not refiner_reject_reasons,
            "refiner_reject_reasons": refiner_reject_reasons,
            "placeholder_retry_count": placeholder_retry_count,
            "refiner_retry_count": refiner_retry_count,
            "refiner_retry_reasons": refiner_retry_reasons,
            "refiner_masked_outputs": refiner_masked_outputs,
            "entity_audit_issues": list(dict.fromkeys(entity_audit_issues)),
            "entity_refinement_hints": list(dict.fromkeys(entity_hints)),
            "entity_normalizations": entity_normalizations,
            "protected_entities": protected_entities,
            "entity_candidates": entity_candidates,
            "entity_matcher_latency_ms": round(matcher_latency_ms, 3),
            "entity_fuzzy_mode": fuzzy_mode.value,
            "entity_matching_config_version": (
                matcher.config.version if matcher is not None else None
            ),
        }

    def fallback_update(
        raw_text: str,
        detected_language: str | None,
        protector: EntityProtector,
        asr_confidence: float | None,
        matcher: EntityCandidateMatcher | None,
        reason: str,
        started_at: float,
    ) -> dict[str, object]:
        # Preserve the complete ASR text and still apply deterministic entity
        # normalization when the neural Refiner exceeds the final deadline.
        clean_parts: list[str] = []
        protected_entities: list[dict[str, str]] = []
        entity_hints: list[str] = []
        entity_normalizations: list[dict[str, str]] = []
        entity_audit_issues: list[str] = []
        entity_candidates: list[dict[str, object]] = []
        matcher_latency_ms = 0.0
        for segment in split_for_refinement(raw_text):
            prepared = prepare_entity_segment(
                segment,
                protector,
                matcher,
                allow_auto=True,
                confidence=asr_confidence,
            )
            protection = prepared.protection
            clean_parts.append(prepared.baseline_text)
            protected_entities.extend(span.public_dict() for span in protection.spans)
            entity_hints.extend(prepared.hints)
            entity_normalizations.extend(prepared.normalizations)
            entity_candidates.extend(
                match.public_dict() for match in prepared.report.matches
            )
            matcher_latency_ms += prepared.report.matcher_latency_ms
            entity_audit_issues.extend(
                protector.audit_unmasked(prepared.baseline_text, protection)
            )
        return {
            "event": "final",
            "raw_text": raw_text,
            "clean_text": join_refined_segments(clean_parts),
            "asr_language": detected_language,
            "asr_confidence": asr_confidence,
            "refiner_latency_ms": round((time.perf_counter() - started_at) * 1000),
            "refiner_accepted": False,
            "refiner_reject_reasons": [reason],
            "placeholder_retry_count": 0,
            "refiner_retry_count": 0,
            "refiner_retry_reasons": [],
            "refiner_masked_outputs": [],
            "entity_audit_issues": list(dict.fromkeys(entity_audit_issues)),
            "entity_refinement_hints": list(dict.fromkeys(entity_hints)),
            "entity_normalizations": entity_normalizations,
            "protected_entities": protected_entities,
            "entity_candidates": entity_candidates,
            "entity_matcher_latency_ms": round(matcher_latency_ms, 3),
            "entity_fuzzy_mode": fuzzy_mode.value,
            "entity_matching_config_version": (
                matcher.config.version if matcher is not None else None
            ),
        }

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(
            WEB_DIR / "index.html",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/health")
    def health() -> dict[str, bool]:
        return {"ok": True, "entity_db": entity_store is not None}

    @app.get("/api/entities")
    def list_entities(
        query: str = "",
        entity_type: str | None = None,
        include_disabled: bool = True,
    ) -> dict[str, object]:
        definitions = managed_entity_store().list_entities(include_disabled=include_disabled)
        keyword = query.strip().casefold()
        type_filter = entity_type.strip().upper() if entity_type and entity_type.strip() else None
        filtered = [
            definition
            for definition in definitions
            if (
                not keyword
                or keyword in definition.canonical_text.casefold()
                or any(keyword in alias.casefold() for alias in definition.aliases)
            )
            and (type_filter is None or definition.entity_type == type_filter)
        ]
        return {"entities": [_entity_dict(definition) for definition in filtered]}

    @app.post("/api/entities", status_code=201)
    def create_entity(payload: EntityInput) -> dict[str, object]:
        store = managed_entity_store()
        try:
            store.get_entity(payload.canonical_text.strip(), payload.domain.strip())
        except KeyError:
            pass
        else:
            raise HTTPException(
                status_code=409,
                detail="该标准名称与领域组合已存在，请使用编辑操作。",
            )
        try:
            definition = store.upsert_entity(
                payload.canonical_text,
                entity_type=payload.entity_type,
                domain=payload.domain,
                normalization_policy=payload.normalization_policy,
                priority=payload.priority,
                aliases=payload.aliases,
                source="ui",
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return _entity_dict(definition)

    @app.put("/api/entities/{entity_id}")
    def update_entity(entity_id: int, payload: EntityInput) -> dict[str, object]:
        try:
            definition = managed_entity_store().update_entity(
                entity_id,
                payload.canonical_text,
                entity_type=payload.entity_type,
                domain=payload.domain,
                normalization_policy=payload.normalization_policy,
                priority=payload.priority,
                aliases=payload.aliases,
                source="ui",
            )
        except KeyError as error:
            raise HTTPException(status_code=404, detail="实体不存在或已被删除。") from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return _entity_dict(definition)

    @app.patch("/api/entities/{entity_id}/enabled")
    def set_entity_enabled(entity_id: int, payload: EntityEnabledInput) -> dict[str, object]:
        store = managed_entity_store()
        if not store.set_enabled_by_id(entity_id, enabled=payload.enabled):
            raise HTTPException(status_code=404, detail="实体不存在或已被删除。")
        return _entity_dict(store.get_entity_by_id(entity_id))

    @app.delete("/api/entities/{entity_id}")
    def delete_entity(entity_id: int) -> dict[str, bool]:
        if not managed_entity_store().delete_entity(entity_id):
            raise HTTPException(status_code=404, detail="实体不存在或已被删除。")
        return {"deleted": True}

    @app.websocket("/ws/stream")
    async def stream(websocket: WebSocket) -> None:
        await websocket.accept()
        send_lock = asyncio.Lock()
        connection_open = True

        async def send_json(payload: dict[str, object]) -> bool:
            nonlocal connection_open
            if not connection_open:
                return False
            async with send_lock:
                if not connection_open:
                    return False
                try:
                    await asyncio.wait_for(
                        websocket.send_json(payload),
                        timeout=WEBSOCKET_SEND_TIMEOUT_SECONDS,
                    )
                except (RuntimeError, WebSocketDisconnect, asyncio.TimeoutError):
                    connection_open = False
                    return False
            return True

        requested_language = websocket.query_params.get("language") or language
        requested_mode = (websocket.query_params.get("mode") or "online").lower()
        requested_domain = (websocket.query_params.get("domain") or "general").strip() or "general"
        if requested_mode not in {"online", "offline", "streaming"}:
            await send_json(
                {"event": "error", "detail": "mode must be online, offline, or streaming"}
            )
            await websocket.close()
            return
        session_id: str | None = None
        last_raw_text = ""
        finished = False
        received_chunk_count = 0
        session_memory = SessionEntityMemory()
        definitions = (
            entity_store.list_entities(domain=requested_domain)
            if entity_store is not None
            else ()
        )
        protector = EntityProtector(definitions, session_memory=session_memory)
        matcher = (
            EntityCandidateMatcher(
                definitions,
                selected_domain=requested_domain,
                mode=fuzzy_mode,
            )
            if definitions and fuzzy_mode is not EntityFuzzyMode.OFF
            else None
        )
        pending_refinement: tuple[
            str, str | None, float | None, str, int
        ] | None = None
        streaming_refiner_task: asyncio.Task[None] | None = None
        streaming_finish_requested = False
        window_refinement = CumulativeWindowRefinement(refine_update)

        def finalize_streaming_windows(
            raw_text: str,
            detected_language: str | None,
            final: bool,
            protector: EntityProtector,
            asr_confidence: float | None,
            matcher_value: EntityCandidateMatcher | None,
        ) -> dict[str, object]:
            """Reuse committed streaming refinements and finalize only the tail.

            Intermediate windows deliberately disable automatic fuzzy entity
            normalization because cumulative ASR hypotheses can still change.
            Run the inexpensive entity-only fallback over the cached clean text
            at finalization so reusing Refiner output does not regress final
            entity replacement.
            """

            if not window_refinement.has_cached_result:
                return refine_update(
                    raw_text,
                    detected_language,
                    final,
                    protector,
                    asr_confidence,
                    matcher_value,
                )
            result = window_refinement.update(
                raw_text,
                detected_language,
                final,
                protector,
                asr_confidence,
                matcher_value,
            )
            entity_result = fallback_update(
                str(result["clean_text"]),
                detected_language,
                protector,
                asr_confidence,
                matcher_value,
                "streaming_final_entity_normalization",
                time.perf_counter(),
            )
            result["clean_text"] = entity_result["clean_text"]
            result["entity_audit_issues"] = entity_result["entity_audit_issues"]
            result["entity_refinement_hints"] = entity_result[
                "entity_refinement_hints"
            ]
            result["entity_normalizations"] = [
                *result.get("entity_normalizations", []),
                *entity_result["entity_normalizations"],
            ]
            result["protected_entities"] = entity_result["protected_entities"]
            result["entity_candidates"] = entity_result["entity_candidates"]
            result["entity_matcher_latency_ms"] = round(
                float(result.get("entity_matcher_latency_ms", 0.0))
                + float(entity_result["entity_matcher_latency_ms"]),
                3,
            )
            return result

        async def process_audio_chunk(audio: bytes) -> tuple[dict[str, object], int]:
            """Forward one chunk while keeping a slow, healthy session visible."""

            nonlocal received_chunk_count
            received_chunk_count += 1
            chunk_index = received_chunk_count
            started_at = time.perf_counter()
            task = asyncio.create_task(
                _stream_request_async(asr_url, "/stream/chunk", session_id, audio)
            )
            while True:
                try:
                    payload = await asyncio.wait_for(
                        asyncio.shield(task),
                        timeout=ASR_CHUNK_STATUS_INTERVAL_SECONDS,
                    )
                    return payload, chunk_index
                except asyncio.TimeoutError:
                    if not await send_json(
                        {
                            "event": "status",
                            "stage": "asr_chunk",
                            "chunk_index": chunk_index,
                            "elapsed_seconds": round(
                                time.perf_counter() - started_at
                            ),
                        }
                    ):
                        raise RuntimeError(
                            "client disconnected while ASR chunk was processing"
                        )

        async def run_streaming_refiner() -> None:
            nonlocal pending_refinement, streaming_finish_requested
            while pending_refinement is not None:
                # Refine only a bounded tail of the cumulative ASR hypothesis.
                # Committed results remain cached for finalization; stale UI
                # updates are discarded once ``finish`` is requested.
                if streaming_finish_requested:
                    pending_refinement = None
                    return
                (
                    tail_value,
                    language_value,
                    confidence_value,
                    full_raw_value,
                    tail_start,
                ) = pending_refinement
                pending_refinement = None
                try:
                    result = await asyncio.to_thread(
                        window_refinement.update,
                        full_raw_value,
                        language_value,
                        False,
                        protector,
                        confidence_value,
                        matcher,
                    )
                    if streaming_finish_requested:
                        return
                    result["raw_text"] = full_raw_value
                    result["event"] = "update"
                    if not await send_json(result):
                        return
                except Exception as error:
                    await send_json({"event": "error", "detail": str(error)})
                if streaming_finish_requested:
                    pending_refinement = None
                    return

        def queue_streaming_refinement(
            tail_value: str,
            language_value: str | None,
            confidence_value: float | None,
            full_raw_value: str,
            tail_start: int,
        ) -> None:
            nonlocal pending_refinement, streaming_refiner_task
            pending_refinement = (
                tail_value,
                language_value,
                confidence_value,
                full_raw_value,
                tail_start,
            )
            if streaming_refiner_task is None or streaming_refiner_task.done():
                streaming_refiner_task = asyncio.create_task(run_streaming_refiner())
        async def refine_final(
            raw_text: str,
            detected_language: str | None,
            protector: EntityProtector,
            asr_confidence: float | None,
        ) -> dict[str, object]:
            started_at = time.perf_counter()
            finalizer = (
                finalize_streaming_windows
                if requested_mode in {"offline", "streaming"}
                else refine_update
            )
            task = asyncio.create_task(
                asyncio.to_thread(
                    finalizer,
                    raw_text,
                    detected_language,
                    True,
                    protector,
                    asr_confidence,
                    matcher,
                )
            )
            loop = asyncio.get_running_loop()
            segment_count = max(1, len(split_for_refinement(raw_text)))
            deadline_seconds = min(
                FINAL_REFINEMENT_MAX_TIMEOUT_SECONDS,
                max(
                    FINAL_REFINEMENT_TIMEOUT_SECONDS,
                    15.0 + segment_count * 8.0,
                ),
            )
            deadline = loop.time() + deadline_seconds
            while True:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    task.cancel()
                    raise asyncio.TimeoutError
                try:
                    return await asyncio.wait_for(
                        asyncio.shield(task), timeout=min(8.0, remaining)
                    )
                except asyncio.TimeoutError:
                    # Keep reverse proxies and the browser informed while a
                    # long transcript is being refined.
                    await send_json({
                        "event": "status",
                        "stage": "final_refinement",
                        "elapsed_seconds": round(time.perf_counter() - started_at),
                    })

        try:
            start = await _stream_request_async(
                asr_url,
                "/stream/start",
                params={"language": requested_language} if requested_language else None,
            )
            session_id = str(start["session_id"])
            await send_json({"event": "ready"})
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    connection_open = False
                    break
                text = message.get("text")
                if text:
                    command = json.loads(text)
                    if command.get("event") != "finish":
                        continue
                    if requested_mode in {"online", "offline", "streaming"}:
                        # Mark intermediate output stale immediately. Do not
                        # wait here: a slow partial generation used to delay
                        # the final ASR result and made the refinement panel
                        # appear frozen. Its worker is bounded by the Refiner
                        # generation timeout and will discard its result once
                        # it observes this flag.
                        streaming_finish_requested = True
                        pending_refinement = None
                    payload = await _stream_request_async(
                        asr_url, "/stream/finish", session_id
                    )
                    final_raw = str(payload.get("text", "")).strip()
                    detected_language = payload.get("language")
                    asr_confidence = _optional_confidence(payload.get("confidence"))
                    if final_raw:
                        # Publish the complete ASR result before starting the
                        # potentially slow neural pass. This keeps the raw
                        # transcript available even while refinement is busy.
                        if not await send_json(
                            {
                                "event": "transcript",
                                "raw_text": final_raw,
                                "asr_language": (
                                    detected_language
                                    if isinstance(detected_language, str)
                                    else None
                                ),
                                "asr_confidence": asr_confidence,
                                "refiner_deferred": True,
                            }
                        ):
                            finished = True
                            break
                        refinement_started = time.perf_counter()
                        try:
                            result = await refine_final(
                                final_raw,
                                detected_language if isinstance(detected_language, str) else None,
                                protector,
                                asr_confidence,
                            )
                        except asyncio.TimeoutError:
                            result = fallback_update(
                                final_raw,
                                detected_language if isinstance(detected_language, str) else None,
                                protector,
                                asr_confidence,
                                matcher,
                                "final_refinement_timeout",
                                refinement_started,
                            )
                        except Exception as error:
                            # A model/runtime failure must not erase a
                            # complete ASR result that was already published.
                            result = fallback_update(
                                final_raw,
                                detected_language if isinstance(detected_language, str) else None,
                                protector,
                                asr_confidence,
                                matcher,
                                f"final_refinement_error:{type(error).__name__}",
                                refinement_started,
                            )
                        if output is not None:
                            _append_record(
                                output,
                                {
                                    "captured_at": datetime.now(timezone.utc).isoformat(),
                                    "mode": requested_mode,
                                    "entity_domain": requested_domain,
                                    "asr_language": result["asr_language"],
                                    "asr_confidence": result["asr_confidence"],
                                    "output": {
                                        "raw_text": result["raw_text"],
                                        "clean_text": result["clean_text"],
                                        "llm_latency_ms": result["refiner_latency_ms"],
                                        "refiner_accepted": result["refiner_accepted"],
                                        "refiner_reject_reasons": result[
                                            "refiner_reject_reasons"
                                        ],
                                        "placeholder_retry_count": result[
                                            "placeholder_retry_count"
                                        ],
                                        "refiner_retry_count": result[
                                            "refiner_retry_count"
                                        ],
                                        "refiner_retry_reasons": result[
                                            "refiner_retry_reasons"
                                        ],
                                        "refiner_masked_outputs": result[
                                            "refiner_masked_outputs"
                                        ],
                                        "entity_audit_issues": result[
                                            "entity_audit_issues"
                                        ],
                                        "entity_refinement_hints": result[
                                            "entity_refinement_hints"
                                        ],
                                        "entity_normalizations": result[
                                            "entity_normalizations"
                                        ],
                                        "protected_entities": result[
                                            "protected_entities"
                                        ],
                                        "entity_candidates": result[
                                            "entity_candidates"
                                        ],
                                        "entity_matcher_latency_ms": result[
                                            "entity_matcher_latency_ms"
                                        ],
                                        "entity_fuzzy_mode": result[
                                            "entity_fuzzy_mode"
                                        ],
                                        "entity_matching_config_version": result[
                                            "entity_matching_config_version"
                                        ],
                                    },
                                },
                            )
                        await send_json(result)
                    else:
                        await send_json({"event": "final"})
                    finished = True
                    break
                audio = message.get("bytes")
                if not audio:
                    continue
                if requested_mode == "offline":
                    payload, chunk_index = await process_audio_chunk(audio)
                    if not await send_json(
                        {"event": "chunk_ack", "chunk_index": chunk_index}
                    ):
                        break
                    raw_text = str(payload.get("text", "")).strip()
                    if raw_text and raw_text != last_raw_text:
                        last_raw_text = raw_text
                        detected_language = payload.get("language")
                        asr_confidence = _optional_confidence(
                            payload.get("confidence")
                        )
                        language_value = (
                            detected_language
                            if isinstance(detected_language, str)
                            else None
                        )
                        if not await send_json(
                            {
                                "event": "transcript",
                                "raw_text": raw_text,
                                "asr_language": language_value,
                                "asr_confidence": asr_confidence,
                                "refiner_deferred": False,
                            }
                        ):
                            break
                        tail_start = max(
                            0, len(raw_text) - STREAMING_INTERMEDIATE_MAX_CHARS
                        )
                        queue_streaming_refinement(
                            raw_text[tail_start:],
                            language_value,
                            asr_confidence,
                            raw_text,
                            tail_start,
                        )
                    continue
                payload, chunk_index = await process_audio_chunk(audio)
                if requested_mode == "streaming":
                    await send_json(
                        {"event": "chunk_ack", "chunk_index": chunk_index}
                    )
                raw_text = str(payload.get("text", "")).strip()
                if raw_text and raw_text != last_raw_text:
                    last_raw_text = raw_text
                    detected_language = payload.get("language")
                    asr_confidence = _optional_confidence(payload.get("confidence"))
                    language_value = (
                        detected_language if isinstance(detected_language, str) else None
                    )
                    if requested_mode in {"online", "streaming"}:
                        if not await send_json(
                            {
                                "event": "transcript",
                                "raw_text": raw_text,
                                "asr_language": language_value,
                                "asr_confidence": asr_confidence,
                                "refiner_deferred": False,
                            }
                        ):
                            break
                        tail_start = max(
                            0, len(raw_text) - STREAMING_INTERMEDIATE_MAX_CHARS
                        )
                        queue_streaming_refinement(
                            raw_text[tail_start:],
                            language_value,
                            asr_confidence,
                            raw_text,
                            tail_start,
                        )
                    else:
                        if not await send_json(
                            await asyncio.to_thread(
                                refine_update,
                                raw_text,
                                language_value,
                                False,
                                protector,
                                asr_confidence,
                                matcher,
                            )
                        ):
                            break
        except (RuntimeError, ValueError, KeyError, json.JSONDecodeError) as error:
            await send_json({"event": "error", "detail": str(error)})
        except WebSocketDisconnect:
            pass
        except Exception as error:
            await send_json({"event": "error", "detail": f"后台处理失败：{error}"})
        finally:
            connection_open = False
            if streaming_refiner_task is not None and not streaming_refiner_task.done():
                streaming_refiner_task.cancel()
            if session_id is not None and not finished:
                try:
                    await _stream_request_async(asr_url, "/stream/cancel", session_id)
                except (RuntimeError, TimeoutError):
                    pass

    return app


def _optional_confidence(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    confidence = float(value)
    return confidence if 0 <= confidence <= 1 else None


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
    parser.add_argument(
        "--entity-db",
        type=Path,
        help="optional SQLite database containing verified protected entities",
    )
    parser.add_argument(
        "--entity-fuzzy-mode",
        choices=[mode.value for mode in EntityFuzzyMode],
        default="shadow",
        help="off, shadow, hint, or final-only automatic entity normalization",
    )
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
        args.entity_db.resolve() if args.entity_db else None,
        args.entity_fuzzy_mode,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
