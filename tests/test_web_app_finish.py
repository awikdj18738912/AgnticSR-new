from __future__ import annotations

import time
import tempfile
from pathlib import Path

import unittest
from unittest.mock import patch

try:
    from fastapi.testclient import TestClient

    import system.web_app as web_app
    from system.entity_store import EntityStore
except ModuleNotFoundError:  # The base data-pipeline environment omits FastAPI.
    TestClient = None
    web_app = None


class _SlowRefiner:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def refine(
        self,
        text: str,
        *,
        entity_hints: tuple[str, ...] = (),
        strict_placeholders: bool = False,
    ) -> tuple[str, float]:
        time.sleep(0.1)
        return text.replace("原始", "精修"), 100.0


class _FailingRefiner:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def refine(
        self,
        text: str,
        *,
        entity_hints: tuple[str, ...] = (),
        strict_placeholders: bool = False,
    ) -> tuple[str, float]:
        raise RuntimeError("simulated model failure")


class _IdentityRefiner:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def refine(
        self,
        text: str,
        *,
        entity_hints: tuple[str, ...] = (),
        strict_placeholders: bool = False,
    ) -> tuple[str, float]:
        return text, 1.0


class _CountingRefiner:
    calls: list[str] = []

    def __init__(self, *args, **kwargs) -> None:
        type(self).calls = []

    def refine(
        self,
        text: str,
        *,
        entity_hints: tuple[str, ...] = (),
        strict_placeholders: bool = False,
    ) -> tuple[str, float]:
        type(self).calls.append(text)
        return text.replace("原始", "精修"), 1.0


class _DropsPlaceholderOnceRefiner:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def refine(
        self,
        text: str,
        *,
        entity_hints: tuple[str, ...] = (),
        strict_placeholders: bool = False,
    ) -> tuple[str, float]:
        if strict_placeholders:
            return text, 2.0
        return text.replace("__ENTITY_000__", "", 1), 1.0


class _CompressesOnceRefiner:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def refine(
        self,
        text: str,
        *,
        entity_hints: tuple[str, ...] = (),
        strict_placeholders: bool = False,
    ) -> tuple[str, float]:
        return (text, 2.0) if strict_placeholders else ("你怎么拼？", 1.0)


def _fake_stream_request(
    asr_url: str,
    endpoint: str,
    session_id: str | None = None,
    data: bytes = b"",
    params: dict[str, str] | None = None,
) -> dict[str, object]:
    if endpoint == "/stream/start":
        return {"session_id": "test-session"}
    if endpoint == "/stream/chunk":
        return {"text": "中间原始文本。", "language": "Chinese"}
    if endpoint == "/stream/finish":
        return {"text": "最终原始文本。", "language": "Chinese"}
    if endpoint == "/stream/cancel":
        return {"cancelled": True}
    raise AssertionError(f"unexpected endpoint: {endpoint}")


def _slow_chunk_stream_request(
    asr_url: str,
    endpoint: str,
    session_id: str | None = None,
    data: bytes = b"",
    params: dict[str, str] | None = None,
) -> dict[str, object]:
    if endpoint == "/stream/chunk":
        time.sleep(0.03)
    return _fake_stream_request(asr_url, endpoint, session_id, data, params)


def _stable_stream_request(
    asr_url: str,
    endpoint: str,
    session_id: str | None = None,
    data: bytes = b"",
    params: dict[str, str] | None = None,
) -> dict[str, object]:
    if endpoint == "/stream/start":
        return {"session_id": "stable-test-session"}
    if endpoint in {"/stream/chunk", "/stream/finish"}:
        return {"text": "第一段原始文本。第二段原始文本。", "language": "Chinese"}
    if endpoint == "/stream/cancel":
        return {"cancelled": True}
    raise AssertionError(f"unexpected endpoint: {endpoint}")


def _entity_stream_request(
    asr_url: str,
    endpoint: str,
    session_id: str | None = None,
    data: bytes = b"",
    params: dict[str, str] | None = None,
) -> dict[str, object]:
    if endpoint == "/stream/start":
        return {"session_id": "entity-test-session"}
    if endpoint in {"/stream/chunk", "/stream/finish"}:
        return {"text": "鬼灵们是这个副本的入口。", "language": "Chinese"}
    if endpoint == "/stream/cancel":
        return {"cancelled": True}
    raise AssertionError(f"unexpected endpoint: {endpoint}")


_LONG_COMPLETE_TRANSCRIPT = (
    "怎么可能？你竟结成了元婴？此青火杖乃墨家的不夜之杖。"
    "你这手段怎么比我还像我道？我记住你了。你认识这位道友？"
    "此人就是我与你说过的那个黄风谷姓韩的，放肆！"
    "韩道友已经是元婴修士，又岂会和你一般见识？"
)


def _content_loss_stream_request(
    asr_url: str,
    endpoint: str,
    session_id: str | None = None,
    data: bytes = b"",
    params: dict[str, str] | None = None,
) -> dict[str, object]:
    if endpoint == "/stream/start":
        return {"session_id": "content-loss-test-session"}
    if endpoint == "/stream/finish":
        return {"text": _LONG_COMPLETE_TRANSCRIPT, "language": "Chinese"}
    if endpoint == "/stream/cancel":
        return {"cancelled": True}
    raise AssertionError(f"unexpected endpoint: {endpoint}")


@unittest.skipIf(web_app is None, "FastAPI is not installed")
class WebAppFinishTest(unittest.TestCase):
    def test_offline_mode_reuses_completed_window_refinement(self) -> None:
        with (
            patch.object(web_app, "TransformersRefiner", _CountingRefiner),
            patch.object(web_app, "_stream_request", _stable_stream_request),
        ):
            app = web_app.create_app(
                Path("/tmp/fake-refiner"),
                "cpu",
                "http://fake-asr",
                "Chinese",
                32,
                None,
            )
            with TestClient(app) as client:
                with client.websocket_connect(
                    "/ws/stream?mode=offline"
                ) as websocket:
                    self.assertEqual(websocket.receive_json()["event"], "ready")
                    websocket.send_bytes(b"pcm")
                    message = websocket.receive_json()
                    while message["event"] != "update":
                        message = websocket.receive_json()
                    calls_before_finish = len(_CountingRefiner.calls)
                    self.assertGreater(calls_before_finish, 0)

                    websocket.send_json({"event": "finish"})
                    self.assertEqual(websocket.receive_json()["event"], "transcript")
                    final = websocket.receive_json()

                    self.assertEqual(final["event"], "final")
                    self.assertEqual(
                        final["clean_text"],
                        "第一段精修文本。第二段精修文本。",
                    )
                    self.assertEqual(len(_CountingRefiner.calls), calls_before_finish)

    def test_streaming_cached_result_still_applies_final_fuzzy_entity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            entity_db = Path(directory) / "entities.db"
            EntityStore(entity_db).upsert_entity(
                "鬼灵门",
                entity_type="TERM",
                normalization_policy="normalize",
            )
            with (
                patch.object(web_app, "TransformersRefiner", _IdentityRefiner),
                patch.object(web_app, "_stream_request", _entity_stream_request),
            ):
                app = web_app.create_app(
                    Path("/tmp/fake-refiner"),
                    "cpu",
                    "http://fake-asr",
                    "Chinese",
                    32,
                    None,
                    entity_db,
                    "auto",
                )
                with TestClient(app) as client:
                    with client.websocket_connect(
                        "/ws/stream?mode=streaming&domain=general"
                    ) as websocket:
                        self.assertEqual(websocket.receive_json()["event"], "ready")
                        websocket.send_bytes(b"pcm")
                        message = websocket.receive_json()
                        while message["event"] != "update":
                            message = websocket.receive_json()
                        self.assertEqual(
                            message["clean_text"], "鬼灵们是这个副本的入口。"
                        )

                        websocket.send_json({"event": "finish"})
                        self.assertEqual(websocket.receive_json()["event"], "transcript")
                        final = websocket.receive_json()

                        self.assertEqual(final["clean_text"], "鬼灵门是这个副本的入口。")
                        self.assertEqual(
                            final["entity_candidates"][0]["decision"],
                            "AUTO_NORMALIZE",
                        )

    def test_streaming_final_reuses_completed_window_refinement(self) -> None:
        with (
            patch.object(web_app, "TransformersRefiner", _CountingRefiner),
            patch.object(web_app, "_stream_request", _stable_stream_request),
        ):
            app = web_app.create_app(
                Path("/tmp/fake-refiner"),
                "cpu",
                "http://fake-asr",
                "Chinese",
                32,
                None,
            )
            with TestClient(app) as client:
                with client.websocket_connect(
                    "/ws/stream?mode=streaming"
                ) as websocket:
                    self.assertEqual(websocket.receive_json()["event"], "ready")
                    websocket.send_bytes(b"pcm")
                    message = websocket.receive_json()
                    while message["event"] != "update":
                        message = websocket.receive_json()
                    calls_before_finish = len(_CountingRefiner.calls)
                    self.assertGreater(calls_before_finish, 0)

                    websocket.send_json({"event": "finish"})
                    self.assertEqual(websocket.receive_json()["event"], "transcript")
                    final = websocket.receive_json()

                    self.assertEqual(final["event"], "final")
                    self.assertEqual(
                        final["clean_text"],
                        "第一段精修文本。第二段精修文本。",
                    )
                    self.assertEqual(len(_CountingRefiner.calls), calls_before_finish)

    def test_severe_content_loss_retries_with_strict_preservation(self) -> None:
        with (
            patch.object(web_app, "TransformersRefiner", _CompressesOnceRefiner),
            patch.object(web_app, "_stream_request", _content_loss_stream_request),
        ):
            app = web_app.create_app(
                Path("/tmp/fake-refiner"),
                "cpu",
                "http://fake-asr",
                "Chinese",
                32,
                None,
            )
            with TestClient(app) as client:
                with client.websocket_connect("/ws/stream?mode=offline") as websocket:
                    self.assertEqual(websocket.receive_json()["event"], "ready")
                    websocket.send_json({"event": "finish"})
                    self.assertEqual(
                        websocket.receive_json()["raw_text"],
                        _LONG_COMPLETE_TRANSCRIPT,
                    )
                    final = websocket.receive_json()
                    self.assertEqual(final["clean_text"], _LONG_COMPLETE_TRANSCRIPT)
                    self.assertTrue(final["refiner_accepted"])
                    self.assertEqual(final["refiner_retry_count"], 2)
                    self.assertEqual(final["placeholder_retry_count"], 0)
                    self.assertIn(
                        "segment_1:severe_content_loss",
                        final["refiner_retry_reasons"],
                    )
                    self.assertIn(
                        "segment_2:severe_content_loss",
                        final["refiner_retry_reasons"],
                    )

    def test_placeholder_failure_retries_once_with_strict_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            entity_db = Path(directory) / "entities.db"
            EntityStore(entity_db).upsert_entity(
                "鬼灵门",
                entity_type="TERM",
                normalization_policy="normalize",
            )
            with (
                patch.object(
                    web_app, "TransformersRefiner", _DropsPlaceholderOnceRefiner
                ),
                patch.object(web_app, "_stream_request", _entity_stream_request),
            ):
                app = web_app.create_app(
                    Path("/tmp/fake-refiner"),
                    "cpu",
                    "http://fake-asr",
                    "Chinese",
                    32,
                    None,
                    entity_db,
                    "auto",
                )
                with TestClient(app) as client:
                    with client.websocket_connect(
                        "/ws/stream?mode=offline&domain=general"
                    ) as websocket:
                        self.assertEqual(websocket.receive_json()["event"], "ready")
                        websocket.send_json({"event": "finish"})
                        self.assertEqual(
                            websocket.receive_json()["raw_text"],
                            "鬼灵们是这个副本的入口。",
                        )
                        final = websocket.receive_json()
                        self.assertEqual(final["clean_text"], "鬼灵门是这个副本的入口。")
                        self.assertTrue(final["refiner_accepted"])
                        self.assertEqual(final["placeholder_retry_count"], 1)
                        self.assertEqual(final["refiner_retry_count"], 1)
                        self.assertEqual(len(final["refiner_masked_outputs"]), 2)

    def test_final_auto_mode_applies_and_restores_fuzzy_entity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            entity_db = Path(directory) / "entities.db"
            EntityStore(entity_db).upsert_entity(
                "鬼灵门",
                entity_type="TERM",
                normalization_policy="normalize",
            )
            with (
                patch.object(web_app, "TransformersRefiner", _IdentityRefiner),
                patch.object(web_app, "_stream_request", _entity_stream_request),
            ):
                app = web_app.create_app(
                    Path("/tmp/fake-refiner"),
                    "cpu",
                    "http://fake-asr",
                    "Chinese",
                    32,
                    None,
                    entity_db,
                    "auto",
                )
                with TestClient(app) as client:
                    with client.websocket_connect(
                        "/ws/stream?mode=offline&domain=general"
                    ) as websocket:
                        self.assertEqual(websocket.receive_json()["event"], "ready")
                        websocket.send_json({"event": "finish"})
                        self.assertEqual(
                            websocket.receive_json()["raw_text"],
                            "鬼灵们是这个副本的入口。",
                        )
                        final = websocket.receive_json()
                        self.assertEqual(final["clean_text"], "鬼灵门是这个副本的入口。")
                        self.assertEqual(
                            final["entity_candidates"][0]["decision"],
                            "AUTO_NORMALIZE",
                        )
                        self.assertEqual(
                            final["protected_entities"][0]["source"],
                            "fuzzy:general",
                        )

    def test_slow_chunk_reports_progress_before_acknowledgement(self) -> None:
        with (
            patch.object(web_app, "TransformersRefiner", _SlowRefiner),
            patch.object(web_app, "_stream_request", _slow_chunk_stream_request),
            patch.object(web_app, "ASR_CHUNK_STATUS_INTERVAL_SECONDS", 0.01),
        ):
            app = web_app.create_app(
                Path("/tmp/fake-refiner"),
                "cpu",
                "http://fake-asr",
                "Chinese",
                32,
                None,
            )
            with TestClient(app) as client:
                with client.websocket_connect(
                    "/ws/stream?mode=streaming"
                ) as websocket:
                    self.assertEqual(websocket.receive_json()["event"], "ready")
                    websocket.send_bytes(b"pcm")
                    progress = websocket.receive_json()
                    self.assertEqual(progress["event"], "status")
                    self.assertEqual(progress["stage"], "asr_chunk")
                    self.assertEqual(progress["chunk_index"], 1)
                    acknowledgement = websocket.receive_json()
                    while acknowledgement["event"] == "status":
                        acknowledgement = websocket.receive_json()
                    self.assertEqual(acknowledgement["event"], "chunk_ack")
                    self.assertEqual(acknowledgement["chunk_index"], 1)

    def test_final_raw_transcript_is_sent_before_slow_refinement(self) -> None:
        with (
            patch.object(web_app, "TransformersRefiner", _SlowRefiner),
            patch.object(web_app, "_stream_request", _fake_stream_request),
        ):
            app = web_app.create_app(
                Path("/tmp/fake-refiner"),
                "cpu",
                "http://fake-asr",
                "Chinese",
                32,
                None,
            )

            with TestClient(app) as client:
                with client.websocket_connect(
                    "/ws/stream?mode=streaming"
                ) as websocket:
                    self.assertEqual(websocket.receive_json()["event"], "ready")
                    websocket.send_bytes(b"pcm")
                    self.assertEqual(websocket.receive_json()['event'], 'chunk_ack')
                    self.assertEqual(
                        websocket.receive_json()["event"], "transcript"
                    )

                    websocket.send_json({"event": "finish"})
                    pending = websocket.receive_json()
                    self.assertEqual(pending["event"], "transcript")
                    self.assertEqual(pending["raw_text"], "最终原始文本。")
                    self.assertTrue(pending["refiner_deferred"])

                    final = websocket.receive_json()
                    self.assertEqual(final["event"], "final")
                    self.assertEqual(final["clean_text"], "最终精修文本。")

    def test_refiner_failure_falls_back_to_complete_raw_text(self) -> None:
        with (
            patch.object(web_app, "TransformersRefiner", _FailingRefiner),
            patch.object(web_app, "_stream_request", _fake_stream_request),
        ):
            app = web_app.create_app(
                Path("/tmp/fake-refiner"),
                "cpu",
                "http://fake-asr",
                "Chinese",
                32,
                None,
            )
            with TestClient(app) as client:
                with client.websocket_connect("/ws/stream?mode=offline") as websocket:
                    self.assertEqual(websocket.receive_json()["event"], "ready")
                    websocket.send_json({"event": "finish"})
                    pending = websocket.receive_json()
                    self.assertEqual(pending["raw_text"], "最终原始文本。")
                    final = websocket.receive_json()
                    self.assertEqual(final["event"], "final")
                    self.assertEqual(final["clean_text"], "最终原始文本。")
                    self.assertFalse(final["refiner_accepted"])
                    self.assertIn(
                        "final_refinement_error:RuntimeError",
                        final["refiner_reject_reasons"],
                    )


if __name__ == "__main__":
    unittest.main()
