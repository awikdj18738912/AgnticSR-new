from __future__ import annotations

import time
from pathlib import Path

import unittest
from unittest.mock import patch

try:
    from fastapi.testclient import TestClient

    import system.web_app as web_app
except ModuleNotFoundError:  # The base data-pipeline environment omits FastAPI.
    TestClient = None
    web_app = None


class _SlowRefiner:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def refine(
        self, text: str, *, entity_hints: tuple[str, ...] = ()
    ) -> tuple[str, float]:
        time.sleep(0.1)
        return text.replace("原始", "精修"), 100.0


class _FailingRefiner:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def refine(
        self, text: str, *, entity_hints: tuple[str, ...] = ()
    ) -> tuple[str, float]:
        raise RuntimeError("simulated model failure")


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


@unittest.skipIf(web_app is None, "FastAPI is not installed")
class WebAppFinishTest(unittest.TestCase):
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
