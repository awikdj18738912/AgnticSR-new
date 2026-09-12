from __future__ import annotations

import unittest
from types import SimpleNamespace

from system.qwen_confidence import (
    QwenVLLMConfidenceDecoder,
    completion_logprobs,
    state_confidence_payload,
    summarize_logprobs,
    update_state_confidence,
)


class QwenConfidenceTest(unittest.TestCase):
    def test_reads_chosen_token_logprobs(self) -> None:
        completion = SimpleNamespace(
            token_ids=[10, 11],
            logprobs=[
                {10: SimpleNamespace(logprob=-0.1)},
                {11: SimpleNamespace(logprob=-0.3)},
            ],
        )

        self.assertEqual(completion_logprobs(completion), [-0.1, -0.3])

    def test_summarizes_geometric_mean_and_marks_raw_score(self) -> None:
        summary = summarize_logprobs([-0.1, -0.3])

        self.assertAlmostEqual(summary.mean_logprob or 0.0, -0.2)
        self.assertAlmostEqual(summary.mean_probability or 0.0, 0.818730753, places=6)
        self.assertFalse(summary.calibrated)
        self.assertEqual(summary.token_count, 2)

    def test_temperature_and_calibration_are_reported(self) -> None:
        state = SimpleNamespace()
        completion = SimpleNamespace(
            token_ids=[1], logprobs=[{1: SimpleNamespace(logprob=-0.2)}]
        )

        update_state_confidence(
            state,
            completion,
            temperature=2.0,
            calibrated=True,
            covers_full_state=True,
        )
        payload = state_confidence_payload(state)

        self.assertTrue(payload["confidence_calibrated"])
        self.assertEqual(payload["confidence_temperature"], 2.0)
        self.assertEqual(payload["confidence_token_count"], 1)
        self.assertTrue(payload["confidence_covers_full_state"])

    def test_decoder_marks_empty_prefix_as_full_state_coverage(self) -> None:
        completion = SimpleNamespace(
            text="你好。",
            token_ids=[1, 2],
            logprobs=[
                {1: SimpleNamespace(logprob=-0.1)},
                {2: SimpleNamespace(logprob=-0.2)},
            ],
        )
        decoder = object.__new__(QwenVLLMConfidenceDecoder)
        decoder.asr = SimpleNamespace(
            model=SimpleNamespace(
                generate=lambda *args, **kwargs: [
                    SimpleNamespace(outputs=[completion])
                ]
            )
        )
        decoder.sampling_params = object()
        decoder.temperature = 1.0
        decoder.calibrated = False
        decoder._parse_asr_output = (
            lambda raw, user_language=None: (user_language or "", raw)
        )
        state = SimpleNamespace(
            prompt_raw="prompt",
            audio_accum=[],
            force_language="Chinese",
        )

        decoder._decode(state, "")
        self.assertTrue(
            state_confidence_payload(state)["confidence_covers_full_state"]
        )

        decoder._decode(state, "前缀")
        self.assertFalse(
            state_confidence_payload(state)["confidence_covers_full_state"]
        )

    def test_partial_token_logprobs_do_not_claim_full_state_coverage(self) -> None:
        state = SimpleNamespace()
        completion = SimpleNamespace(
            token_ids=[1, 2],
            logprobs=[{1: SimpleNamespace(logprob=-0.2)}],
        )

        update_state_confidence(
            state,
            completion,
            temperature=1.0,
            calibrated=False,
            covers_full_state=True,
        )

        self.assertFalse(
            state_confidence_payload(state)["confidence_covers_full_state"]
        )

    def test_missing_logprobs_are_safe(self) -> None:
        state = SimpleNamespace()
        update_state_confidence(
            state,
            SimpleNamespace(token_ids=[], logprobs=None),
            temperature=1.0,
            calibrated=False,
        )

        self.assertIsNone(state_confidence_payload(state)["confidence"])
        self.assertFalse(
            state_confidence_payload(state)["confidence_covers_full_state"]
        )


if __name__ == "__main__":
    unittest.main()
