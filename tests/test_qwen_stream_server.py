from __future__ import annotations

import unittest

from system.qwen_asr_stream_server import (
    _join_transcripts,
    _scope_confidence_payload,
    _should_rotate_segment,
)


class QwenStreamSegmentationTest(unittest.TestCase):
    def test_prefers_sentence_boundary_after_soft_limit(self) -> None:
        self.assertFalse(_should_rotate_segment("还在说", 30, 30, 45))
        self.assertTrue(_should_rotate_segment("这句说完了。", 30, 30, 45))

    def test_hard_limit_rotates_without_punctuation(self) -> None:
        self.assertTrue(_should_rotate_segment("还在说", 45, 30, 45))

    def test_transcript_joining_preserves_chinese_and_english_spacing(self) -> None:
        self.assertEqual(_join_transcripts("第一段。", "第二段。"), "第一段。第二段。")
        self.assertEqual(_join_transcripts("hello", "world"), "hello world")


    def test_full_state_score_covers_full_text_without_committed_prefix(self) -> None:
        payload = _scope_confidence_payload(
            {
                "confidence": 0.97,
                "confidence_token_count": 8,
                "confidence_covers_full_state": True,
            },
            has_committed_prefix=False,
        )

        self.assertTrue(payload["confidence_covers_full_text"])
        self.assertEqual(payload["confidence_scope"], "full_text")

    def test_full_state_score_does_not_cover_prior_committed_text(self) -> None:
        payload = _scope_confidence_payload(
            {
                "confidence": 0.97,
                "confidence_token_count": 8,
                "confidence_covers_full_state": True,
            },
            has_committed_prefix=True,
        )

        self.assertFalse(payload["confidence_covers_full_text"])
        self.assertEqual(payload["confidence_scope"], "current_state")

    def test_prefix_decode_is_reported_as_generated_suffix_only(self) -> None:
        payload = _scope_confidence_payload(
            {
                "confidence": 0.97,
                "confidence_token_count": 8,
                "confidence_covers_full_state": False,
            },
            has_committed_prefix=False,
        )

        self.assertFalse(payload["confidence_covers_full_text"])
        self.assertEqual(payload["confidence_scope"], "generated_suffix")

    def test_missing_score_has_unavailable_scope(self) -> None:
        payload = _scope_confidence_payload(
            {
                "confidence": None,
                "confidence_token_count": 0,
                "confidence_covers_full_state": False,
            },
            has_committed_prefix=False,
        )

        self.assertFalse(payload["confidence_covers_full_text"])
        self.assertEqual(payload["confidence_scope"], "unavailable")


if __name__ == "__main__":
    unittest.main()
