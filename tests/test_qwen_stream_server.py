from __future__ import annotations

import unittest

from system.qwen_asr_stream_server import (
    _join_transcripts,
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


if __name__ == "__main__":
    unittest.main()
