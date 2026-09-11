from __future__ import annotations

import unittest

from system.refinement_guard import join_refined_segments, reject_reasons, split_for_refinement


class RefinementGuardTest(unittest.TestCase):
    def test_default_refinement_segments_are_at_most_eighty_characters(self) -> None:
        source = "这是一句需要完整保留的转录文本。" * 12
        parts = split_for_refinement(source)

        self.assertEqual(join_refined_segments(parts), source)
        self.assertTrue(all(len(part) <= 80 for part in parts))

    def test_sentence_preferred_split_preserves_source(self) -> None:
        source = "第一句需要保留。第二句也需要保留，而且内容稍长。第三句结束。第四句继续说明，确保文本超过分段长度。"
        parts = split_for_refinement(source, max_chars=32)

        self.assertEqual(join_refined_segments(parts), source)
        self.assertGreater(len(parts), 1)
        self.assertTrue(all(len(part) <= 32 for part in parts))

    def test_repeated_generated_sentence_is_rejected(self) -> None:
        raw = "请说明当前情况，然后等待进一步通知。"
        repeated = "请说明当前情况。" * 4

        self.assertIn("repeated_sentence", reject_reasons(raw, repeated))

    def test_truncated_long_complete_source_is_rejected(self) -> None:
        raw = "这是一个需要完整保留的长句。" * 12
        refined = "这是一个不完整的输出"

        self.assertIn("truncated_refiner_output", reject_reasons(raw, refined))

    def test_punctuated_severe_content_loss_is_rejected(self) -> None:
        raw = (
            "怎么可能？你竟结成了元婴？此青火杖乃墨家的不夜之杖。"
            "你这手段怎么比我还像我道？我记住你了。你认识这位道友？"
            "此人就是我与你说过的那个黄风谷姓韩的，放肆！"
            "韩道友已经是元婴修士，又岂会和你一般见识？"
        )

        self.assertIn("severe_content_loss", reject_reasons(raw, "你怎么拼？"))

    def test_normal_refinement_is_accepted(self) -> None:
        self.assertEqual(
            reject_reasons("今天有一个苹果，不对，有一个梨。", "今天有一个梨。"),
            (),
        )


if __name__ == "__main__":
    unittest.main()
