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

    def test_long_self_correction_is_not_content_loss(self) -> None:
        raw = (
            "请记录这段话，我有一个苹果，不对，我有一个梨。"
            "这是后面的补充说明，请一并保留。"
        )

        self.assertNotIn(
            "severe_content_loss",
            reject_reasons(
                raw,
                "请记录这段话，我有一个梨。这是后面的补充说明，请一并保留。",
            ),
        )

    def test_comma_after_self_correction_tail_is_not_content_loss(self) -> None:
        raw = "你好，我有一个苹果，我有一个梨，不对，我有一个香蕉，我有两千一百三十五元。"

        self.assertNotIn(
            "severe_content_loss",
            reject_reasons(raw, "你好，我有一个香蕉，我有2135元。"),
        )

    def test_numeric_value_change_is_rejected(self) -> None:
        reasons = reject_reasons(
            "我有两千一百三十五元。",
            "我有两百三十五元。",
        )
        self.assertIn("numeric_value_mismatch", reasons)

    def test_classifier_quantity_normalization_is_accepted(self) -> None:
        self.assertEqual(
            reject_reasons("我有五个苹果。", "我有5个苹果。"),
            (),
        )

    def test_ambiguous_digit_run_conversion_is_rejected(self) -> None:
        self.assertIn(
            "numeric_value_mismatch",
            reject_reasons("我有二三个人。", "我有23个人。"),
        )

    def test_positional_number_normalization_is_accepted(self) -> None:
        self.assertEqual(
            reject_reasons("他二十三岁。", "他23岁。"),
            (),
        )

    def test_safe_numeric_normalization_is_accepted(self) -> None:
        self.assertEqual(
            reject_reasons(
                "百分之五的概率，日期是二零一五年五月五日。",
                "5%的概率，日期是2015年5月5日。",
            ),
            (),
        )

    def test_semantic_deletions_are_rejected(self) -> None:
        cases = (
            (
                "看，掘地三尺，要把他给我找出来！",
                "看，掘地三尺，要把我找出来！",
            ),
            (
                "你认识这位道友，二伯，此人就是我与你说过，那个黄风谷姓韩的。",
                "你认识这位道友，伯，你我说过，黄风谷姓韩的。",
            ),
        )
        for raw, refined in cases:
            with self.subTest(raw=raw):
                self.assertIn("semantic_content_loss", reject_reasons(raw, refined))

    def test_semantic_particle_substitution_is_rejected(self) -> None:
        self.assertIn(
            "semantic_substitution",
            reject_reasons("你当我傻？", "你当我的？"),
        )

    def test_repetition_cleanup_remains_allowed(self) -> None:
        self.assertEqual(
            reject_reasons(
                "然后哈哈哈，我一五一十的告诉你。",
                "然后我一五一十的告诉你。",
            ),
            (),
        )
        self.assertEqual(
            reject_reasons(
                "今天有一个苹果，不对，有一个梨。",
                "今天有一个梨。",
            ),
            (),
        )

    def test_repeated_content_insertion_is_rejected(self) -> None:
        self.assertIn(
            "unsupported_insertion",
            reject_reasons(
                "随便你们进来打打杀杀，快给我滚！你听见了，我是不会跟你打的。",
                "随便你们进来杀杀，快给我滚！你听见了，我是不会跟你打杀杀的。",
            ),
        )


if __name__ == "__main__":
    unittest.main()


class NumericValidationAlignmentTest(unittest.TestCase):
    """Legal Chinese->Arabic numeric normalizations must pass the guard, while
    real value changes must still be rejected."""

    def test_legal_normalizations_are_accepted(self) -> None:
        cases = (
            ("晚上七点十分开会。", "晚上7点10分开会。"),
            ("价格涨了一到两点五米。", "价格涨了1到2.5米。"),
            ("有百分之十二点五的概率。", "有12.5%的概率。"),
            ("今天是二零一五年十二月五日。", "今天是2015年12月5日。"),
            ("进度是百分之五。", "进度是5%。"),
            ("预算两千一百三十五元。", "预算2135元。"),
            ("他今年三十岁。", "他今年30岁。"),
            ("走了二十分钟。", "走了20分钟。"),
            ("我让他五个法器；再不行，我让他十个，怎么样？", "我让他5个法器；再不行，我让他10个，怎么样？"),
            ("下午三点半开会。", "下午3点半开会。"),
        )
        for raw, refined in cases:
            with self.subTest(raw=raw, refined=refined):
                reasons = reject_reasons(raw, refined)
                self.assertNotIn("numeric_value_mismatch", reasons, reasons)
                self.assertNotIn("severe_content_loss", reasons, reasons)

    def test_real_value_changes_are_rejected(self) -> None:
        cases = (
            ("我有两千一百三十五元。", "我有两百三十五元。"),
            ("我有百分之五的概率。", "我有百分之十五的概率。"),
            ("今天是一月五日。", "今天是12月5日。"),
        )
        for raw, refined in cases:
            with self.subTest(raw=raw, refined=refined):
                self.assertIn(
                    "numeric_value_mismatch", reject_reasons(raw, refined)
                )

    def test_numeric_itn_inside_a_wider_edit_is_accepted(self) -> None:
        # The pair is not numeric-surface-only because a self-correction
        # removed a number-bearing false start.  The surviving numeric
        # rewrite is still legal ITN and must not be flagged as semantic loss.
        raw = "不对，我有一百二十三万五千四百三十一元。应该是二百万零五十元。"
        self.assertEqual(reject_reasons(raw, "应该是2000050元。"), ())


class MultiStageSelfCorrectionTest(unittest.TestCase):
    """A multi-stage correction chain may collapse to its final clause even
    when the refiner only sees the ITN-normalized surface (``一个`` -> ``1个``)."""

    RAW = "你好，你好，我有一个苹果，不对，我有一个梨，不对，我有一个香蕉。"
    RESOLVED = "你好，我有1个香蕉。"

    def test_itn_normalized_correction_tail_is_accepted(self) -> None:
        self.assertEqual(reject_reasons(self.RAW, self.RESOLVED), ())

    def test_multi_stage_chain_keeps_working_with_sentence_punctuation(self) -> None:
        raw = "你好，你好，我有一个苹果。不对，我有一个梨。不对，我有一个香蕉。"
        self.assertEqual(reject_reasons(raw, "你好，我有1个香蕉。"), ())

    def test_single_stage_correction_still_needs_a_reasonable_ratio(self) -> None:
        # Only one correction marker: a deep collapse that keeps the tail is
        # still reported as content loss instead of being silently accepted.
        raw = "今天早上我在家里吃了一个苹果，不对，我吃了一个梨。"
        reasons = reject_reasons(raw, "我吃了1个梨。")
        self.assertIn("severe_content_loss", reasons)

    def test_changed_number_in_correction_tail_is_still_rejected(self) -> None:
        # Numeric normalization must stay value preserving: a real value
        # change inside the retained tail is not a formatting difference.
        reasons = reject_reasons("不对，我有五个梨。", "我有3个梨。")
        self.assertIn("numeric_value_mismatch", reasons)

    def test_dangling_correction_marker_does_not_hide_earlier_correction(self) -> None:
        # A chunk boundary may leave ``不对，`` dangling at the end of a
        # window.  That occurrence has no replacement clause, so the guard
        # must fall back to the earlier complete self-correction instead of
        # reporting the whole window as content loss.
        raw = (
            "你好，我有一个苹果。不对，我有一个梨。"
            "今天是二零一五年十二月五日。"
            "我有百分之三十七的概率会获得一百二十三万五千四百三十一元。不对，"
        )
        refined = (
            "你好，我有1个梨。今天是2015年12月5日。"
            "我有37%的概率会获得1235431元。"
        )
        self.assertEqual(reject_reasons(raw, refined), ())


class NumericSelfCorrectionTest(unittest.TestCase):
    """A numeric self-correction chain may collapse to the final value."""

    RAW = (
        "我有百分之三十七的概率在二零一五年十二月五日这天获得"
        "一千二百三十四万五千四百三十一元。不对，应该是一百五十块。不对，应该是一百块。"
    )

    def test_numeric_correction_chain_collapses_to_final_value(self) -> None:
        # The Refiner resolved the amount to ``100元``.  ``100块`` and
        # ``100元`` are the same currency mention, and the superseded
        # values are exactly what a self-correction discards.
        self.assertEqual(
            reject_reasons(
                self.RAW,
                "我有37%的概率在2015年12月5日这天获得100元。",
            ),
            (),
        )

    def test_invented_value_inside_a_correction_is_rejected(self) -> None:
        reasons = reject_reasons(
            "获得一千二百三十四万五千四百三十一元。不对，应该是一百块。",
            "获得555元。应该是100块。",
        )
        self.assertIn("semantic_content_loss", reasons)

    def test_unit_change_inside_a_correction_is_rejected(self) -> None:
        reasons = reject_reasons(
            "获得一千二百三十四万五千四百三十一元。不对，应该是一百块。",
            "获得100个。应该是100块。",
        )
        self.assertIn("semantic_content_loss", reasons)


if __name__ == "__main__":
    unittest.main()
