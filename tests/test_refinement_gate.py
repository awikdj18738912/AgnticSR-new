from __future__ import annotations

import unittest

from system.refinement_gate import RefinementGate


class RefinementGateTest(unittest.TestCase):
    def test_off_mode_preserves_original_always_refine_baseline(self) -> None:
        decision = RefinementGate("off").decide("好。", asr_confidence=0.99)

        self.assertTrue(decision.should_refine)
        self.assertEqual(decision.reasons, ("gate_disabled",))

    def test_conservative_mode_skips_unsafe_short_fragment(self) -> None:
        decision = RefinementGate("conservative").decide("好。")

        self.assertFalse(decision.should_refine)
        self.assertEqual(decision.reasons, ("too_short_for_safe_refinement",))

    def test_entity_hint_overrides_short_fragment_skip(self) -> None:
        decision = RefinementGate("conservative").decide(
            "韩立。",
            asr_confidence=0.99,
            entity_hints=("韩立",),
            calibrated=True,
            covers_segment=True,
        )

        self.assertTrue(decision.should_refine)
        self.assertEqual(decision.reasons, ("entity_hint_present",))

    def test_high_confidence_clean_complete_segment_is_skipped(self) -> None:
        decision = RefinementGate("conservative").decide(
            "今天天气很好。",
            asr_confidence=0.97,
            calibrated=True,
            covers_segment=True,
        )

        self.assertFalse(decision.should_refine)
        self.assertEqual(decision.reasons, ("high_confidence_clean_segment",))
        self.assertTrue(decision.calibrated)
        self.assertTrue(decision.covers_segment)

    def test_confidence_eligibility_defaults_fail_open(self) -> None:
        decision = RefinementGate("conservative").decide(
            "今天天气很好。",
            asr_confidence=0.99,
        )

        self.assertTrue(decision.should_refine)
        self.assertEqual(decision.reasons, ("confidence_uncalibrated",))
        self.assertFalse(decision.calibrated)
        self.assertFalse(decision.covers_segment)

    def test_uncalibrated_confidence_fails_open_to_refinement(self) -> None:
        decision = RefinementGate("conservative").decide(
            "今天天气很好。",
            asr_confidence=0.99,
            calibrated=False,
            covers_segment=True,
        )

        self.assertTrue(decision.should_refine)
        self.assertEqual(decision.reasons, ("confidence_uncalibrated",))
        self.assertFalse(decision.calibrated)
        self.assertTrue(decision.public_dict()["covers_segment"])

    def test_incomplete_confidence_coverage_fails_open_to_refinement(self) -> None:
        decision = RefinementGate("conservative").decide(
            "今天天气很好。",
            asr_confidence=0.99,
            calibrated=True,
            covers_segment=False,
        )

        self.assertTrue(decision.should_refine)
        self.assertEqual(
            decision.reasons, ("confidence_coverage_incomplete",)
        )
        self.assertFalse(decision.covers_segment)
        self.assertTrue(decision.public_dict()["calibrated"])

    def test_missing_confidence_fails_open_to_refinement(self) -> None:
        decision = RefinementGate("conservative").decide("今天天气很好。")

        self.assertTrue(decision.should_refine)
        self.assertEqual(decision.reasons, ("confidence_unavailable",))

    def test_cleanup_signal_overrides_high_confidence_skip(self) -> None:
        decision = RefinementGate("conservative").decide(
            "我我我知道了。",
            asr_confidence=0.99,
            calibrated=True,
            covers_segment=True,
        )

        self.assertTrue(decision.should_refine)
        self.assertIn("repeated_character", decision.cleanup_signals)

    def test_short_disfluency_is_refined_instead_of_skipped(self) -> None:
        decision = RefinementGate("conservative").decide("啊。")

        self.assertTrue(decision.should_refine)
        self.assertEqual(decision.reasons, ("cleanup_signal_present",))
        self.assertIn("disfluency", decision.cleanup_signals)

    def test_embedded_low_ambiguity_disfluencies_are_refined(self) -> None:
        examples = (
            "我嗯，不太嗯知道。",
            "我呃不知道该怎么说。",
            "嗯我知道了。",
            "我知道嗯。",
        )
        for text in examples:
            with self.subTest(text=text):
                decision = RefinementGate("conservative").decide(
                    text,
                    asr_confidence=0.99,
                    calibrated=True,
                    covers_segment=True,
                )

                self.assertTrue(decision.should_refine)
                self.assertIn(
                    "embedded_disfluency", decision.cleanup_signals
                )

    def test_ambiguous_embedded_words_are_not_disfluencies(self) -> None:
        examples = (
            "余额不足。",
            "额外说明。",
            "请呼叫客服。",
            "请喂养宠物。",
            "这就是答案。",
            "这个方案很好。",
            "好呀，我们走吧。",
            "医学上称为呃逆。",
            "嗯哼。",
        )
        for text in examples:
            with self.subTest(text=text):
                decision = RefinementGate("conservative").decide(
                    text,
                    asr_confidence=0.99,
                    calibrated=True,
                    covers_segment=True,
                )

                self.assertFalse(decision.should_refine)
                self.assertNotIn("disfluency", decision.cleanup_signals)
                self.assertNotIn(
                    "embedded_disfluency", decision.cleanup_signals
                )

    def test_punctuation_separated_single_repeat_is_refined(self) -> None:
        decision = RefinementGate("conservative").decide(
            "不，不，不，不。",
            asr_confidence=0.99,
            calibrated=True,
            covers_segment=True,
        )

        self.assertTrue(decision.should_refine)
        self.assertIn("repeated_character", decision.cleanup_signals)

    def test_two_character_repeat_is_refined(self) -> None:
        decision = RefinementGate("conservative").decide(
            "你你来。",
            asr_confidence=0.99,
            calibrated=True,
            covers_segment=True,
        )

        self.assertTrue(decision.should_refine)
        self.assertIn("repeated_character", decision.cleanup_signals)

    def test_two_phrase_repeat_is_refined(self) -> None:
        decision = RefinementGate("conservative").decide(
            "韩道友，韩道友。",
            asr_confidence=0.99,
            calibrated=True,
            covers_segment=True,
        )

        self.assertTrue(decision.should_refine)
        self.assertIn("repeated_phrase", decision.cleanup_signals)

    def test_incomplete_sentence_is_still_refined(self) -> None:
        decision = RefinementGate("conservative").decide(
            "今天天气很好",
            asr_confidence=0.99,
            calibrated=True,
            covers_segment=True,
        )

        self.assertTrue(decision.should_refine)
        self.assertEqual(decision.reasons, ("sentence_incomplete",))

    def test_numeric_normalization_overrides_high_confidence_skip(self) -> None:
        decision = RefinementGate("conservative").decide(
            "我有一千一百一十五元，百分之五的概率。",
            asr_confidence=0.99,
            calibrated=True,
            covers_segment=True,
        )

        self.assertTrue(decision.should_refine)
        self.assertIn("numeric_normalization", decision.cleanup_signals)


if __name__ == "__main__":
    unittest.main()
