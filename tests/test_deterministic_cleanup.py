from __future__ import annotations

import unittest

from system.deterministic_cleanup import (
    clean_transcript_deterministically,
    collapse_repeated_comma_items,
    collapse_repeated_short_utterances,
)


class DeterministicCleanupTest(unittest.TestCase):
    def test_three_identical_short_exclamations_collapse_to_one(self) -> None:
        self.assertEqual(
            collapse_repeated_short_utterances("前文。可恶！可恶！可恶！后文。"),
            "前文。可恶！后文。",
        )

    def test_trailing_unpunctuated_repetition_is_included(self) -> None:
        self.assertEqual(
            collapse_repeated_short_utterances("可恶！可恶！可恶"),
            "可恶！",
        )

    def test_two_repetitions_collapse_to_one(self) -> None:
        self.assertEqual(
            collapse_repeated_short_utterances("快跑！快跑！"),
            "快跑！",
        )

    def test_comma_separated_short_repetitions_collapse_to_one(self) -> None:
        self.assertEqual(
            collapse_repeated_comma_items("不，不，不，不，我赔。"),
            "不，我赔。",
        )
        self.assertEqual(
            collapse_repeated_comma_items("少主，少主，少主！"),
            "少主！",
        )

    def test_latest_transcript_comma_repetitions_are_cleaned(self) -> None:
        source = (
            "嗯，这嘿，我，呀，不，嘿，不，不，不，不，我赔。"
            "少主，少主！你你来！少主，少主，少主！"
        )
        self.assertEqual(
            clean_transcript_deterministically(source),
            "嗯，这嘿，我，呀，不，嘿，不，我赔。少主！你来！少主！",
        )

    def test_nonidentical_comma_fields_are_preserved(self) -> None:
        source = "他说少主，少主快走，人人平等，天天向上。"
        self.assertEqual(collapse_repeated_comma_items(source), source)

    def test_different_or_long_utterances_are_preserved(self) -> None:
        source = "可恶！快跑！可恶！这是一句超过八个字的重复！" * 3
        self.assertEqual(collapse_repeated_short_utterances(source), source)

    def test_repeated_pronoun_stutter_is_collapsed(self) -> None:
        self.assertEqual(
            clean_transcript_deterministically("你你竟结成了元婴，我我不知道。"),
            "你竟结成了元婴，我不知道。",
        )

    def test_normal_chinese_reduplication_is_preserved(self) -> None:
        source = "人人平等，天天向上，好好学习，让我看看。"
        self.assertEqual(clean_transcript_deterministically(source), source)


if __name__ == "__main__":
    unittest.main()
