from __future__ import annotations

import unittest
from decimal import Decimal

from system.numeric_normalizer import (
    ContextualNumericNormalizer,
    chinese_number_to_decimal,
)


class ContextualNumericNormalizerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.normalizer = ContextualNumericNormalizer()

    def test_compound_currency_is_exact(self) -> None:
        result = self.normalizer.normalize("你好，我有二万二千二百元。")

        self.assertEqual(result.text, "你好，我有22200元。")
        self.assertEqual(result.changes[0].kind, "currency")

    def test_liang_compound_currency_is_exact(self) -> None:
        self.assertEqual(
            self.normalizer.normalize("我有两千一百三十五元。").text,
            "我有2135元。",
        )

    def test_full_date_percent_decimal_and_range(self) -> None:
        source = "今天是二零一五年十二月五日，进度百分之五，长一到两点五米。"

        self.assertEqual(
            self.normalizer.normalize(source).text,
            "今天是2015年12月5日，进度5%，长1到2.5米。",
        )

    def test_idiom_is_protected_but_classifier_quantity_is_normalized(self) -> None:
        source = "他一五一十地说清了经过，做事一心一意，我还有一个苹果。"

        result = self.normalizer.normalize(source)

        self.assertEqual(
            result.text,
            "他一五一十地说清了经过，做事一心一意，我还有1个苹果。",
        )
        self.assertEqual(result.changes[0].kind, "count")

    def test_idiom_does_not_block_real_amount_later(self) -> None:
        source = "他一五一十地说，一共花了二百元。"

        self.assertEqual(
            self.normalizer.normalize(source).text,
            "他一五一十地说，一共花了200元。",
        )

    def test_parser_preserves_positional_value(self) -> None:
        self.assertEqual(chinese_number_to_decimal("二万二千二百"), 22200)
        self.assertEqual(chinese_number_to_decimal("一万零三"), 10003)
        self.assertEqual(chinese_number_to_decimal("二点六"), Decimal("2.6"))

    def test_adjacent_digit_runs_are_not_collapsed(self) -> None:
        cases = (
            ("二三个人。", "二三个人。"),
            ("二三十个人。", "二三十个人。"),
            ("十几个苹果。", "十几个苹果。"),
        )
        for source, expected in cases:
            with self.subTest(source=source):
                self.assertEqual(self.normalizer.normalize(source).text, expected)

    def test_positional_number_without_trailing_unit_is_normalized(self) -> None:
        cases = (
            ("他二十三岁。", "他23岁。"),
            ("获得一百二十三万。", "获得1230000。"),
        )
        for source, expected in cases:
            with self.subTest(source=source):
                self.assertEqual(self.normalizer.normalize(source).text, expected)

    def test_common_numeral_idioms_override_positional_rule(self) -> None:
        source = "十有八九，千言万语，五花八门。"

        result = self.normalizer.normalize(source)

        self.assertEqual(result.text, source)
        self.assertEqual(result.changes, ())


if __name__ == "__main__":
    unittest.main()
