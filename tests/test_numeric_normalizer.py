from __future__ import annotations

import re
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
            # Large units stay as units (policy F2), so the value is not
            # expanded into bare digits any more.
            ("获得一百二十三万。", "获得123万。"),
        )
        for source, expected in cases:
            with self.subTest(source=source):
                self.assertEqual(self.normalizer.normalize(source).text, expected)

    def test_common_numeral_idioms_override_positional_rule(self) -> None:
        source = "十有八九，千言万语，五花八门。"

        result = self.normalizer.normalize(source)

        self.assertEqual(result.text, source)
        self.assertEqual(result.changes, ())


class NumeralBoundaryRegressionTest(unittest.TestCase):
    """Regressions from the 2026-09-13 production transcript.

    The rules used to start matching inside a longer numeral expression, so
    ``六十多万条`` became ``六十多10000条`` and ``一点七亿`` became
    ``一点700000000``.  These cases pin the numeral-boundary guard, the
    large-unit policy, and the lexical forms that only look numeric.
    """

    def setUp(self) -> None:
        self.normalizer = ContextualNumericNormalizer()

    def test_large_units_keep_the_unit_character(self) -> None:
        cases = (
            ("超过十亿吨泥沙沿着六十多万条沟谷", "超过10亿吨泥沙沿着60多万条沟谷"),
            ("从曾经的十六亿吨减少到了两亿吨", "从曾经的16亿吨减少到了2亿吨"),
            ("它的最大库容有四百五十亿立方米", "它的最大库容有450亿立方米"),
            ("每秒超过六万立方米的流量", "每秒超过6万立方米的流量"),
            ("覆盖全流域的三万多个监测站点", "覆盖全流域的3万多个监测站点"),
        )
        for source, expected in cases:
            with self.subTest(source=source):
                self.assertEqual(self.normalizer.normalize(source).text, expected)

    def test_decimal_is_not_split_before_a_large_unit(self) -> None:
        cases = (
            ("一点七亿人口", "1.7亿人口"),
            ("天河之水又化作四点五万条江河", "天河之水又化作4.5万条江河"),
            ("四点五万人次", "4.5万人次"),
        )
        for source, expected in cases:
            with self.subTest(source=source):
                self.assertEqual(self.normalizer.normalize(source).text, expected)

    def test_lexical_usage_keeps_the_spoken_form(self) -> None:
        cases = (
            "为了不让万一发生",
            "让黄河一度成为世界上含沙量最大的河流",
            "这些不到万不得已不会轻易开启的区域",
            "让这些原本相隔千里的大河",
            "数千年来，我们从未停止过",
            "守护着这里大面积的湿地和数千种森林",
            "我们正在以前所未有的方式重修江河百川",
            "他二三十岁的样子",
            "大概三四十米深",
            "而在安徽，数十座泵站将长江水不断举起",
        )
        for source in cases:
            with self.subTest(source=source):
                self.assertEqual(self.normalizer.normalize(source).text, source)

    def test_real_numeric_context_still_converts(self) -> None:
        cases = (
            ("大堤加高了一到两米", "大堤加高了1到2米"),
            ("气温降到了零下三十一度。", "气温降到了零下31度。"),
            ("这一块钱我先垫上", "这1块钱我先垫上"),
            ("我有一箱梨", "我有1箱梨"),
            ("我有二万二千二百元", "我有22200元"),
            ("进度百分之三十六", "进度36%"),
            ("今天是二零一五年十二月五日", "今天是2015年12月5日"),
            ("一共二十三个人", "一共23个人"),
        )
        for source, expected in cases:
            with self.subTest(source=source):
                self.assertEqual(self.normalizer.normalize(source).text, expected)

    def test_production_paragraph_has_no_corrupted_numeral(self) -> None:
        source = (
            "每逢雨季，超过十亿吨泥沙沿着六十多万条沟谷汇入黄河，"
            "让黄河一度成为世界上含沙量最大的河流。"
            "大地之上，天河之水又化作四点五万条江河。"
            "四十四座城市和一点七亿人口用上了长江水。"
            "数千年来，我们从未停止过。为了不让万一发生，"
            "我们正在以前所未有的方式重修江河百川。"
        )
        result = self.normalizer.normalize(source)

        self.assertEqual(
            result.text,
            "每逢雨季，超过10亿吨泥沙沿着60多万条沟谷汇入黄河，"
            "让黄河一度成为世界上含沙量最大的河流。"
            "大地之上，天河之水又化作4.5万条江河。"
            "44座城市和1.7亿人口用上了长江水。"
            "数千年来，我们从未停止过。为了不让万一发生，"
            "我们正在以前所未有的方式重修江河百川。",
        )
        for marker in ("10001", "100川", "700000000", "10000条", "50000条"):
            self.assertNotIn(marker, result.text)
        self.assertIsNone(re.search(r"\d{9,}", result.text))
        self.assertIsNone(re.search(r"数\d", result.text))


class NumeralIdempotenceRegressionTest(unittest.TestCase):
    """The pipeline normalizes a segment more than once.

    ``system/web_app.py`` normalizes the baseline, the masked text and the
    refiner output of the same segment, so ``normalize`` has to be idempotent.
    It was not: the ``10亿`` written by the first pass was re-read by the second
    pass as ``10`` + ``100000000``, producing ``10100000000吨`` and
    ``4.510000条`` in the published transcript.
    """

    def setUp(self) -> None:
        self.normalizer = ContextualNumericNormalizer()

    def test_already_converted_output_is_stable(self) -> None:
        cases = (
            "超过10亿吨泥沙沿着60多万条沟谷",
            "从曾经的16亿吨减少到了2亿吨",
            "天河之水又化作4.5万条江河",
            "它的最大库容有450亿立方米",
            "每秒超过6万立方米的流量",
            "覆盖全流域的3万多个监测站点",
            "44座城市和1.7亿人口用上了长江水",
            "今天是2015年12月5日，我有37%的概率获得1237851元",
        )
        for source in cases:
            with self.subTest(source=source):
                self.assertEqual(self.normalizer.normalize(source).text, source)

    def test_repeated_passes_do_not_corrupt_the_value(self) -> None:
        cases = (
            "超过十亿吨泥沙沿着六十多万条沟谷",
            "从曾经的十六亿吨减少到了两亿吨",
            "天河之水又化作四点五万条江河",
            "每秒超过六万立方米的流量",
            "四百五十亿立方米",
        )
        for source in cases:
            with self.subTest(source=source):
                once = self.normalizer.normalize(source).text
                twice = self.normalizer.normalize(once).text
                thrice = self.normalizer.normalize(twice).text
                self.assertEqual(once, twice)
                self.assertEqual(twice, thrice)
                for marker in ("1010000", "1610000", "2100000", "4.5100"):
                    self.assertNotIn(marker, twice)

    def test_bare_large_unit_is_not_a_value(self) -> None:
        cases = (
            "耗资亿元",
            "亿万吨",
            "亿万人民",
            "万万不可",
            "千千万万的人",
            "万万千千的劳动者",
        )
        for source in cases:
            with self.subTest(source=source):
                self.assertEqual(self.normalizer.normalize(source).text, source)


if __name__ == "__main__":
    unittest.main()
