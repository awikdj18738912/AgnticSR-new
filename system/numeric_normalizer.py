"""Deterministic, context-bound Chinese number normalization.

The neural Refiner is good at transcript cleanup but can lose positional
values in Chinese numerals. This module handles explicit numeric contexts
(money, percentages, full dates/times, measurements, and classifier counts)
plus unambiguous positional forms such as ``二十三``. Approximate adjacent
digit runs and fixed expressions remain untouched.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal


_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
_SMALL_UNITS = {"十": 10, "百": 100, "千": 1_000}
_LARGE_UNITS = {"万": 10_000, "亿": 100_000_000}
_CN_INTEGER = "零〇一二两三四五六七八九十百千万亿"
_CN_DIGIT = "零〇一二两三四五六七八九"
_CN_NUMBER_PATTERN = rf"[{_CN_INTEGER}]+(?:点[{_CN_DIGIT}]+)?"

# Keep an explicit deny-list in addition to contextual matching.  It makes the
# safety policy obvious and protects an idiom even if it happens to be next to
# a word that resembles a supported unit.
_FIXED_EXPRESSIONS = (
    "一五一十",
    "一心一意",
    "三心二意",
    "不三不四",
    "乱七八糟",
    "七上八下",
    "五湖四海",
    "四面八方",
    "九牛一毛",
    "十全十美",
    "百里挑一",
    "千方百计",
    "万无一失",
    "一举两得",
    "三番五次",
    "一清二楚",
    "说一不二",
    # Common fixed expressions whose numeral-looking characters are lexical,
    # not values.  Keep this deny-list ahead of the broad positional-number
    # rule below; the list is intentionally explicit and easy to extend.
    "十有八九",
    "七七八八",
    "三三两两",
    "五花八门",
    "五颜六色",
    "千言万语",
    "千军万马",
    "千变万化",
    "千山万水",
    "万水千山",
    "九死一生",
    "八九不离十",
    "四通八达",
    "四平八稳",
    "两全其美",
    "两肋插刀",
    "一针一线",
    "一朝一夕",
    "一刀两断",
    "一穷二白",
    "一来二去",
    "一干二净",
    "一百年大计",
    "一百年好合",
    "一千年一遇",
    "百年大计",
    "百年好合",
    "千年一遇",
)

_PERCENT_RE = re.compile(rf"百分之(?P<number>{_CN_NUMBER_PATTERN})")
_FULL_DATE_RE = re.compile(
    rf"(?P<year>[{_CN_INTEGER}]+)年"
    rf"(?P<month>[{_CN_INTEGER}]+)月"
    rf"(?P<day>[{_CN_INTEGER}]+)(?P<day_unit>[日号])"
)
_YEAR_MONTH_RE = re.compile(
    rf"(?P<year>[{_CN_INTEGER}]+)年(?P<month>[{_CN_INTEGER}]+)月"
)
_YEAR_RE = re.compile(rf"(?P<year>[{_CN_DIGIT}]{{4}})年")
_FULL_TIME_RE = re.compile(
    rf"(?P<hour>[{_CN_INTEGER}]+)点"
    rf"(?P<minute>[{_CN_INTEGER}]+)分(?:钟)?"
)

_MEASURE_UNITS = (
    "平方公里",
    "平方米",
    "立方米",
    "公里",
    "千米",
    "厘米",
    "毫米",
    "公斤",
    "千克",
    "毫升",
    "小时",
    "分钟",
    "人民币",
    "块钱",
    "平米",
    "米",
    "克",
    "吨",
    "升",
    "秒",
    "度",
    "岁",
    "元",
    "块",
)
_UNIT_PATTERN = "|".join(sorted(map(re.escape, _MEASURE_UNITS), key=len, reverse=True))
# Classifier quantities are unambiguous numeric contexts in transcript output.
# Keep the list focused on common spoken classifiers; fixed expressions are
# checked first and therefore still win on any overlap.
_COUNT_UNITS = (
    "个",
    "只",
    "件",
    "本",
    "张",
    "台",
    "条",
    "位",
    "名",
    "辆",
    "套",
    "箱",
    "袋",
    "杯",
    "份",
    "枚",
    "颗",
    "粒",
    "栋",
    "间",
    "户",
    "艘",
    "架",
    "门",
    "次",
)
_COUNT_UNIT_PATTERN = "|".join(
    sorted(map(re.escape, _COUNT_UNITS), key=len, reverse=True)
)
_COUNT_NUMBER_RE = re.compile(
    rf"(?P<number>{_CN_NUMBER_PATTERN})(?P<unit>{_COUNT_UNIT_PATTERN})"
)
_RANGE_RE = re.compile(
    rf"(?P<left>{_CN_NUMBER_PATTERN})(?P<separator>到|至|[-~～])"
    rf"(?P<right>{_CN_NUMBER_PATTERN})(?P<unit>{_UNIT_PATTERN})"
)
_UNIT_NUMBER_RE = re.compile(
    rf"(?P<number>{_CN_NUMBER_PATTERN})(?P<unit>{_UNIT_PATTERN})"
)
# A positional Chinese number is still a number without a trailing unit:
# ``二十三`` and ``一百二十三万``.  Bare adjacent digit runs such as ``二三``
# are deliberately excluded because they commonly express an approximation,
# list, or lexical phrase rather than one numeric value.
_POSITIONAL_NUMBER_RE = re.compile(
    rf"(?<![{_CN_INTEGER}])"
    rf"(?P<number>[{_CN_INTEGER}]*[十百千万亿][{_CN_INTEGER}]*)"
    rf"(?![{_CN_INTEGER}])"
)
_AMBIGUOUS_NUMBER_SUFFIXES = frozenset("几多来余")


@dataclass(frozen=True, slots=True)
class NumericNormalization:
    original: str
    replacement: str
    kind: str
    start: int
    end: int

    def public_dict(self) -> dict[str, object]:
        return {
            "original": self.original,
            "replacement": self.replacement,
            "kind": self.kind,
            "start": self.start,
            "end": self.end,
        }


@dataclass(frozen=True, slots=True)
class NumericNormalizationResult:
    text: str
    changes: tuple[NumericNormalization, ...]


class ContextualNumericNormalizer:
    """Normalize Chinese numbers only when their numeric role is explicit."""

    def normalize(self, text: str) -> NumericNormalizationResult:
        if not text:
            return NumericNormalizationResult(text, ())

        protected = _fixed_expression_spans(text)
        replacements: list[NumericNormalization] = []
        occupied: list[tuple[int, int]] = []

        def add(start: int, end: int, replacement: str, kind: str) -> None:
            original = text[start:end]
            if original == replacement:
                return
            if _overlaps(start, end, protected) or _overlaps(start, end, occupied):
                return
            replacements.append(
                NumericNormalization(original, replacement, kind, start, end)
            )
            occupied.append((start, end))

        for match in _PERCENT_RE.finditer(text):
            value = chinese_number_to_decimal(match.group("number"))
            if value is not None:
                add(*match.span(), f"{_format_decimal(value)}%", "percent")

        for match in _FULL_DATE_RE.finditer(text):
            year = _parse_year(match.group("year"))
            month = chinese_number_to_decimal(match.group("month"))
            day = chinese_number_to_decimal(match.group("day"))
            if year is not None and _is_integer_in(month, 1, 12) and _is_integer_in(day, 1, 31):
                add(
                    *match.span(),
                    f"{year}年{int(month)}月{int(day)}{match.group('day_unit')}",
                    "date",
                )

        for match in _YEAR_MONTH_RE.finditer(text):
            year = _parse_year(match.group("year"))
            month = chinese_number_to_decimal(match.group("month"))
            if year is not None and _is_integer_in(month, 1, 12):
                add(*match.span(), f"{year}年{int(month)}月", "date")

        for match in _YEAR_RE.finditer(text):
            year = _parse_year(match.group("year"))
            if year is not None:
                add(*match.span(), f"{year}年", "date")

        for match in _FULL_TIME_RE.finditer(text):
            hour = chinese_number_to_decimal(match.group("hour"))
            minute = chinese_number_to_decimal(match.group("minute"))
            if _is_integer_in(hour, 0, 23) and _is_integer_in(minute, 0, 59):
                suffix = "分钟" if match.group(0).endswith("分钟") else "分"
                add(*match.span(), f"{int(hour)}点{int(minute)}{suffix}", "time")

        for match in _RANGE_RE.finditer(text):
            left = chinese_number_to_decimal(match.group("left"))
            right = chinese_number_to_decimal(match.group("right"))
            if left is not None and right is not None:
                add(
                    *match.span(),
                    f"{_format_decimal(left)}{match.group('separator')}"
                    f"{_format_decimal(right)}{match.group('unit')}",
                    "measurement_range",
                )

        for match in _UNIT_NUMBER_RE.finditer(text):
            value = chinese_number_to_decimal(match.group("number"))
            if value is not None:
                add(
                    *match.span(),
                    f"{_format_decimal(value)}{match.group('unit')}",
                    "currency" if match.group("unit") in {"人民币", "块钱", "元", "块"} else "measurement",
                )

        # Classifier quantities such as ``五个苹果`` are safe to render with
        # Arabic digits.  A consecutive bare digit run (``二三个``) is an
        # approximation and stays in the source form instead of becoming
        # ``23个``.
        for match in _COUNT_NUMBER_RE.finditer(text):
            token = match.group("number")
            if _ambiguous_number_token(token, text, match.end()):
                continue
            value = chinese_number_to_decimal(token)
            if value is not None:
                add(
                    *match.span(),
                    f"{_format_decimal(value)}{match.group('unit')}",
                    "count",
                )

        # Positional numbers do not always carry an explicit unit.  Convert
        # ``二十三`` and ``一百二十三万`` while leaving approximate forms such
        # as ``二三十``/``十几`` untouched.  Fixed expressions are excluded by
        # ``add`` through their protected spans.
        for match in _POSITIONAL_NUMBER_RE.finditer(text):
            token = match.group("number")
            if _ambiguous_number_token(token, text, match.end()):
                continue
            value = chinese_number_to_decimal(token)
            if value is not None:
                add(*match.span(), _format_decimal(value), "number")

        if not replacements:
            return NumericNormalizationResult(text, ())
        output = text
        for change in sorted(replacements, key=lambda item: item.start, reverse=True):
            output = output[: change.start] + change.replacement + output[change.end :]
        return NumericNormalizationResult(
            output,
            tuple(sorted(replacements, key=lambda item: item.start)),
        )


def chinese_number_to_decimal(token: str) -> Decimal | None:
    """Parse a conventional Chinese integer or decimal expression."""

    if not token:
        return None
    if "点" in token:
        integer_text, fractional_text = token.split("点", 1)
        if not fractional_text or any(char not in _DIGITS for char in fractional_text):
            return None
        integer = chinese_number_to_decimal(integer_text)
        if integer is None or integer != integer.to_integral_value():
            return None
        fraction = "".join(str(_DIGITS[char]) for char in fractional_text)
        return Decimal(f"{int(integer)}.{fraction}")
    if any(char not in _DIGITS and char not in _SMALL_UNITS and char not in _LARGE_UNITS for char in token):
        return None
    if not any(char in _SMALL_UNITS or char in _LARGE_UNITS for char in token):
        return Decimal("".join(str(_DIGITS[char]) for char in token))

    total = 0
    section = 0
    number = 0
    for char in token:
        if char in _DIGITS:
            number = _DIGITS[char]
        elif char in _SMALL_UNITS:
            section += (number or 1) * _SMALL_UNITS[char]
            number = 0
        else:
            section += number
            section = section or 1
            total += section * _LARGE_UNITS[char]
            section = 0
            number = 0
    return Decimal(total + section + number)


def _parse_year(token: str) -> int | None:
    if all(char in _DIGITS for char in token):
        value = int("".join(str(_DIGITS[char]) for char in token))
    else:
        parsed = chinese_number_to_decimal(token)
        if parsed is None or parsed != parsed.to_integral_value():
            return None
        value = int(parsed)
    return value if 1 <= value <= 9999 else None


def _format_decimal(value: Decimal) -> str:
    if value == value.to_integral_value():
        return str(int(value))
    return format(value.normalize(), "f")


def _is_integer_in(value: Decimal | None, minimum: int, maximum: int) -> bool:
    return bool(
        value is not None
        and value == value.to_integral_value()
        and minimum <= value <= maximum
    )


def _fixed_expression_spans(text: str) -> tuple[tuple[int, int], ...]:
    return tuple(
        match.span()
        for expression in _FIXED_EXPRESSIONS
        for match in re.finditer(re.escape(expression), text)
    )


def _ambiguous_number_token(token: str, text: str, end: int) -> bool:
    """Return whether a numeral is an approximation/list, not one value.

    ``二三`` and ``二三十`` are commonly spoken as ``two or three`` and
    ``twenty or thirty``.  Treating them as the integers 23 and 30 would
    change the meaning.  A single digit before a positional unit (``二十三``)
    remains a normal number and is converted.
    """

    if len(token) >= 2 and all(char in _CN_DIGIT for char in token):
        return True
    first_unit = next(
        (index for index, char in enumerate(token) if char in "十百千万亿"),
        None,
    )
    if first_unit is not None and first_unit >= 2:
        prefix = token[:first_unit]
        if all(char in _CN_DIGIT for char in prefix):
            return True
    return text[end : end + 1] in _AMBIGUOUS_NUMBER_SUFFIXES


def _overlaps(start: int, end: int, spans: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> bool:
    return any(start < other_end and end > other_start for other_start, other_end in spans)
