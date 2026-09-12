"""Bounded Refiner input windows and conservative output-quality checks."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from difflib import SequenceMatcher
from decimal import Decimal, InvalidOperation

from .numeric_normalizer import _COUNT_UNITS, _FIXED_EXPRESSIONS


_SENTENCE_ENDINGS = frozenset("。！？!?；;\n")
_SOFT_BREAKS = frozenset("，、：:）)】] ")
_SENTENCE_RE = re.compile(r"[^。！？!?；;\n]+[。！？!?；;\n]?")
DEFAULT_REFINEMENT_MAX_CHARS = 80
_SEVERE_LOSS_MIN_SOURCE_CHARS = 16
_SHORT_SEGMENT_MIN_SIMILARITY = 0.35
_SEVERE_LOSS_MIN_LENGTH_RATIO = 0.65
_SHORT_LOSS_MIN_SOURCE_CHARS = 4
_SHORT_LOSS_MIN_SIMILARITY = 0.7
_CHINESE_NUMERAL_RE = re.compile(
    r"[零〇一二三四五六七八九十百千万亿两]+(?:点[零〇一二三四五六七八九]+)?"
)
_ARABIC_NUMERAL_RE = re.compile(r"[+-]?\d+(?:\.\d+)?")
_CHINESE_NUMERIC_UNITS = frozenset("十百千万亿年月日号元块点")
_CHINESE_COUNT_UNITS = frozenset(_COUNT_UNITS)
_AMBIGUOUS_NUMBER_SUFFIXES = frozenset("几多来余")
_CHINESE_DIGITS = {
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
_CHINESE_SMALL_UNITS = {"十": 10, "百": 100, "千": 1000}
_CHINESE_LARGE_UNITS = {"万": 10_000, "亿": 100_000_000}
_PLACEHOLDER_RE = re.compile(r"__ENTITY_\d{3}__")
_LOW_RISK_DELETION_CHARS = frozenset(
    "的地得了着过和及而也又很太啊呀吧呢吗嗯呃哈嘿唉哎"
)
_SEMANTIC_ANCHOR_CHARS = frozenset(
    "我你他她它们咱自己不没无未别莫勿会能可要给把被从向与此这那是就"
)
_FUNCTION_CHARS = frozenset(
    "的地得了着过是就才又也都还很太吗呢啊呀吧啦哦嗯呃和与及而给把被从向"
)
_ROLE_NUMBER_SUFFIXES = frozenset("伯叔爷奶哥姐妹弟")


def split_for_refinement(
    text: str, *, max_chars: int = DEFAULT_REFINEMENT_MAX_CHARS
) -> tuple[str, ...]:
    """Split text into bounded, punctuation-preferred Refiner windows.

    The Refiner has a finite generation budget.  Sending a long meeting
    transcript in one request can exhaust that budget and cause repetition or
    a truncated ending.  This preserves all source characters while preferring
    sentence boundaries, then softer clause boundaries.
    """

    if max_chars < 32:
        raise ValueError("max_chars must be at least 32")
    source = text.strip()
    if not source:
        return ()

    parts: list[str] = []
    start = 0
    while start < len(source):
        end = min(len(source), start + max_chars)
        if end == len(source):
            parts.append(source[start:end])
            break
        minimum_break = start + max_chars // 2
        preferred = _last_break(source, start, end, minimum_break, _SENTENCE_ENDINGS)
        soft = _last_break(source, start, end, minimum_break, _SOFT_BREAKS)
        cut = preferred or soft or end
        parts.append(source[start:cut])
        start = cut
    return tuple(part for part in parts if part)


def join_refined_segments(parts: Iterable[str]) -> str:
    """Join segment outputs without inserting unwanted spaces in Chinese text."""

    output = ""
    for part in parts:
        value = part.strip()
        if not value:
            continue
        if output and output[-1].isascii() and value[0].isascii():
            output += " "
        output += value
    return output


def reject_reasons(raw_text: str, refined_text: str) -> tuple[str, ...]:
    """Return deterministic reasons to fall back to a raw source segment."""

    raw = raw_text.strip()
    refined = refined_text.strip()
    if not refined:
        return ("empty_refiner_output",)

    reasons: list[str] = []
    raw_sentences = _sentence_counts(raw)
    for sentence, count in _sentence_counts(refined).items():
        if len(sentence) >= 8 and count >= 3 and count > raw_sentences[sentence]:
            reasons.append("repeated_sentence")
            break

    # Catch a loop without sentence-ending punctuation, such as a generated
    # phrase repeated several times before max_new_tokens is exhausted.
    repeated_phrase = re.search(r"(.{8,80}?)(?:\1){2,}", refined)
    if repeated_phrase and repeated_phrase.group(1) not in raw:
        reasons.append("repeated_phrase")

    # A fluent, punctuated hallucination can still replace most of a source
    # window with one short sentence. Do not let final punctuation bypass the
    # completeness guard. This conservative floor still permits substantial
    # filler and repetition cleanup.
    if len(raw) >= _SHORT_LOSS_MIN_SOURCE_CHARS:
        length_ratio = len(refined) / len(raw)
        similarity = SequenceMatcher(None, raw, refined, autojunk=False).ratio()
        numeric_surface_only = _numeric_surface_only(raw, refined)
        if len(raw) >= _SEVERE_LOSS_MIN_SOURCE_CHARS and (
            length_ratio < _SEVERE_LOSS_MIN_LENGTH_RATIO
            and (len(raw) >= 24 or similarity < _SHORT_SEGMENT_MIN_SIMILARITY)
            and not numeric_surface_only
            # A self-correction intentionally removes the false start.  If
            # the candidate retains the corrected tail, do not classify that
            # deliberate compression as whole-window content loss.
            and not (
                length_ratio >= 0.45
                and _retains_correction_tail(raw, refined)
            )
        ):
            reasons.append("severe_content_loss")
        # Sentence-sized windows can be short enough to bypass the long-source
        # floor above. Reject an unrelated rewrite as well, while still allowing
        # a legitimate self-correction whose final clause is retained verbatim
        # in the source (for example, ``...不对，有一个梨``).
        if (
            similarity < _SHORT_LOSS_MIN_SIMILARITY
            and refined not in raw
            and not _retains_correction_tail(raw, refined)
            and not numeric_surface_only
        ):
            reasons.append("severe_content_loss")

    # LLM cleanup can preserve the surrounding sentence while silently
    # changing a value (for example, ``两千一百三十五`` -> ``两百三十五``).
    # Length and edit-distance checks cannot detect that semantic loss, so
    # compare numeric mentions before accepting the candidate.  This also
    # permits safe Chinese-to-Arabic forms such as ``百分之五`` -> ``5%`` and
    # ``二零一五年五月五日`` -> ``2015年5月5日``.
    if (
        _numeric_values(raw) != _numeric_values(refined)
        and not _retains_correction_tail(raw, refined)
    ):
        reasons.append("numeric_value_mismatch")

    # Length and global similarity are useful for catching a collapsed window,
    # but they miss small deletions that change who did what (``他给``), or a
    # one-character substitution that turns a meaningful word into a particle
    # (``傻`` -> ``的``).  Inspect the aligned edits locally as a second line of
    # defense.  Numeric-only surface changes are handled by the numeric guard
    # above and are deliberately excluded here.
    reasons.extend(_semantic_edit_reasons(raw, refined))

    source_complete = bool(raw) and raw[-1] in _SENTENCE_ENDINGS
    target_complete = bool(refined) and refined[-1] in _SENTENCE_ENDINGS
    if (
        source_complete
        and not target_complete
        and len(raw) >= 100
        and len(refined) < len(raw) * 0.8
    ):
        reasons.append("truncated_refiner_output")
    return tuple(dict.fromkeys(reasons))


def _last_break(
    text: str, start: int, end: int, minimum: int, characters: frozenset[str]
) -> int | None:
    for index in range(end - 1, minimum - 1, -1):
        if text[index] in characters:
            return index + 1
    return None


def _sentence_counts(text: str) -> Counter[str]:
    return Counter(
        value.strip()
        for value in _SENTENCE_RE.findall(text)
        if value.strip()
    )


def _semantic_edit_reasons(raw: str, refined: str) -> tuple[str, ...]:
    """Reject local edits that can remove or invent sentence meaning.

    The Refiner is allowed to remove disfluencies, repeated runs, and the
    false start before a self-correction.  Other deleted CJK spans are treated
    conservatively: a short deletion can be semantically riskier than a large
    rewrite because the surrounding text still looks very similar.
    """

    if _numeric_surface_only(raw, refined):
        return ()

    reasons: list[str] = []
    matcher = SequenceMatcher(None, raw, refined, autojunk=False)
    for tag, source_start, source_end, target_start, target_end in matcher.get_opcodes():
        if tag == "equal":
            continue
        source = raw[source_start:source_end]
        target = refined[target_start:target_end]
        if tag == "delete":
            if _allowed_deletion(raw, refined, source_start, source_end, source):
                continue
            if _meaningful_edit_text(source):
                reasons.append("semantic_content_loss")
        elif tag == "replace":
            if _allowed_replacement(
                raw,
                refined,
                source_start,
                source_end,
                target_start,
                target_end,
                source,
                target,
            ):
                continue
            if _meaningful_edit_text(source) and len(source) - len(target) >= 2:
                reasons.append("semantic_content_loss")
            elif _suspicious_function_substitution(
                refined, target_start, target_end, source, target
            ):
                reasons.append("semantic_substitution")
        elif tag == "insert" and _suspicious_insertion(refined, target_start, target_end, target):
            reasons.append("unsupported_insertion")
    return tuple(dict.fromkeys(reasons))


def _allowed_deletion(
    raw: str, refined: str, start: int, end: int, source: str
) -> bool:
    if not _meaningful_edit_text(source):
        return True
    if _retains_correction_tail(raw, refined) and _deletion_touches_correction(raw, start, end):
        return True
    compact = _strip_edit_punctuation(source)
    if not compact:
        return True
    if all(char in _LOW_RISK_DELETION_CHARS for char in compact):
        return True
    if _is_repeated_deletion(raw, start, end, compact):
        return True
    if len(compact) == 1 and compact in _SEMANTIC_ANCHOR_CHARS:
        return False
    if len(compact) == 1 and compact in _CHINESE_DIGITS:
        following = raw[end : end + 1]
        if following in _ROLE_NUMBER_SUFFIXES:
            return False
        return True
    # Multi-character deletions are high risk unless they were explicitly
    # classified as a correction, filler, or repetition above.
    return len(compact) == 1


def _allowed_replacement(
    raw: str,
    refined: str,
    source_start: int,
    source_end: int,
    target_start: int,
    target_end: int,
    source: str,
    target: str,
) -> bool:
    if _numeric_surface_only(raw, refined):
        return True
    if _retains_correction_tail(raw, refined) and _deletion_touches_correction(
        raw, source_start, source_end
    ):
        return True
    # Pure punctuation and whitespace edits are editorial surface changes.
    if not _meaningful_edit_text(source) and not _meaningful_edit_text(target):
        return True
    # Rewriting two meaningful characters into two other meaningful characters
    # is the normal typo-correction case.  Keep it available to the Refiner;
    # the stricter checks target omission and particle substitutions.
    if len(source) == len(target) and len(source) >= 2:
        return True
    return False


def _suspicious_function_substitution(
    refined: str, target_start: int, target_end: int, source: str, target: str
) -> bool:
    if len(source) != 1 or len(target) != 1:
        return False
    if source in _FUNCTION_CHARS or target not in _FUNCTION_CHARS:
        return False
    # A content word becoming a function particle immediately before sentence
    # punctuation is especially likely to be an accidental semantic rewrite:
    # ``你当我傻？`` -> ``你当我的？``.
    return target_end >= len(refined) or refined[target_end : target_end + 1] in _SENTENCE_ENDINGS


def _suspicious_insertion(text: str, start: int, end: int, target: str) -> bool:
    compact = _strip_edit_punctuation(target)
    if len(compact) < 2:
        return False
    if all(char in _LOW_RISK_DELETION_CHARS for char in compact):
        return False
    # Do not allow a repeated content run to be copied into a second location
    # (``...打打杀杀...不会跟你打`` -> ``...不会跟你打杀杀``).
    if len(set(compact)) == 1 and text[:start].count(compact) + text[end:].count(compact):
        return True
    return False


def _deletion_touches_correction(raw: str, start: int, end: int) -> bool:
    for match in re.finditer(r"不对|不是|而是|应该是", raw):
        marker_start = match.start()
        marker_end = match.end()
        if start <= marker_end and end >= marker_start:
            return True
    return False


def _is_repeated_deletion(raw: str, start: int, end: int, compact: str) -> bool:
    if len(compact) >= 2 and len(set(compact)) == 1:
        return True
    if len(compact) >= 2 and raw[:start].endswith(compact):
        return True
    if len(compact) >= 2 and raw[end:].startswith(compact):
        return True
    # Punctuation can sit between two spoken repetitions (可恶！可恶！).
    pieces = [piece for piece in re.split(r"[，,、。！？!?；;\s]+", compact) if piece]
    return len(pieces) >= 2 and len(set(pieces)) == 1


def _strip_edit_punctuation(value: str) -> str:
    return re.sub(r"[，,、。！？!?；;：:（）()【】\[\]《》\s]", "", value)


def _meaningful_edit_text(value: str) -> bool:
    return bool(re.search(r"[\u3400-\u9fffA-Za-z0-9]", value))


def _retains_correction_tail(raw: str, refined: str) -> bool:
    """Allow a compact rewrite when it keeps the corrected clause verbatim."""

    for marker in ("不对", "不是", "而是", "应该是"):
        index = raw.rfind(marker)
        if index < 0:
            continue
        remainder = raw[index + len(marker):].lstrip(" ，,、")
        # Only the clause immediately following the correction marker is the
        # replacement target.  Do not require later independent sentences to
        # survive verbatim, otherwise a valid correction is mistaken for
        # whole-window loss merely because another clause was cleaned up.
        tail = re.split(r"[,，、.。！？!?;；\n]", remainder, maxsplit=1)[0].strip(
            " ，,、"
        )
        if tail and tail in refined:
            return True
    return False


def _numeric_values(text: str) -> tuple[tuple[Decimal, str], ...]:
    """Extract ordered numeric values with a small amount of unit context.

    Standalone Chinese ``一``/``两`` in ordinary classifier phrases is not
    treated as a numeric assertion unless it is followed by a number unit.
    This keeps normal wording cleanup from being rejected while protecting
    prices, percentages, dates, and multi-digit Chinese numerals.
    """

    values: list[tuple[int, int, Decimal, str]] = []
    occupied: list[tuple[int, int]] = []

    # Protected-span placeholders such as ``__ENTITY_000__`` must not be
    # parsed as Arabic numerals.  Masking a verified entity turns it into a
    # numbered placeholder; treating that id (``000``/``001``) as a numeric
    # value makes the masked baseline and the restored text compare unequal
    # and spuriously rejects an otherwise valid refinement with
    # ``numeric_value_mismatch``.
    for match in _PLACEHOLDER_RE.finditer(text):
        occupied.append((match.start(), match.end()))

    # Fixed idioms contain numerals that are not numbers (一五一十, 三番五次,
    # 乱七八糟).  Exclude their spans so they are never parsed as numeric
    # assertions: otherwise ``一五一十`` can be misread as 10 and a valid
    # unchanged idiom can look like a numeric change.
    for expression in _FIXED_EXPRESSIONS:
        for match in re.finditer(re.escape(expression), text):
            occupied.append((match.start(), match.end()))

    # Chinese percentages may carry a decimal part (百分之十二点五 -> 12.5%).
    for match in re.finditer(
        r"百分之([零〇一二三四五六七八九十百千万亿两]+(?:点[零〇一二三四五六七八九]+)?)",
        text,
    ):
        value = _parse_chinese_number(match.group(1))
        if value is None:
            continue
        start, end = match.span()
        values.append((start, end, value, "percent"))
        occupied.append((start, end))

    # Chinese clock times ``七点十分`` split into (hour, minute) components so
    # they align with the Arabic form ``7点10分``.
    for match in re.finditer(
        r"([零〇一二三四五六七八九十百千万亿两]+)点([零〇一二三四五六七八九十百千万亿两]+)分(?:钟)?",
        text,
    ):
        start, end = match.span()
        if _overlaps(start, end, occupied):
            continue
        hour = _parse_chinese_number(match.group(1))
        minute = _parse_chinese_number(match.group(2))
        if hour is None or minute is None:
            continue
        hour_start = match.start(1)
        minute_start = match.start(2)
        values.append((hour_start, hour_start + 1, hour, ""))
        values.append((minute_start, match.end(2), minute, ""))
        occupied.append((start, end))

    for match in _ARABIC_NUMERAL_RE.finditer(text):
        start, end = match.span()
        if _overlaps(start, end, occupied):
            continue
        token = match.group(0)
        suffix = text[end : end + 1]
        unit = "percent" if suffix in {"%", "％"} else _unit_for(suffix)
        # Mirror the Chinese-numeral guard: a single bare Arabic digit that is
        # not part of an explicit numeric context (unit, percent, decimal,
        # sign, or a multi-digit run) is not treated as a numeric assertion.
        # Otherwise idioms such as ``三番五次 -> 3番5次`` would look like a
        # numeric change and spuriously reject a valid cleanup with
        # ``numeric_value_mismatch``.
        if (
            len(token) == 1
            and not unit
            and "." not in token
            and suffix not in "点"
            and not (token[0] in "+-")
        ):
            continue
        try:
            value = Decimal(token)
        except InvalidOperation:
            continue
        values.append((start, end, value, unit))

    for match in _CHINESE_NUMERAL_RE.finditer(text):
        start, end = match.span()
        if _overlaps(start, end, occupied):
            continue
        token = match.group(0)
        suffix = text[end : end + 1]
        if _ambiguous_chinese_number(token, text, end):
            continue
        # Skip ordinary classifier phrases with an ambiguous digit run
        # (``二三个人``), but keep dates and values with an explicit unit.  A
        # single glyph such as ``十`` is different from a bare digit: it is a
        # positional Chinese numeral whose value is 10.  It must be retained
        # here so ``十个`` and the equivalent ``10个`` compare equally.  The
        # same applies to ``百``/``千``/``万``/``亿``; fixed idioms and
        # approximate forms are excluded above.
        if (
            len(token) == 1
            and token in _CHINESE_DIGITS
            and suffix not in _CHINESE_NUMERIC_UNITS
            and suffix not in _CHINESE_COUNT_UNITS
        ):
            continue
        value = _parse_chinese_number(token)
        if value is None:
            continue
        values.append((start, end, value, _unit_for(suffix)))

    values.sort(key=lambda item: item[0])
    return tuple((value, unit) for _, _, value, unit in values)


def _numeric_surface_only(raw: str, refined: str) -> bool:
    """Return whether the textual difference is only numeric formatting."""

    if not _numeric_values(raw) or _numeric_values(raw) != _numeric_values(refined):
        return False
    return _numeric_skeleton(raw) == _numeric_skeleton(refined)


def _numeric_skeleton(text: str) -> str:
    # Keep protected placeholders intact so their id digits are not treated as
    # numeric tokens and do not leak into the numeric-surface comparison.
    text = _PLACEHOLDER_RE.sub("<ENTITY>", text)
    text = re.sub(
        r"百分之[零〇一二三四五六七八九十百千万亿两]+(?:点[零〇一二三四五六七八九]+)?",
        "<NUM>",
        text,
    )
    text = _ARABIC_NUMERAL_RE.sub("<NUM>", text)
    text = _CHINESE_NUMERAL_RE.sub("<NUM>", text)
    return text.replace("%", "").replace("％", "")


def _overlaps(start: int, end: int, spans: Iterable[tuple[int, int]]) -> bool:
    return any(start < other_end and end > other_start for other_start, other_end in spans)


def _unit_for(value: str) -> str:
    if value in {"%", "％"}:
        return "percent"
    if value in {"年", "月", "日", "号"}:
        return "date"
    if value in {"元", "块"}:
        return "currency"
    if value in _CHINESE_COUNT_UNITS:
        return "count"
    return ""


def _ambiguous_chinese_number(token: str, text: str, end: int) -> bool:
    """Keep approximate/list-like Chinese digit runs out of numeric checks."""

    # A date year such as ``二零一五年`` is an explicit numeric assertion;
    # other bare adjacent digits such as ``二三个人`` commonly mean an
    # approximation and must not be equated with ``23个人``.
    if len(token) >= 2 and all(char in _CHINESE_DIGITS for char in token):
        suffix = text[end : end + 1]
        if not suffix or suffix not in "年月日号":
            return True
    first_unit = next(
        (index for index, char in enumerate(token) if char in "十百千万亿"),
        None,
    )
    if first_unit is not None and first_unit >= 2:
        prefix = token[:first_unit]
        if all(char in _CHINESE_DIGITS for char in prefix):
            return True
    return text[end : end + 1] in _AMBIGUOUS_NUMBER_SUFFIXES


def _parse_chinese_number(token: str) -> Decimal | None:
    if not token:
        return None
    # Chinese decimals: ``十二点五`` -> 12.5, ``零点五`` -> 0.5.
    if "点" in token:
        integer_part, _, fractional_part = token.partition("点")
        integer = _parse_chinese_number(integer_part) if integer_part else Decimal(0)
        if integer is None:
            return None
        if not fractional_part or any(
            char not in _CHINESE_DIGITS for char in fractional_part
        ):
            return None
        fraction = "".join(str(_CHINESE_DIGITS[char]) for char in fractional_part)
        return Decimal(f"{int(integer)}.{fraction}")
    if not any(char in _CHINESE_SMALL_UNITS or char in _CHINESE_LARGE_UNITS for char in token):
        digits = [_CHINESE_DIGITS.get(char) for char in token]
        if any(value is None for value in digits):
            return None
        return Decimal("".join(str(value) for value in digits))

    total = 0
    section = 0
    number = 0
    for char in token:
        if char in _CHINESE_DIGITS:
            number = _CHINESE_DIGITS[char]
        elif char in _CHINESE_SMALL_UNITS:
            number = number or 1
            section += number * _CHINESE_SMALL_UNITS[char]
            number = 0
        elif char in _CHINESE_LARGE_UNITS:
            section += number
            section = section or 1
            total += section * _CHINESE_LARGE_UNITS[char]
            section = 0
            number = 0
        else:
            return None
    return Decimal(total + section + number)
