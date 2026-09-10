"""Bounded Refiner input windows and conservative output-quality checks."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable


_SENTENCE_ENDINGS = frozenset("。！？!?；;\n")
_SOFT_BREAKS = frozenset("，、：:）)】] ")
_SENTENCE_RE = re.compile(r"[^。！？!?；;\n]+[。！？!?；;\n]?")


def split_for_refinement(text: str, *, max_chars: int = 200) -> tuple[str, ...]:
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
