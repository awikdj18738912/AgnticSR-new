"""Low-risk deterministic transcript cleanup independent of the Refiner."""

from __future__ import annotations

import re


_UTTERANCE_RE = re.compile(r"[^。！？!?\uff1b;\n]+(?:[。！？!?\uff1b;]+|\n+|$)")
_TRAILING_BOUNDARY_RE = re.compile(r"[。！？!?\uff1b;\s]+$")
_TERMINAL_BOUNDARY_RE = re.compile(r"([。！？!?\uff1b;\n]+\s*)$")
_COMMA_RE = re.compile(r"([，,])")
_VISIBLE_RE = re.compile(r"[\w\u3400-\u9fff]", re.UNICODE)
_PRONOUN_STUTTER_RE = re.compile(r"([我你您他她它这那])\1+")


def clean_transcript_deterministically(text: str) -> str:
    """Apply only exact, low-ambiguity cleanup rules to refined text."""

    return collapse_repeated_short_utterances(
        collapse_repeated_comma_items(
            collapse_repeated_pronoun_stutters(text)
        )
    )


def collapse_repeated_pronoun_stutters(text: str) -> str:
    """Collapse adjacent repeated pronouns/demonstratives used as stutters.

    The character set is deliberately narrow. Normal lexical reduplication
    such as ``人人``/``天天`` and verb reduplication such as ``看看`` remain intact.
    """

    return _PRONOUN_STUTTER_RE.sub(lambda match: match.group(1), text)


def collapse_repeated_comma_items(
    text: str,
    *,
    min_repetitions: int = 2,
    max_visible_chars: int = 8,
) -> str:
    """Collapse exact adjacent short items separated only by commas.

    Neural cleanup can be rejected when it removes a repeated filler together
    with unrelated content.  This post-merge rule handles the low-ambiguity
    remainder, for example ``不，不，不`` and ``少主，少主，少主！``.  Items
    must occupy complete comma-delimited fields in the same sentence; a word
    repeated inside a longer field therefore does not match.
    """

    if not text or min_repetitions < 2 or max_visible_chars < 1:
        return text

    return "".join(
        _collapse_comma_items_in_utterance(
            match.group(0),
            min_repetitions=min_repetitions,
            max_visible_chars=max_visible_chars,
        )
        for match in _UTTERANCE_RE.finditer(text)
    )


def _collapse_comma_items_in_utterance(
    utterance: str,
    *,
    min_repetitions: int,
    max_visible_chars: int,
) -> str:
    boundary_match = _TERMINAL_BOUNDARY_RE.search(utterance)
    if boundary_match is None:
        body, boundary = utterance, ""
    else:
        body = utterance[: boundary_match.start()]
        boundary = boundary_match.group(1)

    parts = _COMMA_RE.split(body)
    items = parts[0::2]
    separators = parts[1::2]
    if len(items) < min_repetitions:
        return utterance

    output: list[str] = []
    index = 0
    while index < len(items):
        key = _comma_item_key(items[index])
        end = index + 1
        while end < len(items) and _comma_item_key(items[end]) == key:
            end += 1
        visible_chars = len(_VISIBLE_RE.findall(key))
        collapse = (
            bool(key)
            and visible_chars <= max_visible_chars
            and end - index >= min_repetitions
        )
        if collapse:
            output.append(items[index])
            if end < len(items):
                output.append(separators[end - 1])
        else:
            for item_index in range(index, end):
                output.append(items[item_index])
                if item_index < len(separators):
                    output.append(separators[item_index])
        index = end
    return "".join(output) + boundary


def _comma_item_key(value: str) -> str:
    return re.sub(r"\s+", "", value.strip())


def collapse_repeated_short_utterances(
    text: str,
    *,
    min_repetitions: int = 2,
    max_visible_chars: int = 8,
) -> str:
    """Collapse adjacent identical short utterances repeated two times or more.

    Sentence punctuation is a chunk boundary in the streaming pipeline, so a
    run such as ``可恶！可恶！可恶！`` cannot reliably be handled by a
    per-chunk neural cleanup pass. This exact-match rule runs after chunks are
    joined. Every exact adjacent repetition from the second occurrence onward
    is removed, including deliberate spoken emphasis.
    """

    if not text or min_repetitions < 2 or max_visible_chars < 1:
        return text

    utterances = [match.group(0) for match in _UTTERANCE_RE.finditer(text)]
    if not utterances:
        return text

    output: list[str] = []
    index = 0
    while index < len(utterances):
        key = _utterance_key(utterances[index])
        end = index + 1
        while end < len(utterances) and _utterance_key(utterances[end]) == key:
            end += 1
        visible_chars = len(_VISIBLE_RE.findall(key))
        if key and visible_chars <= max_visible_chars and end - index >= min_repetitions:
            output.append(utterances[index])
        else:
            output.extend(utterances[index:end])
        index = end
    return "".join(output)


def _utterance_key(value: str) -> str:
    body = _TRAILING_BOUNDARY_RE.sub("", value).strip()
    return re.sub(r"\s+", "", body)
