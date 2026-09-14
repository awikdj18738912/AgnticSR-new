"""Low-risk deterministic transcript cleanup independent of the Refiner."""

from __future__ import annotations

import re


_UTTERANCE_RE = re.compile(r"[^。！？!?\uff1b;\n]+(?:[。！？!?\uff1b;]+|\n+|$)")
_TRAILING_BOUNDARY_RE = re.compile(r"[。！？!?\uff1b;\s]+$")
_TERMINAL_BOUNDARY_RE = re.compile(r"([。！？!?\uff1b;\n]+\s*)$")
_COMMA_RE = re.compile(r"([，,])")
_VISIBLE_RE = re.compile(r"[\w\u3400-\u9fff]", re.UNICODE)
_PRONOUN_STUTTER_RE = re.compile(r"([我你您他她它这那])\1+")
_REPEATED_CHARACTER_STUTTER_RE = re.compile(
    r"([\u3400-\u9fff])\1(?=[\u3400-\u9fff])"
)
_STANDALONE_FILLER_RE = re.compile(
    r"(?P<left>^|[，,、；;])\s*"
    r"(?P<filler>呃+|额+)\s*"
    r"(?P<right>[，,、；;。！？!?]|$)"
)

# These are common lexical reduplications, not ASR stutters.  The repeated
# character rule below is intentionally disabled for this small allowlist so
# phrases such as ``看看这里`` and ``人人平等`` remain unchanged.
_LEXICAL_REDUPLICATIONS = frozenset(
    {
        "人人", "天天", "年年", "月月", "日日", "家家", "处处", "时时",
        "事事", "步步", "层层", "面面", "头头", "句句", "字字", "件件",
        "次次", "样样", "种种", "常常", "往往", "渐渐", "慢慢", "悄悄",
        "默默", "深深", "紧紧", "牢牢", "早早", "高高", "好好", "看看",
        "听听", "说说", "想想", "试试", "问问", "走走", "聊聊", "等等",
        "刚刚", "仅仅", "偏偏", "重重", "整整", "满满", "稳稳", "远远",
        "多多", "大大", "轻轻", "缓缓", "纷纷", "偷偷", "静静", "悄悄",
        "滚滚", "滔滔", "熊熊", "翩翩", "彬彬", "济济", "赫赫", "茫茫",
        "哈哈", "呵呵", "嘿嘿", "嘻嘻", "爸爸", "妈妈", "哥哥", "姐姐",
        "弟弟", "妹妹", "爷爷", "奶奶", "叔叔", "伯伯", "姑姑", "舅舅",
        "宝宝", "娃娃", "星星", "点点", "团团", "圆圆", "毛毛", "晶晶",
        "明明", "菲菲", "婷婷", "珊珊", "萌萌", "乐乐", "念念", "津津",
        "喃喃", "依依", "楚楚", "冉冉", "芸芸", "寥寥", "区区", "惴惴",
        "惶惶", "惺惺", "铮铮", "凿凿", "孜孜", "佼佼",
    }
)


def clean_transcript_deterministically(text: str) -> str:
    """Apply only exact, low-ambiguity cleanup rules to refined text."""

    return collapse_repeated_short_utterances(
        collapse_repeated_comma_items(
            collapse_repeated_character_stutters(
                collapse_repeated_pronoun_stutters(
                    collapse_standalone_fillers(text)
                )
            )
        )
    )


def collapse_standalone_fillers(text: str) -> str:
    """Remove unambiguous standalone ``呃``/``额`` fillers.

    Only punctuation-delimited fillers are touched.  This deliberately keeps
    attached words such as ``呃逆`` and meaningful phrases such as ``嗯哼``.
    When a filler sits between two commas, one comma is retained; before a
    sentence terminator the preceding comma is removed to avoid ``，。``.
    """

    if not text:
        return text

    def replace(match: re.Match[str]) -> str:
        left = match.group("left")
        right = match.group("right")
        if right in "，,、；;":
            return "" if left == "" else left
        if right in "。！？!?":
            return right
        return ""

    return _STANDALONE_FILLER_RE.sub(replace, text)


def collapse_repeated_pronoun_stutters(text: str) -> str:
    """Collapse adjacent repeated pronouns/demonstratives used as stutters.

    The character set is deliberately narrow. Normal lexical reduplication
    such as ``人人``/``天天`` and verb reduplication such as ``看看`` remain intact.
    """

    return _PRONOUN_STUTTER_RE.sub(lambda match: match.group(1), text)


def collapse_repeated_character_stutters(text: str) -> str:
    """Collapse repeated single-character ASR stutters inside a word.

    A repeated CJK character followed by more CJK text is treated as a
    stutter unless the two-character sequence is a common lexical
    reduplication.  Requiring a following CJK character avoids changing a
    standalone lexical pair at the end of a phrase.
    """

    if not text:
        return text

    def replace(match: re.Match[str]) -> str:
        pair = match.group(0)
        return pair if pair in _LEXICAL_REDUPLICATIONS else match.group(1)

    return _REPEATED_CHARACTER_STUTTER_RE.sub(replace, text)


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
