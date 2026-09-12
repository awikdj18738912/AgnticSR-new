"""Text chunking for partial streaming ASR hypotheses."""

from __future__ import annotations

import re
from dataclasses import dataclass

_SENTENCE_END = re.compile(r"[.!?\u3002\uff01\uff1f]\s*")
_ANY_PUNCTUATION = re.compile(
    r"[,;:!?\uff0c\u3002\uff01\uff1f\uff1b\uff1a\u3001]\s*"
)
_SELF_CORRECTION = re.compile(r"(?:不对|不是|而是|应该是)")
_SELF_CORRECTION_PREFIX = re.compile(r"^(?:不对|不是|而是|应该是)")
_SOFT_CORRECTION_PUNCTUATION = re.compile(r"[,，;；:：、]\s*")
_SENTENCE_END_CHARACTERS = frozenset(".!?。！？")


@dataclass(frozen=True, slots=True)
class Chunk:
    """One immutable raw-ASR chunk."""

    index: int
    text: str


class ChunkManager:
    """Convert evolving ASR hypotheses into bounded, stable text chunks.

    Hypotheses are cumulative within a VAD segment. Sentence-final punctuation
    closes a chunk. Long clauses containing a self-correction marker
    (``不对``/``不是``/``而是``) may close at the next clause punctuation even
    before the hard character limit. Text longer than ``max_chars`` is cut at
    the nearest preceding punctuation, or exactly at the limit when no
    punctuation exists.
    A VAD boundary flushes the remaining text and starts a new hypothesis.
    """

    def __init__(self, max_chars: int = 80) -> None:
        if max_chars < 1:
            raise ValueError("max_chars must be at least 1")
        self.max_chars = max_chars
        self._hypothesis = ""
        self._committed_source = ""
        self._next_index = 0

    @property
    def pending_text(self) -> str:
        """Source text accepted since the most recently emitted chunk."""

        return self._pending

    @property
    def _pending(self) -> str:
        return self._hypothesis[len(self._committed_source) :]

    def update(self, hypothesis: str, *, vad_boundary: bool = False) -> list[Chunk]:
        """Accept a cumulative partial hypothesis and emit newly stable chunks."""

        normalized = hypothesis.strip()
        if self._committed_source and not normalized.startswith(self._committed_source):
            raise ValueError(
                "ASR revised text that has already been committed; start a new VAD "
                "segment or delay chunk emission"
            )
        self._hypothesis = normalized
        chunks = self._emit_available(flush=vad_boundary)
        if vad_boundary:
            self._hypothesis = ""
            self._committed_source = ""
        return chunks

    def flush(self) -> list[Chunk]:
        """Close the current segment even when the ASR produced no VAD event."""

        return self.update(self._hypothesis, vad_boundary=True)

    def _emit_available(self, *, flush: bool) -> list[Chunk]:
        emitted: list[Chunk] = []
        while pending := self._pending:
            split = self._split_point(pending, flush=flush)
            if split is None:
                break
            source = pending[:split]
            text = source.strip()
            self._committed_source += source
            if not text:
                continue
            emitted.append(Chunk(index=self._next_index, text=text))
            self._next_index += 1
        return merge_self_correction_chunks(emitted, self.max_chars)

    def _split_point(self, text: str, *, flush: bool) -> int | None:
        correction_end = self._self_correction_break(text)
        if correction_end is not None:
            return correction_end

        sentence_end = _SENTENCE_END.search(text)
        if sentence_end is not None and sentence_end.end() <= self.max_chars:
            return sentence_end.end()

        if len(text) > self.max_chars:
            candidates = list(_ANY_PUNCTUATION.finditer(text[: self.max_chars]))
            return candidates[-1].end() if candidates else self.max_chars

        return len(text) if flush else None

    def _self_correction_break(self, text: str) -> int | None:
        """Return a soft boundary for a long self-correction clause."""

        # Keep short examples such as ``苹果，不对，梨`` together so the
        # Refiner sees the full correction. For longer hypotheses, separate
        # the corrected clause before another edit is included in the same
        # generation request.
        if len(text) <= max(1, self.max_chars // 2):
            return None
        for marker in _SELF_CORRECTION.finditer(text):
            search_start = marker.end()
            while (
                search_start < len(text)
                and text[search_start] in " \t,，;；:：、"
            ):
                search_start += 1
            punctuation = _SOFT_CORRECTION_PUNCTUATION.search(
                text,
                search_start,
                min(len(text), self.max_chars),
            )
            if punctuation is not None:
                return punctuation.end()
        return None


def merge_self_correction_chunks(
    chunks: list[Chunk], max_chars: int = 80
) -> list[Chunk]:
    """Keep a sentence and a following correction in the same chunk.

    ASR punctuation may end the abandoned clause before emitting ``不对``.
    Merging that prefix prevents the gate from committing the wrong clause
    before the correction becomes visible.
    """

    merged: list[Chunk] = []
    for chunk in chunks:
        if (
            merged
            and _SELF_CORRECTION_PREFIX.match(chunk.text)
            and merged[-1].text.endswith(tuple(_SENTENCE_END_CHARACTERS))
            and len(merged[-1].text) + len(chunk.text) <= max_chars
        ):
            previous = merged[-1]
            merged[-1] = Chunk(previous.index, previous.text + chunk.text)
        else:
            merged.append(chunk)
    return merged
