"""Stable-text scheduling for cumulative streaming ASR hypotheses."""

from __future__ import annotations

import re
from dataclasses import dataclass


_SENTENCE_END_RE = re.compile(r"[。！？!?；;]\s*")


@dataclass(frozen=True, slots=True)
class RefinementScheduleDecision:
    should_refine: bool
    text: str
    reason: str

    def public_dict(self) -> dict[str, object]:
        return {
            "action": "refine" if self.should_refine else "defer",
            "reason": self.reason,
            "scheduled_chars": len(self.text),
        }


class StableTextRefinementScheduler:
    """Submit stable sentence prefixes and bounded unpunctuated tails once.

    Cumulative ASR hypotheses commonly change every audio chunk. Refining each
    revision repeats work on nearly identical text. This scheduler emits a new
    intermediate request only when the complete-sentence prefix changes, or
    when an unfinished tail crosses another bounded character bucket. Final
    refinement bypasses this scheduler and remains mandatory.
    """

    def __init__(self, max_unstable_chars: int = 80) -> None:
        if max_unstable_chars < 1:
            raise ValueError("max_unstable_chars must be at least 1")
        self.max_unstable_chars = max_unstable_chars
        self._stable_prefix = ""
        self._tail_bucket = 0
        self._previous_complete_sentences: tuple[str, ...] = ()

    def decide(self, text: str) -> RefinementScheduleDecision:
        source = text.strip()
        complete_sentences = _complete_sentences(source)
        common_count = 0
        while (
            common_count < len(complete_sentences)
            and common_count < len(self._previous_complete_sentences)
            and complete_sentences[common_count]
            == self._previous_complete_sentences[common_count]
        ):
            common_count += 1
        # Once another sentence follows, every earlier complete sentence is
        # no longer the recognizer's actively revised tail and can be refined.
        confirmed_count = max(common_count, max(0, len(complete_sentences) - 1))
        stable_prefix = "".join(complete_sentences[:confirmed_count]).strip()
        self._previous_complete_sentences = complete_sentences

        if stable_prefix != self._stable_prefix:
            self._stable_prefix = stable_prefix
            self._tail_bucket = 0
            if stable_prefix:
                return RefinementScheduleDecision(
                    True, stable_prefix, "stable_sentence_prefix_changed"
                )

        complete_end = len("".join(complete_sentences))
        tail = source[complete_end:].strip()
        tail_bucket = len(tail) // self.max_unstable_chars
        if tail_bucket > self._tail_bucket:
            self._tail_bucket = tail_bucket
            scheduled_tail_chars = tail_bucket * self.max_unstable_chars
            scheduled = (
                "".join(complete_sentences)
                + tail[:scheduled_tail_chars]
            ).strip()
            return RefinementScheduleDecision(
                True, scheduled, "unstable_tail_length_threshold"
            )

        return RefinementScheduleDecision(
            False, "", "awaiting_stable_sentence_boundary"
        )


def _complete_sentences(text: str) -> tuple[str, ...]:
    sentences: list[str] = []
    start = 0
    for match in _SENTENCE_END_RE.finditer(text):
        sentences.append(text[start:match.end()])
        start = match.end()
    return tuple(sentences)
