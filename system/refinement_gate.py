"""Decoupled, deterministic routing gate for ASR text refinement.

The gate only decides whether the neural Refiner should run.  Entity
normalization and output-integrity validation remain separate stages, so the
gate can be disabled for an exact A/B baseline without changing either one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Iterable


_SENTENCE_ENDINGS = frozenset("。！？!?；;")
_VISIBLE_RE = re.compile(r"[\w\u3400-\u9fff]", re.UNICODE)
_REPEATED_CJK_RE = re.compile(r"([\u3400-\u9fff])\1{2,}")
_REPEATED_SINGLE_RE = re.compile(
    r"(?:^|[，,、。！？!?；;\s])"
    r"([\u3400-\u9fff])(?:\1|[，,、。！？!?；;\s]+\1)+"
)
_REPEATED_PHRASE_RE = re.compile(r"([\u3400-\u9fff]{2,8})(?:[，,、 ]?\1){1,}")
_DISFLUENCY_RE = re.compile(
    r"(?:^|[，,。！？!?；;、\s])(?:嗯+|呃+|额+|啊+|哎+|呀+|嘿+|呼+|喂|这个|那个|就是)(?=$|[，,。！？!?；;、\s])"
)
_EMBEDDED_DISFLUENCY_RE = re.compile(r"(?:嗯+|呃+)")
_SELF_CORRECTION_RE = re.compile(r"(?:不对|我是说|应该是|准确地说|更正一下)")
_NUMERIC_NORMALIZATION_RE = re.compile(
    r"(?:百分之[零〇一二三四五六七八九十百千万亿两0-9]+"
    r"|[零〇一二三四五六七八九十百千万亿两]{2,}(?=[年月日号元块点]|人民币))"
)


class RefinementGateMode(str, Enum):
    """Supported experiment modes."""

    OFF = "off"
    CONSERVATIVE = "conservative"

    @classmethod
    def parse(cls, value: str | "RefinementGateMode") -> "RefinementGateMode":
        if isinstance(value, cls):
            return value
        try:
            return cls(value.strip().lower())
        except (AttributeError, ValueError) as error:
            choices = ", ".join(item.value for item in cls)
            raise ValueError(
                f"refinement gate mode must be one of: {choices}"
            ) from error


@dataclass(frozen=True, slots=True)
class RefinementGateDecision:
    """One auditable routing decision for one source segment."""

    mode: str
    should_refine: bool
    reasons: tuple[str, ...]
    visible_chars: int
    asr_confidence: float | None
    cleanup_signals: tuple[str, ...]
    calibrated: bool = False
    covers_segment: bool = False

    def public_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "action": "refine" if self.should_refine else "skip",
            "reasons": list(self.reasons),
            "visible_chars": self.visible_chars,
            "asr_confidence": self.asr_confidence,
            "calibrated": self.calibrated,
            "covers_segment": self.covers_segment,
            "cleanup_signals": list(self.cleanup_signals),
        }


class RefinementGate:
    """Conservative pre-Refiner gate with no model or database dependency.

    In ``off`` mode every non-empty segment follows the original Refiner path.
    In ``conservative`` mode the gate skips only very short fragments, or
    complete high-confidence text with no visible cleanup signal.  Missing ASR
    confidence, calibration evidence, or complete segment coverage is treated
    as uncertainty and therefore does not skip useful refinement work.
    """

    def __init__(
        self,
        mode: str | RefinementGateMode = RefinementGateMode.OFF,
        *,
        high_confidence: float = 0.92,
        max_short_chars: int = 2,
    ) -> None:
        self.mode = RefinementGateMode.parse(mode)
        if not 0 <= high_confidence <= 1:
            raise ValueError("high_confidence must be between 0 and 1")
        if max_short_chars < 0:
            raise ValueError("max_short_chars must be non-negative")
        self.high_confidence = high_confidence
        self.max_short_chars = max_short_chars

    def config_dict(self) -> dict[str, object]:
        """Return the effective settings for experiment manifests and logs."""

        return {
            "mode": self.mode.value,
            "high_confidence": self.high_confidence,
            "max_short_chars": self.max_short_chars,
        }

    def decide(
        self,
        text: str,
        *,
        asr_confidence: float | None = None,
        entity_hints: Iterable[str] = (),
        calibrated: bool = False,
        covers_segment: bool = False,
    ) -> RefinementGateDecision:
        source = text.strip()
        visible_chars = len(_VISIBLE_RE.findall(source))
        signals = _cleanup_signals(source)
        hints = tuple(item.strip() for item in entity_hints if item.strip())

        if self.mode is RefinementGateMode.OFF:
            return self._decision(
                True, "gate_disabled", visible_chars, asr_confidence,
                calibrated, covers_segment, signals
            )
        if not source:
            return self._decision(
                False, "empty_segment", visible_chars, asr_confidence,
                calibrated, covers_segment, signals
            )
        # An entity hint means the segment contains an unresolved candidate.
        # Never suppress the only context-aware correction opportunity.
        if hints:
            return self._decision(
                True, "entity_hint_present", visible_chars, asr_confidence,
                calibrated, covers_segment, signals
            )
        if signals:
            return self._decision(
                True, "cleanup_signal_present", visible_chars, asr_confidence,
                calibrated, covers_segment, signals
            )
        if visible_chars <= self.max_short_chars:
            return self._decision(
                False, "too_short_for_safe_refinement", visible_chars,
                asr_confidence, calibrated, covers_segment, signals
            )
        if asr_confidence is None:
            return self._decision(
                True, "confidence_unavailable", visible_chars, None,
                calibrated, covers_segment, signals
            )
        if not calibrated:
            return self._decision(
                True, "confidence_uncalibrated", visible_chars, asr_confidence,
                calibrated, covers_segment, signals
            )
        if not covers_segment:
            return self._decision(
                True, "confidence_coverage_incomplete", visible_chars,
                asr_confidence, calibrated, covers_segment, signals
            )
        if asr_confidence < self.high_confidence:
            return self._decision(
                True, "confidence_below_skip_threshold", visible_chars,
                asr_confidence, calibrated, covers_segment, signals
            )
        if source[-1] not in _SENTENCE_ENDINGS:
            return self._decision(
                True, "sentence_incomplete", visible_chars, asr_confidence,
                calibrated, covers_segment, signals
            )
        return self._decision(
            False, "high_confidence_clean_segment", visible_chars,
            asr_confidence, calibrated, covers_segment, signals
        )

    def _decision(
        self,
        should_refine: bool,
        reason: str,
        visible_chars: int,
        asr_confidence: float | None,
        calibrated: bool,
        covers_segment: bool,
        signals: tuple[str, ...],
    ) -> RefinementGateDecision:
        return RefinementGateDecision(
            mode=self.mode.value,
            should_refine=should_refine,
            reasons=(reason,),
            visible_chars=visible_chars,
            asr_confidence=asr_confidence,
            cleanup_signals=signals,
            calibrated=calibrated,
            covers_segment=covers_segment,
        )


def _cleanup_signals(text: str) -> tuple[str, ...]:
    signals: list[str] = []
    if _REPEATED_CJK_RE.search(text) or _REPEATED_SINGLE_RE.search(text):
        signals.append("repeated_character")
    if _REPEATED_PHRASE_RE.search(text):
        signals.append("repeated_phrase")
    if _DISFLUENCY_RE.search(text):
        signals.append("disfluency")
    if _has_embedded_disfluency(text):
        signals.append("embedded_disfluency")
    if _SELF_CORRECTION_RE.search(text):
        signals.append("self_correction")
    if _NUMERIC_NORMALIZATION_RE.search(text):
        signals.append("numeric_normalization")
    if "  " in text or "，，" in text or "。。" in text:
        signals.append("malformed_spacing_or_punctuation")
    return tuple(signals)


def _has_embedded_disfluency(text: str) -> bool:
    """Detect low-ambiguity fillers attached to surrounding Chinese text."""

    for match in _EMBEDDED_DISFLUENCY_RE.finditer(text):
        suffix = text[match.end() : match.end() + 1]
        # Preserve the meaningful response 嗯哼 and the medical term 呃逆.
        if (match.group().startswith("嗯") and suffix == "哼") or (
            match.group().startswith("呃") and suffix == "逆"
        ):
            continue
        remainder = text[: match.start()] + text[match.end() :]
        if _VISIBLE_RE.search(remainder):
            return True
    return False
