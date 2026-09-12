"""Shared entity preparation and restoration used by every inference entry point."""

from __future__ import annotations

from dataclasses import dataclass

from .entity_matcher import EntityCandidateMatcher, EntityMatchReport
from .protection import EntityProtector, ProtectionResult
from .refinement_guard import reject_reasons


@dataclass(frozen=True, slots=True)
class PreparedEntitySegment:
    source_text: str
    protection: ProtectionResult
    baseline_text: str
    hints: tuple[str, ...]
    report: EntityMatchReport

    @property
    def normalizations(self) -> tuple[dict[str, object], ...]:
        return tuple(
            {
                "observed": span.original,
                "canonical": span.replacement,
                "entity_type": span.entity_type,
                "entity_id": span.entity_id,
                "match_type": span.match_type,
                "match_score": span.match_score,
            }
            for span in self.protection.spans
            if span.original != span.replacement
        )


@dataclass(frozen=True, slots=True)
class FinalizedEntitySegment:
    text: str
    accepted: bool
    reject_reasons: tuple[str, ...]


_PLACEHOLDER_REJECT_PREFIXES = (
    "placeholder_count:",
    "unknown_placeholders:",
    "placeholder_order_changed",
    "malformed_placeholder",
)


def has_placeholder_failure(reasons: tuple[str, ...]) -> bool:
    """Return whether rejection was caused by damaged protected markers."""

    return any(reason.startswith(_PLACEHOLDER_REJECT_PREFIXES) for reason in reasons)


def has_retryable_integrity_failure(reasons: tuple[str, ...]) -> bool:
    """Return whether a stricter second refinement attempt can recover output."""

    return has_placeholder_failure(reasons) or any(
        reason in {
            "severe_content_loss",
            "truncated_refiner_output",
            "entity_sentence_boundary_lost",
            "numeric_value_mismatch",
            "semantic_content_loss",
            "semantic_substitution",
            "unsupported_insertion",
        }
        for reason in reasons
    )


def prepare_entity_segment(
    text: str,
    protector: EntityProtector,
    matcher: EntityCandidateMatcher | None,
    *,
    allow_auto: bool,
    confidence: float | None = None,
) -> PreparedEntitySegment:
    exact = protector.protect(text, confidence=confidence)
    if matcher is None:
        report = EntityMatchReport((), 0.0)
    else:
        report = matcher.match(
            text,
            allow_auto=allow_auto,
            blocked_spans=((span.start, span.end) for span in exact.spans),
        )
    protection = protector.protect(
        text,
        confidence=confidence,
        fuzzy_matches=report.auto_matches,
    )
    baseline = protector.restore(protection.masked_text, protection)
    if not baseline.accepted:
        raise RuntimeError(
            "internal entity baseline restoration failed: "
            + ",".join(baseline.reject_reasons)
        )
    max_hints = matcher.config.max_refiner_hints if matcher is not None else 16
    hints = tuple(
        dict.fromkeys(
            (*protector.refinement_hints(protection), *report.hint_canonicals)
        )
    )[:max_hints]
    return PreparedEntitySegment(text, protection, baseline.text, hints, report)


def finalize_entity_segment(
    refined_masked_text: str,
    prepared: PreparedEntitySegment,
    protector: EntityProtector,
) -> FinalizedEntitySegment:
    restored = protector.restore(refined_masked_text, prepared.protection)
    if not restored.accepted:
        return FinalizedEntitySegment(
            prepared.baseline_text,
            False,
            restored.reject_reasons,
        )
    reasons = (
        *_introduced_hint_reasons(prepared, restored.text),
        *reject_reasons(prepared.baseline_text, restored.text),
    )
    if reasons:
        return FinalizedEntitySegment(prepared.baseline_text, False, reasons)
    return FinalizedEntitySegment(restored.text, True, ())


def _introduced_hint_reasons(
    prepared: PreparedEntitySegment, restored_text: str
) -> tuple[str, ...]:
    """Reject glossary entities added without consuming a matching ASR span."""

    reasons: list[str] = []
    for canonical in prepared.hints:
        added = max(
            0,
            restored_text.count(canonical) - prepared.baseline_text.count(canonical),
        )
        if not added:
            continue
        allowed = 0
        for match in prepared.report.matches:
            if match.canonical != canonical:
                continue
            removed = max(
                0,
                prepared.baseline_text.count(match.observed)
                - restored_text.count(match.observed),
            )
            allowed += min(1, removed)
        if added > allowed:
            reasons.append(f"introduced_entity_from_hint:{canonical}")
    return tuple(reasons)
