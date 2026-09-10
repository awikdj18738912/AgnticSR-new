"""Deterministic protected-span masking and restoration for ASR refinement."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

from .entity_store import EntityDefinition
from .session_memory import SessionEntityMemory

if TYPE_CHECKING:
    from .entity_matcher import EntityMatch


_PLACEHOLDER_RE = re.compile(r"__ENTITY_\d{3}__")
_REFINER_KEY_SUFFIX_RE = re.compile(r"\s*<KEY>\[[^\]]*\]\s*$")
_RULE_PATTERNS: tuple[tuple[str, re.Pattern[str], int], ...] = (
    ("URL", re.compile(r"https?://[^\s，。！？；]+", re.IGNORECASE), 900),
    (
        "EMAIL",
        re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
        890,
    ),
    (
        "DATE",
        re.compile(
            r"(?<!\d)(?:\d{4}[./-]\d{1,2}[./-]\d{1,2}"
            r"|(?:\d{4}年)?\d{1,2}月\d{1,2}日?)(?!\d)"
        ),
        880,
    ),
    (
        "TIME",
        re.compile(r"(?<!\d)(?:[01]?\d|2[0-3])[:：][0-5]\d(?:[:：][0-5]\d)?(?!\d)"),
        870,
    ),
    (
        "NUMBER",
        re.compile(
            r"(?<![A-Za-z0-9_.])[+-]?\d+(?:\.\d+)?(?:%|％)?"
            r"(?![A-Za-z0-9_.])"
        ),
        700,
    ),
    (
        "IDENTIFIER",
        re.compile(
            r"(?<![A-Za-z0-9])[A-Za-z][A-Za-z0-9]*"
            r"(?:[-_.][A-Za-z0-9]+)+(?![A-Za-z0-9])"
        ),
        650,
    ),
    (
        "ACRONYM",
        re.compile(r"(?<![A-Za-z0-9])[A-Z][A-Z0-9]{1,}(?![A-Za-z0-9])"),
        640,
    ),
)


@dataclass(frozen=True, slots=True)
class ProtectedSpan:
    placeholder: str
    start: int
    end: int
    original: str
    replacement: str
    entity_type: str
    source: str
    entity_id: int | None = None
    match_type: str = "RULE"
    match_score: float | None = None
    decision: str = "EXACT"

    def public_dict(self) -> dict[str, object]:
        return {
            "placeholder": self.placeholder,
            "original": self.original,
            "replacement": self.replacement,
            "entity_type": self.entity_type,
            "source": self.source,
            "entity_id": self.entity_id,
            "match_type": self.match_type,
            "match_score": self.match_score,
            "decision": self.decision,
        }


@dataclass(frozen=True, slots=True)
class ProtectionResult:
    original_text: str
    masked_text: str
    spans: tuple[ProtectedSpan, ...]


@dataclass(frozen=True, slots=True)
class RestorationResult:
    text: str
    accepted: bool
    reject_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Candidate:
    start: int
    end: int
    original: str
    replacement: str
    entity_type: str
    source: str
    priority: int
    entity_id: int | None = None
    match_type: str = "RULE"
    match_score: float | None = None
    decision: str = "EXACT"


class EntityProtector:
    """Mask verified entities and deterministic sensitive token patterns."""

    def __init__(
        self,
        entities: Iterable[EntityDefinition] = (),
        *,
        session_memory: SessionEntityMemory | None = None,
    ) -> None:
        self.entities = tuple(entities)
        self.session_memory = session_memory

    def protect(
        self,
        text: str,
        *,
        confidence: float | None = None,
        now_ms: float | None = None,
        fuzzy_matches: Iterable["EntityMatch"] = (),
    ) -> ProtectionResult:
        observed_at = now_ms if now_ms is not None else time.monotonic() * 1000
        candidates: list[_Candidate] = []

        for definition in self.entities:
            surfaces = (definition.canonical_text, *definition.aliases)
            for surface in dict.fromkeys(surfaces):
                for start, end, original in _literal_matches(text, surface):
                    replacement = (
                        definition.canonical_text
                        if definition.normalization_policy == "normalize"
                        else original
                    )
                    candidates.append(
                        _Candidate(
                            start=start,
                            end=end,
                            original=original,
                            replacement=replacement,
                            entity_type=definition.entity_type,
                            source=f"database:{definition.domain}",
                            priority=1000 + definition.priority,
                            entity_id=definition.entity_id,
                            match_type=(
                                "EXACT_CANONICAL"
                                if original.casefold() == definition.canonical_text.casefold()
                                else "EXACT_ALIAS"
                            ),
                            match_score=1.0,
                            decision="EXACT",
                        )
                    )
                    if self.session_memory is not None:
                        self.session_memory.observe(
                            definition.canonical_text,
                            entity_type=definition.entity_type,
                            aliases=definition.aliases,
                            confidence=confidence,
                            verified=True,
                            source=definition.source,
                            now_ms=observed_at,
                        )

        if self.session_memory is not None:
            for memory in self.session_memory.trusted(now_ms=observed_at):
                surfaces = (memory.canonical_text, *sorted(memory.aliases))
                for surface in dict.fromkeys(surfaces):
                    for start, end, original in _literal_matches(text, surface):
                        candidates.append(
                            _Candidate(
                                start=start,
                                end=end,
                                original=original,
                                replacement=original,
                                entity_type=memory.entity_type,
                                source=f"session:{memory.trust_level.value}",
                                priority=950,
                                match_type="SESSION_EXACT",
                                match_score=1.0,
                            )
                        )

        for entity_type, pattern, priority in _RULE_PATTERNS:
            for match in pattern.finditer(text):
                candidates.append(
                    _Candidate(
                        start=match.start(),
                        end=match.end(),
                        original=match.group(),
                        replacement=match.group(),
                        entity_type=entity_type,
                        source="rule",
                        priority=priority,
                        match_type="RULE",
                    )
                )

        rule_ranges = tuple(
            (candidate.start, candidate.end)
            for candidate in candidates
            if candidate.source == "rule"
        )
        definitions = {definition.entity_id: definition for definition in self.entities}
        for match in fuzzy_matches:
            definition = definitions.get(match.entity_id)
            if (
                match.decision.value != "AUTO_NORMALIZE"
                or definition is None
                or not definition.enabled
                or definition.normalization_policy != "normalize"
                or definition.canonical_text != match.canonical
                or not (0 <= match.start < match.end <= len(text))
                or text[match.start : match.end] != match.observed
                or any(
                    match.start < rule_end and rule_start < match.end
                    for rule_start, rule_end in rule_ranges
                )
            ):
                continue
            candidates.append(
                _Candidate(
                    start=match.start,
                    end=match.end,
                    original=match.observed,
                    replacement=definition.canonical_text,
                    entity_type=definition.entity_type,
                    source=f"fuzzy:{definition.domain}",
                    priority=980 + definition.priority,
                    entity_id=definition.entity_id,
                    match_type=match.match_type,
                    match_score=match.final_score,
                    decision=match.decision.value,
                )
            )

        selected = _select_non_overlapping(candidates)
        spans: list[ProtectedSpan] = []
        parts: list[str] = []
        cursor = 0
        for index, candidate in enumerate(selected):
            placeholder = f"__ENTITY_{index:03d}__"
            parts.append(text[cursor : candidate.start])
            parts.append(placeholder)
            spans.append(
                ProtectedSpan(
                    placeholder=placeholder,
                    start=candidate.start,
                    end=candidate.end,
                    original=candidate.original,
                    replacement=candidate.replacement,
                    entity_type=candidate.entity_type,
                    source=candidate.source,
                    entity_id=candidate.entity_id,
                    match_type=candidate.match_type,
                    match_score=candidate.match_score,
                    decision=candidate.decision,
                )
            )
            cursor = candidate.end
        parts.append(text[cursor:])
        return ProtectionResult(text, "".join(parts), tuple(spans))

    def restore(self, refined_text: str, protection: ProtectionResult) -> RestorationResult:
        output = _REFINER_KEY_SUFFIX_RE.sub("", refined_text).strip()
        if not output or "<KEY>" in output:
            return RestorationResult(protection.original_text, False, ("empty_or_metadata_output",))
        remainder = _PLACEHOLDER_RE.sub("", output)
        if "__ENTITY_" in remainder:
            return RestorationResult(protection.original_text, False, ("malformed_placeholder",))
        if not protection.spans:
            return RestorationResult(output, True, ())

        expected = [span.placeholder for span in protection.spans]
        reasons: list[str] = []
        unknown = sorted(set(_PLACEHOLDER_RE.findall(output)) - set(expected))
        if unknown:
            reasons.append(f"unknown_placeholders:{','.join(unknown)}")

        positions: list[int] = []
        for placeholder in expected:
            count = output.count(placeholder)
            if count != 1:
                reasons.append(f"placeholder_count:{placeholder}:{count}")
            else:
                positions.append(output.index(placeholder))
        if len(positions) == len(expected) and positions != sorted(positions):
            reasons.append("placeholder_order_changed")

        if reasons:
            return RestorationResult(protection.original_text, False, tuple(reasons))

        restored = output
        for span in protection.spans:
            restored = restored.replace(span.placeholder, span.replacement)
        return RestorationResult(restored, True, ())

    def audit_unmasked(
        self, refined_text: str, protection: ProtectionResult
    ) -> tuple[str, ...]:
        """Report trusted-entity losses after a Refiner update."""
        output = refined_text.strip()
        if not output:
            return ("empty_refiner_output",)
        issues: list[str] = []
        for span in protection.spans:
            if span.source == "rule":
                continue
            if span.original not in output and span.replacement not in output:
                issues.append(f"missing_{span.entity_type}")
        return tuple(dict.fromkeys(issues))

    def refinement_hints(self, protection: ProtectionResult) -> tuple[str, ...]:
        """Return verified canonical names that were explicitly found in the ASR text.

        Only database spans with a ``normalize`` policy have a canonical value
        different from their observed surface.  Supplying only these exact
        matches keeps the Refiner glossary short and avoids turning unrelated
        database entries into a source of hallucinated corrections.
        """

        return tuple(
            dict.fromkeys(
                span.replacement
                for span in protection.spans
                if span.source.startswith("database:")
                and span.replacement != span.original
            )
        )

    def normalize_verified_aliases(
        self, refined_text: str, protection: ProtectionResult
    ) -> tuple[str, tuple[dict[str, str], ...]]:
        """Normalize only explicitly detected database aliases in Refiner output.

        This is deliberately not fuzzy matching: a database record can change
        output only after its exact alias appeared in the ASR hypothesis.  The
        returned records make every deterministic replacement observable in
        the API and session log.
        """

        # ``<KEY>`` is a Refiner input annotation, not user-facing text. Some
        # checkpoints echo it despite the prompt, so remove a trailing echo
        # before applying a verified replacement.
        normalized = _REFINER_KEY_SUFFIX_RE.sub("", refined_text).strip()
        changes: list[dict[str, str]] = []
        for span in protection.spans:
            if (
                not span.source.startswith("database:")
                or span.original == span.replacement
                or span.original not in normalized
            ):
                continue
            normalized = normalized.replace(span.original, span.replacement)
            changes.append(
                {
                    "observed": span.original,
                    "canonical": span.replacement,
                    "entity_type": span.entity_type,
                }
            )
        return normalized, tuple(changes)


def _literal_matches(text: str, surface: str) -> list[tuple[int, int, str]]:
    if not surface:
        return []
    flags = re.IGNORECASE if surface.isascii() else 0
    return [
        (match.start(), match.end(), match.group())
        for match in re.finditer(re.escape(surface), text, flags=flags)
    ]


def _select_non_overlapping(candidates: Iterable[_Candidate]) -> tuple[_Candidate, ...]:
    ranked = sorted(
        candidates,
        key=lambda item: (-item.priority, -(item.end - item.start), item.start),
    )
    selected: list[_Candidate] = []
    for candidate in ranked:
        if any(
            candidate.start < existing.end and existing.start < candidate.end
            for existing in selected
        ):
            continue
        selected.append(candidate)
    return tuple(sorted(selected, key=lambda item: item.start))
