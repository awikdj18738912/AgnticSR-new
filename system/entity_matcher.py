"""Deterministic, dictionary-constrained fuzzy matching for ASR entities."""

from __future__ import annotations

import json
import time
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable

from .entity_store import EntityDefinition


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "entity_matching.json"
_HARD_BOUNDARIES = frozenset("。！？!?\n\r")
_IGNORED_CATEGORIES = frozenset({"P", "Z"})


class EntityDecision(str, Enum):
    EXACT = "EXACT"
    AUTO_NORMALIZE = "AUTO_NORMALIZE"
    HINT_ONLY = "HINT_ONLY"
    KEEP_RAW = "KEEP_RAW"


class EntityFuzzyMode(str, Enum):
    OFF = "off"
    SHADOW = "shadow"
    HINT = "hint"
    AUTO = "auto"

    @classmethod
    def parse(cls, value: str | "EntityFuzzyMode") -> "EntityFuzzyMode":
        if isinstance(value, cls):
            return value
        try:
            return cls(value.strip().lower())
        except (AttributeError, ValueError) as error:
            choices = ", ".join(item.value for item in cls)
            raise ValueError(f"entity fuzzy mode must be one of: {choices}") from error


@dataclass(frozen=True, slots=True)
class EntityMatchConfig:
    version: int
    pinyin_weight: float
    character_weight: float
    length_weight: float
    domain_weight: float
    priority_weight: float
    hint_score: float
    auto_score: float
    auto_min_pinyin: float
    auto_min_margin: float
    max_entities: int
    max_candidates_per_span: int
    max_logged_candidates_per_segment: int
    max_refiner_hints: int
    auto_entity_types: frozenset[str]
    high_risk_entity_types: frozenset[str]

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CONFIG_PATH) -> "EntityMatchConfig":
        with Path(path).open(encoding="utf-8") as handle:
            payload = json.load(handle)
        try:
            weights = payload["weights"]
            thresholds = payload["thresholds"]
            limits = payload["limits"]
            config = cls(
                version=int(payload["version"]),
                pinyin_weight=float(weights["pinyin"]),
                character_weight=float(weights["character"]),
                length_weight=float(weights["length"]),
                domain_weight=float(weights["domain"]),
                priority_weight=float(weights["priority"]),
                hint_score=float(thresholds["hint_score"]),
                auto_score=float(thresholds["auto_score"]),
                auto_min_pinyin=float(thresholds["auto_min_pinyin"]),
                auto_min_margin=float(thresholds["auto_min_margin"]),
                max_entities=int(limits["max_entities"]),
                max_candidates_per_span=int(limits["max_candidates_per_span"]),
                max_logged_candidates_per_segment=int(
                    limits["max_logged_candidates_per_segment"]
                ),
                max_refiner_hints=int(limits["max_refiner_hints"]),
                auto_entity_types=frozenset(
                    str(value).upper() for value in payload["auto_entity_types"]
                ),
                high_risk_entity_types=frozenset(
                    str(value).upper() for value in payload["high_risk_entity_types"]
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid entity matching config: {error}") from error
        config.validate()
        return config

    def validate(self) -> None:
        weights = (
            self.pinyin_weight,
            self.character_weight,
            self.length_weight,
            self.domain_weight,
            self.priority_weight,
        )
        if any(value < 0 or value > 1 for value in weights):
            raise ValueError("entity matching weights must be between 0 and 1")
        if abs(sum(weights) - 1.0) > 1e-9:
            raise ValueError("entity matching weights must sum to 1")
        thresholds = (
            self.hint_score,
            self.auto_score,
            self.auto_min_pinyin,
            self.auto_min_margin,
        )
        if any(value < 0 or value > 1 for value in thresholds):
            raise ValueError("entity matching thresholds must be between 0 and 1")
        if self.hint_score > self.auto_score:
            raise ValueError("hint_score must not exceed auto_score")
        limits = (
            self.max_entities,
            self.max_candidates_per_span,
            self.max_logged_candidates_per_segment,
            self.max_refiner_hints,
        )
        if any(value < 1 for value in limits):
            raise ValueError("entity matching limits must be positive")


@dataclass(frozen=True, slots=True)
class EntityMatch:
    start: int
    end: int
    observed: str
    entity_id: int
    canonical: str
    entity_type: str
    domain: str
    matched_surface: str
    match_type: str
    char_score: float
    pinyin_score: float
    length_score: float
    domain_score: float
    priority_score: float
    final_score: float
    runner_up_score: float | None
    margin: float | None
    priority: int
    decision: EntityDecision
    reasons: tuple[str, ...]

    def public_dict(self) -> dict[str, object]:
        return {
            "start": self.start,
            "end": self.end,
            "observed": self.observed,
            "entity_id": self.entity_id,
            "canonical": self.canonical,
            "entity_type": self.entity_type,
            "domain": self.domain,
            "matched_surface": self.matched_surface,
            "match_type": self.match_type,
            "char_score": round(self.char_score, 4),
            "pinyin_score": round(self.pinyin_score, 4),
            "length_score": round(self.length_score, 4),
            "domain_score": round(self.domain_score, 4),
            "priority_score": round(self.priority_score, 4),
            "final_score": round(self.final_score, 4),
            "runner_up_score": (
                round(self.runner_up_score, 4)
                if self.runner_up_score is not None
                else None
            ),
            "margin": round(self.margin, 4) if self.margin is not None else None,
            "decision": self.decision.value,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True, slots=True)
class EntityMatchReport:
    matches: tuple[EntityMatch, ...]
    matcher_latency_ms: float

    @property
    def auto_matches(self) -> tuple[EntityMatch, ...]:
        return tuple(
            match
            for match in self.matches
            if match.decision is EntityDecision.AUTO_NORMALIZE
        )

    @property
    def hint_canonicals(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                match.canonical
                for match in self.matches
                if match.decision is EntityDecision.HINT_ONLY
            )
        )


@dataclass(frozen=True, slots=True)
class _Surface:
    definition: EntityDefinition
    value: str
    normalized: str
    pinyin: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Scored:
    surface: _Surface
    char_score: float
    pinyin_score: float
    length_score: float
    domain_score: float
    priority_score: float
    final_score: float


class EntityCandidateMatcher:
    """Recall and rank fuzzy spans against verified entities only."""

    def __init__(
        self,
        entities: Iterable[EntityDefinition],
        *,
        selected_domain: str = "general",
        config: EntityMatchConfig | None = None,
        mode: str | EntityFuzzyMode = EntityFuzzyMode.SHADOW,
    ) -> None:
        self.config = config or EntityMatchConfig.load()
        self.mode = EntityFuzzyMode.parse(mode)
        self.selected_domain = selected_domain.strip() or "general"
        definitions = tuple(entities)
        if len(definitions) > self.config.max_entities:
            raise ValueError(
                f"entity count {len(definitions)} exceeds configured limit "
                f"{self.config.max_entities}"
            )
        invalid_domains = {
            item.domain
            for item in definitions
            if item.domain not in {self.selected_domain, "general"}
        }
        if invalid_domains:
            raise ValueError(
                "matcher received entities outside selected domain: "
                + ", ".join(sorted(invalid_domains))
            )
        self._entities = {item.entity_id: item for item in definitions if item.enabled}
        self._surfaces: list[_Surface] = []
        self._char_index: dict[str, set[int]] = defaultdict(set)
        self._pinyin_index: dict[str, set[int]] = defaultdict(set)
        self._lengths: set[int] = set()
        for definition in self._entities.values():
            for value in dict.fromkeys((definition.canonical_text, *definition.aliases)):
                normalized = _normalize(value)
                if not normalized:
                    continue
                surface = _Surface(definition, value, normalized, _pinyin_tokens(normalized))
                surface_index = len(self._surfaces)
                self._surfaces.append(surface)
                self._lengths.add(len(normalized))
                for gram in _bigrams(normalized):
                    self._char_index[gram].add(surface_index)
                for gram in _bigrams(surface.pinyin):
                    self._pinyin_index[gram].add(surface_index)

    def match(
        self,
        text: str,
        *,
        allow_auto: bool,
        blocked_spans: Iterable[tuple[int, int]] = (),
    ) -> EntityMatchReport:
        started = time.perf_counter()
        if self.mode is EntityFuzzyMode.OFF or not text or not self._surfaces:
            return EntityMatchReport((), (time.perf_counter() - started) * 1000)
        blocked = tuple(blocked_spans)
        span_scores: dict[tuple[int, int], dict[int, _Scored]] = defaultdict(dict)
        for run_start, run_text in _text_runs(text):
            normalized, offsets = _normalize_with_offsets(run_text, run_start)
            if len(normalized) < 2:
                continue
            # Pinyin conversion is the expensive part of matching. Convert a
            # bounded run once and slice the aligned token sequence for every
            # candidate window instead of invoking pypinyin thousands of times.
            run_pinyin = _pinyin_tokens(normalized)
            possible_lengths = _possible_window_lengths(self._lengths)
            for start_index in range(len(normalized)):
                for length in possible_lengths:
                    end_index = start_index + length
                    if end_index > len(normalized):
                        continue
                    start = offsets[start_index]
                    end = offsets[end_index - 1] + 1
                    if _overlaps(start, end, blocked):
                        continue
                    observed_normalized = normalized[start_index:end_index]
                    if len(observed_normalized) < 2:
                        continue
                    observed_pinyin = run_pinyin[start_index:end_index]
                    candidate_indexes: set[int] = set()
                    for gram in _bigrams(observed_normalized):
                        candidate_indexes.update(self._char_index.get(gram, ()))
                    for gram in _bigrams(observed_pinyin):
                        candidate_indexes.update(self._pinyin_index.get(gram, ()))
                    if not candidate_indexes:
                        continue
                    observed = text[start:end]
                    for surface_index in candidate_indexes:
                        surface = self._surfaces[surface_index]
                        # Literal occurrences are handled by EntityProtector.
                        # Keep normalization-only variants (spaces, width,
                        # underscores) because correcting those is one of the
                        # matcher's intended jobs.
                        if observed.casefold() == surface.value.casefold():
                            continue
                        if not _length_compatible(
                            len(observed_normalized), len(surface.normalized)
                        ):
                            continue
                        scored = self._score(
                            observed_normalized, observed_pinyin, surface
                        )
                        previous = span_scores[(start, end)].get(
                            surface.definition.entity_id
                        )
                        if previous is None or scored.final_score > previous.final_score:
                            span_scores[(start, end)][surface.definition.entity_id] = scored

        matches: list[EntityMatch] = []
        for (start, end), by_entity in span_scores.items():
            ranked = sorted(
                by_entity.values(),
                key=lambda item: (
                    -item.final_score,
                    -item.surface.definition.priority,
                    item.surface.definition.entity_id,
                ),
            )[: self.config.max_candidates_per_span]
            if not ranked or ranked[0].final_score < self.config.hint_score:
                continue
            best = ranked[0]
            runner_up = ranked[1].final_score if len(ranked) > 1 else None
            margin = (
                best.final_score - runner_up if runner_up is not None else 1.0
            )
            decision, reasons = self._decide(best, margin, allow_auto=allow_auto)
            matches.append(
                EntityMatch(
                    start=start,
                    end=end,
                    observed=text[start:end],
                    entity_id=best.surface.definition.entity_id,
                    canonical=best.surface.definition.canonical_text,
                    entity_type=best.surface.definition.entity_type,
                    domain=best.surface.definition.domain,
                    matched_surface=best.surface.value,
                    match_type=(
                        "PINYIN_FUZZY"
                        if best.pinyin_score >= best.char_score
                        else "CHAR_FUZZY"
                    ),
                    char_score=best.char_score,
                    pinyin_score=best.pinyin_score,
                    length_score=best.length_score,
                    domain_score=best.domain_score,
                    priority_score=best.priority_score,
                    final_score=best.final_score,
                    runner_up_score=runner_up,
                    margin=margin,
                    priority=best.surface.definition.priority,
                    decision=decision,
                    reasons=reasons,
                )
            )
        matches.sort(
            key=lambda item: (
                -_decision_rank(item.decision),
                -item.final_score,
                -(item.end - item.start),
                -item.priority,
                item.entity_id,
                item.start,
            )
        )
        selected = _select_non_overlapping_matches(
            matches[: self.config.max_logged_candidates_per_segment]
        )
        return EntityMatchReport(selected, (time.perf_counter() - started) * 1000)

    def _score(
        self,
        observed: str,
        observed_pinyin: tuple[str, ...],
        surface: _Surface,
    ) -> _Scored:
        char_score = _similarity(observed, surface.normalized)
        pinyin_score = _similarity(observed_pinyin, surface.pinyin)
        length_score = 1.0 - abs(len(observed) - len(surface.normalized)) / max(
            len(observed), len(surface.normalized)
        )
        domain_score = 1.0 if surface.definition.domain == self.selected_domain else 0.8
        priority_score = min(20, max(0, surface.definition.priority)) / 20
        final_score = (
            self.config.pinyin_weight * pinyin_score
            + self.config.character_weight * char_score
            + self.config.length_weight * length_score
            + self.config.domain_weight * domain_score
            + self.config.priority_weight * priority_score
        )
        return _Scored(
            surface,
            char_score,
            pinyin_score,
            length_score,
            domain_score,
            priority_score,
            final_score,
        )

    def _decide(
        self, scored: _Scored, margin: float, *, allow_auto: bool
    ) -> tuple[EntityDecision, tuple[str, ...]]:
        definition = scored.surface.definition
        reasons: list[str] = []
        normalized_length = len(scored.surface.normalized)
        too_short = (
            normalized_length == 1
            or (scored.surface.normalized.isascii() and normalized_length < 4)
        )
        if too_short:
            reasons.append("entity_too_short")
        if definition.normalization_policy != "normalize":
            reasons.append("preserve_policy")
        if definition.entity_type.upper() not in self.config.auto_entity_types:
            reasons.append("entity_type_not_auto")
        if definition.entity_type.upper() in self.config.high_risk_entity_types:
            reasons.append("high_risk_entity_type")
        if scored.final_score < self.config.auto_score:
            reasons.append("auto_score_below_threshold")
        if scored.pinyin_score < self.config.auto_min_pinyin:
            reasons.append("pinyin_score_below_threshold")
        if margin < self.config.auto_min_margin:
            reasons.append("candidate_margin_too_small")
        auto_eligible = not reasons
        if auto_eligible and self.mode is EntityFuzzyMode.AUTO and allow_auto:
            return EntityDecision.AUTO_NORMALIZE, (
                "score_pass",
                "margin_pass",
                "final_event",
            )
        if self.mode is EntityFuzzyMode.SHADOW:
            if auto_eligible:
                reasons.append("shadow_would_auto_normalize")
            else:
                reasons.append("shadow_mode")
            return EntityDecision.KEEP_RAW, tuple(dict.fromkeys(reasons))
        if auto_eligible and not allow_auto:
            reasons.append("partial_event")
        elif auto_eligible and self.mode is EntityFuzzyMode.HINT:
            reasons.append("hint_mode")
        return EntityDecision.HINT_ONLY, tuple(dict.fromkeys(reasons))


def _normalize(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(
        character
        for character in normalized
        if unicodedata.category(character)[0] not in _IGNORED_CATEGORIES
    )


def _normalize_with_offsets(value: str, base_offset: int) -> tuple[str, tuple[int, ...]]:
    characters: list[str] = []
    offsets: list[int] = []
    for index, original in enumerate(value):
        folded = unicodedata.normalize("NFKC", original).casefold()
        for character in folded:
            if unicodedata.category(character)[0] in _IGNORED_CATEGORIES:
                continue
            characters.append(character)
            offsets.append(base_offset + index)
    return "".join(characters), tuple(offsets)


def _pinyin_tokens(value: str) -> tuple[str, ...]:
    try:
        from pypinyin import Style, lazy_pinyin
    except ImportError as error:
        raise RuntimeError(
            "fuzzy entity matching requires pypinyin; install project requirements"
        ) from error
    return tuple(lazy_pinyin(value, style=Style.NORMAL, errors=lambda item: list(item)))


def _similarity(left: object, right: object) -> float:
    try:
        from rapidfuzz.distance import Levenshtein
    except ImportError as error:
        raise RuntimeError(
            "fuzzy entity matching requires rapidfuzz; install project requirements"
        ) from error
    return float(Levenshtein.normalized_similarity(left, right))


def _bigrams(value: str | tuple[str, ...]) -> tuple[str, ...]:
    if len(value) < 2:
        return ()
    return tuple(f"{value[index]}\0{value[index + 1]}" for index in range(len(value) - 1))


def _text_runs(text: str) -> tuple[tuple[int, str], ...]:
    runs: list[tuple[int, str]] = []
    start = 0
    for index, character in enumerate(text):
        if character not in _HARD_BOUNDARIES:
            continue
        if start < index:
            runs.append((start, text[start:index]))
        start = index + 1
    if start < len(text):
        runs.append((start, text[start:]))
    return tuple(runs)


def _possible_window_lengths(entity_lengths: Iterable[int]) -> tuple[int, ...]:
    lengths: set[int] = set()
    for length in entity_lengths:
        delta = max(1, min(2, int(length * 0.25)))
        lengths.update(range(max(2, length - delta), length + delta + 1))
    return tuple(sorted(lengths))


def _length_compatible(observed: int, entity: int) -> bool:
    delta = max(1, min(2, int(entity * 0.25)))
    return abs(observed - entity) <= delta


def _overlaps(start: int, end: int, ranges: Iterable[tuple[int, int]]) -> bool:
    return any(start < blocked_end and blocked_start < end for blocked_start, blocked_end in ranges)


def _decision_rank(decision: EntityDecision) -> int:
    return {
        EntityDecision.EXACT: 4,
        EntityDecision.AUTO_NORMALIZE: 3,
        EntityDecision.HINT_ONLY: 2,
        EntityDecision.KEEP_RAW: 1,
    }[decision]


def _select_non_overlapping_matches(matches: Iterable[EntityMatch]) -> tuple[EntityMatch, ...]:
    selected: list[EntityMatch] = []
    for match in matches:
        if any(
            match.start < existing.end and existing.start < match.end
            for existing in selected
        ):
            continue
        selected.append(match)
    return tuple(sorted(selected, key=lambda item: item.start))
