"""Bounded, trust-aware entity memory for one transcription session."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum


class TrustLevel(str, Enum):
    PROVISIONAL = "PROVISIONAL"
    OBSERVED = "OBSERVED"
    ACCEPTED = "ACCEPTED"
    VERIFIED = "VERIFIED"


@dataclass(slots=True)
class EntityMemory:
    canonical_text: str
    entity_type: str
    aliases: set[str] = field(default_factory=set)
    first_seen_ms: float = 0.0
    last_seen_ms: float = 0.0
    occurrence_count: int = 0
    confidence_sum: float = 0.0
    confidence_count: int = 0
    observation_ids: set[str] = field(default_factory=set, repr=False)
    trust_level: TrustLevel = TrustLevel.PROVISIONAL
    source: str = "asr"

    @property
    def mean_confidence(self) -> float | None:
        if self.confidence_count == 0:
            return None
        return self.confidence_sum / self.confidence_count


class SessionEntityMemory:
    """Maintain entity observations without promoting unverified corrections."""

    def __init__(
        self,
        *,
        capacity: int = 128,
        ttl_seconds: float = 20 * 60,
        promote_after: int = 2,
        min_confidence: float = 0.8,
    ) -> None:
        if capacity < 1 or ttl_seconds <= 0 or promote_after < 1:
            raise ValueError("memory limits must be positive")
        if not 0 <= min_confidence <= 1:
            raise ValueError("min_confidence must be between 0 and 1")
        self.capacity = capacity
        self.ttl_ms = ttl_seconds * 1000
        self.promote_after = promote_after
        self.min_confidence = min_confidence
        self._entries: dict[str, EntityMemory] = {}

    def observe(
        self,
        canonical_text: str,
        *,
        entity_type: str = "TERM",
        aliases: tuple[str, ...] = (),
        confidence: float | None = None,
        observation_id: str | None = None,
        verified: bool = False,
        source: str = "asr",
        now_ms: float | None = None,
    ) -> EntityMemory:
        canonical = canonical_text.strip()
        if not canonical:
            raise ValueError("canonical_text must not be empty")
        if confidence is not None and not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        observed_at = now_ms if now_ms is not None else time.monotonic() * 1000
        self._evict_expired(observed_at)

        entry = self._entries.get(canonical)
        if entry is None:
            entry = EntityMemory(
                canonical_text=canonical,
                entity_type=entity_type,
                first_seen_ms=observed_at,
                last_seen_ms=observed_at,
                source=source,
            )
            self._entries[canonical] = entry
        entry.last_seen_ms = observed_at
        entry.occurrence_count += 1
        entry.aliases.update(alias.strip() for alias in aliases if alias.strip())
        independent_observation = (
            observation_id is not None and observation_id not in entry.observation_ids
        )
        if independent_observation:
            entry.observation_ids.add(observation_id)
        if confidence is not None and independent_observation:
            entry.confidence_sum += confidence
            entry.confidence_count += 1

        if verified:
            entry.trust_level = TrustLevel.VERIFIED
        elif entry.trust_level is not TrustLevel.VERIFIED:
            mean_confidence = entry.mean_confidence
            if (
                entry.occurrence_count >= self.promote_after
                and entry.confidence_count >= self.promote_after
                and mean_confidence is not None
                and mean_confidence >= self.min_confidence
            ):
                entry.trust_level = TrustLevel.ACCEPTED
            elif entry.occurrence_count > 1:
                entry.trust_level = TrustLevel.OBSERVED

        self._evict_over_capacity()
        return entry

    def trusted(self, *, now_ms: float | None = None) -> tuple[EntityMemory, ...]:
        current = now_ms if now_ms is not None else time.monotonic() * 1000
        self._evict_expired(current)
        return tuple(
            entry
            for entry in self._entries.values()
            if entry.trust_level in {TrustLevel.ACCEPTED, TrustLevel.VERIFIED}
        )

    def snapshot(self, *, now_ms: float | None = None) -> tuple[EntityMemory, ...]:
        current = now_ms if now_ms is not None else time.monotonic() * 1000
        self._evict_expired(current)
        return tuple(self._entries.values())

    def _evict_expired(self, now_ms: float) -> None:
        expired = [
            key
            for key, entry in self._entries.items()
            if entry.trust_level is not TrustLevel.VERIFIED
            and now_ms - entry.last_seen_ms > self.ttl_ms
        ]
        for key in expired:
            del self._entries[key]

    def _evict_over_capacity(self) -> None:
        while len(self._entries) > self.capacity:
            candidates = [
                entry
                for entry in self._entries.values()
                if entry.trust_level is not TrustLevel.VERIFIED
            ]
            if not candidates:
                candidates = list(self._entries.values())
            victim = min(
                candidates,
                key=lambda entry: (
                    entry.trust_level is TrustLevel.ACCEPTED,
                    entry.last_seen_ms,
                    entry.occurrence_count,
                ),
            )
            del self._entries[victim.canonical_text]
