"""Persistent user-managed entities for transcript protection.

The database is deliberately small and deterministic.  It stores canonical
spellings and explicit aliases; online inference loads these records into
memory instead of querying SQLite for every partial ASR hypothesis.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


_POLICIES = frozenset({"preserve", "normalize"})


@dataclass(frozen=True, slots=True)
class EntityDefinition:
    """One verified entity and the surfaces that may refer to it."""

    entity_id: int
    canonical_text: str
    entity_type: str
    domain: str
    normalization_policy: str
    priority: int
    aliases: tuple[str, ...]
    enabled: bool = True
    source: str = "user"
    created_at: str = ""
    updated_at: str = ""


class EntityStore:
    """SQLite-backed verified entity dictionary."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS protected_entities (
                    id INTEGER PRIMARY KEY,
                    canonical_text TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    domain TEXT NOT NULL DEFAULT 'general',
                    normalization_policy TEXT NOT NULL DEFAULT 'preserve'
                        CHECK (normalization_policy IN ('preserve', 'normalize')),
                    priority INTEGER NOT NULL DEFAULT 0,
                    enabled INTEGER NOT NULL DEFAULT 1 CHECK (enabled IN (0, 1)),
                    source TEXT NOT NULL DEFAULT 'user',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE (canonical_text, domain)
                );

                CREATE TABLE IF NOT EXISTS entity_aliases (
                    id INTEGER PRIMARY KEY,
                    entity_id INTEGER NOT NULL,
                    alias TEXT NOT NULL,
                    FOREIGN KEY (entity_id) REFERENCES protected_entities(id)
                        ON DELETE CASCADE,
                    UNIQUE (entity_id, alias)
                );

                CREATE INDEX IF NOT EXISTS idx_entity_alias
                    ON entity_aliases(alias);
                CREATE INDEX IF NOT EXISTS idx_entity_domain_enabled
                    ON protected_entities(domain, enabled);
                """
            )

    def upsert_entity(
        self,
        canonical_text: str,
        *,
        entity_type: str = "TERM",
        domain: str = "general",
        normalization_policy: str = "preserve",
        priority: int = 0,
        aliases: Iterable[str] = (),
        source: str = "user",
    ) -> EntityDefinition:
        canonical = _required_text(canonical_text, "canonical_text")
        kind = _required_text(entity_type, "entity_type").upper()
        entity_domain = _required_text(domain, "domain")
        policy = normalization_policy.strip().lower()
        if policy not in _POLICIES:
            raise ValueError("normalization_policy must be preserve or normalize")
        source_value = _required_text(source, "source")
        alias_values = tuple(
            dict.fromkeys(
                value
                for alias in aliases
                if (value := alias.strip()) and value != canonical
            )
        )
        now = datetime.now(timezone.utc).isoformat()

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO protected_entities (
                    canonical_text, entity_type, domain, normalization_policy,
                    priority, enabled, source, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?)
                ON CONFLICT(canonical_text, domain) DO UPDATE SET
                    entity_type = excluded.entity_type,
                    normalization_policy = excluded.normalization_policy,
                    priority = excluded.priority,
                    enabled = 1,
                    source = excluded.source,
                    updated_at = excluded.updated_at
                """,
                (
                    canonical,
                    kind,
                    entity_domain,
                    policy,
                    int(priority),
                    source_value,
                    now,
                    now,
                ),
            )
            row = connection.execute(
                """
                SELECT id FROM protected_entities
                WHERE canonical_text = ? AND domain = ?
                """,
                (canonical, entity_domain),
            ).fetchone()
            if row is None:
                raise RuntimeError("failed to read entity after upsert")
            entity_id = int(row["id"])
            connection.execute(
                "DELETE FROM entity_aliases WHERE entity_id = ?", (entity_id,)
            )
            connection.executemany(
                "INSERT INTO entity_aliases (entity_id, alias) VALUES (?, ?)",
                ((entity_id, alias) for alias in alias_values),
            )

        return self.get_entity(canonical, entity_domain)

    def get_entity(self, canonical_text: str, domain: str = "general") -> EntityDefinition:
        definitions = self.list_entities(domain=domain, include_disabled=True)
        for definition in definitions:
            if (
                definition.canonical_text == canonical_text
                and definition.domain == domain
            ):
                return definition
        raise KeyError(f"unknown entity: {canonical_text!r} in domain {domain!r}")

    def get_entity_by_id(self, entity_id: int) -> EntityDefinition:
        for definition in self.list_entities(include_disabled=True):
            if definition.entity_id == entity_id:
                return definition
        raise KeyError(f"unknown entity id: {entity_id}")

    def list_entities(
        self,
        *,
        domain: str | None = None,
        include_disabled: bool = False,
    ) -> tuple[EntityDefinition, ...]:
        conditions: list[str] = []
        parameters: list[object] = []
        if domain is not None:
            conditions.append("e.domain IN (?, 'general')")
            parameters.append(domain)
        if not include_disabled:
            conditions.append("e.enabled = 1")
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT e.id, e.canonical_text, e.entity_type, e.domain,
                       e.normalization_policy, e.priority, e.enabled, e.source,
                       e.created_at, e.updated_at,
                       a.alias
                FROM protected_entities AS e
                LEFT JOIN entity_aliases AS a ON a.entity_id = e.id
                {where}
                ORDER BY e.priority DESC, e.id ASC, a.id ASC
                """,
                parameters,
            ).fetchall()

        grouped: dict[int, dict[str, object]] = {}
        for row in rows:
            entity_id = int(row["id"])
            current = grouped.setdefault(
                entity_id,
                {
                    "canonical_text": str(row["canonical_text"]),
                    "entity_type": str(row["entity_type"]),
                    "domain": str(row["domain"]),
                    "normalization_policy": str(row["normalization_policy"]),
                    "priority": int(row["priority"]),
                    "enabled": bool(row["enabled"]),
                    "source": str(row["source"]),
                    "created_at": str(row["created_at"]),
                    "updated_at": str(row["updated_at"]),
                    "aliases": [],
                },
            )
            if row["alias"] is not None:
                aliases = current["aliases"]
                assert isinstance(aliases, list)
                aliases.append(str(row["alias"]))

        return tuple(
            EntityDefinition(
                entity_id=entity_id,
                canonical_text=str(values["canonical_text"]),
                entity_type=str(values["entity_type"]),
                domain=str(values["domain"]),
                normalization_policy=str(values["normalization_policy"]),
                priority=int(values["priority"]),
                aliases=tuple(values["aliases"]),
                enabled=bool(values["enabled"]),
                source=str(values["source"]),
                created_at=str(values["created_at"]),
                updated_at=str(values["updated_at"]),
            )
            for entity_id, values in grouped.items()
        )

    def set_enabled(
        self, canonical_text: str, *, domain: str = "general", enabled: bool
    ) -> bool:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE protected_entities
                SET enabled = ?, updated_at = ?
                WHERE canonical_text = ? AND domain = ?
                """,
                (int(enabled), now, canonical_text, domain),
            )
            return cursor.rowcount > 0

    def set_enabled_by_id(self, entity_id: int, *, enabled: bool) -> bool:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE protected_entities SET enabled = ?, updated_at = ? WHERE id = ?",
                (int(enabled), now, entity_id),
            )
            return cursor.rowcount > 0

    def update_entity(
        self,
        entity_id: int,
        canonical_text: str,
        *,
        entity_type: str = "TERM",
        domain: str = "general",
        normalization_policy: str = "preserve",
        priority: int = 0,
        aliases: Iterable[str] = (),
        source: str = "ui",
    ) -> EntityDefinition:
        canonical = _required_text(canonical_text, "canonical_text")
        kind = _required_text(entity_type, "entity_type").upper()
        entity_domain = _required_text(domain, "domain")
        policy = normalization_policy.strip().lower()
        if policy not in _POLICIES:
            raise ValueError("normalization_policy must be preserve or normalize")
        source_value = _required_text(source, "source")
        alias_values = tuple(
            dict.fromkeys(
                value
                for alias in aliases
                if (value := alias.strip()) and value != canonical
            )
        )
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            conflict = connection.execute(
                """
                SELECT id FROM protected_entities
                WHERE canonical_text = ? AND domain = ? AND id != ?
                """,
                (canonical, entity_domain, entity_id),
            ).fetchone()
            if conflict is not None:
                raise ValueError("an entity with this canonical text and domain already exists")
            cursor = connection.execute(
                """
                UPDATE protected_entities
                SET canonical_text = ?, entity_type = ?, domain = ?,
                    normalization_policy = ?, priority = ?, source = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    canonical,
                    kind,
                    entity_domain,
                    policy,
                    int(priority),
                    source_value,
                    now,
                    entity_id,
                ),
            )
            if cursor.rowcount == 0:
                raise KeyError(f"unknown entity id: {entity_id}")
            connection.execute("DELETE FROM entity_aliases WHERE entity_id = ?", (entity_id,))
            connection.executemany(
                "INSERT INTO entity_aliases (entity_id, alias) VALUES (?, ?)",
                ((entity_id, alias) for alias in alias_values),
            )
        return self.get_entity_by_id(entity_id)

    def delete_entity(self, entity_id: int) -> bool:
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM protected_entities WHERE id = ?", (entity_id,))
            return cursor.rowcount > 0


def _required_text(value: str, name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    return normalized
