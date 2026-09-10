"""Command-line management for the verified protected-entity database."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .entity_store import EntityStore


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage protected AgenticASR entities")
    parser.add_argument("--db", type=Path, required=True, help="SQLite database path")
    subparsers = parser.add_subparsers(dest="command", required=True)

    add = subparsers.add_parser("add", help="add or update a verified entity")
    add.add_argument("canonical")
    add.add_argument("--alias", action="append", default=[])
    add.add_argument("--type", default="TERM", dest="entity_type")
    add.add_argument("--domain", default="general")
    add.add_argument("--policy", choices=("preserve", "normalize"), default="preserve")
    add.add_argument("--priority", type=int, default=0)

    list_parser = subparsers.add_parser("list", help="list stored entities")
    list_parser.add_argument("--domain")
    list_parser.add_argument("--all", action="store_true", dest="include_disabled")

    for command in ("enable", "disable"):
        change = subparsers.add_parser(command, help=f"{command} one entity")
        change.add_argument("canonical")
        change.add_argument("--domain", default="general")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    store = EntityStore(args.db)
    if args.command == "add":
        definition = store.upsert_entity(
            args.canonical,
            entity_type=args.entity_type,
            domain=args.domain,
            normalization_policy=args.policy,
            priority=args.priority,
            aliases=args.alias,
        )
        print(json.dumps(_as_dict(definition), ensure_ascii=False))
        return 0
    if args.command == "list":
        for definition in store.list_entities(
            domain=args.domain, include_disabled=args.include_disabled
        ):
            print(json.dumps(_as_dict(definition), ensure_ascii=False))
        return 0

    enabled = args.command == "enable"
    changed = store.set_enabled(args.canonical, domain=args.domain, enabled=enabled)
    if not changed:
        print(f"entity not found: {args.canonical!r} ({args.domain})")
        return 1
    print(f"{args.command}d: {args.canonical}")
    return 0


def _as_dict(definition: object) -> dict[str, object]:
    return {
        "entity_id": getattr(definition, "entity_id"),
        "canonical_text": getattr(definition, "canonical_text"),
        "entity_type": getattr(definition, "entity_type"),
        "domain": getattr(definition, "domain"),
        "normalization_policy": getattr(definition, "normalization_policy"),
        "priority": getattr(definition, "priority"),
        "aliases": list(getattr(definition, "aliases")),
    }


if __name__ == "__main__":
    raise SystemExit(main())
