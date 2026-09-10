from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from system.entity_store import EntityStore
from system.protection import EntityProtector
from system.session_memory import SessionEntityMemory, TrustLevel


class EntityStoreTest(unittest.TestCase):
    def test_upsert_alias_and_enable_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = EntityStore(Path(directory) / "entities.db")
            definition = store.upsert_entity(
                "AgenticASR",
                entity_type="PROJECT",
                aliases=("Agentic SR", "agentic asr"),
                normalization_policy="normalize",
                priority=10,
            )

            self.assertEqual(definition.canonical_text, "AgenticASR")
            self.assertEqual(definition.aliases, ("Agentic SR", "agentic asr"))
            self.assertEqual(len(store.list_entities()), 1)
            self.assertTrue(
                store.set_enabled("AgenticASR", domain="general", enabled=False)
            )
            self.assertEqual(store.list_entities(), ())
            self.assertEqual(len(store.list_entities(include_disabled=True)), 1)

    def test_update_toggle_and_delete_by_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = EntityStore(Path(directory) / "entities.db")
            created = store.upsert_entity("Qwen ASR", entity_type="MODEL")
            updated = store.update_entity(
                created.entity_id,
                "Qwen3-ASR",
                entity_type="MODEL",
                normalization_policy="normalize",
                priority=7,
                aliases=("Qwen ASR",),
            )

            self.assertEqual(updated.canonical_text, "Qwen3-ASR")
            self.assertEqual(updated.aliases, ("Qwen ASR",))
            self.assertTrue(store.set_enabled_by_id(updated.entity_id, enabled=False))
            self.assertFalse(store.get_entity_by_id(updated.entity_id).enabled)
            self.assertTrue(store.delete_entity(updated.entity_id))
            with self.assertRaises(KeyError):
                store.get_entity_by_id(updated.entity_id)

    def test_domain_lookup_includes_general_only_for_selected_domain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = EntityStore(Path(directory) / "entities.db")
            store.upsert_entity("通用术语", domain="general")
            store.upsert_entity("医疗术语", domain="medical")
            store.upsert_entity("法律术语", domain="legal")

            names = {item.canonical_text for item in store.list_entities(domain="medical")}
            self.assertEqual(names, {"通用术语", "医疗术语"})


class EntityProtectorTest(unittest.TestCase):
    def test_rules_and_verified_alias_are_restored(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = EntityStore(Path(directory) / "entities.db")
            definition = store.upsert_entity(
                "AgenticASR",
                entity_type="PROJECT",
                aliases=("Agentic SR",),
                normalization_policy="normalize",
            )
            protector = EntityProtector((definition,))
            protection = protector.protect(
                "预算是12.5万元，使用 Agentic SR，在2026年9月8日测试。"
            )

            self.assertNotIn("12.5", protection.masked_text)
            self.assertNotIn("Agentic SR", protection.masked_text)
            self.assertNotIn("2026年9月8日", protection.masked_text)
            restored = protector.restore(protection.masked_text, protection)
            self.assertTrue(restored.accepted)
            self.assertIn("12.5", restored.text)
            self.assertIn("AgenticASR", restored.text)
            self.assertIn("2026年9月8日", restored.text)

    def test_missing_placeholder_rejects_refiner_output(self) -> None:
        protector = EntityProtector()
        protection = protector.protect("预算是12.5万元。")
        self.assertEqual(len(protection.spans), 1)

        restored = protector.restore("预算是一万元。", protection)
        self.assertFalse(restored.accepted)
        self.assertEqual(restored.text, protection.original_text)
        self.assertTrue(
            any(reason.startswith("placeholder_count:") for reason in restored.reject_reasons)
        )

    def test_unknown_placeholder_is_rejected(self) -> None:
        protector = EntityProtector()
        protection = protector.protect("版本是2.0。")
        restored = protector.restore(
            f"{protection.masked_text} ⟦P999⟧", protection
        )
        self.assertFalse(restored.accepted)
        self.assertTrue(
            any(reason.startswith("unknown_placeholders:") for reason in restored.reject_reasons)
        )

    def test_reordered_placeholders_are_rejected(self) -> None:
        protector = EntityProtector()
        protection = protector.protect("版本2.0，预算12.5万。")
        self.assertEqual(len(protection.spans), 2)

        reordered = (
            protection.masked_text.replace("⟦P000⟧", "__TEMP__")
            .replace("⟦P001⟧", "⟦P000⟧")
            .replace("__TEMP__", "⟦P001⟧")
        )
        restored = protector.restore(reordered, protection)
        self.assertFalse(restored.accepted)
        self.assertIn("placeholder_order_changed", restored.reject_reasons)

    def test_decimal_and_chinese_adjacent_identifier_types(self) -> None:
        protector = EntityProtector()
        protection = protector.protect("预算12.5万，使用Qwen3-ASR模型。")

        spans = {span.original: span.entity_type for span in protection.spans}
        self.assertEqual(spans["12.5"], "NUMBER")
        self.assertEqual(spans["Qwen3-ASR"], "IDENTIFIER")

    def test_unmasked_audit_checks_only_trusted_entities(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = EntityStore(Path(directory) / "entities.db")
            definition = store.upsert_entity(
                "AgenticASR",
                entity_type="PROJECT",
                aliases=("Agentic SR",),
                normalization_policy="normalize",
            )
            protector = EntityProtector((definition,))
            protection = protector.protect("预算12.5万，项目是 Agentic SR。")

            self.assertEqual(
                protector.audit_unmasked("预算十二点五万，项目是 AgenticASR。", protection),
                (),
            )
            self.assertEqual(
                protector.audit_unmasked("预算十二点五万。", protection),
                ("missing_PROJECT",),
            )

    def test_verified_alias_is_exposed_to_refiner_and_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = EntityStore(Path(directory) / "entities.db")
            definition = store.upsert_entity(
                "北京航空航天大学",
                entity_type="ORG",
                aliases=("北航",),
                normalization_policy="normalize",
            )
            protector = EntityProtector((definition,))
            protection = protector.protect("北航的张老师介绍实验。")

            self.assertEqual(protector.refinement_hints(protection), ("北京航空航天大学",))
            normalized, changes = protector.normalize_verified_aliases(
                "北航的张老师介绍实验。", protection
            )
            self.assertEqual(normalized, "北京航空航天大学的张老师介绍实验。")
            self.assertEqual(
                changes,
                (
                    {
                        "observed": "北航",
                        "canonical": "北京航空航天大学",
                        "entity_type": "ORG",
                    },
                ),
            )

    def test_preserve_policy_does_not_force_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = EntityStore(Path(directory) / "entities.db")
            definition = store.upsert_entity(
                "北京航空航天大学",
                entity_type="ORG",
                aliases=("北航",),
                normalization_policy="preserve",
            )
            protector = EntityProtector((definition,))
            protection = protector.protect("北航的张老师介绍实验。")

            self.assertEqual(protector.refinement_hints(protection), ())
            self.assertEqual(
                protector.normalize_verified_aliases("北航的张老师介绍实验。", protection),
                ("北航的张老师介绍实验。", ()),
            )

    def test_normalization_removes_refiner_key_metadata(self) -> None:
        protector = EntityProtector()
        protection = protector.protect("正常文本。")

        self.assertEqual(
            protector.normalize_verified_aliases("正常文本。<KEY>[测试词]", protection),
            ("正常文本。", ()),
        )


class SessionEntityMemoryTest(unittest.TestCase):
    def test_two_independent_high_confidence_observations_promote_entity(self) -> None:
        memory = SessionEntityMemory(promote_after=2, min_confidence=0.8)
        first = memory.observe(
            "Qwen3-ASR", confidence=0.91, observation_id="segment-1", now_ms=1000
        )
        self.assertEqual(first.trust_level, TrustLevel.PROVISIONAL)
        second = memory.observe(
            "Qwen3-ASR", confidence=0.89, observation_id="segment-2", now_ms=2000
        )

        self.assertEqual(second.trust_level, TrustLevel.ACCEPTED)
        self.assertEqual(len(memory.trusted(now_ms=2000)), 1)

    def test_repeated_partial_does_not_count_as_independent_evidence(self) -> None:
        memory = SessionEntityMemory(promote_after=2, min_confidence=0.8)
        memory.observe(
            "Qwen3-ASR", confidence=0.95, observation_id="segment-1", now_ms=1000
        )
        entry = memory.observe(
            "Qwen3-ASR", confidence=0.95, observation_id="segment-1", now_ms=1100
        )

        self.assertEqual(entry.confidence_count, 1)
        self.assertNotEqual(entry.trust_level, TrustLevel.ACCEPTED)

    def test_missing_confidence_never_auto_promotes(self) -> None:
        memory = SessionEntityMemory(promote_after=2)
        memory.observe("一个新实体", now_ms=1000)
        entry = memory.observe("一个新实体", now_ms=2000)

        self.assertEqual(entry.trust_level, TrustLevel.OBSERVED)
        self.assertEqual(memory.trusted(now_ms=2000), ())

    def test_verified_entity_is_immediately_trusted(self) -> None:
        memory = SessionEntityMemory()
        entry = memory.observe("Whisper", verified=True, now_ms=1000)
        self.assertEqual(entry.trust_level, TrustLevel.VERIFIED)
        self.assertEqual(len(memory.trusted(now_ms=1000)), 1)


if __name__ == "__main__":
    unittest.main()
