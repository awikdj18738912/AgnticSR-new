from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from system.entity_matcher import (
    EntityCandidateMatcher,
    EntityDecision,
    EntityFuzzyMode,
    EntityMatchConfig,
)
from system.entity_pipeline import finalize_entity_segment, prepare_entity_segment
from system.entity_store import EntityDefinition, EntityStore
from system.protection import EntityProtector


def _definition(
    canonical: str,
    *,
    entity_id: int = 1,
    entity_type: str = "TERM",
    domain: str = "game",
    policy: str = "normalize",
    aliases: tuple[str, ...] = (),
    priority: int = 0,
) -> EntityDefinition:
    return EntityDefinition(
        entity_id=entity_id,
        canonical_text=canonical,
        entity_type=entity_type,
        domain=domain,
        normalization_policy=policy,
        priority=priority,
        aliases=aliases,
    )


class EntityCandidateMatcherTest(unittest.TestCase):
    def test_same_pinyin_single_character_error_can_auto_normalize(self) -> None:
        entity = _definition("鬼灵门")
        matcher = EntityCandidateMatcher(
            (entity,), selected_domain="game", mode=EntityFuzzyMode.AUTO
        )

        report = matcher.match("鬼灵们是副本入口。", allow_auto=True)

        self.assertEqual(len(report.auto_matches), 1)
        match = report.auto_matches[0]
        self.assertEqual(match.observed, "鬼灵们")
        self.assertEqual(match.canonical, "鬼灵门")
        self.assertGreaterEqual(match.pinyin_score, 0.99)
        self.assertGreaterEqual(match.final_score, matcher.config.auto_score)

    def test_shadow_mode_reports_without_changing_decision(self) -> None:
        matcher = EntityCandidateMatcher(
            (_definition("鬼灵门"),), selected_domain="game", mode="shadow"
        )

        report = matcher.match("鬼灵们是入口", allow_auto=True)

        self.assertEqual(report.auto_matches, ())
        self.assertEqual(report.matches[0].decision, EntityDecision.KEEP_RAW)
        self.assertIn("shadow_would_auto_normalize", report.matches[0].reasons)

    def test_ascii_spacing_variant_can_be_normalized(self) -> None:
        entity = _definition("AgenticASR", entity_type="PROJECT")
        matcher = EntityCandidateMatcher(
            (entity,), selected_domain="game", mode="auto"
        )

        report = matcher.match("Agentic ASR 项目", allow_auto=True)

        self.assertEqual(report.auto_matches[0].observed, "Agentic ASR")
        self.assertEqual(report.auto_matches[0].canonical, "AgenticASR")

    def test_partial_event_is_hint_only_even_in_auto_mode(self) -> None:
        matcher = EntityCandidateMatcher(
            (_definition("鬼灵门"),), selected_domain="game", mode="auto"
        )

        report = matcher.match("鬼灵们是入口", allow_auto=False)

        self.assertEqual(report.matches[0].decision, EntityDecision.HINT_ONLY)
        self.assertEqual(report.hint_canonicals, ("鬼灵门",))
        self.assertIn("partial_event", report.matches[0].reasons)

    def test_high_risk_entity_never_auto_normalizes(self) -> None:
        matcher = EntityCandidateMatcher(
            (_definition("张三丰", entity_type="PERSON"),),
            selected_domain="game",
            mode="auto",
        )

        report = matcher.match("请张三风发言", allow_auto=True)

        self.assertEqual(report.auto_matches, ())
        self.assertEqual(report.matches[0].decision, EntityDecision.HINT_ONLY)
        self.assertIn("high_risk_entity_type", report.matches[0].reasons)

    def test_character_score_is_not_a_hard_auto_gate(self) -> None:
        config = EntityMatchConfig.load()
        matcher = EntityCandidateMatcher(
            (_definition("厉飞羽"),),
            selected_domain="game",
            mode="auto",
            config=config,
        )

        report = matcher.match("李飞鱼", allow_auto=True)

        self.assertEqual(report.auto_matches[0].canonical, "厉飞羽")
        self.assertAlmostEqual(report.auto_matches[0].char_score, 1 / 3)

    def test_ambiguous_homophones_are_downgraded(self) -> None:
        matcher = EntityCandidateMatcher(
            (
                _definition("鬼灵门", entity_id=1),
                _definition("鬼灵焖", entity_id=2),
            ),
            selected_domain="game",
            mode="auto",
        )

        report = matcher.match("鬼灵们开启", allow_auto=True)

        self.assertEqual(report.auto_matches, ())
        self.assertEqual(report.matches[0].decision, EntityDecision.HINT_ONLY)
        self.assertIn("candidate_margin_too_small", report.matches[0].reasons)

    def test_entities_outside_selected_domain_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside selected domain"):
            EntityCandidateMatcher(
                (_definition("法律术语", domain="legal"),),
                selected_domain="game",
            )

    def test_blocked_rule_span_is_not_a_fuzzy_candidate(self) -> None:
        entity = _definition("Qwen3-ASR", entity_type="MODEL")
        matcher = EntityCandidateMatcher((entity,), selected_domain="game", mode="auto")
        protector = EntityProtector((entity,))
        exact = protector.protect("使用 Qwen2-ASR 模型")

        report = matcher.match(
            "使用 Qwen2-ASR 模型",
            allow_auto=True,
            blocked_spans=((span.start, span.end) for span in exact.spans),
        )

        self.assertEqual(report.matches, ())


class EntityPipelineTest(unittest.TestCase):
    def test_auto_match_is_masked_and_restored_to_canonical(self) -> None:
        entity = _definition("鬼灵门")
        protector = EntityProtector((entity,))
        matcher = EntityCandidateMatcher(
            (entity,), selected_domain="game", mode="auto"
        )

        prepared = prepare_entity_segment(
            "鬼灵们是副本入口。",
            protector,
            matcher,
            allow_auto=True,
        )

        self.assertNotIn("鬼灵们", prepared.protection.masked_text)
        self.assertEqual(prepared.baseline_text, "鬼灵门是副本入口。")
        finalized = finalize_entity_segment(
            prepared.protection.masked_text, prepared, protector
        )
        self.assertTrue(finalized.accepted)
        self.assertEqual(finalized.text, "鬼灵门是副本入口。")

    def test_broken_placeholder_falls_back_to_deterministic_baseline(self) -> None:
        entity = _definition("鬼灵门")
        protector = EntityProtector((entity,))
        matcher = EntityCandidateMatcher(
            (entity,), selected_domain="game", mode="auto"
        )
        prepared = prepare_entity_segment(
            "鬼灵们是入口", protector, matcher, allow_auto=True
        )

        finalized = finalize_entity_segment("这是一段丢失占位符的文本", prepared, protector)

        self.assertFalse(finalized.accepted)
        self.assertEqual(finalized.text, "鬼灵门是入口")
        self.assertTrue(
            any(reason.startswith("placeholder_count:") for reason in finalized.reject_reasons)
        )

    def test_refiner_cannot_add_a_hint_without_replacing_its_observed_span(self) -> None:
        entity = _definition("鬼灵门")
        protector = EntityProtector((entity,))
        matcher = EntityCandidateMatcher(
            (entity,), selected_domain="game", mode="hint"
        )
        prepared = prepare_entity_segment(
            "这里没有实体", protector, matcher, allow_auto=True
        )
        # Simulate an untrusted Refiner appending a glossary name. There is no
        # fuzzy observed span in this source, so the addition must be rejected.
        prepared_with_hint = type(prepared)(
            prepared.source_text,
            prepared.protection,
            prepared.baseline_text,
            ("鬼灵门",),
            prepared.report,
        )

        finalized = finalize_entity_segment(
            "这里没有实体，鬼灵门。", prepared_with_hint, protector
        )

        self.assertFalse(finalized.accepted)
        self.assertEqual(finalized.text, "这里没有实体")
        self.assertIn(
            "introduced_entity_from_hint:鬼灵门", finalized.reject_reasons
        )

    def test_hint_correction_is_allowed_when_it_consumes_the_observed_span(self) -> None:
        entity = _definition("鬼灵门")
        protector = EntityProtector((entity,))
        matcher = EntityCandidateMatcher(
            (entity,), selected_domain="game", mode="hint"
        )
        prepared = prepare_entity_segment(
            "鬼灵们是入口", protector, matcher, allow_auto=True
        )

        finalized = finalize_entity_segment("鬼灵门是入口", prepared, protector)

        self.assertTrue(finalized.accepted)
        self.assertEqual(finalized.text, "鬼灵门是入口")

    def test_exact_alias_still_has_precedence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = EntityStore(Path(directory) / "entities.db")
            entity = store.upsert_entity(
                "AgenticASR",
                entity_type="PROJECT",
                domain="general",
                normalization_policy="normalize",
                aliases=("Agentic SR",),
            )
            protector = EntityProtector((entity,))
            matcher = EntityCandidateMatcher(
                (entity,), selected_domain="general", mode="auto"
            )

            prepared = prepare_entity_segment(
                "使用 Agentic SR 项目", protector, matcher, allow_auto=True
            )

            self.assertEqual(prepared.report.matches, ())
            self.assertEqual(prepared.baseline_text, "使用 AgenticASR 项目")


if __name__ == "__main__":
    unittest.main()
