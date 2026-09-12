import unittest
from system.window_refinement import CumulativeWindowRefinement


class WindowTest(unittest.TestCase):
    def setUp(self):
        self.calls = []
        def refine(text, *args, **kwargs):
            self.calls.append(text)
            return dict(raw_text=text, clean_text=text.replace('苹果', '梨'),
                        refiner_latency_ms=1, refiner_accepted=True,
                        refiner_reject_reasons=[], entity_audit_issues=[],
                        entity_refinement_hints=[], entity_normalizations=[],
                        protected_entities=[], entity_candidates=[],
                        entity_matcher_latency_ms=0)
        self.session = CumulativeWindowRefinement(refine)

    def update(self, text, final=False):
        return self.session.update(text, 'Chinese', final, None, None)

    def test_shift_preserves_prefix_and_suffix_without_duplicate(self):
        for text in ['苹果。', '苹果。天气好。', '苹果。天气好。出门。',
                     '苹果。天气好。出门。结束。']:
            result = self.update(text)
            self.assertEqual(result['clean_text'], text.replace('苹果', '梨'))
        count = len(self.calls)
        result = self.update(text, final=True)
        self.assertEqual(result['event'], 'final')
        self.assertEqual(len(self.calls), count)

    def test_three_identical_short_sentence_chunks_are_deduplicated(self):
        text = "前文。可恶！可恶！可恶！后文。"

        result = self.update(text, final=True)

        self.assertEqual(result["raw_text"], text)
        self.assertEqual(result["clean_text"], "前文。可恶！后文。")

    def test_asr_revision_invalidates_cached_source(self):
        self.update('苹果。天气好。出门。结束。')
        text = '香蕉。天气好。出门。新末尾。'
        self.assertEqual(self.update(text, True)['clean_text'], text)

    def test_prefix_revision_reuses_unchanged_committed_chunks(self):
        self.session.window_size = 2
        self.update('甲。乙。丙。丁。')
        self.calls.clear()

        result = self.update('新甲。乙。丙。丁。', True)

        self.assertEqual(result['clean_text'], '新甲。乙。丙。丁。')
        self.assertEqual(self.calls, ['新甲。'])

    def test_rejected_active_window_recovers_per_chunk(self):
        calls = []

        def refine(text, *args, **kwargs):
            calls.append(text)
            rejected = text == "丙。丁。"
            return {
                "raw_text": text,
                "clean_text": text,
                "refiner_latency_ms": 1,
                "refiner_accepted": not rejected,
                "refiner_reject_reasons": ["segment_1:severe_content_loss"]
                if rejected
                else [],
                "entity_audit_issues": [],
                "entity_refinement_hints": [],
                "entity_normalizations": [],
                "protected_entities": [],
                "entity_candidates": [],
                "refiner_masked_outputs": [],
                "refiner_retry_reasons": [],
                "refinement_gate_decisions": [],
                "entity_matcher_latency_ms": 0,
            }

        session = CumulativeWindowRefinement(refine, window_size=2)
        result = session.update("甲。乙。丙。丁。", "Chinese", True, None, None)

        self.assertEqual(result["clean_text"], "甲。乙。丙。丁。")
        self.assertTrue(result["refiner_accepted"])
        self.assertEqual(calls, ["甲。", "乙。", "丙。丁。", "丙。", "丁。"])

    def test_long_text_keeps_all_content_and_bounds_model_input(self):
        text = '甲乙丙丁' * 300
        self.assertEqual(self.update(text, True)['clean_text'], text)
        self.assertTrue(all(len(call) <= 80 for call in self.calls))

    def test_active_window_uses_sentence_chunks_with_eighty_char_limit(self):
        text = "第一句。第二句内容较长但仍然完整。第三句。第四句。"
        result = self.update(text, True)

        self.assertEqual(result["clean_text"], text)
        self.assertEqual(result["window_max_chars"], 80)
        self.assertTrue(all(len(call) <= 80 for call in self.calls))
        self.assertTrue(any(call.endswith("第三句。第四句。") for call in self.calls))

    def test_self_correction_chunks_are_refined_independently(self):
        text = (
            "你好，我有一个苹果，不对，我有一个梨，我有一箱苹果，不对，我有一箱梨，"
            "我有两千一百三十五元，我有百分之三的几率获得五万四千三百二十一元。"
        )
        result = self.update(text, True)

        self.assertEqual(result["clean_text"], text.replace("苹果", "梨"))
        self.assertEqual(
            self.calls,
            [
                "你好，我有一个苹果，不对，我有一个梨，",
                "我有一箱苹果，不对，我有一箱梨，",
                "我有两千一百三十五元，我有百分之三的几率获得五万四千三百二十一元。",
            ],
        )

    def test_correction_after_sentence_boundary_is_not_committed_separately(self):
        text = "你好，我有一个苹果。不对，我有一个梨。我有一千元。"
        self.update(text, True)

        self.assertIn("你好，我有一个苹果。不对，我有一个梨。", self.calls)
        self.assertNotIn("你好，我有一个苹果。", self.calls)

    def test_final_refreshes_rejected_intermediate_window(self):
        calls = []

        def refine(text, *args, **kwargs):
            final = bool(args[1])
            calls.append((text, final))
            return {
                "raw_text": text,
                "clean_text": text if final else "错误的短输出。",
                "refiner_latency_ms": 1,
                "refiner_accepted": final,
                "refiner_reject_reasons": [] if final else [
                    "segment_1:severe_content_loss"
                ],
                "entity_audit_issues": [],
                "entity_refinement_hints": [],
                "entity_normalizations": [],
                "protected_entities": [],
                "entity_candidates": [],
                "entity_matcher_latency_ms": 0,
            }

        session = CumulativeWindowRefinement(refine)
        first = session.update("第一句。", "Chinese", False, None, None)
        self.assertFalse(first["refiner_accepted"])

        final = session.update("第一句。", "Chinese", True, None, None)
        self.assertTrue(final["refiner_accepted"])
        self.assertEqual(final["clean_text"], "第一句。")
        self.assertEqual(calls, [("第一句。", False), ("第一句。", True)])
