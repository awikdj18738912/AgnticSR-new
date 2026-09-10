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

    def test_asr_revision_invalidates_cached_source(self):
        self.update('苹果。天气好。出门。结束。')
        text = '香蕉。天气好。出门。新末尾。'
        self.assertEqual(self.update(text, True)['clean_text'], text)

    def test_long_text_keeps_all_content_and_bounds_model_input(self):
        text = '甲乙丙丁' * 300
        self.assertEqual(self.update(text, True)['clean_text'], text)
        self.assertTrue(all(len(call) <= 240 for call in self.calls))
