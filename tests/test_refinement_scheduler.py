from __future__ import annotations

import unittest

from system.refinement_scheduler import StableTextRefinementScheduler


class StableTextRefinementSchedulerTest(unittest.TestCase):
    def test_partial_revisions_wait_for_sentence_boundary(self) -> None:
        scheduler = StableTextRefinementScheduler()

        self.assertFalse(scheduler.decide("今天").should_refine)
        self.assertFalse(scheduler.decide("今天天气").should_refine)
        self.assertFalse(scheduler.decide("今天天气很好。继续").should_refine)
        decision = scheduler.decide("今天天气很好。继续说")

        self.assertTrue(decision.should_refine)
        self.assertEqual(decision.text, "今天天气很好。")
        self.assertEqual(decision.reason, "stable_sentence_prefix_changed")
        self.assertFalse(scheduler.decide("今天天气很好。继续说下去").should_refine)

    def test_changed_complete_prefix_is_rescheduled(self) -> None:
        scheduler = StableTextRefinementScheduler()

        self.assertFalse(scheduler.decide("天气很好。").should_refine)
        self.assertTrue(scheduler.decide("天气很好。后续").should_refine)
        self.assertFalse(scheduler.decide("今天天气很好。后续").should_refine)
        revised = scheduler.decide("今天天气很好。后续内容")

        self.assertTrue(revised.should_refine)
        self.assertEqual(revised.text, "今天天气很好。")

    def test_all_but_last_complete_sentence_are_immediately_stable(self) -> None:
        scheduler = StableTextRefinementScheduler()

        decision = scheduler.decide("第一句。第二句。")

        self.assertTrue(decision.should_refine)
        self.assertEqual(decision.text, "第一句。")

    def test_long_unpunctuated_tail_is_scheduled_once_per_bucket(self) -> None:
        scheduler = StableTextRefinementScheduler(max_unstable_chars=8)

        self.assertFalse(scheduler.decide("甲乙丙丁").should_refine)
        first = scheduler.decide("甲乙丙丁戊己庚辛")
        self.assertTrue(first.should_refine)
        self.assertEqual(first.reason, "unstable_tail_length_threshold")
        self.assertFalse(scheduler.decide("甲乙丙丁戊己庚辛壬").should_refine)
        self.assertTrue(
            scheduler.decide("甲乙丙丁戊己庚辛壬癸子丑寅卯辰巳").should_refine
        )


if __name__ == "__main__":
    unittest.main()
