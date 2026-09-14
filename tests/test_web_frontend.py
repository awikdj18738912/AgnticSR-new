from pathlib import Path
import unittest


INDEX_HTML = Path(__file__).resolve().parents[1] / "system" / "web" / "index.html"


class WebFrontendMetricsTests(unittest.TestCase):
    def test_session_metric_cards_use_server_side_counters(self) -> None:
        html = INDEX_HTML.read_text(encoding="utf-8")

        self.assertIn('id="refiner-call-count"', html)
        self.assertIn('id="gate-skip-count"', html)
        self.assertIn("function updateSessionMetrics(data)", html)
        self.assertIn("stats.call_count", html)
        self.assertIn("stats.gate_skipped_segment_count", html)
        self.assertIn("updateSessionMetrics(data);", html)

    def test_streaming_refinement_renders_pending_tail_separately(self) -> None:
        html = INDEX_HTML.read_text(encoding="utf-8")

        self.assertIn("function renderRefinedText(data)", html)
        self.assertIn("data.display_refined_text", html)
        self.assertIn("data.pending_raw_text", html)
        self.assertIn("pending.className = 'pending-refinement'", html)


if __name__ == "__main__":
    unittest.main()
