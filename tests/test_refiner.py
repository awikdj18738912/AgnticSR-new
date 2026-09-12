import unittest

from system.refiner import StreamingRefinementSession


class StreamingRefinerWindowTest(unittest.TestCase):
    def test_active_window_respects_sentence_chunk_character_limit(self):
        calls = []

        class Refiner:
            def refine(self, text):
                calls.append(text)
                return text

        session = StreamingRefinementSession(
            Refiner(), window_size=3, window_max_chars=10
        )
        for chunk in ("第一句。", "第二句。", "第三句。", "第四句。"):
            update = session.add(chunk)

        self.assertEqual(update.raw_chunks, ("第三句。", "第四句。"))
        self.assertEqual(update.transcript, "第一句。第二句。第三句。第四句。")
        self.assertTrue(all(len(text) <= 10 for text in calls))


if __name__ == "__main__":
    unittest.main()
