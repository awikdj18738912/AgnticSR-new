import unittest

from system.chunking import ChunkManager


class ChunkManagerTest(unittest.TestCase):
    def test_short_self_correction_stays_together(self):
        source = "我有一个苹果，不对，我有一个梨。"
        chunks = ChunkManager(80).update(source, vad_boundary=True)

        self.assertEqual([chunk.text for chunk in chunks], [source])

    def test_long_self_corrections_use_soft_clause_boundaries(self):
        source = (
            "你好，我有一个苹果，不对，我有一个梨，我有一箱苹果，不对，我有一箱梨，"
            "我有两千一百三十五元，我有百分之三的几率获得五万四千三百二十一元。"
        )
        chunks = ChunkManager(80).update(source, vad_boundary=True)

        self.assertEqual(
            [chunk.text for chunk in chunks],
            [
                "你好，我有一个苹果，不对，我有一个梨，",
                "我有一箱苹果，不对，我有一箱梨，",
                "我有两千一百三十五元，我有百分之三的几率获得五万四千三百二十一元。",
            ],
        )
        self.assertTrue(all(len(chunk.text) <= 80 for chunk in chunks))

    def test_nearest_clause_boundary_keeps_numeric_correction_together(self):
        # A sentence terminator must close the corrected clause.  Searching
        # only soft punctuation jumps to a far-away comma, producing an
        # oversized chunk that pushes the corrected value out of the
        # K-window refinement range.
        source = (
            "你好，我有一个苹果。不对，我有一个梨。我有百分之三十七的概率在"
            "二零一五年十二月五日这天获得一千二百三十四万五千四百三十一元。"
            "不对，应该是一百五十块。不对，应该是一百块。"
        )
        chunks = ChunkManager(80).update(source, vad_boundary=True)

        self.assertEqual(
            [chunk.text for chunk in chunks],
            [
                "你好，我有一个苹果。不对，我有一个梨。",
                "我有百分之三十七的概率在二零一五年十二月五日这天获得"
                "一千二百三十四万五千四百三十一元。不对，应该是一百五十块。"
                "不对，应该是一百块。",
            ],
        )
        self.assertTrue(all(len(chunk.text) <= 80 for chunk in chunks))

    def test_sentence_before_correction_is_merged(self):
        source = "你好，我有一个苹果。不对，我有一个梨。我有一千元。"
        chunks = ChunkManager(80).update(source, vad_boundary=True)

        self.assertEqual(
            [chunk.text for chunk in chunks],
            ["你好，我有一个苹果。不对，我有一个梨。", "我有一千元。"],
        )


if __name__ == "__main__":
    unittest.main()
