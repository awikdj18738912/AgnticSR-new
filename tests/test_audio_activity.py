from __future__ import annotations

import unittest

import numpy as np

from system.audio_activity import is_silence, pcm_peak, pcm_rms


class AudioActivityTest(unittest.TestCase):
    def test_zero_chunk_is_silence(self) -> None:
        samples = np.zeros(1600, dtype=np.float32)

        self.assertEqual(pcm_rms(samples), 0.0)
        self.assertEqual(pcm_peak(samples), 0.0)
        self.assertTrue(is_silence(samples, rms_threshold=0.002))

    def test_signal_above_threshold_is_not_silence(self) -> None:
        samples = np.full(1600, 0.01, dtype=np.float32)

        self.assertAlmostEqual(pcm_rms(samples), 0.01)
        self.assertFalse(is_silence(samples, rms_threshold=0.002))

    def test_negative_threshold_is_invalid(self) -> None:
        with self.assertRaises(ValueError):
            is_silence(np.zeros(1, dtype=np.float32), rms_threshold=-1)


if __name__ == "__main__":
    unittest.main()
