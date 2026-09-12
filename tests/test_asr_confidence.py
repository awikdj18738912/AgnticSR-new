from __future__ import annotations

import unittest

from system.asr_confidence import (
    TemperatureCalibrator,
    aggregate_confidence,
    expected_calibration_error,
    extract_confidence,
)


class ASRConfidenceTest(unittest.TestCase):
    def test_extracts_explicit_confidence(self) -> None:
        evidence = extract_confidence({"output": {"confidence": 0.83}})

        self.assertEqual(evidence.source, "confidence")
        self.assertEqual(evidence.value, 0.83)
        self.assertTrue(evidence.calibrated)

    def test_aggregates_nemo_style_word_confidence(self) -> None:
        evidence = extract_confidence(
            {"word_confidence": [0.8, 0.9, 0.7]}, aggregation="min"
        )

        self.assertEqual(evidence.source, "word_confidence")
        self.assertEqual(evidence.value, 0.7)
        self.assertEqual(evidence.count, 3)

    def test_avg_logprob_is_marked_uncalibrated(self) -> None:
        evidence = extract_confidence({"avg_logprob": -0.69314718056})

        self.assertAlmostEqual(evidence.value or 0.0, 0.5, places=5)
        self.assertFalse(evidence.calibrated)
        self.assertEqual(evidence.raw_score, -0.69314718056)

    def test_temperature_calibration_can_be_fitted(self) -> None:
        raw = [0.95, 0.9, 0.8, 0.2, 0.1, 0.05]
        labels = [1, 0, 1, 0, 0, 0]
        calibrator = TemperatureCalibrator.fit(raw, labels)

        self.assertGreater(calibrator.temperature, 0)
        self.assertTrue(0 < calibrator.transform(0.8) < 1)

    def test_ece(self) -> None:
        self.assertAlmostEqual(
            expected_calibration_error([0.9, 0.1], [1, 0], bins=2), 0.1
        )

    def test_aggregation_rejects_unknown_method(self) -> None:
        with self.assertRaises(ValueError):
            aggregate_confidence([0.5], "median")


if __name__ == "__main__":
    unittest.main()
