from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.scripts.calibrate_confidence import main


class CalibrateConfidenceTest(unittest.TestCase):
    def test_calibration_script_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "scores.jsonl"
            target = root / "calibration.json"
            source.write_text(
                "\n".join(
                    json.dumps({"asr_confidence": score, "confidence_label": label})
                    for score, label in (
                        (0.95, 1),
                        (0.90, 0),
                        (0.70, 1),
                        (0.20, 0),
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            self.assertEqual(main([str(source), str(target)]), 0)
            result = json.loads(target.read_text(encoding="utf-8"))
            self.assertTrue(result["calibrated"])
            self.assertEqual(result["count"], 4)
            self.assertGreater(result["temperature"], 0)


if __name__ == "__main__":
    unittest.main()
