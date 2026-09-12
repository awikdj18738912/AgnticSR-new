"""Fit temperature scaling for recorded ASR confidence observations.

Input is JSONL with a probability-like score and a binary correctness label.
The output JSON can be copied into a run manifest and used to start the Qwen
streamer with ``--confidence-temperature`` and ``--confidence-calibrated``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from system.asr_confidence import (
    TemperatureCalibrator,
    expected_calibration_error,
)


def _nested_value(record: dict[str, Any], field: str) -> Any:
    value: Any = record
    for part in field.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate ASR confidence scores")
    parser.add_argument("input", type=Path, help="JSONL with score and binary label")
    parser.add_argument("output", type=Path, help="JSON calibration manifest")
    parser.add_argument("--score-field", default="asr_confidence")
    parser.add_argument("--label-field", default="confidence_label")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    probabilities: list[float] = []
    labels: list[int] = []
    with args.input.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            score = _nested_value(record, args.score_field)
            label = _nested_value(record, args.label_field)
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise ValueError(f"line {line_number}: invalid score")
            if isinstance(label, bool):
                label = int(label)
            if label not in (0, 1):
                raise ValueError(f"line {line_number}: label must be 0 or 1")
            probabilities.append(min(1.0, max(0.0, float(score))))
            labels.append(int(label))
    calibrator = TemperatureCalibrator.fit(probabilities, labels)
    calibrated = [calibrator.transform(value) for value in probabilities]
    result = {
        "temperature": calibrator.temperature,
        "count": len(probabilities),
        "score_field": args.score_field,
        "label_field": args.label_field,
        "ece_before": expected_calibration_error(probabilities, labels),
        "ece_after": expected_calibration_error(calibrated, labels),
        "calibrated": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
