"""Confidence evidence extraction and post-hoc calibration helpers.

This module is intentionally independent from the online gate.  ASR backends
do not agree on confidence field names or score semantics: NeMo exposes token,
word and frame confidence, Whisper exposes segment ``avg_logprob`` and some
third-party wrappers expose word probabilities.  The extractor keeps those
signals auditable and never treats a raw log probability as a calibrated
probability by itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite, log
from typing import Any, Iterable, Mapping, Sequence


_EXPLICIT_KEYS = ("asr_confidence", "confidence")
_LIST_KEYS = ("word_confidence", "token_confidence", "frame_confidence")
_PROBABILITY_KEYS = ("confidence", "probability", "prob", "score")


@dataclass(frozen=True, slots=True)
class ConfidenceEvidence:
    """One normalized score and its provenance.

    ``value`` is safe to use as a probability only when ``calibrated`` is true
    or the backend explicitly documents the score as a probability.  A value
    derived from ``avg_logprob`` is deliberately marked uncalibrated.
    """

    value: float | None
    source: str | None
    aggregation: str | None
    count: int
    calibrated: bool
    raw_score: float | None = None

    def public_dict(self) -> dict[str, object]:
        return {
            "value": self.value,
            "source": self.source,
            "aggregation": self.aggregation,
            "count": self.count,
            "calibrated": self.calibrated,
            "raw_score": self.raw_score,
        }


def _finite_probability(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not isfinite(number):
        return None
    # Backend confidence occasionally has tiny numerical excursions outside
    # [0, 1]. Clamp those, but reject values that clearly use another scale.
    if -1e-6 <= number <= 1.0 + 1e-6:
        return min(1.0, max(0.0, number))
    return None


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if isfinite(number) else None


def aggregate_confidence(values: Iterable[float], method: str = "mean") -> float:
    """Aggregate token/word probabilities using NeMo-compatible methods."""

    numbers = [float(value) for value in values]
    if not numbers:
        raise ValueError("at least one confidence value is required")
    if any(not isfinite(value) or not 0 <= value <= 1 for value in numbers):
        raise ValueError("confidence values must be finite and in [0, 1]")
    method = method.strip().lower()
    if method == "mean":
        return sum(numbers) / len(numbers)
    if method == "min":
        return min(numbers)
    if method == "max":
        return max(numbers)
    if method == "prod":
        product = 1.0
        for value in numbers:
            product *= value
        return product
    raise ValueError("aggregation must be one of: mean, min, max, prod")


def _nested_mapping(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    output = payload.get("output")
    return output if isinstance(output, Mapping) else payload


def _extract_list_values(value: Any) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    values: list[float] = []
    for item in value:
        if isinstance(item, Mapping):
            candidate = next(
                (_finite_probability(item.get(key)) for key in _PROBABILITY_KEYS),
                None,
            )
        else:
            candidate = _finite_probability(item)
        if candidate is not None:
            values.append(candidate)
    return values


def extract_confidence(
    payload: Mapping[str, Any],
    *,
    aggregation: str = "mean",
) -> ConfidenceEvidence:
    """Extract confidence from common ASR result shapes.

    Explicit ``confidence``/``asr_confidence`` and token/word confidence are
    considered probability-like.  Whisper's ``avg_logprob`` is returned as an
    uncalibrated ``exp(avg_logprob)`` value for diagnostics only; callers must
    calibrate it before using it to skip refinement.
    """

    source = _nested_mapping(payload)
    for key in _EXPLICIT_KEYS:
        value = _finite_probability(source.get(key))
        if value is not None:
            return ConfidenceEvidence(value, key, None, 1, True, value)

    for key in _LIST_KEYS:
        values = _extract_list_values(source.get(key))
        if values:
            return ConfidenceEvidence(
                aggregate_confidence(values, aggregation),
                key,
                aggregation,
                len(values),
                True,
                None,
            )

    avg_logprob = _finite_number(source.get("avg_logprob"))
    if avg_logprob is not None:
        # This is a monotonic diagnostic score, not a calibrated probability.
        return ConfidenceEvidence(
            min(1.0, max(0.0, exp(avg_logprob))),
            "avg_logprob",
            None,
            1,
            False,
            avg_logprob,
        )

    return ConfidenceEvidence(None, None, None, 0, False, None)


def _clamp_probability(value: float) -> float:
    return min(1.0 - 1e-6, max(1e-6, float(value)))


@dataclass(frozen=True, slots=True)
class TemperatureCalibrator:
    """Post-hoc temperature scaling for probability-like ASR scores."""

    temperature: float = 1.0

    def __post_init__(self) -> None:
        if not isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")

    def transform(self, value: float) -> float:
        probability = _clamp_probability(value)
        logit = log(probability / (1.0 - probability))
        scaled = logit / self.temperature
        if scaled >= 0:
            return 1.0 / (1.0 + exp(-scaled))
        exponent = exp(scaled)
        return exponent / (1.0 + exponent)

    @classmethod
    def fit(
        cls,
        probabilities: Sequence[float],
        labels: Sequence[int | bool],
    ) -> "TemperatureCalibrator":
        """Fit temperature by a deterministic log-spaced NLL grid search.

        This small implementation keeps the experiment dependency-free.  It
        is sufficient for a first calibration study; production experiments
        can replace it with sklearn/scipy without changing the data contract.
        """

        if len(probabilities) != len(labels) or len(probabilities) < 2:
            raise ValueError("probabilities and labels must have at least two items")
        if not {int(bool(label)) for label in labels} == {0, 1}:
            raise ValueError("calibration labels must contain both 0 and 1")
        values = [_clamp_probability(value) for value in probabilities]
        targets = [int(bool(label)) for label in labels]

        def nll(temperature: float) -> float:
            total = 0.0
            for value, target in zip(values, targets):
                calibrated = _clamp_probability(cls(temperature).transform(value))
                total -= target * log(calibrated) + (1 - target) * log(1 - calibrated)
            return total / len(values)

        # A broad deterministic grid is robust for small development sets.
        candidates = [10 ** (-2 + 4 * index / 80) for index in range(81)]
        best = min(candidates, key=nll)
        return cls(best)


def expected_calibration_error(
    probabilities: Sequence[float],
    labels: Sequence[int | bool],
    *,
    bins: int = 10,
) -> float:
    """Compute a simple equal-width expected calibration error."""

    if len(probabilities) != len(labels) or not probabilities:
        raise ValueError("probabilities and labels must have the same non-zero length")
    if bins < 1:
        raise ValueError("bins must be positive")
    total = len(probabilities)
    error = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        members = [
            (float(probability), int(bool(label)))
            for probability, label in zip(probabilities, labels)
            if lower <= probability < upper or (index == bins - 1 and probability == 1)
        ]
        if members:
            confidence = sum(item[0] for item in members) / len(members)
            accuracy = sum(item[1] for item in members) / len(members)
            error += len(members) / total * abs(confidence - accuracy)
    return error
