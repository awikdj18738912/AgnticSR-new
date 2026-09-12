"""Cheap pre-ASR speech activity checks for PCM chunks."""

from __future__ import annotations

from math import sqrt

import numpy as np


def pcm_rms(samples: np.ndarray) -> float:
    """Return RMS amplitude for normalized float PCM in a safe scalar form."""

    values = np.asarray(samples, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return 0.0
    return float(sqrt(float(np.mean(np.square(values, dtype=np.float32)))))


def pcm_peak(samples: np.ndarray) -> float:
    values = np.asarray(samples, dtype=np.float32).reshape(-1)
    return float(np.max(np.abs(values))) if values.size else 0.0


def is_silence(samples: np.ndarray, *, rms_threshold: float) -> bool:
    if rms_threshold < 0:
        raise ValueError("rms_threshold must be non-negative")
    return pcm_rms(samples) < rms_threshold
