"""Stateful VAD gating for chunked HTTP audio streams.

The browser sends relatively large PCM chunks (normally 0.5--1 second), while
the VAD itself works on smaller internal windows.  ``StreamingVADGate`` keeps a
small pre-roll so that the first syllable is not lost while the VAD is waiting
for its minimum-speech duration.  The chunk that ends speech is still
forwarded, so trailing syllables are not dropped when the detector changes to
the non-speech state.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VADDecision:
    """Result of accepting one PCM chunk."""

    forward: np.ndarray | None
    speech_detected: bool
    active: bool
    started: bool
    ended: bool


class StreamingVADGate:
    """Wrap a streaming detector and return only speech-bearing chunks.

    ``detector`` follows the small interface already used by ``backends.py``:
    ``accept_waveform``, ``is_speech_detected``, ``empty`` and ``pop``.  The
    detector is deliberately injected so this class can be tested without a
    native VAD dependency.
    """

    def __init__(
        self,
        detector: object,
        *,
        sample_rate: int = 16_000,
        preroll_seconds: float = 0.35,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if preroll_seconds < 0:
            raise ValueError("preroll_seconds must be non-negative")
        self.detector = detector
        self.sample_rate = sample_rate
        self.preroll_seconds = preroll_seconds
        self.active = False
        self._preroll: deque[np.ndarray] = deque()
        self._preroll_samples = 0

    def _remember_preroll(self, samples: np.ndarray) -> None:
        if self.preroll_seconds <= 0 or samples.size == 0:
            return
        limit = max(1, round(self.preroll_seconds * self.sample_rate))
        chunk = samples[-limit:].copy()
        self._preroll.append(chunk)
        self._preroll_samples += chunk.size
        while self._preroll and self._preroll_samples > limit:
            removed = self._preroll.popleft()
            excess = self._preroll_samples - limit
            if removed.size <= excess:
                self._preroll_samples -= removed.size
                continue
            kept = removed[excess:]
            self._preroll.appendleft(kept)
            self._preroll_samples -= excess
            break

    def _take_preroll(self) -> np.ndarray:
        if not self._preroll:
            return np.zeros(0, dtype=np.float32)
        buffered = np.concatenate(tuple(self._preroll)).astype(np.float32, copy=False)
        self._preroll.clear()
        self._preroll_samples = 0
        return buffered

    def accept(self, samples: np.ndarray) -> VADDecision:
        """Accept a normalized float PCM chunk and decide whether to forward it."""

        chunk = np.asarray(samples, dtype=np.float32).reshape(-1)
        if chunk.size == 0:
            return VADDecision(
                forward=None,
                speech_detected=self.active,
                active=self.active,
                started=False,
                ended=False,
            )
        was_active = self.active
        self.detector.accept_waveform(chunk)
        speech = bool(self.detector.is_speech_detected())

        # Clear one-shot endpoint events exposed by sherpa/FireRed adapters.
        while not self.detector.empty():
            self.detector.pop()

        started = speech and not was_active
        ended = was_active and not speech
        if started:
            prefix = self._take_preroll()
            forward = (
                chunk.copy()
                if prefix.size == 0
                else np.concatenate((prefix, chunk)).astype(np.float32, copy=False)
            )
        elif was_active:
            # Keep the endpoint chunk: it can contain the final syllable before
            # the detector's hangover timer changes to non-speech.
            forward = chunk.copy()
        else:
            self._remember_preroll(chunk)
            forward = None

        self.active = speech
        return VADDecision(
            forward=forward,
            speech_detected=speech,
            active=self.active,
            started=started,
            ended=ended,
        )

    def reset(self) -> None:
        """Forget stream-local state before reusing the gate."""

        self.active = False
        self._preroll.clear()
        self._preroll_samples = 0
