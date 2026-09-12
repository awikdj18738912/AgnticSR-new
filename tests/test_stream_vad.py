from __future__ import annotations

import unittest

import numpy as np

from system.stream_vad import StreamingVADGate


class FakeDetector:
    def __init__(self, states: list[bool]) -> None:
        self.states = iter(states)
        self.speech = False
        self.pending = 0

    def accept_waveform(self, samples: np.ndarray) -> None:
        del samples
        self.speech = next(self.states)
        if not self.speech:
            self.pending += 1

    def is_speech_detected(self) -> bool:
        return self.speech

    def empty(self) -> bool:
        return self.pending == 0

    def pop(self) -> None:
        self.pending -= 1


class StreamingVADGateTest(unittest.TestCase):
    def test_preroll_is_replayed_when_speech_starts(self) -> None:
        detector = FakeDetector([False, True, True])
        gate = StreamingVADGate(detector, sample_rate=10, preroll_seconds=1.0)
        first = np.arange(5, dtype=np.float32)
        second = np.arange(5, 10, dtype=np.float32)
        self.assertIsNone(gate.accept(first).forward)
        decision = gate.accept(second)
        np.testing.assert_array_equal(decision.forward, np.arange(10, dtype=np.float32))
        self.assertTrue(decision.started)

    def test_endpoint_chunk_is_forwarded(self) -> None:
        detector = FakeDetector([True, False, False])
        gate = StreamingVADGate(detector, sample_rate=10, preroll_seconds=0)
        first = np.ones(4, dtype=np.float32)
        tail = np.full(4, 2, dtype=np.float32)
        self.assertIsNotNone(gate.accept(first).forward)
        decision = gate.accept(tail)
        np.testing.assert_array_equal(decision.forward, tail)
        self.assertTrue(decision.ended)
        self.assertFalse(decision.active)
        self.assertIsNone(gate.accept(tail).forward)

    def test_preroll_is_bounded(self) -> None:
        detector = FakeDetector([False, False, True])
        gate = StreamingVADGate(detector, sample_rate=10, preroll_seconds=0.5)
        gate.accept(np.full(10, 1, dtype=np.float32))
        gate.accept(np.full(10, 2, dtype=np.float32))
        decision = gate.accept(np.full(10, 3, dtype=np.float32))
        np.testing.assert_array_equal(decision.forward, np.concatenate((np.full(5, 2), np.full(10, 3))))


if __name__ == "__main__":
    unittest.main()
