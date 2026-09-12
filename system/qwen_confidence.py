"""Optional token-logprob capture for the local Qwen vLLM streamer.

The public qwen-asr streaming helper returns only ``state.text`` and
``state.language``.  This adapter mirrors its small decode loop so vLLM's
``CompletionOutput.logprobs`` can be retained without changing the default
text-generation path.  The resulting score is an uncalibrated diagnostic
unless a temperature learned on labeled development data is supplied.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class QwenConfidenceSummary:
    """Aggregate score for one generated ASR update."""

    mean_logprob: float | None
    mean_probability: float | None
    min_probability: float | None
    token_count: int
    temperature: float
    calibrated: bool
    covers_full_state: bool

    def public_dict(self) -> dict[str, object]:
        return {
            "confidence": self.mean_probability,
            "confidence_mean_logprob": self.mean_logprob,
            "confidence_min_probability": self.min_probability,
            "confidence_token_count": self.token_count,
            "confidence_temperature": self.temperature,
            "confidence_calibrated": self.calibrated,
            "confidence_source": "qwen_vllm_token_logprob",
            "confidence_covers_full_state": self.covers_full_state,
        }


def _logprob_value(value: Any) -> float | None:
    candidate = getattr(value, "logprob", value)
    if isinstance(candidate, bool) or not isinstance(candidate, (int, float)):
        return None
    candidate = float(candidate)
    return candidate if isfinite(candidate) else None


def completion_logprobs(completion: Any) -> list[float]:
    """Read chosen-token logprobs from a vLLM CompletionOutput.

    vLLM versions expose ``logprobs`` as a list of dictionaries keyed by token
    id.  The helper also accepts string keys and scalar dictionary values to
    keep tests and minor vLLM releases compatible.
    """

    rows = getattr(completion, "logprobs", None)
    token_ids = list(getattr(completion, "token_ids", ()) or ())
    if not rows or not token_ids:
        return []
    values: list[float] = []
    for token_id, row in zip(token_ids, rows):
        value: float | None = None
        if isinstance(row, dict):
            item = row.get(token_id)
            if item is None:
                item = row.get(str(token_id))
            if item is None and len(row) == 1:
                item = next(iter(row.values()))
            value = _logprob_value(item)
        else:
            value = _logprob_value(row)
        if value is not None:
            values.append(value)
    return values


def summarize_logprobs(
    logprobs: list[float],
    *,
    temperature: float = 1.0,
    calibrated: bool = False,
    covers_full_state: bool = False,
) -> QwenConfidenceSummary:
    """Convert token logprobs into a diagnostic or calibrated score."""

    if temperature <= 0 or not isfinite(temperature):
        raise ValueError("temperature must be finite and positive")
    if not logprobs:
        return QwenConfidenceSummary(
            None, None, None, 0, temperature, calibrated, False
        )
    mean_logprob = sum(logprobs) / len(logprobs)
    # exp(mean log p) is the geometric mean token probability.  It is useful
    # as a monotonic score, but only calibrated=True makes it gate-eligible.
    scaled = mean_logprob / temperature
    mean_probability = min(1.0, max(0.0, exp(scaled)))
    min_probability = min(
        min(1.0, max(0.0, exp(value / temperature))) for value in logprobs
    )
    return QwenConfidenceSummary(
        mean_logprob,
        mean_probability,
        min_probability,
        len(logprobs),
        temperature,
        calibrated,
        covers_full_state,
    )


def update_state_confidence(
    state: Any,
    completion: Any,
    *,
    temperature: float,
    calibrated: bool,
    covers_full_state: bool = False,
) -> None:
    """Attach the latest confidence summary to an ASRStreamingState."""

    logprobs = completion_logprobs(completion)
    token_ids = list(getattr(completion, "token_ids", ()) or ())
    complete_token_coverage = bool(token_ids) and len(logprobs) == len(token_ids)
    summary = summarize_logprobs(
        logprobs,
        temperature=temperature,
        calibrated=calibrated,
        # A blank decode prefix means every character in the current streaming
        # state came from this completion. Missing token logprobs invalidate
        # that claim even when the textual prefix was blank.
        covers_full_state=covers_full_state and complete_token_coverage,
    )
    state._qwen_confidence = summary


def state_confidence_payload(state: Any) -> dict[str, object]:
    summary = getattr(state, "_qwen_confidence", None)
    if not isinstance(summary, QwenConfidenceSummary):
        return {
            "confidence": None,
            "confidence_mean_logprob": None,
            "confidence_min_probability": None,
            "confidence_token_count": 0,
            "confidence_calibrated": False,
            "confidence_source": None,
            "confidence_covers_full_state": False,
        }
    return summary.public_dict()


class QwenVLLMConfidenceDecoder:
    """Drop-in streaming adapter that preserves vLLM token logprobs.

    ``qwen_asr`` intentionally exposes only text from its streaming helper.
    This adapter mirrors that helper's bounded decode loop and is enabled only
    when the service is started with ``--confidence-logprobs``.
    """

    def __init__(
        self,
        asr: Any,
        *,
        top_logprobs: int,
        temperature: float = 1.0,
        calibrated: bool = False,
    ) -> None:
        if top_logprobs < 1:
            raise ValueError("top_logprobs must be positive")
        if temperature <= 0 or not isfinite(temperature):
            raise ValueError("temperature must be finite and positive")
        try:
            from vllm import SamplingParams
            from qwen_asr.inference.qwen3_asr import parse_asr_output
        except ImportError as error:
            raise RuntimeError(
                "Qwen confidence capture requires qwen-asr[vllm]"
            ) from error
        self.asr = asr
        self.temperature = temperature
        self.calibrated = calibrated
        self._parse_asr_output = parse_asr_output
        base = getattr(asr, "sampling_params", None)
        max_tokens = int(getattr(base, "max_tokens", 32) or 32)
        self.sampling_params = SamplingParams(
            temperature=float(getattr(base, "temperature", 0.0)),
            max_tokens=max_tokens,
            logprobs=top_logprobs,
        )

    def init_streaming_state(self, **kwargs: Any) -> Any:
        return self.asr.init_streaming_state(**kwargs)

    def _prefix(self, state: Any, *, finishing: bool = False) -> str:
        if state.chunk_id < state.unfixed_chunk_num:
            return ""
        tokenizer = self.asr.processor.tokenizer
        cur_ids = tokenizer.encode(state._raw_decoded)
        if finishing:
            end_idx = max(1, len(cur_ids) - int(state.unfixed_token_num))
            return tokenizer.decode(cur_ids[:end_idx])
        rollback = int(state.unfixed_token_num)
        while True:
            end_idx = max(0, len(cur_ids) - rollback)
            prefix = tokenizer.decode(cur_ids[:end_idx]) if end_idx > 0 else ""
            if "\ufffd" not in prefix or end_idx == 0:
                return prefix
            rollback += 1

    def _decode(self, state: Any, prefix: str) -> None:
        prompt = state.prompt_raw + prefix
        request = {
            "prompt": prompt,
            "multi_modal_data": {"audio": [state.audio_accum]},
        }
        outputs = self.asr.model.generate(
            [request], sampling_params=self.sampling_params, use_tqdm=False
        )
        completion = outputs[0].outputs[0]
        state._raw_decoded = prefix + completion.text
        language, text = self._parse_asr_output(
            state._raw_decoded, user_language=state.force_language
        )
        state.language = language
        state.text = text
        update_state_confidence(
            state,
            completion,
            temperature=self.temperature,
            calibrated=self.calibrated,
            covers_full_state=not prefix,
        )

    def streaming_transcribe(self, pcm16k: Any, state: Any) -> Any:
        if getattr(self.asr, "backend", None) != "vllm":
            raise ValueError("confidence streaming requires the vLLM backend")
        if state is None:
            raise ValueError("state must not be None")
        x = np.asarray(pcm16k)
        if x.ndim != 1:
            x = x.reshape(-1)
        if x.dtype == np.int16:
            x = x.astype(np.float32) / 32768.0
        else:
            x = x.astype(np.float32, copy=False)
        if x.size:
            state.buffer = np.concatenate([state.buffer, x], axis=0)
        while state.buffer.shape[0] >= state.chunk_size_samples:
            chunk = state.buffer[: state.chunk_size_samples]
            state.buffer = state.buffer[state.chunk_size_samples :]
            state.audio_accum = (
                chunk
                if state.audio_accum.size == 0
                else np.concatenate([state.audio_accum, chunk], axis=0)
            )
            self._decode(state, self._prefix(state))
            state.chunk_id += 1
        return state

    def finish_streaming_transcribe(self, state: Any) -> Any:
        if state is None:
            raise ValueError("state must not be None")
        if state.buffer is None or state.buffer.size == 0:
            return state
        tail = state.buffer
        state.buffer = np.zeros((0,), dtype=np.float32)
        state.audio_accum = (
            tail
            if state.audio_accum.size == 0
            else np.concatenate([state.audio_accum, tail], axis=0)
        )
        self._decode(state, self._prefix(state, finishing=True))
        state.chunk_id += 1
        return state
