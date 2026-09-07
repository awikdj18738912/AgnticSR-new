"""Transformers Whisper backend shared by offline and streaming entry points."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

SAMPLE_RATE = 16_000

# Whisper accepts language names in its decoder prompt.  Keep the public CLI
# forgiving while passing the canonical names expected by Transformers.
_LANGUAGE_NAMES = {
    "zh": "chinese",
    "zh-cn": "chinese",
    "chinese": "chinese",
    "en": "english",
    "english": "english",
    "ja": "japanese",
    "japanese": "japanese",
    "ko": "korean",
    "korean": "korean",
    "fr": "french",
    "french": "french",
    "de": "german",
    "german": "german",
    "es": "spanish",
    "spanish": "spanish",
}


def whisper_language(language: str | None) -> str | None:
    """Return a canonical Whisper language name, or ``None`` for auto mode."""

    if language is None or language.lower() == "auto":
        return None
    normalized = language.strip().lower()
    return _LANGUAGE_NAMES.get(normalized, normalized)


class WhisperBackend:
    """A local Hugging Face Whisper backend.

    The backend deliberately exposes only ``transcribe`` so it can be used by
    both the batch wrapper and the streaming HTTP service.  ``model_path`` may
    be a local directory or a Hugging Face model id such as ``openai/whisper-small``.
    """

    def __init__(
        self,
        model_path: str | Path,
        device: str = "cuda:0",
        dtype: str = "auto",
        max_new_tokens: int = 256,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        except ImportError as error:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "Whisper requires torch and transformers in the active environment"
            ) from error

        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"Whisper device {device!r} requested but CUDA is unavailable")

        self._torch = torch
        self._device = torch.device(device)
        if dtype == "auto":
            model_dtype = torch.float16 if self._device.type == "cuda" else torch.float32
        else:
            try:
                model_dtype = getattr(torch, dtype)
            except AttributeError as error:
                raise ValueError(f"unsupported torch dtype: {dtype}") from error
        self._dtype = model_dtype
        self._max_new_tokens = max_new_tokens
        self._processor = AutoProcessor.from_pretrained(str(model_path))
        self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
            str(model_path),
            dtype=model_dtype,
            low_cpu_mem_usage=True,
        ).to(self._device).eval()

    def _forced_decoder_ids(self, language: str | None) -> Any:
        if language is None:
            return None
        getter = getattr(self._processor, "get_decoder_prompt_ids", None)
        if getter is None:
            getter = getattr(getattr(self._processor, "tokenizer", None), "get_decoder_prompt_ids", None)
        if getter is None:
            return None
        try:
            return getter(language=language, task="transcribe")
        except (KeyError, TypeError, ValueError):
            # Some Whisper-compatible checkpoints do not expose all language
            # aliases.  In that case allow the model's default language logic.
            return None

    def transcribe(self, samples: np.ndarray, language: str | None = None) -> str:
        waveform = np.asarray(samples, dtype=np.float32)
        if waveform.ndim != 1:
            waveform = waveform.reshape(-1)
        if waveform.size == 0:
            return ""

        inputs = self._processor(
            waveform,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
        )
        model_inputs: dict[str, Any] = {}
        for key, value in inputs.items():
            if hasattr(value, "to"):
                value = value.to(self._device)
                if key == "input_features":
                    value = value.to(dtype=self._dtype)
            model_inputs[key] = value

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self._max_new_tokens,
        }
        forced_decoder_ids = self._forced_decoder_ids(whisper_language(language))
        if forced_decoder_ids is not None:
            generation_kwargs["forced_decoder_ids"] = forced_decoder_ids

        with self._torch.inference_mode():
            generated = self._model.generate(**model_inputs, **generation_kwargs)
        return self._processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
