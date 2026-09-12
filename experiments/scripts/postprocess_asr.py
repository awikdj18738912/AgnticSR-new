#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmark_io import JsonObject, read_jsonl
from postprocess_contract import (
    InferenceOutcome,
    PostprocessError,
    build_result_record,
    completed_record_ids,
    extract_raw_text,
    source_record_id,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from system.entity_store import EntityStore
from system.entity_matcher import EntityCandidateMatcher, EntityFuzzyMode
from system.entity_pipeline import (
    PreparedEntitySegment,
    finalize_entity_segment,
    has_placeholder_failure,
    has_retryable_integrity_failure,
    prepare_entity_segment,
)
from system.protection import EntityProtector
from system.refinement_gate import (
    RefinementGate,
    RefinementGateDecision,
    RefinementGateMode,
)

SYSTEM_PROMPT = (
    "你是 ASR 文本纠错助手。保留原意，最小修改：去口癖/重复，修错字，补必要标点，"
    "处理自我修正。不要总结、扩写或解释。数字、日期、术语和代码符号已由系统规则"
    "处理，不得自行转换数字，成语中的汉字数字（如三番五次）必须保持原样。"
    "重要易错实体在末尾追加 <KEY>[词1、词2]；没有则不加。"
    "输入中形如 __ENTITY_000__ 的内容是不可编辑的受保护标记；"
    "每个标记必须原样保留一次，不得删除、改写、重复或调整顺序。"
)
STRICT_PLACEHOLDER_PROMPT = (
    "最高优先级：完整保留输入中的每个句子和信息，不得总结、缩写、"
    "合并或删除内容。先逐字复制所有 __ENTITY_NNN__ 标记到对应位置，"
    "再仅修正其他文字的错字和标点。任何标记都不能省略或改变。"
)
Conversation: TypeAlias = list[dict[str, str]]


@dataclass(frozen=True, slots=True)
class ModelConfig:
    model_path: str
    device_map: str
    dtype: str
    trust_remote_code: bool
    max_new_tokens: int
    do_sample: bool
    temperature: float
    top_p: float


@dataclass(frozen=True, slots=True)
class RunConfig:
    input_path: Path
    output_path: Path
    batch_size: int
    limit: int | None
    overwrite: bool
    validate_only: bool
    entity_db: Path | None
    entity_domain: str
    entity_fuzzy_mode: str
    refinement_gate_mode: str
    model: ModelConfig


def build_conversations(
    raw_texts: list[str],
    entity_hints: list[tuple[str, ...]] | None = None,
    *,
    strict_placeholders: bool = False,
) -> list[Conversation]:
    hints_by_text = entity_hints or [() for _ in raw_texts]
    if len(hints_by_text) != len(raw_texts):
        raise ValueError("entity_hints must align with raw_texts")
    return [
        [
            {
                "role": "system",
                "content": (
                    f"{SYSTEM_PROMPT}{STRICT_PLACEHOLDER_PROMPT}"
                    if strict_placeholders
                    else SYSTEM_PROMPT
                ),
            },
            {
                "role": "user",
                "content": (
                    f"{raw_text}\n<KEY>[{'、'.join(hints[:16])}]"
                    if hints
                    else raw_text
                ),
            },
        ]
        for raw_text, hints in zip(raw_texts, hints_by_text)
    ]


def thinking_template_kwargs(template: str) -> dict[str, bool]:
    return {"enable_thinking": False} if "enable_thinking" in template else {}


class TransformersPostprocessor:
    # A malformed or unusually long sample must not leave the batch process
    # waiting forever after ASR has already written the raw transcript.
    GENERATION_MAX_TIME_SECONDS = 30.0

    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self._tokenizer = AutoTokenizer.from_pretrained(
            config.model_path,
            padding_side="left",
            trust_remote_code=config.trust_remote_code,
        )
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            dtype=config.dtype,
            device_map=config.device_map,
            trust_remote_code=config.trust_remote_code,
        ).eval()
        template = self._tokenizer.get_chat_template()
        self._template_kwargs = thinking_template_kwargs(template)

    def generate(
        self,
        raw_texts: list[str],
        entity_hints: list[tuple[str, ...]] | None = None,
        *,
        strict_placeholders: bool = False,
    ) -> tuple[list[str], float]:
        conversations = build_conversations(
            raw_texts,
            entity_hints,
            strict_placeholders=strict_placeholders,
        )
        inputs = self._tokenizer.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            padding=True,
            return_dict=True,
            return_tensors="pt",
            **self._template_kwargs,
        )
        input_device = self._model.get_input_embeddings().weight.device
        inputs = inputs.to(input_device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        generation_args = {
            "max_new_tokens": self._config.max_new_tokens,
            "max_time": self.GENERATION_MAX_TIME_SECONDS,
            "do_sample": self._config.do_sample,
        }
        if self._config.do_sample:
            generation_args.update(
                {
                    "temperature": self._config.temperature,
                    "top_p": self._config.top_p,
                }
            )
        with torch.inference_mode():
            generated = self._model.generate(**inputs, **generation_args)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000
        input_width = inputs["input_ids"].shape[1]
        generated_texts = self._tokenizer.batch_decode(
            generated[:, input_width:], skip_special_tokens=True
        )
        if len(generated_texts) != len(raw_texts):
            raise PostprocessError("Model returned a different number of outputs than inputs")
        return [text.strip() for text in generated_texts], elapsed_ms


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(description="Post-process ASR raw_text with a local LM.")
    parser.add_argument("input_jsonl", type=Path)
    parser.add_argument("output_jsonl", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--entity-db",
        type=Path,
        help="optional SQLite database containing verified protected entities",
    )
    parser.add_argument("--entity-domain", default="general")
    parser.add_argument(
        "--entity-fuzzy-mode",
        choices=[mode.value for mode in EntityFuzzyMode],
        default="shadow",
    )
    parser.add_argument(
        "--refinement-gate-mode",
        choices=[mode.value for mode in RefinementGateMode],
        default="off",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be at least 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if not args.entity_domain.strip():
        parser.error("--entity-domain must not be empty")
    return RunConfig(
        input_path=args.input_jsonl.resolve(),
        output_path=args.output_jsonl.resolve(),
        batch_size=args.batch_size,
        limit=args.limit,
        overwrite=args.overwrite,
        validate_only=args.validate_only,
        entity_db=args.entity_db.resolve() if args.entity_db else None,
        entity_domain=args.entity_domain.strip(),
        entity_fuzzy_mode=args.entity_fuzzy_mode,
        refinement_gate_mode=args.refinement_gate_mode,
        model=ModelConfig(
            model_path=args.model,
            device_map=args.device_map,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
        ),
    )


def _append_records(path: Path, records: list[JsonObject]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _pending_records(config: RunConfig) -> list[JsonObject]:
    if config.input_path == config.output_path:
        raise PostprocessError("Input and output paths must be different")
    input_records = read_jsonl(config.input_path)
    existing = (
        read_jsonl(config.output_path)
        if config.output_path.exists() and not config.overwrite
        else []
    )
    completed = completed_record_ids(existing)
    seen: set[str] = set()
    pending: list[JsonObject] = []
    for record in input_records:
        record_id = source_record_id(record)
        if record_id in seen:
            raise PostprocessError(f"Duplicate source_record_id in input: {record_id}")
        seen.add(record_id)
        if record_id not in completed:
            pending.append(record)
    return pending[: config.limit] if config.limit is not None else pending


def _record_asr_confidence(record: JsonObject) -> float | None:
    """Read common confidence locations without changing the input contract."""

    output = record.get("output")
    values = [record.get("asr_confidence")]
    if isinstance(output, dict):
        values.extend((output.get("asr_confidence"), output.get("confidence")))
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        confidence = float(value)
        if 0 <= confidence <= 1:
            return confidence
    return None


def run(config: RunConfig) -> int:
    if config.input_path == config.output_path:
        raise PostprocessError("Input and output paths must be different")
    if config.overwrite and not config.validate_only and config.output_path.exists():
        config.output_path.unlink()
    pending = _pending_records(config)
    valid_records: list[JsonObject] = []
    invalid_records: list[JsonObject] = []
    for record in pending:
        try:
            extract_raw_text(record)
        except PostprocessError:
            invalid_records.append(record)
        else:
            valid_records.append(record)
    if config.validate_only:
        print(
            f"Validated input; {len(valid_records)} records ready, "
            f"{len(invalid_records)} records have no raw_text."
        )
        return 0
    if not pending:
        print("No pending records.")
        return 0
    definitions = (
        EntityStore(config.entity_db).list_entities(domain=config.entity_domain)
        if config.entity_db is not None
        else ()
    )
    protector = EntityProtector(definitions)
    fuzzy_mode = EntityFuzzyMode.parse(config.entity_fuzzy_mode)
    matcher = (
        EntityCandidateMatcher(
            definitions,
            selected_domain=config.entity_domain,
            mode=fuzzy_mode,
        )
        if definitions and fuzzy_mode is not EntityFuzzyMode.OFF
        else None
    )
    refinement_gate = RefinementGate(config.refinement_gate_mode)
    processor = TransformersPostprocessor(config.model) if valid_records else None
    for offset in range(0, len(pending), config.batch_size):
        batch = pending[offset : offset + config.batch_size]
        batch_inputs: list[str] = []
        batch_hints: list[tuple[str, ...]] = []
        batch_prepared: list[PreparedEntitySegment] = []
        batch_gate_decisions: list[RefinementGateDecision] = []
        for record in batch:
            try:
                raw_text = extract_raw_text(record)
            except PostprocessError:
                continue
            asr_confidence = _record_asr_confidence(record)
            prepared = prepare_entity_segment(
                raw_text,
                protector,
                matcher,
                allow_auto=True,
                confidence=asr_confidence,
            )
            batch_prepared.append(prepared)
            gate_decision = refinement_gate.decide(
                prepared.baseline_text,
                asr_confidence=asr_confidence,
                entity_hints=prepared.hints,
            )
            batch_gate_decisions.append(gate_decision)
            if gate_decision.should_refine:
                batch_inputs.append(prepared.protection.masked_text)
                batch_hints.append(prepared.hints)
        generated_texts: list[str] = []
        latency_ms = 0.0
        if batch_inputs:
            if processor is None:
                raise PostprocessError("Postprocessor was not initialized")
            print(
                f"Refining records {offset + 1}-{offset + len(batch)} "
                f"({len(batch_inputs)} valid inputs)...",
                flush=True,
            )
            generated_texts, latency_ms = processor.generate(batch_inputs, batch_hints)
        generated_iterator = iter(generated_texts)
        prepared_iterator = iter(batch_prepared)
        gate_iterator = iter(batch_gate_decisions)
        outcomes: list[InferenceOutcome] = []
        for record in batch:
            try:
                raw_text = extract_raw_text(record)
            except PostprocessError:
                outcomes.append(
                    InferenceOutcome(
                        None,
                        0.0,
                        "input_error: missing non-empty output.raw_text",
                        refiner_executed=False,
                        refinement_gate_mode=refinement_gate.mode.value,
                        refinement_gate_config=refinement_gate.config_dict(),
                    )
                )
            else:
                prepared = next(prepared_iterator)
                gate_decision = next(gate_iterator)
                gate_metadata = (gate_decision.public_dict(),)
                if not gate_decision.should_refine:
                    outcomes.append(
                        InferenceOutcome(
                            prepared.baseline_text,
                            0.0,
                            None,
                            refiner_accepted=True,
                            protected_entities=tuple(
                                span.public_dict()
                                for span in prepared.protection.spans
                            ),
                            entity_candidates=tuple(
                                match.public_dict()
                                for match in prepared.report.matches
                            ),
                            entity_normalizations=prepared.normalizations,
                            entity_refinement_hints=prepared.hints,
                            entity_audit_issues=protector.audit_unmasked(
                                prepared.baseline_text, prepared.protection
                            ),
                            entity_matcher_latency_ms=round(
                                prepared.report.matcher_latency_ms, 3
                            ),
                            entity_fuzzy_mode=fuzzy_mode.value,
                            entity_matching_config_version=(
                                matcher.config.version if matcher is not None else None
                            ),
                            refiner_executed=False,
                            refinement_gate_mode=refinement_gate.mode.value,
                            refinement_gate_config=refinement_gate.config_dict(),
                            refinement_gate_decisions=gate_metadata,
                            refinement_gate_skipped_segments=1,
                        )
                    )
                    continue
                refined_text = next(generated_iterator)
                refiner_masked_outputs = [refined_text]
                finalized = finalize_entity_segment(
                    refined_text, prepared, protector
                )
                sample_latency_ms = latency_ms
                placeholder_retry_count = 0
                refiner_retry_count = 0
                refiner_retry_reasons: tuple[str, ...] = ()
                if has_retryable_integrity_failure(finalized.reject_reasons):
                    refiner_retry_reasons = finalized.reject_reasons
                    placeholder_failed = has_placeholder_failure(
                        finalized.reject_reasons
                    )
                    retry_texts, retry_latency_ms = processor.generate(
                        [prepared.protection.masked_text],
                        [prepared.hints],
                        strict_placeholders=True,
                    )
                    refined_text = retry_texts[0]
                    refiner_masked_outputs.append(refined_text)
                    sample_latency_ms += retry_latency_ms
                    refiner_retry_count = 1
                    placeholder_retry_count = int(placeholder_failed)
                    finalized = finalize_entity_segment(
                        refined_text, prepared, protector
                    )
                outcomes.append(
                    InferenceOutcome(
                        finalized.text,
                        sample_latency_ms,
                        None,
                        refiner_accepted=finalized.accepted,
                        refiner_reject_reasons=finalized.reject_reasons,
                        protected_entities=tuple(
                            span.public_dict() for span in prepared.protection.spans
                        ),
                        entity_candidates=tuple(
                            match.public_dict() for match in prepared.report.matches
                        ),
                        entity_normalizations=prepared.normalizations,
                        entity_refinement_hints=prepared.hints,
                        entity_audit_issues=protector.audit_unmasked(
                            finalized.text, prepared.protection
                        ),
                        entity_matcher_latency_ms=round(
                            prepared.report.matcher_latency_ms, 3
                        ),
                        entity_fuzzy_mode=fuzzy_mode.value,
                        entity_matching_config_version=(
                            matcher.config.version if matcher is not None else None
                        ),
                        placeholder_retry_count=placeholder_retry_count,
                        refiner_retry_count=refiner_retry_count,
                        refiner_retry_reasons=refiner_retry_reasons,
                        refiner_masked_outputs=tuple(refiner_masked_outputs),
                        refiner_executed=True,
                        refinement_gate_mode=refinement_gate.mode.value,
                        refinement_gate_config=refinement_gate.config_dict(),
                        refinement_gate_decisions=gate_metadata,
                    )
                )
        _append_records(
            config.output_path,
            [build_result_record(record, outcome) for record, outcome in zip(batch, outcomes)],
        )
        completed = min(offset + len(batch), len(pending))
        print(f"Completed {completed}/{len(pending)} records.")
    print(f"Output: {config.output_path}")
    return 0


def main() -> int:
    try:
        return run(parse_args())
    except (OSError, PostprocessError, json.JSONDecodeError) as error:
        print(f"error: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
