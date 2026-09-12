"""Sentence-bounded K=3 refinement for revisable cumulative ASR hypotheses.

Committed source chunks and their outputs have explicit ownership. Active
windows replace, never append to, the previous active output. Like the local
StreamingRefinementSession, chunks leaving the window are refined separately
to establish an unambiguous committed boundary. The active tail is limited to
three sentence chunks and at most 80 characters by default.
"""
import re
from threading import Lock
from .chunking import ChunkManager, merge_self_correction_chunks
from .deterministic_cleanup import clean_transcript_deterministically
from .refinement_guard import join_refined_segments


_SELF_CORRECTION = re.compile(r"(?:不对|不是|而是|应该是)")


class CumulativeWindowRefinement:
    def __init__(self, refine, window_size=3, window_max_chars=80):
        if window_size < 1:
            raise ValueError("window_size must be at least 1")
        if window_max_chars < 1:
            raise ValueError("window_max_chars must be at least 1")
        self.refine = refine
        self.window_size = window_size
        self.window_max_chars = window_max_chars
        self.committed = []
        self.active = None
        self.lock = Lock()

    @property
    def has_cached_result(self):
        """Return whether at least one window refinement has completed."""

        with self.lock:
            return self.active is not None

    def update(
        self,
        text,
        language,
        final,
        protector,
        confidence,
        matcher=None,
        *,
        confidence_metadata=None,
    ):
        with self.lock:
            manager = ChunkManager(max_chars=self.window_max_chars)
            chunks = [
                c.text
                for c in merge_self_correction_chunks(
                    manager.update(text, vad_boundary=True),
                    self.window_max_chars,
                )
            ]
            start = self._active_start(chunks)
            # A recognizer may revise earlier text. Invalidate affected cached
            # spans using source equality, never refined-text character offsets.
            # Keep a source-indexed view of the old cache as well: a small
            # punctuation revision can change the first chunk and otherwise
            # needlessly force every unchanged committed chunk through the
            # Refiner again.
            previous_committed = list(self.committed)
            shared = 0
            while (shared < min(start, len(self.committed))
                   and self.committed[shared][0] == chunks[shared]):
                shared += 1
            used_cached_indices = set(range(shared))
            self.committed = self.committed[:shared]
            for chunk in chunks[shared:start]:
                result = self._reuse_cached_chunk(
                    chunk, previous_committed, used_cached_indices
                )
                # A streaming hypothesis can be shorter or less stable than
                # the final hypothesis.  Do not carry a rejected intermediate
                # result into the final transcript: give that source window a
                # fresh final pass before deciding whether to fall back.
                if result is None or (
                    final and not result.get("refiner_accepted", True)
                ):
                    result = self.refine(
                        chunk,
                        language,
                        final,
                        protector,
                        confidence,
                        matcher,
                        single_window=True,
                        confidence_metadata=confidence_metadata,
                    )
                self.committed.append((chunk, result))
            source = join_refined_segments(chunks[start:])
            active_needs_refresh = (
                self.active is not None
                and final
                and self.active[0] == source
                and not self.active[1].get("refiner_accepted", True)
            )
            if (
                self.active is None
                or self.active[0] != source
                or active_needs_refresh
            ):
                active_chunks = chunks[start:]
                split_self_correction = (
                    len(active_chunks) > 1
                    and _SELF_CORRECTION.search(source) is not None
                )
                if split_self_correction:
                    # Keep each complete self-correction clause together, but
                    # do not ask one generation to rewrite several corrections
                    # and numeric edits at once.
                    refined_parts = [
                        self.refine(
                            chunk,
                            language,
                            final,
                            protector,
                            confidence,
                            matcher,
                            single_window=True,
                            confidence_metadata=confidence_metadata,
                        )
                        for chunk in active_chunks
                    ]
                    result = self._aggregate(
                        refined_parts,
                        source,
                        final=final,
                        committed_chunks=0,
                    )
                else:
                    result = self.refine(
                        source,
                        language,
                        final,
                        protector,
                        confidence,
                        matcher,
                        single_window=True,
                        confidence_metadata=confidence_metadata,
                    )
                # A long active window can trigger the content-loss guard even
                # when most of its individual chunks are safe. Recover it at
                # chunk granularity so one bad generation does not restore the
                # entire rolling window to raw ASR text.
                if (
                    not result.get("refiner_accepted", True)
                    and len(active_chunks) > 1
                    and not split_self_correction
                ):
                    recovered = [
                        self.refine(
                            chunk,
                            language,
                            final,
                            protector,
                            confidence,
                            matcher,
                            single_window=True,
                            confidence_metadata=confidence_metadata,
                        )
                        for chunk in chunks[start:]
                    ]
                    result = self._aggregate(
                        recovered,
                        source,
                        final=final,
                        committed_chunks=0,
                    )
                self.active = (source, result)
            parts = [result for _, result in self.committed] + [self.active[1]]
            result = self._aggregate(
                parts,
                text,
                final=final,
                committed_chunks=start,
            )
            return result

    def _aggregate(self, parts, raw_text, *, final, committed_chunks):
        result = dict(parts[-1])
        result.update(
            event="final" if final else "update",
            raw_text=raw_text,
            clean_text=clean_transcript_deterministically(
                join_refined_segments(p["clean_text"] for p in parts)
            ),
            refiner_accepted=all(p["refiner_accepted"] for p in parts),
            refiner_executed=any(p.get("refiner_executed", True) for p in parts),
            refiner_latency_ms=sum(p["refiner_latency_ms"] for p in parts),
            placeholder_retry_count=sum(
                p.get("placeholder_retry_count", 0) for p in parts
            ),
            refiner_retry_count=sum(p.get("refiner_retry_count", 0) for p in parts),
            refinement_gate_skipped_segments=sum(
                p.get("refinement_gate_skipped_segments", 0) for p in parts
            ),
            window_size=self.window_size,
            window_max_chars=self.window_max_chars,
            committed_chunks=committed_chunks,
        )
        for key in (
            "refiner_reject_reasons",
            "entity_audit_issues",
            "entity_refinement_hints",
            "entity_normalizations",
            "numeric_normalizations",
            "protected_entities",
            "entity_candidates",
            "refiner_masked_outputs",
            "refiner_retry_reasons",
            "refinement_gate_decisions",
        ):
            result[key] = [item for p in parts for item in p.get(key, [])]
        result["entity_matcher_latency_ms"] = sum(
            p.get("entity_matcher_latency_ms", 0.0) for p in parts
        )
        return result

    @staticmethod
    def _reuse_cached_chunk(chunk, cached, used_indices):
        """Reuse an unchanged committed source chunk after a prefix revision."""

        for index, (source, result) in enumerate(cached):
            if index in used_indices or source != chunk:
                continue
            used_indices.add(index)
            return result
        return None

    def _active_start(self, chunks):
        """Choose a sentence-oriented tail bounded by count and characters."""

        start = len(chunks)
        total_chars = 0
        active_count = 0
        while start > 0 and active_count < self.window_size:
            candidate = chunks[start - 1]
            candidate_chars = len(candidate)
            if active_count and total_chars + candidate_chars > self.window_max_chars:
                break
            start -= 1
            total_chars += candidate_chars
            active_count += 1
        return start
