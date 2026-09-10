"""Source-indexed K=3 refinement for revisable cumulative ASR hypotheses.

Committed source chunks and their outputs have explicit ownership. Active
windows replace, never append to, the previous active output. Like the local
StreamingRefinementSession, chunks leaving the window are refined separately
to establish an unambiguous committed boundary.
"""
from threading import Lock
from .chunking import ChunkManager
from .refinement_guard import join_refined_segments


class CumulativeWindowRefinement:
    def __init__(self, refine, window_size=3):
        self.refine = refine
        self.window_size = window_size
        self.committed = []
        self.active = None
        self.lock = Lock()

    def update(self, text, language, final, protector, confidence, matcher=None):
        with self.lock:
            manager = ChunkManager(max_chars=80)
            chunks = [c.text for c in manager.update(text, vad_boundary=True)]
            start = max(0, len(chunks) - self.window_size)
            # A recognizer may revise earlier text. Invalidate affected cached
            # spans using source equality, never refined-text character offsets.
            shared = 0
            while (shared < min(start, len(self.committed))
                   and self.committed[shared][0] == chunks[shared]):
                shared += 1
            self.committed = self.committed[:shared]
            for chunk in chunks[shared:start]:
                result = self.refine(chunk, language, False, protector, confidence, matcher,
                                     single_window=True)
                self.committed.append((chunk, result))
            source = join_refined_segments(chunks[start:])
            if self.active is None or self.active[0] != source:
                result = self.refine(source, language, False, protector, confidence, matcher,
                                     single_window=True)
                self.active = (source, result)
            parts = [result for _, result in self.committed] + [self.active[1]]
            result = dict(self.active[1])
            result.update(event='final' if final else 'update', raw_text=text,
                          clean_text=join_refined_segments(p['clean_text'] for p in parts),
                          refiner_accepted=all(p['refiner_accepted'] for p in parts),
                          refiner_latency_ms=sum(p['refiner_latency_ms'] for p in parts),
                          placeholder_retry_count=sum(
                              p.get('placeholder_retry_count', 0) for p in parts
                          ),
                          refiner_retry_count=sum(
                              p.get('refiner_retry_count', 0) for p in parts
                          ),
                          window_size=self.window_size, committed_chunks=start)
            for key in ('refiner_reject_reasons', 'entity_audit_issues',
                        'entity_refinement_hints', 'entity_normalizations',
                        'protected_entities', 'entity_candidates',
                        'refiner_masked_outputs', 'refiner_retry_reasons'):
                result[key] = [item for p in parts for item in p.get(key, [])]
            result['entity_matcher_latency_ms'] = sum(
                p.get('entity_matcher_latency_ms', 0.0) for p in parts
            )
            return result
