"""
Wraps the EXISTING, UNMODIFIED retrieval/rerank/generation research code
into the one callable the API layer uses. No retrieval or reranking logic
is reimplemented here — this is purely composition:
  - scripts/fusion.py::HybridRetriever   (dense + BM25 + RRF, Step 2)
  - scripts/rerank.py::Reranker          (bge-reranker-large, Step 5)
  - app/generation/*                     (this milestone's new code)

"Insufficient evidence" detection reuses the EXISTING prompt instruction
(scripts/llm_client.py's build_prompt already tells the model to say
"I don't know" when the context doesn't contain the answer) rather than
inventing a new, unvalidated numeric confidence threshold — no such
threshold was established anywhere in the Step 4/5 research, so a fresh
one here would be a guess dressed up as a metric.

Document-ambiguity detection (a question that could have a genuinely
different correct answer depending on which retrieved document it refers
to) follows the exact same principle: app/disambiguation.py only supplies
the candidate-document LIST from retrieval metadata, never a hardcoded
question/date pattern, and the final "would these actually differ"
judgment is still made by the LLM reading their real content — not a
second invented numeric threshold.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from fusion import HybridRetriever  # noqa: E402
from rerank import Reranker  # noqa: E402

from app.disambiguation import cluster_candidate_documents, format_candidates
from app.excerpt import select_excerpt
from app.generation import GenerationError, get_provider

CLARIFICATION_PREFIX = "CLARIFICATION_NEEDED:"


class RAGPipeline:
    def __init__(self, settings):
        self.settings = settings
        self._retriever = None
        self._reranker = None
        self._provider = None

    # Lazy-loaded so object construction (used by health checks) is instant;
    # the first real query pays the model-loading cost, matching the
    # cold-start behavior already measured and documented in PROGRESS.md.
    @property
    def retriever(self):
        if self._retriever is None:
            self._retriever = HybridRetriever(
                index_dir=self.settings.index_dir, chunks_path=self.settings.chunks_path
            )
        return self._retriever

    @property
    def reranker(self):
        if self._reranker is None and self.settings.use_reranker:
            self._reranker = Reranker(model_name=self.settings.reranker_model)
        return self._reranker

    @property
    def provider(self):
        if self._provider is None:
            self._provider = get_provider(self.settings)
        return self._provider

    def is_loaded(self) -> dict:
        """For the readiness endpoint — reports what's actually loaded in
        memory right now, without forcing a load."""
        return {
            "retriever_loaded": self._retriever is not None,
            "reranker_loaded": self._reranker is not None,
            "provider_configured": self._provider is not None,
        }

    def answer(self, question: str, top_k: int | None = None, use_reranker: bool | None = None,
               skip_disambiguation: bool = False):
        """Demo/research mode — retrieves from the configured corpus."""
        t0 = time.monotonic()
        pool = self.retriever.search(question, k=self.settings.retrieval_pool_size, pool_size=100)
        retrieval_ms = (time.monotonic() - t0) * 1000
        return self.answer_from_pool(question, pool, retrieval_ms, top_k, use_reranker, skip_disambiguation)

    def answer_from_retriever(self, question: str, retriever, top_k: int | None = None,
                               use_reranker: bool | None = None, skip_disambiguation: bool = False):
        """Upload mode — same rerank/generate/format logic as answer(), but
        against a CALLER-SUPPLIED retriever (a session-scoped HybridRetriever
        from app/upload/retriever_cache.py) instead of the configured
        demo/research one. This is the actual session-isolation mechanism:
        there is no code path from here back into self.retriever."""
        t0 = time.monotonic()
        pool = retriever.search(question, k=self.settings.retrieval_pool_size, pool_size=100)
        retrieval_ms = (time.monotonic() - t0) * 1000
        return self.answer_from_pool(question, pool, retrieval_ms, top_k, use_reranker, skip_disambiguation)

    def answer_from_pool(self, question: str, pool: list, retrieval_ms: float,
                          top_k: int | None = None, use_reranker: bool | None = None,
                          skip_disambiguation: bool = False):
        top_k = top_k or self.settings.top_k
        use_rerank = self.settings.use_reranker if use_reranker is None else use_reranker

        rerank_ms = None
        if use_rerank and pool:
            t0 = time.monotonic()
            pool, rerank_ms = self.reranker.rerank(question, pool)
        results = pool[:top_k]

        if not results:
            return {
                "answer": "I don't have any relevant documents to answer that question.",
                "sources": [],
                "has_sufficient_evidence": False,
                "retrieval_latency_ms": retrieval_ms,
                "rerank_latency_ms": rerank_ms,
                "generation_latency_ms": None,
                "route": "insufficient_evidence",
                "clarification_question": None,
            }

        # skip_disambiguation=True on a clarification-resume call: the user
        # already named a specific document, so don't ask again even if the
        # resolved question still superficially matches more than one.
        candidates = [] if skip_disambiguation else cluster_candidate_documents(pool)
        prompt_candidates = format_candidates(candidates) if len(candidates) >= 2 else None

        try:
            answer, generation_ms = self.provider.generate(question, results, candidates=prompt_candidates)
        except GenerationError as exc:
            return {
                "answer": None,
                # No answer to ground against yet — falls back to
                # question-only excerpt matching (see app/excerpt.py).
                "sources": self._format_sources(results, question, answer=None),
                "has_sufficient_evidence": None,
                "retrieval_latency_ms": retrieval_ms,
                "rerank_latency_ms": rerank_ms,
                "generation_latency_ms": None,
                "error": str(exc),
                "route": "insufficient_evidence",
                "clarification_question": None,
            }

        if answer.strip().startswith(CLARIFICATION_PREFIX):
            clarification_question = answer.strip()[len(CLARIFICATION_PREFIX):].strip()
            return {
                "answer": None,
                "sources": self._format_sources(results, question, answer=None),
                "has_sufficient_evidence": None,
                "retrieval_latency_ms": retrieval_ms,
                "rerank_latency_ms": rerank_ms,
                "generation_latency_ms": generation_ms,
                "route": "clarify",
                "clarification_question": clarification_question,
            }

        # Reuses the model's own grounded "I don't know" response (see
        # module docstring) rather than a fabricated confidence score.
        has_evidence = "i don't know" not in answer.strip().lower()

        return {
            "answer": answer,
            "sources": self._format_sources(results, question, answer=answer),
            "has_sufficient_evidence": has_evidence,
            "retrieval_latency_ms": retrieval_ms,
            "rerank_latency_ms": rerank_ms,
            "generation_latency_ms": generation_ms,
            "route": "answer" if has_evidence else "insufficient_evidence",
            "clarification_question": None,
        }

    @staticmethod
    def _format_sources(results, question: str, answer: str | None):
        return [
            {
                "chunk_id": r["chunk_id"],
                "source": r["source"],
                "score": r.get("rerank_score", r.get("score")),
                # Finds the passage that actually supports the answer,
                # rather than truncating from the start of the chunk —
                # see app/excerpt.py for why and how.
                "excerpt": select_excerpt(r["text"], question, answer),
            }
            for r in results
        ]
