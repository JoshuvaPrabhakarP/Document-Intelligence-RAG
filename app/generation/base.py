"""
Abstract generation-provider interface. The API layer and RAG pipeline only
ever talk to this interface — never to Ollama (or any future provider)
directly. This is what makes "swap in a hosting-compatible provider later"
a one-file change instead of a rewrite: add a new class implementing
`generate()`, point GENERATION_PROVIDER at it, done.

Only one concrete provider exists right now (OllamaProvider) — per the
milestone scope, no paid provider is wired in yet.
"""

from abc import ABC, abstractmethod
from typing import Sequence


class GenerationProvider(ABC):
    @abstractmethod
    def generate(self, query: str, context_chunks: Sequence[dict],
                  candidates: Sequence[dict] | None = None) -> tuple[str, float]:
        """
        query: the user's question.
        context_chunks: list of dicts with at least a "text" key (as
            returned by the retrieval pipeline), already filtered/reranked
            — this function does not do any retrieval itself.
        candidates: optional list of distinct candidate DOCUMENTS (not
            chunks) that retrieval found comparably relevant to each other
            — see app/disambiguation.py::cluster_candidate_documents() /
            format_candidates(). None or a single-item list means no real
            ambiguity signal. When len(candidates) >= 2, the provider
            should instruct the model that if these documents could give
            genuinely different answers, it must respond with EXACTLY
            "CLARIFICATION_NEEDED: <question>" instead of a normal answer
            (see scripts/llm_client.py::build_prompt() for the exact
            protocol this implements). The CALLER (app/rag_pipeline.py) is
            responsible for detecting that prefix in the returned
            answer_text and treating it as a clarification request, not
            an answer — this method's return shape does not change.

        Returns (answer_text, latency_ms). Must raise GenerationError (not
        a bare exception) on failure, so the API layer can return a clean
        error response instead of a raw stack trace.
        """
        raise NotImplementedError


class GenerationError(RuntimeError):
    """Raised by a GenerationProvider on timeout, connection failure, or
    any other generation failure — lets the API layer distinguish "the
    model failed to answer" from an unrelated bug."""
