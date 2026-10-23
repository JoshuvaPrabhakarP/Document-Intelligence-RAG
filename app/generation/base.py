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
    def generate(self, query: str, context_chunks: Sequence[dict]) -> tuple[str, float]:
        """
        query: the user's question.
        context_chunks: list of dicts with at least a "text" key (as
            returned by the retrieval pipeline), already filtered/reranked
            — this function does not do any retrieval itself.

        Returns (answer_text, latency_ms). Must raise GenerationError (not
        a bare exception) on failure, so the API layer can return a clean
        error response instead of a raw stack trace.
        """
        raise NotImplementedError


class GenerationError(RuntimeError):
    """Raised by a GenerationProvider on timeout, connection failure, or
    any other generation failure — lets the API layer distinguish "the
    model failed to answer" from an unrelated bug."""
