"""
The two tools exposed to the model: search_documents and fetch_source.
Both reuse the EXISTING, unmodified retrieval code (scripts/fusion.py,
scripts/metadata_filter.py, app/upload/retriever_cache.py) — no retrieval
logic is reimplemented here, only the tool-calling glue.

Session binding: the JSON schemas handed to the model (TOOL_SCHEMAS) only
ever expose `query`/`vendor_filter`/`passage_id` — there is no session_id
parameter in what the model can see or set. The actual Python functions
receive session_mode/session_id as explicit arguments from the GRAPH STATE
(set once, server-side, at graph invocation — see state.py), never from
parsed model output. A model cannot ask for another session's documents
because it has no way to even name one.
"""

import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

from fusion import HybridRetriever  # noqa: E402
from metadata_filter import MetadataRetriever  # noqa: E402
from rerank import Reranker  # noqa: E402

from app.config import get_settings
from app.upload import retriever_cache as upload_retriever_cache
from app.upload.session import sessions as upload_sessions

TOOL_RESULT_K = 5  # how many passages the model sees per search_documents call

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": (
                "Search the document corpus for passages relevant to a query. "
                "Returns a list of passages, each with a passage_id you can "
                "pass to fetch_source for the full text."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for"},
                    "vendor_filter": {
                        "type": "string",
                        "description": "Optional: restrict results to a specific vendor/business name, if known.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_source",
            "description": (
                "Fetch the full text of a specific passage you already saw in "
                "a search_documents result, by its passage_id."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "passage_id": {"type": "string", "description": "A passage_id from a prior search_documents result"},
                },
                "required": ["passage_id"],
            },
        },
    },
]

TOOL_NAMES = {t["function"]["name"] for t in TOOL_SCHEMAS}


class ToolArgumentError(ValueError):
    """Malformed or missing arguments from the model's tool call — handled
    as a normal tool-error result fed back to the model, not a crash."""


_demo_retriever = None
_demo_metadata_retriever = None
_reranker = None


def _get_demo_retriever():
    global _demo_retriever
    if _demo_retriever is None:
        # Must match app/rag_pipeline.py's retriever construction exactly —
        # HybridRetriever() with no args defaults to the RESEARCH corpus
        # paths (data/index, data/chunks/chunks.jsonl), not the app's
        # configured demo corpus. Passing settings explicitly is what keeps
        # demo-mode agent search_documents() consistent with the baseline
        # /query endpoint's demo results.
        settings = get_settings()
        _demo_retriever = HybridRetriever(index_dir=settings.index_dir, chunks_path=settings.chunks_path)
    return _demo_retriever


def _get_demo_metadata_retriever():
    global _demo_metadata_retriever
    if _demo_metadata_retriever is None:
        # Same corpus-path bug as _get_demo_retriever() above: must be
        # pointed at the app's configured demo corpus, not MetadataRetriever's
        # own defaults (which are the research paths).
        settings = get_settings()
        _demo_metadata_retriever = MetadataRetriever(
            metadata_source="ground_truth", index_dir=settings.index_dir, chunks_path=settings.chunks_path
        )
    return _demo_metadata_retriever


def _get_reranker():
    global _reranker
    if _reranker is None:
        settings = get_settings()
        _reranker = Reranker(model_name=settings.reranker_model)
    return _reranker


def _rerank_and_cut(query: str, pool: list[dict]) -> list[dict]:
    """Mirrors app/rag_pipeline.py's retrieve-a-large-pool-then-rerank
    pattern exactly, so Standard RAG and the agent's search_documents tool
    use COMPARABLE retrieval settings (Milestone 4 requirement) — reranking
    was reused in name only before this fix: the tool asked the retriever
    for a final k=5 directly and never reranked at all."""
    settings = get_settings()
    if settings.use_reranker and pool:
        pool, _ = _get_reranker().rerank(query, pool)
    return pool[:TOOL_RESULT_K]


def execute_search_documents(args: dict, *, session_mode: str, session_id: str | None) -> list[dict]:
    query = args.get("query")
    if not query or not isinstance(query, str):
        raise ToolArgumentError("search_documents requires a non-empty 'query' string.")
    vendor_filter = args.get("vendor_filter")
    settings = get_settings()

    if session_mode == "upload":
        if not session_id:
            raise ToolArgumentError("No upload session is active.")

        retriever = upload_retriever_cache.get_retriever(
            session_id, str(upload_sessions.index_dir(session_id)), str(upload_sessions.chunks_path(session_id))
        )
        # k=settings.retrieval_pool_size (a large fused pool), pool_size=100 —
        # same shape as rag_pipeline.py's answer_from_retriever(), so reranking
        # below has the same amount of material to work with as Standard RAG.
        pool = retriever.search(query, k=settings.retrieval_pool_size, pool_size=100)
        # Upload-mode chunks carry no metadata (no extraction model runs on
        # uploads, per Milestone 2's documented scope) — a vendor_filter
        # here has nothing to filter against, so it's accepted (the model
        # can still pass it) but has no effect, not an error.
    else:
        if vendor_filter:
            # Vendor-filtered search is pre-narrowed by exact metadata match,
            # not a broad ranking problem — reranking a single vendor's
            # already-small result set adds latency without changing the
            # outcome, so this path intentionally skips the rerank step
            # (same reasoning Standard RAG would apply if it had this filter).
            return [
                {"chunk_id": r["chunk_id"], "source": r["source"], "text": r["text"], "score": r.get("score")}
                for r in _get_demo_metadata_retriever().search(query, {"vendor": vendor_filter}, base="hybrid", k=TOOL_RESULT_K)
            ]
        pool = _get_demo_retriever().search(query, k=settings.retrieval_pool_size, pool_size=100)

    results = _rerank_and_cut(query, pool)
    return [
        {"chunk_id": r["chunk_id"], "source": r["source"], "text": r["text"],
         "score": r.get("rerank_score", r.get("score"))}
        for r in results
    ]


def execute_fetch_source(args: dict, *, retrieved_passages: dict) -> dict:
    passage_id = args.get("passage_id")
    if not passage_id or not isinstance(passage_id, str):
        raise ToolArgumentError("fetch_source requires a non-empty 'passage_id' string.")

    # Scoped lookup: only passages THIS run's search_documents calls have
    # already surfaced are fetchable — not an unscoped global chunk lookup,
    # and definitionally never another session's data, since
    # retrieved_passages only ever contains this run's own search results.
    passage = retrieved_passages.get(passage_id)
    if passage is None:
        raise ToolArgumentError(
            f"passage_id {passage_id!r} was not found among previously retrieved passages. "
            f"Use search_documents first, or check the ID."
        )
    return passage


def new_passage_id() -> str:
    return uuid.uuid4().hex[:10]
