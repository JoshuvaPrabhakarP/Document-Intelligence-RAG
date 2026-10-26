"""
Shared, metadata-based ambiguity detection used by BOTH Standard RAG
(app/rag_pipeline.py) and the Agent (app/agent/graph.py) — one function,
one behavior, not two independently-tuned heuristics.

Why this exists: Milestone 4's evaluation found a case (demo question
"What was purchased on 03/14/2026?") where two unrelated receipts happen
to match equally well, and the agent silently answered using one of them
with no indication a second, equally valid candidate existed. The fix is
NOT "detect this specific date" or "detect this specific question" — it's
to notice, from the retrieval results themselves, that more than one
DISTINCT DOCUMENT is comparably relevant, and surface that fact so the
answering step (an LLM call, in both systems) can decide whether those
documents would actually give different answers.

Deliberately NOT a numeric confidence/relevance floor on its own: a
reranker's raw logit scale varies a lot by query, and an off-topic
question's top-ranked (but still irrelevant) chunks can score only
slightly apart from each other too (verified directly: "What is the
capital of France?" against the demo corpus has its top two chunks only
~0.4 points apart, same ballpark as a genuinely ambiguous question's
top-two gap). A margin-only check would therefore flag plenty of
off-topic questions as "ambiguous" just as readily as genuinely ambiguous
ones. This module only supplies the CANDIDATE LIST as a signal; the final
call on whether to actually ask for clarification is still made by the
downstream LLM step, which reads the candidates' real content and can
(and must) say "none of these are actually relevant" instead of asking a
pointless clarifying question — the same "let the model decide from
content, don't invent an unvalidated numeric threshold for the real
decision" principle app/rag_pipeline.py's module docstring already
documents for insufficient-evidence detection.
"""

import re

_PAGE_SUFFIX_RE = re.compile(r"_p(\d+)$")

# How close (in rerank_score / RRF-score units) another document's best
# chunk must be to the top result's to count as "comparably relevant."
# Calibrated against real rerank scores (BAAI/bge-reranker-large) on this
# project's demo corpus: genuinely ambiguous cases (two receipts sharing a
# date, or a bare "what is the tax?") show top-two gaps of ~0.2-2.5;
# clearly-unambiguous cases (the question names the vendor) show gaps of
# ~7-8. 3.0 sits between those two clusters. This is a signal threshold,
# not a relevance floor -- see module docstring.
DEFAULT_MARGIN = 3.0

# Only consider this many distinct documents at most, even if more are
# within margin -- a clarifying question listing 8 choices isn't useful,
# and the LLM step's context window shouldn't be spent on long tails.
MAX_CANDIDATES = 4


def _document_key(chunk_id: str) -> str:
    """Upload-mode chunk_ids are "{file_id}_p{page_number}" (see
    app/main.py's upload handler) -- multiple pages of the SAME uploaded
    file must count as ONE candidate document, not several. Demo-mode
    chunk_ids (e.g. "demo_000") have no such suffix and already identify
    one document each."""
    return _PAGE_SUFFIX_RE.sub("", chunk_id)


def _page_number(chunk_id: str) -> int | None:
    m = _PAGE_SUFFIX_RE.search(chunk_id)
    return int(m.group(1)) if m else None


def _score_of(result: dict) -> float:
    return result.get("rerank_score", result.get("score", 0.0)) or 0.0


def cluster_candidate_documents(pool: list[dict], margin: float = DEFAULT_MARGIN) -> list[dict]:
    """
    pool: search results (reranked if available), each with at least
    'chunk_id', 'source', 'text', and a score field ('rerank_score' if
    reranked, else 'score').

    Groups by DOCUMENT identity (see _document_key), keeping each
    document's best-scoring chunk. Returns the distinct documents whose
    best score is within `margin` of the top document's score, sorted
    best-first, capped at MAX_CANDIDATES.

    Returns a list of length 0 or 1 when there is no real ambiguity (no
    results, or one document's best chunk clearly dominates). Returns a
    list of length >= 2 when multiple distinct documents are comparably
    relevant -- the actual decision about whether to treat that as
    "ambiguous enough to ask about" is NOT made here (see module
    docstring).
    """
    if not pool:
        return []

    best_per_doc: dict[str, dict] = {}
    for r in pool:
        key = _document_key(r["chunk_id"])
        if key not in best_per_doc or _score_of(r) > _score_of(best_per_doc[key]):
            best_per_doc[key] = r

    ranked = sorted(best_per_doc.values(), key=_score_of, reverse=True)
    if not ranked:
        return []

    top_score = _score_of(ranked[0])
    candidates = [r for r in ranked if top_score - _score_of(r) <= margin]
    return candidates[:MAX_CANDIDATES]


def _distinguishes(candidates: list[dict]) -> bool:
    """True if 'source' actually tells the candidates apart (e.g. upload
    mode, where it's the original filename). False when every candidate
    shares the same value -- the demo/research corpus sets 'source' to a
    fixed dataset identifier ("synthetic_demo") for every document, not a
    per-document name, so using it as a label there would present the
    model (and, if ever shown directly, the user) with N identical
    choices -- not a hint, just noise that looks like a broken signal.
    Trivially true for fewer than 2 candidates -- there's nothing to
    distinguish from, so 'source' is as good a label as any."""
    return len(candidates) < 2 or len({c["source"] for c in candidates}) > 1


def format_candidates(candidates: list[dict]) -> list[dict]:
    """Human-readable choices for a clarifying question -- vendor/source,
    filename, and page where applicable (per the explicit "vendor,
    filename, and page" requirement). Pass the result to a prompt, not to
    the user directly -- callers build the actual clarifying question text.

    Falls back to the first line of the chunk's own text as the label when
    'source' doesn't distinguish the candidates (see _distinguishes()) --
    a receipt/invoice's own first line is reliably its vendor/business
    name by construction of the document type, for ANY corpus (demo,
    research, or upload), not a lookup keyed to a specific question or a
    privileged "ground truth" field that only exists for synthetic data."""
    use_source = _distinguishes(candidates)
    out = []
    for c in candidates:
        page = _page_number(c["chunk_id"])
        base = c["source"] if use_source else c.get("text", "").strip().splitlines()[0][:60]
        label = base + (f" (page {page})" if page else "")
        out.append({"source": c["source"], "chunk_id": c["chunk_id"], "label": label})
    return out
